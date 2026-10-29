"""CityFix core: validation, inference, routing policy, SQLite persistence, hotspots, summary facts.

Every report read/update/aggregate is scoped by city_id.
The model predicts shared categories; the city config picks the department.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config" / "cities.json"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
META_PATH = ROOT / "artifacts" / "model_metadata.json"
DB_PATH = ROOT / "data" / "cityfix.db"
TRAIN_CSV = ROOT / "data" / "gpt_train.csv"
VALIDATION_CSV = ROOT / "data" / "gpt_validation.csv"
TEST_CSV = ROOT / "data" / "gpt_tel_aviv_test.csv"
DEMO_CSV = ROOT / "data" / "gpt_tel_aviv_demo.csv"
PROVENANCE_PATH = ROOT / "data" / "gpt_data_provenance.json"

MAX_TEXT = 1000
EARTH_RADIUS_M = 6_371_000
URGENT_NOTICE = ("This demo is not monitored for emergencies; contact the appropriate emergency "
                 "service if there is immediate danger.")
# Demonstration phrase guard only - NOT reliable emergency detection.
URGENT_PATTERNS = [
    r"\b(exposed|live|bare|sparking|hanging|fallen|downed)\s+(live\s+)?(electric(al)?\s+)?(wire|wires|cable|cables)\b",
    r"\bgas\s+(leak|smell)\b", r"\bsmell(s)?\s+of\s+gas\b",
    r"\b(fire|on fire|burning)\b.*\b(building|car|bin|pole)\b",
    r"\bsomeone\s+(is\s+)?(injured|hurt|trapped|unconscious)\b",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- config / model
def load_cities(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def load_model(model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
    model = joblib.load(model_path)
    meta = json.loads(Path(meta_path).read_text()) if Path(meta_path).exists() else {}
    return model, meta


def thresholds_for(city_cfg: dict, meta: dict) -> tuple[float, float]:
    """Thresholds validated with this model (metadata) take precedence over city defaults."""
    t = (meta or {}).get("thresholds", {})
    return float(t.get("min_score", city_cfg["min_score"])), float(t.get("min_margin", city_cfg["min_margin"]))


# ---------------------------------------------------------------- validation
def validate_text(text) -> tuple[bool, str, str]:
    t = (text or "").strip()
    if not t:
        return False, "", "Please enter a complaint (1-1,000 characters)."
    if len(t) > MAX_TEXT:
        return False, t, f"Complaint is {len(t)} characters; the limit is {MAX_TEXT}."
    return True, t, ""


def location_status(lat, lon, city_cfg: dict) -> str:
    if lat is None and lon is None:
        return "missing"
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return "invalid"
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return "invalid"
    a, b, c, d = city_cfg["demo_bbox"]
    return "valid" if (a <= lat <= c and b <= lon <= d) else "outside_bbox"


# ---------------------------------------------------------------- inference + policy
def classify(model, text: str) -> dict:
    vec = model.named_steps["tfidf"].transform([text])
    known_terms = int(vec.nnz)
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    order = np.argsort(probs)[::-1]
    top, second = probs[order[0]], probs[order[1]] if len(order) > 1 else 0.0
    return {"category": classes[order[0]], "score": float(top), "margin": float(top - second),
            "known_terms": known_terms, "probs": {c: float(p) for c, p in zip(classes, probs)}}


def is_urgent(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in URGENT_PATTERNS)


def route(text: str, city_cfg: dict, model, meta: dict | None = None, unsure: bool = False) -> dict:
    """Apply model + policy. Returns prediction, routing state, department and reasons."""
    ok, text, err = validate_text(text)
    if not ok:
        raise ValueError(err)
    min_score, min_margin = thresholds_for(city_cfg, meta or {})
    pred = classify(model, text)
    reasons = []
    if is_urgent(text):
        state = "urgent_review"
        reasons.append("urgent_phrase_guard")
    else:
        if pred["known_terms"] == 0:
            reasons.append("no_known_vocabulary")
        if pred["score"] < min_score:
            reasons.append(f"low_score<{min_score}")
        if pred["margin"] < min_margin:
            reasons.append(f"low_margin<{min_margin}")
        if pred["category"] not in city_cfg["departments"]:
            reasons.append("no_department_mapping")
        if unsure:
            reasons.append("citizen_marked_multiple_or_unsure")
        state = "review_required" if reasons else "routed"
    dept = city_cfg["departments"][pred["category"]] if state == "routed" else city_cfg["review_department"]
    return {"predicted_category": pred["category"], "score": pred["score"], "margin": pred["margin"],
            "known_terms": pred["known_terms"], "probs": pred["probs"], "routing_state": state,
            "proposed_department": dept, "reasons": reasons,
            "approved_category": pred["category"] if state == "routed" else None,
            "notice": URGENT_NOTICE if state == "urgent_review" else None}


# ---------------------------------------------------------------- SQLite
SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    city_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    latitude REAL, longitude REAL,
    location_status TEXT NOT NULL,
    predicted_category TEXT, predicted_score REAL, predicted_margin REAL,
    approved_category TEXT,
    category_source TEXT NOT NULL,
    proposed_department TEXT NOT NULL,
    routing_state TEXT NOT NULL,
    routing_reasons TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    data_version TEXT, model_version TEXT, config_version TEXT,
    dedupe_key TEXT UNIQUE,
    attachment_path TEXT
);
CREATE INDEX IF NOT EXISTS ix_reports_city_time ON reports(city_id, created_at_utc);
CREATE TABLE IF NOT EXISTS corrections (
    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL, city_id TEXT NOT NULL,
    old_category TEXT, new_category TEXT NOT NULL,
    old_department TEXT, new_department TEXT NOT NULL,
    operator TEXT NOT NULL, corrected_at_utc TEXT NOT NULL,
    FOREIGN KEY(report_id) REFERENCES reports(report_id)
);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(reports)")}
    if "attachment_path" not in cols:  # migrate databases created before attachments existed
        conn.execute("ALTER TABLE reports ADD COLUMN attachment_path TEXT")
        conn.commit()
    return conn


# ---------------------------------------------------------------- attachments (stored, NOT analyzed)
UPLOAD_DIR = ROOT / "data" / "uploads"
ALLOWED_ATTACHMENTS = {"jpg", "jpeg", "png", "webp", "mp4", "mov"}
MAX_ATTACHMENT_MB = 10


def save_attachment(conn, city_id: str, report_id: str, filename: str, data: bytes,
                    upload_dir: Path | None = None) -> str:
    """Store a photo/clip for staff review. Never used for routing or sent to the LLM."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_ATTACHMENTS:
        raise ValueError(f"File type .{ext} not allowed ({', '.join(sorted(ALLOWED_ATTACHMENTS))}).")
    if len(data) > MAX_ATTACHMENT_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_ATTACHMENT_MB} MB.")
    if get_report(conn, city_id, report_id) is None:
        raise KeyError(f"Report {report_id} not found in {city_id}")
    folder = Path(upload_dir or UPLOAD_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{report_id}.{ext}"  # server-chosen name; the original filename is not used
    path.write_bytes(data)
    conn.execute("UPDATE reports SET attachment_path=? WHERE city_id=? AND report_id=?",
                 (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), city_id, report_id))
    conn.commit()
    return str(path)


def import_seed(conn, csv_path: Path = DEMO_CSV, city_cfg: dict | None = None, data_version: str = "") -> int:
    """Idempotent: INSERT OR IGNORE on report_id. Returns rows newly inserted.
    Fixture rows from a different fixture file version (older demo data) are removed first, together with
    their corrections, so the map always shows exactly the current supplied fixtures. Citizen reports are kept."""
    df = pd.read_csv(csv_path, dtype={"report_id": str, "city_id": str, "category": str, "created_at_utc": str})
    data_version = data_version or hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:16]
    if city_cfg:
        stale = [r[0] for r in conn.execute(
            "SELECT report_id FROM reports WHERE city_id=? AND dedupe_key LIKE 'seed:%' "
            "AND IFNULL(data_version,'') != ?", (city_cfg["city_id"], data_version))]
        for rid in stale:
            conn.execute("DELETE FROM corrections WHERE city_id=? AND report_id=?", (city_cfg["city_id"], rid))
            conn.execute("DELETE FROM reports WHERE city_id=? AND report_id=?", (city_cfg["city_id"], rid))
    inserted = 0
    for r in df.itertuples(index=False):
        cfg = city_cfg if city_cfg and city_cfg["city_id"] == r.city_id else None
        if cfg is None:
            continue
        cur = conn.execute(
            """INSERT OR IGNORE INTO reports (report_id, city_id, text, created_at_utc, latitude, longitude,
               location_status, approved_category, category_source, proposed_department, routing_state,
               routing_reasons, status, data_version, config_version, dedupe_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r.report_id, r.city_id, r.text, r.created_at_utc, float(r.latitude), float(r.longitude),
             location_status(r.latitude, r.longitude, cfg), r.category, "seed_fixture",
             cfg["departments"][r.category], "routed", "[]", "open", data_version, cfg["config_version"],
             "seed:" + r.report_id))
        inserted += cur.rowcount
    conn.commit()
    return inserted


def make_dedupe_key(city_id, text, lat, lon, when: datetime) -> str:
    norm_text = re.sub(r"\s+", " ", text.strip().lower())
    raw = f"{city_id}|{norm_text}|{lat}|{lon}|{when.strftime('%Y-%m-%d')}"
    return hashlib.sha256(raw.encode()).hexdigest()


def save_report(conn, city_cfg: dict, text: str, lat, lon, routing: dict, meta: dict | None = None,
                now: datetime | None = None) -> tuple[str, bool]:
    """Save a citizen report. Same city+text+location on the same UTC day is treated as a duplicate.
    Returns (report_id, is_new)."""
    ok, text, err = validate_text(text)
    if not ok:
        raise ValueError(err)
    now = now or utcnow()
    loc = location_status(lat, lon, city_cfg)
    if loc in ("missing", "invalid"):
        lat_v = lon_v = None
    else:
        lat_v, lon_v = float(lat), float(lon)
    key = make_dedupe_key(city_cfg["city_id"], text, lat_v, lon_v, now)
    existing = conn.execute("SELECT report_id FROM reports WHERE dedupe_key=? AND city_id=?",
                            (key, city_cfg["city_id"])).fetchone()
    if existing:
        return existing["report_id"], False
    report_id = "CF-" + key[:8].upper()
    meta = meta or {}
    conn.execute(
        """INSERT INTO reports (report_id, city_id, text, created_at_utc, latitude, longitude, location_status,
           predicted_category, predicted_score, predicted_margin, approved_category, category_source,
           proposed_department, routing_state, routing_reasons, status, data_version, model_version,
           config_version, dedupe_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (report_id, city_cfg["city_id"], text, iso(now), lat_v, lon_v, loc,
         routing["predicted_category"], routing["score"], routing["margin"], routing["approved_category"],
         "model" if routing["routing_state"] == "routed" else "pending_review",
         routing["proposed_department"], routing["routing_state"], json.dumps(routing["reasons"]), "open",
         meta.get("data_hash", ""), meta.get("model_version", ""), city_cfg["config_version"], key))
    conn.commit()
    return report_id, True


def get_report(conn, city_id: str, report_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM reports WHERE city_id=? AND report_id=?", (city_id, report_id)).fetchone()
    return dict(row) if row else None


def apply_correction(conn, city_cfg: dict, report_id: str, new_category: str, operator: str,
                     now: datetime | None = None) -> dict:
    """Operator approves/corrects a category. Original prediction is preserved; department recalculated
    from config; audit row written."""
    if new_category not in city_cfg["departments"]:
        raise ValueError(f"No department mapping for '{new_category}' in {city_cfg['city_id']}")
    operator = (operator or "").strip() or "demo-operator"
    rep = get_report(conn, city_cfg["city_id"], report_id)
    if rep is None:
        raise KeyError(f"Report {report_id} not found in {city_cfg['city_id']}")
    new_dept = city_cfg["departments"][new_category]
    now = now or utcnow()
    reasons = json.loads(rep["routing_reasons"] or "[]") + ["operator_correction"]
    conn.execute(
        """UPDATE reports SET approved_category=?, category_source='operator_correction', proposed_department=?,
           routing_state='routed', routing_reasons=?, config_version=? WHERE city_id=? AND report_id=?""",
        (new_category, new_dept, json.dumps(reasons), city_cfg["config_version"], city_cfg["city_id"], report_id))
    conn.execute(
        """INSERT INTO corrections (report_id, city_id, old_category, new_category, old_department, new_department,
           operator, corrected_at_utc) VALUES (?,?,?,?,?,?,?,?)""",
        (report_id, city_cfg["city_id"], rep["approved_category"], new_category, rep["proposed_department"],
         new_dept, operator, iso(now)))
    conn.commit()
    return get_report(conn, city_cfg["city_id"], report_id)


def fetch_reports(conn, city_id: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    sql, params = "SELECT * FROM reports WHERE city_id=?", [city_id]
    if start is not None:
        sql += " AND created_at_utc >= ?"; params.append(iso(start))
    if end is not None:
        sql += " AND created_at_utc <= ?"; params.append(iso(end))
    return pd.read_sql_query(sql + " ORDER BY created_at_utc DESC", conn, params=params)


def fetch_corrections(conn, city_id: str) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM corrections WHERE city_id=? ORDER BY correction_id DESC",
                             conn, params=[city_id])


# ---------------------------------------------------------------- hotspots
def compute_hotspots(df: pd.DataFrame, city_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """df must already be scoped to one city and the window. Returns (hotspots, exclusions)."""
    hs = city_cfg["hotspots"]
    cols = ["hotspot_id", "category", "department", "report_count", "centroid_lat", "centroid_lon",
            "first_utc", "last_utc", "report_ids"]
    excl = {"not_open_routed": 0, "no_approved_category": 0, "missing_location": 0, "outside_or_invalid": 0}
    if df.empty:
        return pd.DataFrame(columns=cols), excl
    d = df[df["city_id"] == city_cfg["city_id"]].drop_duplicates("report_id")
    mask_status = (d["status"] == "open") & (d["routing_state"] == "routed")
    excl["not_open_routed"] = int((~mask_status).sum())
    d = d[mask_status]
    mask_cat = d["approved_category"].notna()
    excl["no_approved_category"] = int((~mask_cat).sum())
    d = d[mask_cat]
    missing = d["latitude"].isna() | d["longitude"].isna()
    excl["missing_location"] = int(missing.sum())
    d = d[~missing]
    lat, lon = d["latitude"].astype(float), d["longitude"].astype(float)
    a, b, c, e = city_cfg["demo_bbox"]
    inside = np.isfinite(lat) & np.isfinite(lon) & lat.between(a, c) & lon.between(b, e)
    excl["outside_or_invalid"] = int((~inside).sum())
    d = d[inside]
    out = []
    for cat, group in d.groupby("approved_category"):
        if len(group) < hs["min_samples"]:
            continue
        labels = DBSCAN(eps=hs["eps_meters"] / EARTH_RADIUS_M, min_samples=hs["min_samples"],
                        metric="haversine", algorithm="ball_tree").fit_predict(
            np.radians(group[["latitude", "longitude"]].to_numpy(dtype=float)))
        for lab in sorted(set(labels) - {-1}):
            m = group[labels == lab]
            out.append({"category": cat, "department": city_cfg["departments"].get(cat, city_cfg["review_department"]),
                        "report_count": len(m), "centroid_lat": round(m["latitude"].mean(), 5),
                        "centroid_lon": round(m["longitude"].mean(), 5), "first_utc": m["created_at_utc"].min(),
                        "last_utc": m["created_at_utc"].max(), "report_ids": ",".join(sorted(m["report_id"]))})
    res = pd.DataFrame(out, columns=[c for c in cols if c != "hotspot_id"])
    if not res.empty:
        res = res.sort_values(["report_count", "category"], ascending=[False, True]).reset_index(drop=True)
    res.insert(0, "hotspot_id", [f"H{i+1}" for i in range(len(res))])
    return res, excl


# ---------------------------------------------------------------- summary facts (for UI + LLM)
def summary_facts(df: pd.DataFrame, hotspots: pd.DataFrame, city_cfg: dict, start: datetime, end: datetime) -> list[dict]:
    """Python-computed facts with stable IDs. The LLM may only reference these IDs."""
    facts = [
        {"id": "F1", "fact": "City", "value": city_cfg["city_name"]},
        {"id": "F2", "fact": "Window (UTC)", "value": f"{iso(start)} to {iso(end)}"},
        {"id": "F3", "fact": "Data", "value": "Simulated demo reports (not real municipal data)"},
        {"id": "F4", "fact": "Reports in window", "value": int(len(df))},
        {"id": "F5", "fact": "Routed", "value": int((df["routing_state"] == "routed").sum()) if len(df) else 0},
        {"id": "F6", "fact": "Awaiting staff review", "value": int((df["routing_state"] == "review_required").sum()) if len(df) else 0},
        {"id": "F7", "fact": "Urgent review", "value": int((df["routing_state"] == "urgent_review").sum()) if len(df) else 0},
        {"id": "F8", "fact": "Hotspots found", "value": int(len(hotspots))},
    ]
    n = 9
    if len(df):
        for cat, cnt in df["approved_category"].fillna("unassigned").value_counts().items():
            facts.append({"id": f"F{n}", "fact": f"Reports with category '{cat}'", "value": int(cnt)}); n += 1
        for dept, cnt in df["proposed_department"].value_counts().items():
            facts.append({"id": f"F{n}", "fact": f"Queue: {dept}", "value": int(cnt)}); n += 1
    for h in hotspots.itertuples(index=False):
        facts.append({"id": h.hotspot_id, "fact": f"Hotspot: {h.category} cluster -> {h.department}",
                      "value": f"{h.report_count} reports near {h.centroid_lat},{h.centroid_lon}; "
                               f"{h.first_utc} to {h.last_utc}"})
    return facts


def demo_anchor() -> datetime:
    try:
        return parse_iso(json.loads(PROVENANCE_PATH.read_text())["demo_anchor_utc"])
    except Exception:
        return utcnow()
