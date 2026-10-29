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
CREATE TABLE IF NOT EXISTS status_updates (
    update_id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL, city_id TEXT NOT NULL,
    old_status TEXT, new_status TEXT NOT NULL,
    resident_note TEXT, operator TEXT NOT NULL, updated_at_utc TEXT NOT NULL,
    FOREIGN KEY(report_id) REFERENCES reports(report_id)
);
"""

# Case lifecycle (separate from routing_state). "Active" cases count in queues and hotspots.
CASE_STATUSES = {"open": "Received", "in_progress": "In progress", "resolved": "Resolved",
                 "closed": "Closed – not actionable"}
ACTIVE_STATUSES = ("open", "in_progress")
MAX_NOTE = 300


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


def is_simulated(row) -> bool:
    """Fixture (simulated history) vs live report classified by the model."""
    return str(row.get("dedupe_key") or "").startswith("seed:")


def update_status(conn, city_id: str, report_id: str, new_status: str, resident_note: str, operator: str,
                  now: datetime | None = None) -> dict:
    """Staff changes the case status. The note is shown to the resident on 'Check my report'."""
    if new_status not in CASE_STATUSES:
        raise ValueError(f"Unknown status '{new_status}'")
    note = (resident_note or "").strip()
    if len(note) > MAX_NOTE:
        raise ValueError(f"Note is longer than {MAX_NOTE} characters.")
    rep = get_report(conn, city_id, report_id)
    if rep is None:
        raise KeyError(f"Report {report_id} not found in {city_id}")
    conn.execute("UPDATE reports SET status=? WHERE city_id=? AND report_id=?", (new_status, city_id, report_id))
    conn.execute("""INSERT INTO status_updates (report_id, city_id, old_status, new_status, resident_note, operator,
                    updated_at_utc) VALUES (?,?,?,?,?,?,?)""",
                 (report_id, city_id, rep["status"], new_status, note, (operator or "").strip() or "demo-operator",
                  iso(now or utcnow())))
    conn.commit()
    return get_report(conn, city_id, report_id)


SPAM = "spam"
SPAM_DEPARTMENT = "None – spam / not a city issue"
SPAM_NOTE = "This message isn't a street issue the city handles, so it has been closed."


def staff_decision(conn, city_cfg: dict, report_id: str, category: str, status: str, note: str, operator: str,
                   now: datetime | None = None) -> tuple[dict, list[str]]:
    """One save step for staff: confirm/correct the category (or mark spam) and update the case status.
    Spam is not routed to any department and always closes the case. Returns (report, actions taken)."""
    city_id, actions = city_cfg["city_id"], []
    rep = get_report(conn, city_id, report_id)
    if rep is None:
        raise KeyError(f"Report {report_id} not found in {city_id}")
    operator = (operator or "").strip() or "demo-operator"
    now = now or utcnow()
    if category == SPAM:
        if rep["approved_category"] != SPAM:
            reasons = json.loads(rep["routing_reasons"] or "[]") + ["operator_marked_spam"]
            conn.execute("""UPDATE reports SET approved_category=?, category_source='operator_correction',
                            proposed_department=?, routing_state='spam', routing_reasons=? WHERE city_id=? AND report_id=?""",
                         (SPAM, SPAM_DEPARTMENT, json.dumps(reasons), city_id, report_id))
            conn.execute("""INSERT INTO corrections (report_id, city_id, old_category, new_category, old_department,
                            new_department, operator, corrected_at_utc) VALUES (?,?,?,?,?,?,?,?)""",
                         (report_id, city_id, rep["approved_category"], SPAM, rep["proposed_department"],
                          SPAM_DEPARTMENT, operator, iso(now)))
            conn.commit()
            actions.append("marked as spam")
        status, note = "closed", (note or "").strip() or SPAM_NOTE
    elif category != rep["approved_category"] or rep["routing_state"] != "routed":
        apply_correction(conn, city_cfg, report_id, category, operator, now)
        actions.append(f"assigned to {city_cfg['departments'][category]}")
    rep = get_report(conn, city_id, report_id)
    if status != rep["status"] or (note or "").strip():
        update_status(conn, city_id, report_id, status, note, operator, now)
        actions.append(f"status {CASE_STATUSES.get(status, status)}")
    return get_report(conn, city_id, report_id), actions


def staff_actions(conn, city_id: str, report_id: str | None = None) -> pd.DataFrame:
    """Simplified audit trail: one row per staff action (category decisions + status changes)."""
    corr = fetch_corrections(conn, city_id)
    upd = fetch_status_updates(conn, city_id)
    rows = [{"when (UTC)": c.corrected_at_utc, "report": c.report_id, "by": c.operator,
             "action": ("Marked as spam" if c.new_category == SPAM else f"Assigned to {c.new_department}"),
             "note for resident": "", "_k": (c.corrected_at_utc, c.report_id, 0, c.correction_id)} for c in corr.itertuples()]
    rows += [{"when (UTC)": u.updated_at_utc, "report": u.report_id, "by": u.operator,
              "action": f"Status → {CASE_STATUSES.get(u.new_status, u.new_status)}",
              "note for resident": u.resident_note or "", "_k": (u.updated_at_utc, u.report_id, 1, u.update_id)}
             for u in upd.itertuples()]
    rows.sort(key=lambda r: r["_k"], reverse=True)  # newest first; per report, status after category decision
    out = pd.DataFrame(rows, columns=["when (UTC)", "report", "action", "note for resident", "by", "_k"]).drop(columns="_k")
    if report_id:
        out = out[out["report"] == report_id]
    return out.reset_index(drop=True)


def fetch_status_updates(conn, city_id: str, report_id: str | None = None) -> pd.DataFrame:
    sql, params = "SELECT * FROM status_updates WHERE city_id=?", [city_id]
    if report_id:
        sql += " AND report_id=?"; params.append(report_id)
    return pd.read_sql_query(sql + " ORDER BY update_id", conn, params=params)


def find_by_reference(conn, city_id: str, reference: str) -> dict | None:
    ref = (reference or "").strip()
    if not ref:
        return None
    row = conn.execute("SELECT * FROM reports WHERE city_id=? AND UPPER(report_id)=UPPER(?)", (city_id, ref)).fetchone()
    return dict(row) if row else None


def case_timeline(conn, city_cfg: dict, rep: dict) -> list[dict]:
    """Resident-facing history: received, initial routing, staff decisions, status updates (oldest first)."""
    reasons = [r for r in json.loads(rep.get("routing_reasons") or "[]")
               if r not in ("operator_correction", "operator_marked_spam")]
    events = [{"at": rep["created_at_utc"], "title": "Report received", "detail": ""}]
    if is_simulated(rep):
        events.append({"at": rep["created_at_utc"], "title": "Simulated history",
                       "detail": "Sample report used to fill the map."})
    elif "urgent_phrase_guard" in reasons:
        events.append({"at": rep["created_at_utc"], "title": "Flagged for urgent staff review", "detail": URGENT_NOTICE})
    elif reasons:
        events.append({"at": rep["created_at_utc"], "title": "Waiting for staff review",
                       "detail": "The system was not sure enough to choose a department on its own."})
    else:
        first_dept = city_cfg["departments"].get(rep["predicted_category"], rep["proposed_department"])
        events.append({"at": rep["created_at_utc"], "title": f"Proposed for {first_dept}",
                       "detail": "Matched automatically by the text model."})
    corr = fetch_corrections(conn, city_cfg["city_id"])
    for c in corr[corr["report_id"] == rep["report_id"]].itertuples():
        events.append({"at": c.corrected_at_utc, "detail": "",
                       "title": ("Marked as not a city street issue" if c.new_category == SPAM
                                 else f"Staff assigned it to {c.new_department}")})
    for u in fetch_status_updates(conn, city_cfg["city_id"], rep["report_id"]).itertuples():
        events.append({"at": u.updated_at_utc, "title": f"Status: {CASE_STATUSES.get(u.new_status, u.new_status)}",
                       "detail": u.resident_note or ""})
    return sorted(events, key=lambda e: e["at"])


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
    mask_status = d["status"].isin(ACTIVE_STATUSES) & (d["routing_state"] == "routed")
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
    res.insert(2, "near_area", [nearest_area(la, lo, city_cfg) for la, lo in zip(res["centroid_lat"], res["centroid_lon"])])
    return res, excl


def distance_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def nearest_area(lat, lon, city_cfg: dict, max_m: float = 1500) -> str:
    """Name of the closest configured area point (approximate; not geocoding). '' if none within max_m."""
    if lat is None or lon is None or not city_cfg.get("map_presets"):
        return ""
    name, (plat, plon) = min(city_cfg["map_presets"].items(),
                             key=lambda kv: distance_m(lat, lon, kv[1][0], kv[1][1]))
    return name if distance_m(lat, lon, plat, plon) <= max_m else ""


# ---------------------------------------------------------------- summary facts (for UI + LLM)
def summary_facts(df: pd.DataFrame, hotspots: pd.DataFrame, city_cfg: dict, start: datetime, end: datetime) -> list[dict]:
    """Python-computed facts with stable IDs. The LLM may only reference these IDs."""
    cm = case_metrics(df)
    facts = [
        {"id": "F1", "fact": "City", "value": city_cfg["city_name"]},
        {"id": "F2", "fact": "Window (UTC)", "value": f"{iso(start)} to {iso(end)}"},
        {"id": "F3", "fact": "Data", "value": "Simulated demo history plus reports typed into this demo (not real municipal data)"},
        {"id": "F4", "fact": "Reports in window", "value": int(len(df))},
        {"id": "F5", "fact": "Active cases (received or in progress)", "value": cm["active"]},
        {"id": "F6", "fact": "Awaiting staff review now (incl. urgent)", "value": cm["awaiting_review"]},
        {"id": "F7", "fact": "Urgent review now", "value": cm["urgent_waiting"]},
        {"id": "F8", "fact": "Hotspots found (active, routed reports)", "value": int(len(hotspots))},
        {"id": "F9", "fact": "Previously sent to review, already decided by staff", "value": cm["previously_reviewed"]},
    ]
    n = 10
    if len(df):
        for st_, cnt in df["status"].value_counts().items():
            facts.append({"id": f"F{n}", "fact": f"Case status '{CASE_STATUSES.get(st_, st_)}'", "value": int(cnt)}); n += 1
    if len(df):
        for cat, cnt in df["approved_category"].fillna("unassigned").value_counts().items():
            facts.append({"id": f"F{n}", "fact": f"Reports with category '{cat}'", "value": int(cnt)}); n += 1
        for dept, cnt in df["proposed_department"].value_counts().items():
            facts.append({"id": f"F{n}", "fact": f"Queue: {dept}", "value": int(cnt)}); n += 1
    for h in hotspots.itertuples(index=False):
        area = f" near {h.near_area} (approximate)" if h.near_area else ""
        facts.append({"id": h.hotspot_id, "fact": f"Hotspot: {h.category} cluster{area} -> {h.department}",
                      "value": f"{h.report_count} reports near {h.centroid_lat},{h.centroid_lon}; "
                               f"{h.first_utc} to {h.last_utc}"})
    return facts


def demo_anchor() -> datetime:
    try:
        return parse_iso(json.loads(PROVENANCE_PATH.read_text())["demo_anchor_utc"])
    except Exception:
        return utcnow()


# ---------------------------------------------------------------- shared case metrics (dashboard + brief)
REVIEW_STATES = ("review_required", "urgent_review")
_STAFF_TAGS = ("operator_correction", "operator_marked_spam")


def _initially_reviewed(reasons_json: str) -> bool:
    return any(r not in _STAFF_TAGS for r in json.loads(reasons_json or "[]"))


def case_metrics(df: pd.DataFrame) -> dict:
    """One definition used everywhere:
    active            = status Received or In progress
    awaiting_review   = active AND routing state is review_required/urgent_review (waiting for a staff decision now)
    previously_reviewed = was sent to review on arrival, staff have since decided (no longer waiting)"""
    keys = ("total", "active", "awaiting_review", "urgent_waiting", "previously_reviewed", "routed_by_model",
            "resolved", "closed")
    if not len(df):
        return dict.fromkeys(keys, 0)
    active = df["status"].isin(ACTIVE_STATUSES)
    waiting = active & df["routing_state"].isin(REVIEW_STATES)
    initially = df["routing_reasons"].map(_initially_reviewed)
    return {"total": int(len(df)), "active": int(active.sum()), "awaiting_review": int(waiting.sum()),
            "urgent_waiting": int((waiting & (df["routing_state"] == "urgent_review")).sum()),
            "previously_reviewed": int((initially & ~waiting).sum()),
            "routed_by_model": int((df["category_source"] == "model").sum()),
            "resolved": int((df["status"] == "resolved").sum()), "closed": int((df["status"] == "closed").sum())}


def awaiting_review(df: pd.DataFrame) -> pd.DataFrame:
    if not len(df):
        return df
    return df[df["status"].isin(ACTIVE_STATUSES) & df["routing_state"].isin(REVIEW_STATES)]


INVESTIGATION_STEP = {
    "roads": "Inspect the road surface near {area} and record whether the reports describe one defect or several.",
    "waste": "Check the bins and collection points near {area} and note which ones the reports refer to.",
    "lighting": "Survey the street lights near {area} and list which poles are not working.",
    "noise": "Record where and at what times the noise is reported near {area}, and which premises residents mention.",
}


def priority_hotspots(df: pd.DataFrame, hotspots: pd.DataFrame, city_cfg: dict, top: int = 3) -> list[dict]:
    """Rank hotspots by active report count, ties broken by the oldest active report (older first).
    Only active (received/in progress), routed reports count - resolved and closed cases are excluded.
    This is a workload ranking, not a danger assessment."""
    out = []
    for h in hotspots.itertuples(index=False):
        ids = h.report_ids.split(",")
        sub = df[df["report_id"].isin(ids) & df["status"].isin(ACTIVE_STATUSES)]
        if not len(sub):
            continue
        area = h.near_area or "an unnamed area"
        out.append({"evidence_id": h.hotspot_id, "category": h.category, "area": h.near_area,
                    "active_count": int(len(sub)), "oldest_utc": sub["created_at_utc"].min(),
                    "newest_utc": sub["created_at_utc"].max(),
                    "department": city_cfg["departments"].get(h.category, city_cfg["review_department"]),
                    "step": INVESTIGATION_STEP.get(h.category, "Inspect the area near {area}.").format(
                        area=area.split("·")[-1].strip()),
                    "report_ids": sorted(sub["report_id"]), "centroid": (h.centroid_lat, h.centroid_lon),
                    "live_count": int((~sub["dedupe_key"].astype(str).str.startswith("seed:")).sum())})
    out.sort(key=lambda p: (-p["active_count"], p["oldest_utc"]))
    for i, p in enumerate(out[:top], start=1):
        p["rank"] = i
    return out[:top]


def last_updates(conn, city_id: str) -> dict:
    """report_id -> latest of: staff category decision, status change (UTC ISO). Missing = never updated."""
    rows = conn.execute("""SELECT report_id, MAX(t) FROM (
                             SELECT report_id, corrected_at_utc AS t FROM corrections WHERE city_id=?
                             UNION ALL SELECT report_id, updated_at_utc AS t FROM status_updates WHERE city_id=?)
                           GROUP BY report_id""", (city_id, city_id)).fetchall()
    return {r[0]: r[1] for r in rows}
