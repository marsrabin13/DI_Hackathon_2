"""Focused smoke checks. Uses a temporary SQLite DB; never fits or tunes on data/gpt_tel_aviv_test.csv.
  python smoke_check.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import timedelta
from pathlib import Path

import pandas as pd

import ai_service
import core

ROOT = Path(__file__).parent
results = []


def check(name, cond, detail=""):
    results.append({"check": name, "pass": bool(cond), "detail": str(detail)})
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(t).lower())).strip()


def offset(lat, lon, dn_m, de_m):
    import math
    return lat + dn_m / 111_320, lon + de_m / (111_320 * math.cos(math.radians(lat)))


cities = core.load_cities()
tlv, ex = cities["il_tel_aviv"], cities["example_city"]
model_hash_before = hashlib.sha256((ROOT / "artifacts/model.joblib").read_bytes()).hexdigest()
model, meta = core.load_model()

# 1. split isolation ------------------------------------------------------------
missing = [p.name for p in (core.TRAIN_CSV, core.VALIDATION_CSV, core.TEST_CSV, core.DEMO_CSV, core.PROVENANCE_PATH)
           if not p.exists()]
check("required gpt_ data files present", not missing, missing or "all 5")
if missing:
    raise SystemExit(1)
tr = pd.read_csv(core.TRAIN_CSV, keep_default_na=False, dtype=str)
va = pd.read_csv(core.VALIDATION_CSV, keep_default_na=False, dtype=str)
te = pd.read_csv(core.TEST_CSV, keep_default_na=False, dtype=str)
prov = json.loads(core.PROVENANCE_PATH.read_text())
check("counts match provenance (train/val/test/demo)",
      [len(tr), len(va), len(te), len(pd.read_csv(core.DEMO_CSV))] ==
      [prov["actual_counts"][k]["rows"] for k in ("train", "validation", "tel_aviv_test", "tel_aviv_demo")],
      f"{len(tr)}/{len(va)}/{len(te)}")
overl = {"fam_tr_va": len(set(tr.family_id) & set(va.family_id)), "fam_tr_te": len(set(tr.family_id) & set(te.family_id)),
         "fam_va_te": len(set(va.family_id) & set(te.family_id)),
         "txt_tr_va": len(set(tr.text.map(norm)) & set(va.text.map(norm))),
         "txt_tr_te": len(set(tr.text.map(norm)) & set(te.text.map(norm))),
         "txt_va_te": len(set(va.text.map(norm)) & set(te.text.map(norm)))}
check("split isolation (family + normalized text)", all(v == 0 for v in overl.values()), overl)
check("model fitted on gpt_train.csv only (hash in metadata)",
      meta["file_hashes"].get(core.TRAIN_CSV.name) == hashlib.sha256(core.TRAIN_CSV.read_bytes()).hexdigest()[:16])

tmp = Path(tempfile.mkdtemp()) / "smoke.db"
conn = core.connect(tmp)
now = core.utcnow()


def seed_point(cfg, rid, cat, lat, lon, when=None):
    conn.execute("""INSERT INTO reports (report_id, city_id, text, created_at_utc, latitude, longitude, location_status,
        approved_category, category_source, proposed_department, routing_state, routing_reasons, status, dedupe_key)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                 (rid, cfg["city_id"], "fixture", core.iso(when or now), lat, lon,
                  core.location_status(lat, lon, cfg), cat, "seed_fixture", cfg["departments"][cat], "routed", "[]",
                  "open", "smoke:" + rid))
    conn.commit()


# 2. six nearby same-category points vs a distant point ---------------------------
base = (32.090, 34.800)
for i in range(6):
    seed_point(tlv, f"S-R{i}", "roads", *offset(*base, 20 * i - 50, 15 * (i % 3)))
seed_point(tlv, "S-RFAR", "roads", *offset(*base, 3000, 0))
df = core.fetch_reports(conn, "il_tel_aviv", now - timedelta(days=7), now + timedelta(minutes=1))
hs, _ = core.compute_hotspots(df, tlv)
roads_hs = hs[hs.category == "roads"]
check("6 nearby same-category points form one hotspot", len(roads_hs) == 1 and roads_hs.iloc[0].report_count == 6,
      roads_hs[["report_count"]].to_dict("records"))
check("distant point is not in the hotspot", "S-RFAR" not in roads_hs.iloc[0].report_ids if len(roads_hs) else False)

# 3. category / city separation ---------------------------------------------------
base2 = (32.120, 34.810)
for i in range(3):
    seed_point(tlv, f"S-MW{i}", "waste", *offset(*base2, 10 * i, 0))
    seed_point(tlv, f"S-ML{i}", "lighting", *offset(*base2, 10 * i, 10))
for i in range(6):
    seed_point(ex, f"S-EX{i}", "noise", *offset(*base2, 10 * i, 5))
df = core.fetch_reports(conn, "il_tel_aviv", now - timedelta(days=7), now + timedelta(minutes=1))
hs, _ = core.compute_hotspots(df, tlv)
check("3 waste + 3 lighting at one spot do not merge into a hotspot",
      not set(hs.category) & {"waste", "lighting"}, hs.category.tolist())
check("other city's 6 points absent from Tel Aviv reads/hotspots",
      (df.city_id == "il_tel_aviv").all() and "noise" not in set(hs.category))
dfx = core.fetch_reports(conn, "example_city", now - timedelta(days=7), now + timedelta(minutes=1))
hsx, _ = core.compute_hotspots(dfx, ex)
check("other city clusters only its own points", len(hsx) == 1 and hsx.iloc[0].report_count == 6)

# 4. missing location -------------------------------------------------------------
txt = "There is a deep pothole on our street near the bus stop, cars keep swerving."
r = core.route(txt, tlv, model, meta)
rid, new = core.save_report(conn, tlv, txt, None, None, r, meta)
rep = core.get_report(conn, "il_tel_aviv", rid)
check("missing location does not block routing", rep["routing_state"] == "routed" and rep["location_status"] == "missing",
      f"{rep['routing_state']}, {rep['location_status']}")
df = core.fetch_reports(conn, "il_tel_aviv", now - timedelta(days=7), core.utcnow() + timedelta(minutes=1))
_, excl = core.compute_hotspots(df, tlv)
check("missing location excluded from clustering with explanation", excl["missing_location"] >= 1, excl)
rid_out, _ = core.save_report(conn, tlv, txt + " (outside)", 31.5, 34.5, core.route(txt, tlv, model, meta), meta)
check("outside-bbox location flagged", core.get_report(conn, "il_tel_aviv", rid_out)["location_status"] == "outside_bbox")

# 5. duplicate-save prevention ----------------------------------------------------
rid2, new2 = core.save_report(conn, tlv, txt, None, None, r, meta)
n = conn.execute("SELECT COUNT(*) FROM reports WHERE city_id='il_tel_aviv' AND text=?", (txt,)).fetchone()[0]
check("duplicate save returns existing report", rid2 == rid and not new2 and n == 1)
seed_n = core.import_seed(conn, core.DEMO_CSV, tlv)
seed_again = core.import_seed(conn, core.DEMO_CSV, tlv)
check("seed import idempotent", seed_n == 90 and seed_again == 0, f"{seed_n} then {seed_again}")
src = pd.read_csv(core.DEMO_CSV, dtype=str)
fx = core.fetch_reports(conn, "il_tel_aviv")
fx = fx[fx["report_id"].isin(src["report_id"])]
check("fixtures stored as category_source=seed_fixture", (fx["category_source"] == "seed_fixture").all())
merged = src.merge(fx, on="report_id", suffixes=("_src", ""))
check("fixtures preserve IDs, coordinates, categories, timestamps",
      len(merged) == 90 and (merged["category"] == merged["approved_category"]).all()
      and (merged["created_at_utc_src"] == merged["created_at_utc"]).all()
      and (merged["latitude_src"].astype(float) == merged["latitude"]).all()
      and (merged["longitude_src"].astype(float) == merged["longitude"]).all())
check("fixtures imported open + routed + seed_fixture",
      (fx["status"] == "open").all() and (fx["routing_state"] == "routed").all() and len(fx) == 90)
anchor = core.demo_anchor()
check("default window end = provenance demo anchor", core.iso(anchor) == prov["demo_anchor_utc"], core.iso(anchor))
win = core.fetch_reports(conn, "il_tel_aviv", anchor - timedelta(days=7), anchor)
win = win[win["report_id"].isin(src["report_id"])]
fhs, _ = core.compute_hotspots(win, tlv)
check("planted fixture hotspots: roads 12, waste 10, lighting 8, no noise",
      sorted(zip(fhs.category, fhs.report_count)) == [("lighting", 8), ("roads", 12), ("waste", 10)],
      list(zip(fhs.category, fhs.report_count)))
# older fixture version is replaced, citizen reports kept
conn.execute("UPDATE reports SET data_version='old' WHERE report_id=?", (src.report_id[0],))
conn.commit()
again = core.import_seed(conn, core.DEMO_CSV, tlv)
check("stale fixture version replaced; citizen report kept",
      again == 1 and core.get_report(conn, "il_tel_aviv", rid) is not None, again)

# 6. correction audit -------------------------------------------------------------
vague = "Something seems off around here."
rv = core.route(vague, tlv, model, meta)
rid_v, _ = core.save_report(conn, tlv, vague, None, None, rv, meta)
before = core.get_report(conn, "il_tel_aviv", rid_v)
after = core.apply_correction(conn, tlv, rid_v, "waste", "smoke-operator")
audit = core.fetch_corrections(conn, "il_tel_aviv")
check("correction recalculates route from config",
      after["approved_category"] == "waste" and after["proposed_department"] == tlv["departments"]["waste"]
      and after["routing_state"] == "routed")
check("original prediction preserved after correction",
      after["predicted_category"] == before["predicted_category"] and after["predicted_score"] == before["predicted_score"])
check("correction audit row saved", len(audit) == 1 and audit.iloc[0].operator == "smoke-operator"
      and audit.iloc[0].old_department == before["proposed_department"])
try:
    core.apply_correction(conn, ex, rid_v, "roads", "x")
    check("correction cannot cross cities", False)
except KeyError:
    check("correction cannot cross cities", True)

# 6b. attachments: stored, not analyzed --------------------------------------------
up = Path(tempfile.mkdtemp())
before_route = core.get_report(conn, "il_tel_aviv", rid)
core.save_attachment(conn, "il_tel_aviv", rid, "../../evil.PNG", b"\x89PNG fake", upload_dir=up)
after_att = core.get_report(conn, "il_tel_aviv", rid)
check("attachment stored under server-chosen name",
      (up / f"{rid}.png").exists() and after_att["attachment_path"] and "evil" not in after_att["attachment_path"])
check("attachment does not change routing",
      all(after_att[k] == before_route[k] for k in ("routing_state", "proposed_department", "approved_category")))
bad = 0
for name, data in (("x.exe", b"x"), ("big.jpg", b"0" * (core.MAX_ATTACHMENT_MB * 1024 * 1024 + 1))):
    try:
        core.save_attachment(conn, "il_tel_aviv", rid, name, data, upload_dir=up)
    except ValueError:
        bad += 1
check("disallowed type and oversize file rejected", bad == 2)
check("acknowledgment facts exclude attachment", "attachment_path" not in ai_service.ack_facts(after_att))

# 6c. named area points -------------------------------------------------------------
a_, b_, c_, d_ = tlv["demo_bbox"]
check("all named area points inside the demo bbox",
      all(a_ <= la <= c_ and b_ <= lo <= d_ for la, lo in tlv["map_presets"].values()), len(tlv["map_presets"]))
check("nearest area: planted roads centre -> Rothschild Boulevard",
      core.nearest_area(32.068, 34.774, tlv) == "Center · Rothschild Boulevard")
check("nearest area: far point -> none", core.nearest_area(32.0, 34.5, tlv) == "")

# 6d. case status, resident lookup, history -----------------------------------------
core.update_status(conn, "il_tel_aviv", rid_v, "in_progress", "A team will check the area.", "smoke-op")
lk = core.find_by_reference(conn, "il_tel_aviv", rid_v.lower() + "  ")
tl = core.case_timeline(conn, tlv, lk)
check("reference lookup is case/space-insensitive and city-scoped",
      lk and lk["report_id"] == rid_v and core.find_by_reference(conn, "example_city", rid_v) is None)
check("timeline shows review, staff decision and status note in order",
      [e["title"] for e in tl] == ["Report received", "Waiting for staff review",
                                   "Staff assigned it to Sanitation (demo)", "Status: In progress"]
      and tl[-1]["detail"] == "A team will check the area.", [e["title"] for e in tl])
check("status change logged with old/new status",
      core.fetch_status_updates(conn, "il_tel_aviv", rid_v).iloc[-1][["old_status", "new_status"]].tolist()
      == ["open", "in_progress"])
for bad_status, bad_note in (("done", ""), ("resolved", "x" * (core.MAX_NOTE + 1))):
    try:
        core.update_status(conn, "il_tel_aviv", rid_v, bad_status, bad_note, "x")
        check(f"invalid status/note rejected ({bad_status})", False)
    except ValueError:
        check(f"invalid status/note rejected ({bad_status})", True)
# resolved cases leave hotspots; in-progress cases stay
for i in range(5):
    core.update_status(conn, "il_tel_aviv", f"S-R{i}", "resolved", "", "smoke-op")
df_r = core.fetch_reports(conn, "il_tel_aviv", now - timedelta(days=7), core.utcnow() + timedelta(minutes=1))
hs_r, _ = core.compute_hotspots(df_r, tlv)
check("resolved cases drop out of hotspots", "S-R5" not in ",".join(hs_r.report_ids) if len(hs_r) else True)
core.update_status(conn, "il_tel_aviv", "S-R0", "in_progress", "", "smoke-op")
check("simulated fixture recognised; live report not", core.is_simulated(core.get_report(conn, "il_tel_aviv", src.report_id[0]))
      and not core.is_simulated(lk))

# 6e. combined staff decision + spam ------------------------------------------------
sp_txt = "I need to renew my passport, which office?"
sp_id, _ = core.save_report(conn, tlv, sp_txt, None, None, core.route(sp_txt, tlv, model, meta), meta)
sp, sp_acts = core.staff_decision(conn, tlv, sp_id, core.SPAM, "open", "", "smoke-op")
check("spam: no department, case closed, default resident note",
      sp["routing_state"] == "spam" and sp["status"] == "closed" and sp["proposed_department"] == core.SPAM_DEPARTMENT
      and core.fetch_status_updates(conn, "il_tel_aviv", sp_id).iloc[-1]["resident_note"] == core.SPAM_NOTE, sp_acts)
check("spam leaves the staff queue and keeps the model prediction",
      sp["status"] not in core.ACTIVE_STATUSES and sp["predicted_category"] is not None)
cd_txt = "Something odd near the corner, not sure what."
cd_id, _ = core.save_report(conn, tlv, cd_txt, None, None, core.route(cd_txt, tlv, model, meta), meta)
cd, cd_acts = core.staff_decision(conn, tlv, cd_id, "noise", "in_progress", "Checking.", "smoke-op")
_, again_acts = core.staff_decision(conn, tlv, cd_id, "noise", "in_progress", "", "smoke-op")
check("one save sets department and status; repeat save changes nothing",
      cd["proposed_department"] == tlv["departments"]["noise"] and cd["status"] == "in_progress"
      and len(cd_acts) == 2 and again_acts == [], cd_acts)
trail = core.staff_actions(conn, "il_tel_aviv", cd_id)
check("simplified audit trail lists both actions with staff name",
      trail["action"].tolist() == ["Status → In progress", "Assigned to Environmental Review (demo)"]
      and set(trail["by"]) == {"smoke-op"}, trail["action"].tolist())

# 6f. shared counts + Weekly Action Brief priorities --------------------------------
cmx = core.case_metrics(core.fetch_reports(conn, "il_tel_aviv"))
allx = core.fetch_reports(conn, "il_tel_aviv")
check("Needs review count == awaiting-review list (one definition)",
      cmx["awaiting_review"] == len(core.awaiting_review(allx))
      and all(r in core.ACTIVE_STATUSES for r in core.awaiting_review(allx)["status"]))
check("spam/decided cases counted as previously reviewed, not waiting",
      sp_id not in set(core.awaiting_review(allx)["report_id"]) and cmx["previously_reviewed"] >= 2)
pb = Path(tempfile.mkdtemp()) / "brief.db"
bconn = core.connect(pb)
core.import_seed(bconn, core.DEMO_CSV, tlv)
anc = core.demo_anchor()
def _prio(c):
    d = core.fetch_reports(c, "il_tel_aviv", anc - timedelta(days=7), anc)
    h, _ = core.compute_hotspots(d, tlv)
    return core.priority_hotspots(d, h, tlv)
p0 = _prio(bconn)
check("priorities: up to 3, ranked by active count", [p["active_count"] for p in p0] == [12, 10, 8]
      and [p["rank"] for p in p0] == [1, 2, 3], [(p["category"], p["active_count"]) for p in p0])
for r in p0[0]["report_ids"][:3]:
    core.update_status(bconn, "il_tel_aviv", r, "resolved", "", "t")
core.update_status(bconn, "il_tel_aviv", p0[0]["report_ids"][3], "closed", "", "t")
p1 = _prio(bconn)
roads = [p for p in p1 if p["category"] == "roads"][0]
check("resolved/closed cases leave priorities; ranking updates",
      roads["active_count"] == 8 and p1[0]["category"] == "waste"
      and not set(p0[0]["report_ids"][:4]) & set(roads["report_ids"]), [(p["category"], p["active_count"]) for p in p1])
check("tie broken by oldest active report", [p["category"] for p in p1 if p["active_count"] == 8] ==
      [p["category"] for p in sorted([p for p in p1 if p["active_count"] == 8], key=lambda p: p["oldest_utc"])])
check("priority evidence lists only that hotspot's active reports",
      all(core.get_report(bconn, "il_tel_aviv", r)["approved_category"] == p1[0]["category"]
          and core.get_report(bconn, "il_tel_aviv", r)["status"] in core.ACTIVE_STATUSES for r in p1[0]["report_ids"]))
check("suggested department comes from city config", all(p["department"] == tlv["departments"][p["category"]] for p in p1))
ctx = {"city": "x", "period": "y", "awaiting_review": 0, "oldest_active_days": 1,
       "priorities": [{k: p[k] for k in ("evidence_id", "rank", "category", "area", "active_count", "oldest_utc",
                                         "department")} for p in p1]}
ho = ai_service.generate_handover(ctx)
check("handover without AI: template draft keeps every priority + evidence ID",
      ho["label"] == ai_service.FALLBACK and [i["evidence_id"] for i in ho["items"]] == [p["evidence_id"] for p in p1])
ai_service._GENERATOR = lambda p: "These reports keep getting worse because of heavy trucks."
check("handover rejects causes/trends", ai_service.generate_handover(ctx)["label"] == ai_service.FALLBACK)
ai_service._GENERATOR = lambda p: "Several residents describe the same stretch, so a site visit can confirm it."
ho2 = ai_service.generate_handover(ctx)
check("handover accepts bounded text; facts stay with code",
      ho2["label"] == ai_service.LIVE and all(not re.search(r"\d", i["why"] + i["verify"]) for i in ho2["items"]))
ai_service._GENERATOR = None
check("no priorities -> honest empty handover", ai_service.generate_handover({**ctx, "priorities": []})["items"] == [])

# 7. config change route, unchanged model -----------------------------------------
r_tlv, r_ex = core.route(txt, tlv, model, meta), core.route(txt, ex, model, meta)
same_model = hashlib.sha256((ROOT / "artifacts/model.joblib").read_bytes()).hexdigest() == model_hash_before
check("config change: same category, different department, model unchanged",
      r_tlv["predicted_category"] == r_ex["predicted_category"] == "roads"
      and r_tlv["proposed_department"] == "Road Maintenance (demo)"
      and r_ex["proposed_department"] == "Public Works (demo)" and same_model,
      f"{r_tlv['proposed_department']} vs {r_ex['proposed_department']}")

# 8. labeled API-failure fallback -------------------------------------------------
saved_env = {k: os.environ.get(k) for k in ("AI_PROVIDER", "FLAN_T5_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")}
facts = core.summary_facts(df, hs, tlv, now - timedelta(days=7), now)
# local FLAN-T5 unavailable (bad model path / not installed) -> labeled fallback
os.environ["AI_PROVIDER"], os.environ["FLAN_T5_MODEL"] = "flan_t5", "/nonexistent/flan-model"
ai_service._load_flan.cache_clear()
a_f, b_f = ai_service.draft_acknowledgment(rep), ai_service.generate_brief(facts)
check("local model unavailable -> Template fallback (ack + brief)",
      a_f["label"] == b_f["label"] == ai_service.FALLBACK, a_f["detail"][:60])
# stand-in generator: checks the FLAN wiring and safety filters without downloading the model
ai_service._GENERATOR = lambda p: "thank you for letting us know about this"
a_ok = ai_service.draft_acknowledgment(rep)
check("local model text accepted -> Live AI draft with code-filled facts",
      a_ok["label"] == ai_service.LIVE and rep["report_id"] in a_ok["text"]
      and rep["proposed_department"] in a_ok["text"] and len(a_ok["text"].split()) <= 60, a_ok["text"][:80])
b_ok = ai_service.generate_brief(facts)
check("local model brief: Live label, evidence IDs from code, passes validator",
      b_ok["label"] == ai_service.LIVE and bool(ai_service.validate_brief(b_ok["brief"], facts)), b_ok["detail"])
ai_service._GENERATOR = lambda p: "It will be fixed by tomorrow by our crew"
a_bad, b_bad = ai_service.draft_acknowledgment(rep), ai_service.generate_brief(facts)
check("local model promise/deadline text rejected -> fallback (ack + brief)",
      a_bad["label"] == b_bad["label"] == ai_service.FALLBACK and "tomorrow" not in a_bad["text"])
ai_service._GENERATOR = lambda p: "There were 12 complaints in 3 days"
check("local model numbers rejected", ai_service.generate_brief(facts)["label"] == ai_service.FALLBACK)
ai_service._GENERATOR = None
# Claude provider without key / unreachable -> labeled fallback
os.environ["AI_PROVIDER"] = "anthropic"
os.environ.pop("ANTHROPIC_API_KEY", None)
a0 = ai_service.draft_acknowledgment(rep)
check("Claude without credentials -> Template fallback label", a0["label"] == ai_service.FALLBACK, a0["detail"])
os.environ["ANTHROPIC_API_KEY"] = "sk-invalid-smoke"
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
a1 = ai_service.draft_acknowledgment(rep)
b1 = ai_service.generate_brief(facts)
check("API failure -> labeled Template fallback (ack)", a1["label"] == ai_service.FALLBACK, a1["detail"][:60])
check("API failure -> labeled Template fallback (brief)", b1["label"] == ai_service.FALLBACK, b1["detail"][:60])
for k, v in saved_env.items():
    os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
ai_service._load_flan.cache_clear()

# 9. no invented route/deadline claims --------------------------------------------
f_routed = ai_service.ack_facts(rep)
f_review = ai_service.ack_facts(core.get_report(conn, "il_tel_aviv", rid_v) | {"routing_state": "review_required"})
bad_acks = [{"message": "Recorded in this demo. A crew will be fixed within 2 days."},
            {"message": "Recorded in this demo; it will be repaired by tomorrow."},
            {"message": "Your issue has been dispatched to the team in this demo."}]
rejected = 0
for b in bad_acks:
    try:
        ai_service.validate_ack(b, f_routed)
    except ValueError:
        rejected += 1
check("deadline/dispatch/completion claims rejected", rejected == len(bad_acks), f"{rejected}/{len(bad_acks)}")
try:
    ai_service.validate_ack({"message": "Recorded in this demo and sent to Sanitation (demo)."}, f_review)
    check("department not named for non-routed case", False)
except ValueError:
    check("department not named for non-routed case", True)
bad_brief = {"observations": [{"evidence_id": "F99", "text": "Simulated data shows a pattern."}],
             "recommended_actions": [{"evidence_id": "F4", "text": "Investigate."}]}
num_brief = {"observations": [{"evidence_id": "F4", "text": "Simulated data shows 40% more reports."}],
             "recommended_actions": [{"evidence_id": "F4", "text": "Investigate."}]}
rej = 0
for bb in (bad_brief, num_brief):
    try:
        ai_service.validate_brief(bb, facts)
    except ValueError:
        rej += 1
check("brief with unknown evidence id or numbers rejected", rej == 2)
tmpl_ok = not ai_service.FORBIDDEN.search(ai_service.ack_template(f_routed)) and \
    not ai_service.FORBIDDEN.search(ai_service.ack_template(f_review))
check("fallback templates contain no promise/deadline claims", tmpl_ok)
check("fallback brief passes its own validator", bool(ai_service.validate_brief(ai_service.brief_template(facts), facts)))

# 10. policy inputs with actual predictions -----------------------------------------
cases = [("known", "The streetlight outside my building is not working and the sidewalk is dark at night.", False, "routed"),
         ("unclear", "Please help, it is a problem.", False, "review_required"),
         ("unsupported", "How do I pay my property tax bill online?", False, "review_required"),
         ("multi-issue (checkbox)", "Garbage everywhere and the lamp is broken and it's loud.", True, "review_required"),
         ("urgent", "There is an exposed live wire on the sidewalk near the school.", False, "urgent_review")]
policy_rows = []
for name, t, unsure, expected in cases:
    rr = core.route(t, tlv, model, meta, unsure=unsure)
    policy_rows.append({"case": name, "predicted": rr["predicted_category"], "score": round(rr["score"], 3),
                        "margin": round(rr["margin"], 3), "state": rr["routing_state"], "expected": expected,
                        "reasons": rr["reasons"]})
    check(f"policy: {name} -> {expected}", rr["routing_state"] == expected,
          f"{rr['predicted_category']} {rr['score']:.2f}/{rr['margin']:.2f} {rr['routing_state']}")
try:
    core.route("", tlv, model, meta); check("empty text rejected", False)
except ValueError:
    check("empty text rejected", True)
try:
    core.route("x" * 1001, tlv, model, meta); check(">1000 chars rejected", False)
except ValueError:
    check(">1000 chars rejected", True)

passed = sum(r["pass"] for r in results)
print(f"\n{passed}/{len(results)} checks passed")
(ROOT / "artifacts/smoke_results.json").write_text(json.dumps({"passed": passed, "total": len(results),
                                                                "results": results, "policy_inputs": policy_rows}, indent=2))
raise SystemExit(0 if passed == len(results) else 1)
