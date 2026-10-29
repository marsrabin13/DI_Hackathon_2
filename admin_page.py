"""Admin dashboard (demo, no login): statistics, map/hotspots, review queue, weekly brief, model & data."""
from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import pydeck as pdk
import streamlit as st

import ai_service
import core
import ui_common as ui

st.set_page_config(page_title="CityFix · Admin", page_icon="📊", layout="wide")
ui.apply_style()
model, meta = ui.require_model()
conn = ui.get_conn()
cities = core.load_cities()

ui.header("CityFix Admin", "Operations dashboard · demo, no login · department labels are illustrative")

# ---------------------------------------------------------------- filters
f1, f2, f3 = st.columns([2, 3, 2])
city_id = f1.selectbox("City", list(cities), format_func=lambda c: cities[c]["city_name"])
cfg = cities[city_id]
anchor = core.demo_anchor()
win = f2.radio("Window end", ["Demo anchor", "Now"], horizontal=True,
               help=f"Demo anchor = {core.iso(anchor)} (end of the simulated history)")
end = anchor if win == "Demo anchor" else core.utcnow()
days = cfg["hotspots"]["window_days"]
start = end - timedelta(days=days)
f3.caption(f"Window: {core.iso(start)[:10]} → {core.iso(end)[:10]} ({days} days, UTC)")

df = core.fetch_reports(conn, city_id, start, end)
hotspots, excl = core.compute_hotspots(df, cfg)
facts = core.summary_facts(df, hotspots, cfg, start, end)
n = len(df)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reports", n)
k2.metric("Routed", int((df["routing_state"] == "routed").sum()) if n else 0)
k3.metric("Staff review", int((df["routing_state"] == "review_required").sum()) if n else 0)
k4.metric("Urgent review", int((df["routing_state"] == "urgent_review").sum()) if n else 0)
k5.metric("Hotspots", len(hotspots))

t_over, t_map, t_review, t_brief, t_model = st.tabs(
    ["Overview", "Map & hotspots", "Review queue", "Weekly brief", "Model & data"])

# ---------------------------------------------------------------- overview
with t_over:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Department queue (open)")
        if n:
            q = (df[df["status"] == "open"].pivot_table(index="proposed_department", columns="routing_state",
                                                        values="report_id", aggfunc="count", fill_value=0))
            q["total"] = q.sum(axis=1)
            st.dataframe(q.sort_values("total", ascending=False), width="stretch")
        else:
            st.info("No reports in this window.")
    with c2:
        st.markdown("#### Reports by approved category")
        if n:
            st.bar_chart(df["approved_category"].fillna("unassigned").value_counts(), color=ui.TEAL,
                         horizontal=True)
    st.markdown("#### Latest reports")
    if n:
        st.dataframe(df[["report_id", "created_at_utc", "text", "approved_category", "proposed_department",
                         "routing_state", "location_status"]].head(15), hide_index=True, width="stretch")

# ---------------------------------------------------------------- map
with t_map:
    pts = df.dropna(subset=["latitude", "longitude"]).copy()
    pts["color"] = pts["approved_category"].map(lambda c: ui.CAT_COLORS.get(c, [150, 150, 150]))
    layers = [pdk.Layer("ScatterplotLayer", data=pts, get_position="[longitude, latitude]",  # pydeck: lon, lat
                        get_fill_color="color", get_radius=45, pickable=True, opacity=0.85)]
    if len(hotspots):
        layers.append(pdk.Layer("ScatterplotLayer", data=hotspots, get_position="[centroid_lon, centroid_lat]",
                                get_radius=cfg["hotspots"]["eps_meters"], filled=False, stroked=True,
                                get_line_color=[20, 40, 75], line_width_min_pixels=3, pickable=True))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(
        latitude=cfg["map_center"][0], longitude=cfg["map_center"][1], zoom=12),
        tooltip={"text": "{report_id} {approved_category} {hotspot_id}"}))
    st.caption("Navy = roads · teal = waste · amber = lighting · rose = noise · rings = hotspots. "
               "Simulated, illustrative points — not verified incident or street locations.")
    st.markdown("#### Hotspots")
    st.caption(f"DBSCAN (haversine) per category, neighbor distance {cfg['hotspots']['eps_meters']} m, "
               f"min {cfg['hotspots']['min_samples']} reports; open + routed only; ranked by report count.")
    st.dataframe(hotspots.drop(columns=["report_ids"]) if len(hotspots) else hotspots,
                 hide_index=True, width="stretch")
    st.caption(f"Not clustered: {excl['missing_location']} without location, {excl['outside_or_invalid']} "
               f"outside demo area, {excl['not_open_routed']} not open+routed. Hotspots are recurring report "
               "patterns, not confirmed unique incidents or causes.")

# ---------------------------------------------------------------- review queue
with t_review:
    allrep = core.fetch_reports(conn, city_id)
    review = allrep[allrep["routing_state"].isin(["review_required", "urgent_review"]) & (allrep["status"] == "open")]
    st.markdown(f"#### Waiting for staff ({len(review)}, all dates)")
    review = review.assign(media=review["attachment_path"].notna().map({True: "📎", False: ""}))
    st.dataframe(review[["report_id", "media", "created_at_utc", "text", "predicted_category", "predicted_score",
                         "routing_state", "routing_reasons"]], hide_index=True, width="stretch")

    with_media = allrep[allrep["attachment_path"].notna()]
    if len(with_media):
        with st.expander(f"📎 Reports with a photo or clip ({len(with_media)})"):
            pick = st.selectbox("Report", with_media["report_id"].tolist(), key="media_pick")
            row = with_media[with_media["report_id"] == pick].iloc[0]
            st.write(f"**{row['report_id']}** · {row['routing_state']} · {row['proposed_department']}")
            st.write(row["text"])
            path = core.ROOT / row["attachment_path"]
            if not path.exists():
                st.warning("File not found on disk.")
            elif path.suffix.lower() in (".mp4", ".mov"):
                st.video(str(path))
            else:
                st.image(str(path), width=420)
            st.caption("Stored for staff review only — not analyzed by AI and not used for routing. "
                       "Roadmap: AI image analysis to suggest a category, after a privacy review and an "
                       "evaluation on real local photos.")
    with st.form("correct"):
        st.markdown("**Approve or correct a category** — the department is recalculated from the city config; "
                    "the original prediction is kept.")
        ids = review["report_id"].tolist() + [i for i in allrep["report_id"] if i not in set(review["report_id"])]
        c1, c2, c3 = st.columns(3)
        rid = c1.selectbox("Report", ids) if ids else None
        new_cat = c2.selectbox("Approved category", list(cfg["departments"]))
        op = c3.text_input("Operator", value="demo-operator")
        if st.form_submit_button("Save decision") and rid:
            upd = core.apply_correction(conn, cfg, rid, new_cat, op)
            st.success(f"{rid}: '{new_cat}' → {upd['proposed_department']}")
    corr = core.fetch_corrections(conn, city_id)
    if len(corr):
        st.markdown(f"#### Correction audit ({len(corr)})")
        st.dataframe(corr, hide_index=True, width="stretch")

# ---------------------------------------------------------------- weekly brief
with t_brief:
    c1, c2 = st.columns([3, 2])
    with c2:
        st.markdown("#### Computed facts")
        st.dataframe(pd.DataFrame(facts).astype(str), hide_index=True, width="stretch", height=360)
        st.caption("These numbers are authoritative; the brief only references them by ID.")
    with c1:
        st.markdown("#### Weekly brief")
        if st.button("Generate weekly brief", type="primary"):
            key = json.dumps(facts, sort_keys=True, default=str)
            cache = st.session_state.setdefault("brief_cache", {})
            if key not in cache:
                cache[key] = ai_service.generate_brief(facts)
            st.session_state["brief"] = cache[key]
        b = st.session_state.get("brief")
        if b:
            st.markdown(ui.draft_tag(b["label"], b["detail"]), unsafe_allow_html=True)
            st.markdown("**Observations**")
            for it in b["brief"]["observations"]:
                st.markdown(f"- `{it['evidence_id']}` {it['text']}")
            st.markdown("**Suggested investigations**")
            for it in b["brief"]["recommended_actions"]:
                st.markdown(f"- `{it['evidence_id']}` {it['text']}")
            st.caption("Draft for staff review. No action is promised or taken.")

# ---------------------------------------------------------------- model & data
with t_model:
    v = meta.get("validation", {})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Model")
        st.write(f"TF-IDF (1–2 grams) → Logistic Regression · `{meta.get('model_version')}`")
        st.write(f"Thresholds: score ≥ {meta['thresholds']['min_score']}, margin ≥ {meta['thresholds']['min_margin']}")
        st.caption(meta.get("threshold_decision", ""))
        rows = [{"set": "Validation (24)", "accuracy": v.get("accuracy"), "macro-F1": v.get("macro_f1"),
                 "routed": v.get("routing_coverage"), "routed correctly": v.get("accepted_route_accuracy")}]
        tm = core.ROOT / "artifacts" / "test_metrics.json"
        if tm.exists():
            t = json.loads(tm.read_text())
            rows.append({"set": "Held-out test (20 + 4 policy)", "accuracy": t["accuracy"], "macro-F1": t["macro_f1"],
                         "routed": t["routing_coverage"], "routed correctly": t["accepted_route_accuracy"]})
            st.caption(f"Policy cases on test: {t['policy_cases_passed']}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Synthetic evaluations share generation bias; they don't establish real-city accuracy.")
        st.markdown("#### Roadmap (not built)")
        st.markdown("- **AI image analysis:** suggest a category from the resident's photo or clip. Needs a privacy "
                    "review (faces, car plates), consent wording, and an evaluation on real local photos before use.\n"
                    "- Multi-issue detection without the resident's checkbox.\n"
                    "- Real local reports, calibrated scores, and shadow testing beside staff.")
    with c2:
        st.markdown("#### Data")
        if core.PROVENANCE_PATH.exists():
            p = json.loads(core.PROVENANCE_PATH.read_text())
            ac = p.get("actual_counts", {})
            tr = ac.get("train", {})
            st.write(f"Mode: **{p.get('mode')}** · NYC download: {p.get('download', {}).get('status')} "
                     f"({p.get('sources', {}).get('raw_rows_retained')} raw rows retained)")
            st.dataframe(pd.DataFrame([
                {"file": f"data/{core.TRAIN_CSV.name}", "rows": tr.get("rows"),
                 "use": f"training · {tr.get('by_source', {}).get('nyc_structured')} NYC details + "
                        f"{tr.get('by_source', {}).get('synthetic')} synthetic"},
                {"file": f"data/{core.VALIDATION_CSV.name}", "rows": ac.get("validation", {}).get("rows"),
                 "use": "validation + threshold selection (synthetic)"},
                {"file": f"data/{core.TEST_CSV.name}", "rows": ac.get("tel_aviv_test", {}).get("rows"),
                 "use": "held-out test, evaluated once (synthetic)"},
                {"file": f"data/{core.DEMO_CSV.name}", "rows": ac.get("tel_aviv_demo", {}).get("rows"),
                 "use": "map/dashboard fixtures only (simulated)"},
            ]), hide_index=True, width="stretch")
            st.caption(f"Seed {p.get('seed')} · demo anchor {p.get('demo_anchor_utc')} · provenance: "
                       f"data/{core.PROVENANCE_PATH.name}")
        st.write("AI drafts: " + ("API key configured" if ai_service.has_credentials()
                                  else "no API key → Template fallback"))
        st.markdown("#### Portability check")
        test_text = st.text_input("Same text, every city config", "There is a deep pothole on our street.")
        if test_text.strip():
            st.dataframe(pd.DataFrame([
                {"city config": c["city_name"], **{k: v for k, v in core.route(test_text, c, model, meta).items()
                                                   if k in ("predicted_category", "routing_state",
                                                            "proposed_department")}}
                for c in cities.values()]), hide_index=True, width="stretch")
            st.caption("Same model, same category — the department comes from each city's config. No retraining.")
