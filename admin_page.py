"""Admin dashboard (no login): filters + summary cards, map beside the review queue, staff decisions,
readable report tables, the Weekly Action Brief, and model/data/diagnostics.

Counts come from core.case_metrics / core.awaiting_review so cards, queues and the brief always agree."""
from __future__ import annotations

import html
import json
from datetime import datetime, timedelta

import pandas as pd
import pydeck as pdk
import streamlit as st

import ai_service
import core
import ui_common as ui

st.set_page_config(page_title="CityFix · Admin", page_icon=":material/dashboard:", layout="wide")
ui.apply_style()
model, meta = ui.require_model()
conn = ui.get_conn()
cities = core.load_cities()

LIVE, SIM = "Live", "Simulated"
ASSIGNED = {"model": "Model (automatic)", "operator_correction": "Staff decision",
            "pending_review": "Waiting for staff", "seed_fixture": "Simulated history"}
SECTIONS = {"overview": ":material/map: Overview", "review": ":material/rule: For Staff Review",
            "reports": ":material/table_rows: Reports", "brief": ":material/assignment: Weekly Action Brief",
            "model": ":material/analytics: Model & data"}
st.session_state.setdefault("admin_section", "overview")


# ---------------------------------------------------------------- navigation actions (used by the brief)
def open_reports(label: str, ids: list[str]):
    st.session_state["report_filter"] = {"label": label, "ids": list(ids)}
    st.session_state["report_pick"] = ids[0]
    st.session_state["admin_section"] = "reports"


def view_on_map(hotspot_id: str, lat: float, lon: float):
    st.session_state["map_focus"] = {"id": hotspot_id, "lat": lat, "lon": lon}
    st.session_state["admin_section"] = "overview"


def go(section: str):
    st.session_state["admin_section"] = section


# ---------------------------------------------------------------- filters
f1, f2 = st.columns([1, 3])
city_id = f1.selectbox("City", list(cities), format_func=lambda c: cities[c]["city_name"])
cfg = cities[city_id]
view = f2.radio("Show", ["Sample week + live reports", "Live reports only (last 7 days)", "Sample week only"],
                horizontal=True,
                help="The sample week is simulated history so the map isn't empty. Live reports are the ones "
                     "submitted on the Report page and classified by the trained model.")
ui.header("Admin dashboard", cfg["city_name"])

anchor, now = core.demo_anchor(), core.utcnow()
days = cfg["hotspots"]["window_days"]
if view == "Sample week + live reports":
    start, end = anchor - timedelta(days=days), max(now, anchor)
    period = (f"Sample week {start:%d %b} – {anchor:%d %b %Y}, plus live reports up to "
              f"{end:%d %b %H:%M} UTC")
elif view == "Live reports only (last 7 days)":
    start, end = now - timedelta(days=days), now
    period = f"Live reports, {start:%d %b} – {end:%d %b %Y, %H:%M} UTC"
else:
    start, end = anchor - timedelta(days=days), anchor
    period = f"Sample week {start:%d %b} – {end:%d %b %Y} (simulated history)"


def enrich(d: pd.DataFrame) -> pd.DataFrame:
    if not len(d):
        return d.assign(origin=[], assigned_by=[], area=[])
    d = d.copy()
    d["origin"] = [SIM if str(k).startswith("seed:") else LIVE for k in d["dedupe_key"]]
    d["assigned_by"] = [ASSIGNED["seed_fixture"] if o == SIM else ASSIGNED.get(c, c)
                        for o, c in zip(d["origin"], d["category_source"])]
    d["area"] = [core.nearest_area(la, lo, cfg) if pd.notna(la) else "" for la, lo in zip(d["latitude"], d["longitude"])]
    return d


updates = core.last_updates(conn, city_id)


def readable(d: pd.DataFrame, extra: tuple = ()) -> pd.DataFrame:
    """Plain column names for tables; scores and assignment details live in the report view."""
    cols = {"report_id": "Reference", "text": "Issue", "area": "Area", "proposed_department": "Department",
            "status": "Status", "created_at_utc": "Submitted", **{e: e for e in extra}}
    out = d[[c for c in cols if c in d.columns]].rename(columns=cols)
    if "Issue" in out:
        out["Issue"] = out["Issue"].map(lambda t: t if len(t) <= 90 else t[:87] + "…")
    if "Status" in out:
        out["Status"] = out["Status"].map(lambda s: core.CASE_STATUSES.get(s, s))
    if "Area" in out:
        out["Area"] = out["Area"].replace("", "—")
    if "Department" in out:
        out["Department"] = out["Department"].map(ui.dept)
    if "Submitted" in out:
        out["Submitted"] = out["Submitted"].str[:16].str.replace("T", " ")
    return out


df = enrich(core.fetch_reports(conn, city_id, start, end))
if view == "Live reports only (last 7 days)":
    df = df[df["origin"] == LIVE]
elif view == "Sample week only":
    df = df[df["origin"] == SIM]
hotspots, excl = core.compute_hotspots(df, cfg)
facts = core.summary_facts(df, hotspots, cfg, start, end)
cm = core.case_metrics(df)
n = len(df)
live = df[df["origin"] == LIVE] if n else df
active = df[df["status"].isin(core.ACTIVE_STATUSES)] if n else df
waiting = core.awaiting_review(df)
priorities = core.priority_hotspots(df, hotspots, cfg, top=3)

# ---------------------------------------------------------------- summary cards (same numbers as the brief)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(ui.kpi("Needs review", cm["awaiting_review"], "waiting for a staff decision now", review=True),
            unsafe_allow_html=True)
c2.markdown(ui.kpi("Active cases", cm["active"], "received or in progress"), unsafe_allow_html=True)
c3.markdown(ui.kpi("Resolved", cm["resolved"], "in this view"), unsafe_allow_html=True)
c4.markdown(ui.kpi("Hotspots", len(hotspots), "recurring same-category clusters"), unsafe_allow_html=True)
st.caption(f"{period} · {len(live)} live reports classified by the trained model "
           f"({int((live['routing_state'] == 'routed').sum()) if len(live) else 0} routed automatically) and "
           f"{n - len(live)} simulated history reports.")

section = st.segmented_control("Section", list(SECTIONS), format_func=SECTIONS.get, label_visibility="collapsed",
                               key="admin_section") or "overview"
st.write("")


def report_details(rep: dict):
    """Expandable detail view: scores, assignment source, reasons, audit trail."""
    min_s, min_m = core.thresholds_for(cfg, meta)
    reasons = [ui.reason_text(x, min_s, min_m) for x in json.loads(rep["routing_reasons"] or "[]")
               if x not in ("operator_correction", "operator_marked_spam")]
    d1, d2 = st.columns(2)
    d1.markdown(
        f"**Model's best match:** {ui.CATEGORY_LABEL.get(rep['predicted_category'], rep['predicted_category'] or '—')}"
        + (f" · score {rep['predicted_score']:.2f} · lead {rep['predicted_margin']:.2f}"
           if rep["predicted_score"] is not None else " · (simulated history, not scored)")
        + f"  \n**Approved category:** {ui.CATEGORY_LABEL.get(rep['approved_category'], '—')}"
        + f"  \n**Assigned by:** {ASSIGNED.get(rep['category_source'], rep['category_source'])}"
        + f"  \n**Routing:** {ui.STATE_UI.get(rep['routing_state'], (rep['routing_state'],))[0]}")
    d2.markdown(f"**Location:** {ui.LOCATION_LABEL.get(rep['location_status'], rep['location_status'])}"
                + (f"  \n**Why it needed a person:** {'; '.join(reasons)}" if reasons else "")
                + f"  \n**Last update:** {(updates.get(rep['report_id']) or 'never')[:16].replace('T', ' ')}"
                + f"  \n**Versions:** model `{rep.get('model_version') or '—'}` · config `{rep.get('config_version')}`")
    trail = core.staff_actions(conn, city_id, rep["report_id"])
    if len(trail):
        trail["action"] = trail["action"].map(ui.dept)
        st.markdown("**Audit trail**")
        st.dataframe(trail.drop(columns="report"), hide_index=True, width="stretch")
    else:
        st.caption("No staff actions yet.")


# ======================================================================== OVERVIEW
if section == "overview":
    m1, m2 = st.columns([3, 2])
    with m1:
        focus = st.session_state.get("map_focus")
        if focus:
            fc1, fc2 = st.columns([3, 1], vertical_alignment="center")
            fc1.caption(f"Map focused on hotspot {focus['id']}.")
            if fc2.button("Reset map", icon=":material/zoom_out_map:", width="stretch"):
                st.session_state.pop("map_focus", None)
                st.rerun()
        pts = df.dropna(subset=["latitude", "longitude"]).copy()
        pts["color"] = pts["approved_category"].map(lambda c: ui.CAT_COLORS.get(c, [150, 150, 150]))
        pts["dot_m"] = [75 if o == LIVE else 45 for o in pts["origin"]] if len(pts) else []
        pts["edge"] = [[20, 40, 75, 255] if o == LIVE else [0, 0, 0, 0] for o in pts["origin"]] if len(pts) else []
        pts["label"] = [f"{r} · {ui.CATEGORY_LABEL.get(c, 'not decided')} · {a or 'no area'} · {o}"
                        for r, c, a, o in zip(pts["report_id"], pts["approved_category"], pts["area"], pts["origin"])] \
            if len(pts) else []
        layers = [pdk.Layer("ScatterplotLayer", data=pts, get_position="[longitude, latitude]",  # pydeck: lon, lat
                            get_fill_color="color", get_radius="dot_m", radius_min_pixels=3, radius_max_pixels=11,
                            stroked=True, get_line_color="edge", line_width_min_pixels=2, pickable=True,
                            opacity=0.85)]
        if len(hotspots):
            hs_map = hotspots.assign(
                label=[f"Hotspot {h}: {ui.CATEGORY_LABEL.get(c, c)} near {a or 'unnamed area'}"
                       for h, c, a in zip(hotspots["hotspot_id"], hotspots["category"], hotspots["near_area"])],
                ring_w=[30.0 if focus and h == focus["id"] else 12.0 for h in hotspots["hotspot_id"]],
                ring_m=float(cfg["hotspots"]["eps_meters"]))
            layers.append(pdk.Layer("ScatterplotLayer", data=hs_map, get_position="[centroid_lon, centroid_lat]",
                                    get_radius="ring_m", radius_min_pixels=8, radius_max_pixels=120, filled=False,
                                    stroked=True, get_line_color=[20, 40, 75], get_line_width="ring_w",
                                    line_width_min_pixels=2, line_width_max_pixels=6, pickable=True))
        vs = (pdk.ViewState(latitude=focus["lat"], longitude=focus["lon"], zoom=15) if focus else
              pdk.ViewState(latitude=cfg["map_center"][0], longitude=cfg["map_center"][1], zoom=12))
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=vs, tooltip={"text": "{label}"}), height=430,
                        key=f"overview_map_{focus['id'] if focus else 'all'}")
        st.caption("Large outlined dots: live reports. Small dots: simulated history. Navy roads · teal waste · "
                   "blue lighting · plum noise · rings = hotspots. Area names are approximate.")
    with m2:
        st.markdown(f"##### Needs review ({len(waiting)})")
        if not len(waiting):
            st.markdown("<div class='cf-card'>Nothing is waiting for a staff decision.</div>", unsafe_allow_html=True)
        for r in waiting.head(6).itertuples():
            st.markdown(
                f"<div class='cf-card review'><span class='cf-ref' style='font-size:.9rem'>{r.report_id}</span> "
                f"{ui.pill(r.routing_state)}<div style='margin-top:6px'>{html.escape(r.text[:110])}</div>"
                f"<div class='cf-tag'>{r.created_at_utc[:16].replace('T', ' ')} UTC"
                + (f" · {html.escape(r.area)}" if r.area else "") + "</div></div>", unsafe_allow_html=True)
        if len(waiting) > 6:
            st.caption(f"+ {len(waiting) - 6} more")
        if len(waiting):
            st.button("Open For Staff Review", icon=":material/rule:", on_click=go, args=("review",))

    q1, q2 = st.columns([3, 2])
    with q1:
        st.markdown("##### Active department queues")
        rows = []
        for d in dict.fromkeys([*cfg["departments"].values(), cfg["review_department"]]):
            a_ = active[active["proposed_department"] == d] if len(active) else active
            rows.append({"Department": ui.dept(d) + (" (For Staff Review)" if d == cfg["review_department"] else ""),
                         "Active": len(a_), "Live": int((a_["origin"] == LIVE).sum()) if len(a_) else 0,
                         "Simulated": int((a_["origin"] == SIM).sum()) if len(a_) else 0})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with q2:
        st.markdown("##### Hotspots")
        if len(hotspots):
            st.dataframe(pd.DataFrame({
                "Hotspot": hotspots["hotspot_id"], "Issue": hotspots["category"].map(ui.CATEGORY_LABEL),
                "Area": hotspots["near_area"].replace("", "—"), "Reports": hotspots["report_count"],
                "Department": hotspots["department"].map(ui.dept)}), hide_index=True, width="stretch")
        else:
            st.markdown("<div class='cf-card'>No hotspot in this view.</div>", unsafe_allow_html=True)
        st.caption("Hotspots are recurring report patterns, not confirmed incidents or causes.")

# ======================================================================== FOR STAFF REVIEW
elif section == "review":
    st.markdown(f"##### Waiting for a staff decision ({len(waiting)})")
    st.caption("Same count as the Needs review card: active cases whose routing is Needs staff review or Urgent "
               f"review, in this view. Already decided by staff (no longer waiting): {cm['previously_reviewed']}.")
    if len(waiting):
        st.dataframe(readable(waiting), hide_index=True, width="stretch")
    else:
        st.markdown("<div class='cf-card'>Nothing is waiting for a staff decision.</div>", unsafe_allow_html=True)

    st.markdown("##### Work on a case")
    if msg := st.session_state.pop("staff_msg", None):
        st.success(ui.dept(msg), icon=":material/check_circle:")
    order = (waiting["report_id"].tolist()
             + [r for r in df[df["origin"] == LIVE]["report_id"] if r not in set(waiting["report_id"])]
             + [r for r in df[df["origin"] == SIM]["report_id"] if r not in set(waiting["report_id"])]) if n else []
    if not order:
        st.info("No reports in this view.")
    else:
        origin_of = dict(zip(df["report_id"], df["origin"]))
        if st.session_state.get("case_pick") not in order:
            st.session_state.pop("case_pick", None)
        rid = st.selectbox("Case", order, key="case_pick", format_func=lambda r: f"{r} · " + (
            "needs decision" if r in set(waiting["report_id"]) else origin_of.get(r, "").lower()))
        rep = core.get_report(conn, city_id, rid)
        area = core.nearest_area(rep["latitude"], rep["longitude"], cfg) or "—"
        is_review = rep["routing_state"] in core.REVIEW_STATES and rep["status"] in core.ACTIVE_STATUSES
        st.markdown(
            f"<div class='cf-card{' review' if is_review else ''}'><span class='cf-ref'>{rid}</span> &nbsp;"
            f"{ui.pill(rep['routing_state'])} {ui.case_pill(rep['status'])}"
            f"<div style='margin:8px 0 4px 0'>{html.escape(rep['text'])}</div>"
            f"<div class='cf-row'><span><span class='cf-label'>Department</span><br>"
            f"{html.escape(ui.dept(rep['proposed_department']))}</span>"
            f"<span><span class='cf-label'>Area</span><br>{html.escape(area)}</span>"
            f"<span><span class='cf-label'>Submitted</span><br>{rep['created_at_utc'][:16].replace('T', ' ')} UTC</span>"
            f"</div></div>", unsafe_allow_html=True)
        ui.show_attachment(rep.get("attachment_path"))
        with st.expander("Report details: scores, assignment and audit trail"):
            report_details(rep)

        with st.form("staff_decision"):
            cats = [*cfg["departments"], core.SPAM]
            cur = rep["approved_category"] or rep["predicted_category"]
            g1, g2 = st.columns(2)
            new_cat = g1.selectbox(
                "Category → department", cats, index=cats.index(cur) if cur in cats else 0,
                format_func=lambda c: ("Spam / not a city issue → close the case" if c == core.SPAM
                                       else f"{ui.CATEGORY_LABEL.get(c, c)} → {ui.dept(cfg['departments'][c])}"))
            sts = list(core.CASE_STATUSES)
            new_st = g2.selectbox("Status", sts, index=sts.index(rep["status"]) if rep["status"] in sts else 0,
                                  format_func=core.CASE_STATUSES.get, help="Spam always closes the case.")
            note = st.text_area("Note for the resident (optional — shown when they track the report)",
                                max_chars=core.MAX_NOTE, height=80,
                                placeholder="e.g. The area was inspected and the bin was emptied.")
            op = st.text_input("Staff name", value="demo-operator")
            if st.form_submit_button("Save decision", type="primary", icon=":material/save:"):
                _, acts = core.staff_decision(conn, cfg, rid, new_cat, new_st, note, op)
                st.session_state["staff_msg"] = (f"{rid}: " + ", ".join(acts)) if acts else f"{rid}: nothing changed"
                st.rerun()
        st.caption("The model's original prediction is always kept. No SMS or email is sent (roadmap).")

# ======================================================================== REPORTS
elif section == "reports":
    flt = st.session_state.get("report_filter")
    shown = df[df["report_id"].isin(flt["ids"])] if (flt and n) else df
    h1, h2 = st.columns([4, 1], vertical_alignment="center")
    h1.markdown(f"##### Reports ({len(shown)})" + (f" — {flt['label']}" if flt else ""))
    if flt and h2.button("Clear filter", icon=":material/filter_alt_off:", width="stretch"):
        st.session_state.pop("report_filter", None)
        st.rerun()
    if len(shown):
        tbl = readable(shown, extra=("origin",)).rename(columns={"origin": "Source"})
        tbl.insert(len(tbl.columns) - 1, "Last update",
                   shown["report_id"].map(lambda r: (updates.get(r) or "—")[:16].replace("T", " ")).values)
        st.dataframe(tbl, hide_index=True, width="stretch", height=min(420, 40 + 35 * len(tbl)))
        ids = shown["report_id"].tolist()
        if st.session_state.get("report_pick") not in ids:
            st.session_state["report_pick"] = ids[0]
        pick = st.selectbox("Open a report", ids, key="report_pick")
        with st.expander(f"Report {pick}", expanded=True):
            r_ = core.get_report(conn, city_id, pick)
            st.write(r_["text"])
            ui.show_attachment(r_.get("attachment_path"))
            report_details(r_)
    else:
        st.info("No reports in this view.")
    with st.expander("All staff actions"):
        acts_all = core.staff_actions(conn, city_id)
        acts_all["action"] = acts_all["action"].map(ui.dept)
        st.dataframe(acts_all, hide_index=True, width="stretch")

# ======================================================================== WEEKLY ACTION BRIEF
elif section == "brief":
    st.markdown("### Weekly Action Brief")
    st.caption(f"{cfg['city_name']} · {period}")

    # 2 · short operational summary (all numbers computed in Python)
    live_active = int((active["origin"] == LIVE).sum()) if len(active) else 0
    n_depts = int(active["proposed_department"].nunique()) if len(active) else 0
    urgent_txt = f", including {cm['urgent_waiting']} urgent" if cm["urgent_waiting"] else ""
    st.markdown(
        f"<div class='cf-card'><b>{cm['active']} active cases</b> ({live_active} live, {cm['active'] - live_active} "
        f"simulated history) across {n_depts} queues. <b>{cm['awaiting_review']} waiting for staff review now</b>"
        f"{urgent_txt}; {cm['previously_reviewed']} earlier review "
        f"{'case is' if cm['previously_reviewed'] == 1 else 'cases are'} already decided. "
        f"<b>{len(priorities)} priority hotspot{'s' if len(priorities) != 1 else ''}</b> below.</div>",
        unsafe_allow_html=True)

    # 3 · top priorities
    st.markdown("#### Top priorities")
    st.caption("Ranked by the number of active reports in each hotspot; ties go to the hotspot whose oldest active "
               "report is older. This reflects workload, not how dangerous an issue is. Resolved and closed cases "
               "are excluded.")
    if not priorities:
        st.markdown("<div class='cf-card'>No hotspot meets the clustering rule in this period, so there are no "
                    "priority areas to report. Check the cases needing attention below.</div>", unsafe_allow_html=True)
    for p in priorities:
        cat = ui.CATEGORY_LABEL.get(p["category"], p["category"])
        area = p["area"] or "Unnamed area"
        live_txt = f" ({p['live_count']} live)" if p["live_count"] else ""
        with st.container(border=True):
            st.markdown(
                f"<span class='cf-label'>Priority {p['rank']}</span><br><b style='font-size:1.1rem'>{html.escape(cat)}"
                f" · {html.escape(area)}</b>"
                f"<div class='cf-row'><span><span class='cf-label'>Active reports</span><br><b>{p['active_count']}</b>"
                f"{live_txt}</span>"
                f"<span><span class='cf-label'>Oldest active report</span><br>{p['oldest_utc'][:10]}</span>"
                f"<span><span class='cf-label'>Suggested department</span><br>{html.escape(ui.dept(p['department']))}</span>"
                f"<span><span class='cf-label'>Evidence</span><br>{p['evidence_id']}</span></div>"
                f"<div style='margin-top:10px'><b>Next step:</b> {html.escape(p['step'])}</div>",
                unsafe_allow_html=True)
            b1, b2, _ = st.columns([1, 1, 2])
            b1.button("Open related reports", key=f"open_{p['evidence_id']}", icon=":material/table_rows:",
                      on_click=open_reports, args=(f"{p['evidence_id']} · {cat} near {area}", p["report_ids"]),
                      width="stretch")
            b2.button("View on map", key=f"map_{p['evidence_id']}", icon=":material/map:",
                      on_click=view_on_map, args=(p["evidence_id"], *p["centroid"]), width="stretch")

    # 4 · cases needing attention
    st.markdown("#### Cases needing attention")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f"**Waiting for staff review now ({len(waiting)})**")
        if len(waiting):
            w = readable(waiting)[["Reference", "Issue", "Area", "Submitted"]]
            w.insert(3, "Routing", waiting["routing_state"].map(lambda s: ui.STATE_UI[s][0]).values)
            st.dataframe(w, hide_index=True, width="stretch")
            st.button("Open For Staff Review", key="brief_to_review", icon=":material/rule:", on_click=go,
                      args=("review",))
        else:
            st.markdown("<div class='cf-card'>Nothing is waiting for a staff decision.</div>", unsafe_allow_html=True)
        st.caption(f"Previously sent to review and already decided by staff: {cm['previously_reviewed']} "
                   "(not counted as waiting).")
    with a2:
        st.markdown("**Oldest active cases**")
        if len(active):
            old = active.sort_values("created_at_utc").head(5)
            ref_end = end
            o = readable(old)[["Reference", "Issue", "Department", "Status", "Submitted"]]
            o["Last update"] = old["report_id"].map(lambda r: (updates.get(r) or "none")[:16].replace("T", " ")).values
            o["Age (days)"] = old["created_at_utc"].map(lambda t: (ref_end - core.parse_iso(t)).days).values
            st.dataframe(o, hide_index=True, width="stretch")
            st.caption("Age is measured to the end of the selected period. \"none\" = no staff update yet.")
        else:
            st.markdown("<div class='cf-card'>No active cases in this period.</div>", unsafe_allow_html=True)

    # 5 · optional handover summary (bounded GenAI)
    st.markdown("#### Handover summary")
    oldest_age = int((end - core.parse_iso(active["created_at_utc"].min())).days) if len(active) else 0
    context = {"city": cfg["city_name"], "period": period, "awaiting_review": cm["awaiting_review"],
               "oldest_active_days": oldest_age,
               "priorities": [{k: p[k] for k in ("evidence_id", "rank", "category", "area", "active_count",
                                                 "oldest_utc", "department")} for p in priorities]}
    if st.button("Draft handover summary", type="primary", icon=":material/auto_awesome:",
                 disabled=not priorities, help=None if priorities else "No priorities to summarise."):
        key = json.dumps(context, sort_keys=True)
        cache = st.session_state.setdefault("handover_cache", {})
        if key not in cache:
            with st.spinner("Drafting the handover summary…"):
                cache[key] = ai_service.generate_handover(context)
        st.session_state["handover"] = {"key": key, **cache[key]}
        st.session_state["handover_detail"] = cache[key]["detail"]
    h = st.session_state.get("handover")
    if h and h["key"] == json.dumps(context, sort_keys=True):
        live_draft = h["label"] == ai_service.LIVE
        st.markdown(f"<span class='cf-tag'><span class='{'live' if live_draft else 'fb'}'>"
                    f"{'AI-written draft' if live_draft else 'Template draft'}</span></span>"
                    + ("" if live_draft else " <span class='cf-tag'>AI model unavailable or output rejected — "
                                             "details in Model & data.</span>"), unsafe_allow_html=True)
        by_id = {p["evidence_id"]: p for p in priorities}
        for it in h["items"]:
            p = by_id.get(it["evidence_id"])
            if not p:
                continue
            cat = ui.CATEGORY_LABEL.get(p["category"], p["category"])
            st.markdown(f"**{p['rank']}. {cat} near {p['area'] or 'an unnamed area'}** — {p['active_count']} active "
                        f"reports, oldest {p['oldest_utc'][:10]}, {ui.dept(p['department'])}.  \n"
                        f"*Why it's on the list:* {it['why']}  \n*Verify next:* {it['verify']}")
            st.button(f"Evidence {p['evidence_id']} · {p['active_count']} reports", key=f"ev_{p['evidence_id']}",
                      icon=":material/link:", on_click=open_reports,
                      args=(f"{p['evidence_id']} · {cat} near {p['area'] or 'unnamed area'}", p["report_ids"]))
        st.markdown(f"**Also:** {cm['awaiting_review']} cases are waiting for staff review; the oldest active case "
                    f"is {oldest_age} days old.")
        st.caption("Draft for handover only. Counts, dates, departments and ranking are calculated by the system; "
                   "the drafted sentences explain and suggest checks only. Nothing is scheduled or dispatched.")
    elif not priorities:
        st.caption("No priorities to summarise in this period.")

    # 6 · evidence and methodology
    with st.expander("Evidence and methodology"):
        st.markdown(
            "- **Active case:** status Received or In progress. Resolved and Closed cases are excluded everywhere "
            "on this page.\n"
            "- **Waiting for staff review:** an active case whose routing is *Needs staff review* or *Urgent review*. "
            "The dashboard's *Needs review* card uses the same definition.\n"
            "- **Previously sent to review:** arrived needing review, already decided by staff.\n"
            f"- **Hotspot:** DBSCAN (haversine) per category over active, routed reports with a location; neighbour "
            f"distance {cfg['hotspots']['eps_meters']} m, at least {cfg['hotspots']['min_samples']} reports.\n"
            "- **Priority ranking:** active report count, then oldest active report. Workload, not danger.\n"
            "- **Suggested department:** from this city's routing table, not from the model.\n"
            "- **Data:** the sample week is simulated history ending at the demo anchor "
            f"({core.iso(anchor)}); live reports are those submitted in this app. No week-over-week comparison is "
            "shown until comparable history exists.")
        st.markdown("**Computed facts** (IDs used as evidence)")
        st.dataframe(pd.DataFrame(facts).astype(str).rename(columns={"id": "ID", "fact": "Fact", "value": "Value"})
                     .assign(Fact=lambda d: d["Fact"].map(ui.dept)), hide_index=True, width="stretch")

# ======================================================================== MODEL & DATA
elif section == "model":
    min_score, min_margin = core.thresholds_for(cfg, meta)
    st.markdown("##### How a report reaches a department")
    s1, s2, s3 = st.columns(3)
    s1.markdown("<div class='cf-card'><b>1 · The model reads the text</b><br>A trained text model scores the "
                "complaint against roads, waste, lighting and noise.</div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='cf-card'><b>2 · Is it confident?</b><br>Best match ≥ {min_score:.2f}, a lead of ≥ "
                f"{min_margin:.2f} over the next category, no danger words, not marked <i>multiple issues</i>. "
                "Otherwise it goes to For Staff Review.</div>", unsafe_allow_html=True)
    s3.markdown("<div class='cf-card'><b>3 · Category → department</b><br>The city's routing table turns the "
                "category into a department. Staff decisions use the same table.</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Category": ui.CATEGORY_LABEL.get(c, c), "Department": ui.dept(d)}
                               for c, d in cfg["departments"].items()]
                              + [{"Category": "Not confident / danger words / multiple issues",
                                  "Department": f"{ui.dept(cfg['review_department'])} (For Staff Review)"},
                                 {"Category": ui.CATEGORY_LABEL["spam"], "Department": "None — case closed by staff"}]),
                 hide_index=True, width="stretch")
    st.caption(f"Routing table from `config/cities.json` · {cfg['city_name']}")

    v = meta.get("validation", {})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Model")
        st.write(f"TF-IDF (1–2 grams) → Logistic Regression · `{meta.get('model_version')}`")
        st.caption(f"Thresholds score ≥ {min_score}, lead ≥ {min_margin}. {meta.get('threshold_decision', '')}")
        rows = [{"Set": "Validation (24)", "Accuracy": v.get("accuracy"), "Macro-F1": v.get("macro_f1"),
                 "Routed": v.get("routing_coverage"), "Routed correctly": v.get("accepted_route_accuracy")}]
        tm = core.ROOT / "artifacts" / "test_metrics.json"
        if tm.exists():
            t = json.loads(tm.read_text())
            rows.append({"Set": "Held-out test (20 + 4 policy)", "Accuracy": t["accuracy"], "Macro-F1": t["macro_f1"],
                         "Routed": t["routing_coverage"], "Routed correctly": t["accepted_route_accuracy"]})
            st.caption(f"Policy cases on test: {t['policy_cases_passed']}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Synthetic evaluations share generation bias; they don't establish real-city accuracy.")
        st.markdown("##### Roadmap (not built)")
        st.markdown("- **AI image analysis:** suggest a category from the resident's photo or clip, after a privacy "
                    "review and evaluation on real local photos.\n"
                    "- **Retrieval over responsibility rules and past staff decisions** to suggest a department for "
                    "unclear cases (staff decide).\n"
                    "- Resident notifications by SMS/email when the status changes (needs contact details and consent).\n"
                    "- Multi-issue detection without the resident's checkbox.\n"
                    "- Week-over-week comparisons once comparable history exists.\n"
                    "- Real local reports, calibrated scores, and shadow testing beside staff.")
    with c2:
        st.markdown("##### AI drafts — diagnostics")
        st.write(ai_service.status_text())
        acks = list(st.session_state.get("ack_cache", {}).items())
        if acks:
            rid_last, last = acks[-1]
            st.caption(f"Last resident reply in this session ({rid_last}): **{last['label']}** — {last['detail']}")
        if st.session_state.get("handover_detail"):
            st.caption(f"Last handover summary in this session: {st.session_state['handover_detail']}")
        if st.button("Run AI check", icon=":material/troubleshoot:"):
            with st.spinner("Checking the AI provider…"):
                st.session_state["ai_check"] = ai_service.check_access()
        if st.session_state.get("ai_check"):
            st.code(st.session_state["ai_check"], language=None)
        st.caption("Residents see an \"AI-written reply\" or \"Template response\" badge and the brief shows "
                   "\"AI-written draft\" or \"Template draft\"; technical reasons are shown only here.")
        st.markdown("##### Data")
        if core.PROVENANCE_PATH.exists():
            pv = json.loads(core.PROVENANCE_PATH.read_text())
            ac = pv.get("actual_counts", {})
            tr = ac.get("train", {})
            st.write(f"Mode: **{pv.get('mode')}** · NYC download: {pv.get('download', {}).get('status')} "
                     f"({pv.get('sources', {}).get('raw_rows_retained')} raw rows retained)")
            st.dataframe(pd.DataFrame([
                {"File": f"data/{core.TRAIN_CSV.name}", "Rows": tr.get("rows"),
                 "Use": f"training · {tr.get('by_source', {}).get('nyc_structured')} NYC details + "
                        f"{tr.get('by_source', {}).get('synthetic')} synthetic"},
                {"File": f"data/{core.VALIDATION_CSV.name}", "Rows": ac.get("validation", {}).get("rows"),
                 "Use": "validation + threshold selection (synthetic)"},
                {"File": f"data/{core.TEST_CSV.name}", "Rows": ac.get("tel_aviv_test", {}).get("rows"),
                 "Use": "held-out test, evaluated once (synthetic)"},
                {"File": f"data/{core.DEMO_CSV.name}", "Rows": ac.get("tel_aviv_demo", {}).get("rows"),
                 "Use": "map/dashboard fixtures only (simulated)"},
            ]), hide_index=True, width="stretch")
            st.caption(f"Seed {pv.get('seed')} · demo anchor {pv.get('demo_anchor_utc')} · provenance: "
                       f"data/{core.PROVENANCE_PATH.name}")
        st.markdown("##### Portability check")
        test_text = st.text_input("Same text, every city configuration", "There is a deep pothole on our street.")
        if test_text.strip():
            st.dataframe(pd.DataFrame([
                {"City configuration": c["city_name"],
                 **{lbl: (ui.dept(val) if lbl == "Department" else val) for lbl, val in
                    zip(("Category", "Routing", "Department"),
                        (lambda r: (r["predicted_category"], r["routing_state"], r["proposed_department"]))(
                            core.route(test_text, c, model, meta)))}}
                for c in cities.values()]), hide_index=True, width="stretch")
            st.caption("Same model, same category — the department comes from each city's settings. No retraining.")
