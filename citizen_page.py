"""Resident page: chat-style reporting with an automatic receipt reply, an area map picker, and report tracking.
Submitting routes and saves the report straight away; nothing is sent to a municipality."""
from __future__ import annotations

import html

import pydeck as pdk
import streamlit as st

import ai_service
import core
import ui_common as ui

st.set_page_config(page_title="CityFix · Report an issue", page_icon=":material/edit_note:", layout="centered")
ui.apply_style(max_width_px=820)
model, meta = ui.require_model()
conn = ui.get_conn()
cfg = core.load_cities()["il_tel_aviv"]
AREAS = cfg["map_presets"]
BOT, YOU = ":material/support_agent:", ":material/person:"

ui.header("Report an issue", cfg["city_name"])

history = st.session_state.setdefault("history", [])  # oldest first: {"text","where","report_id","routing","ack",...}
st.session_state.setdefault("form_key", 0)
st.session_state.setdefault("mode", "new")
VIEWS = {"new": ":material/chat: Report", "track": ":material/search: Track a report"}


def go_track(ref: str):
    st.session_state["mode"] = "track"
    st.session_state["ref_in"] = ref


def new_visit():
    st.session_state["history"] = []
    st.session_state["form_key"] += 1
    st.session_state["mode"] = "new"


top1, top2 = st.columns([3, 1], vertical_alignment="center")
with top1:
    mode = st.segmented_control("View", list(VIEWS), format_func=VIEWS.get, label_visibility="collapsed",
                                key="mode") or "new"
top2.button("Start new visit", icon=":material/restart_alt:", on_click=new_visit, width="stretch",
            help="Clears this conversation. Saved reports can still be tracked by reference.")


def status_row(rep: dict) -> str:
    d = ui.dept(rep["proposed_department"]) if rep["routing_state"] == "routed" else "Staff will choose"
    return (f"<div class='cf-row'><span><span class='cf-label'>Routing</span><br>{ui.pill(rep['routing_state'])}</span>"
            f"<span><span class='cf-label'>Progress</span><br>{ui.case_pill(rep['status'])}</span>"
            f"<span><span class='cf-label'>Department</span><br><b>{html.escape(d)}</b></span></div>"
            f"<div style='height:14px'></div>")


# ======================================================================== track a report
if mode == "track":
    st.markdown("#### Track a report")
    mine = list(dict.fromkeys(m["report_id"] for m in reversed(history)))
    c1, c2 = st.columns([3, 2])
    typed = c1.text_input("Reference number", key="ref_in", placeholder="CF-1A2B3C4D")
    picked = c2.selectbox("Or pick one from this visit", ["—", *mine], key="ref_pick")
    ref = typed.strip() or (picked if picked != "—" else "")
    if ref:
        rep = core.find_by_reference(conn, cfg["city_id"], ref)
        if rep is None:
            st.warning(f"No report found with reference {ref}. Check the number and try again.")
        else:
            area = core.nearest_area(rep["latitude"], rep["longitude"], cfg) or "Not given"
            st.markdown(
                f"<div class='cf-card'><span class='cf-label'>Reference</span><br>"
                f"<span class='cf-ref'>{rep['report_id']}</span>{status_row(rep)}"
                f"<div class='cf-row'><span><span class='cf-label'>Area</span><br>{html.escape(area)}</span>"
                f"<span><span class='cf-label'>Submitted</span><br>{rep['created_at_utc'][:16].replace('T', ' ')} UTC</span></div>"
                f"<div class='cf-tag' style='font-size:.85rem;color:#2B3445'>{html.escape(rep['text'])}</div></div>",
                unsafe_allow_html=True)
            ui.show_attachment(rep.get("attachment_path"), thumb_px=140)
            st.markdown("**What has happened so far**")
            for e in core.case_timeline(conn, cfg, rep):
                st.markdown(f"- `{e['at'][:16].replace('T', ' ')}` **{ui.dept(e['title'])}**"
                            + (f" — {e['detail']}" if e["detail"] else ""))
            st.caption("Times are UTC. Updates appear here when staff change the status.")
    st.stop()

# ======================================================================== chat
with st.chat_message("assistant", avatar=BOT):
    st.write("Hi! Tell me what's wrong — potholes, garbage, street lights or noise. I'll record it and "
             "suggest the right department straight away. You can add an area or a photo below.")

for i, m in enumerate(history):
    with st.chat_message("user", avatar=YOU):
        st.write(m["text"])
        if m.get("where"):
            st.caption(f":material/location_on: {m['where']}")
    rep = core.get_report(conn, cfg["city_id"], m["report_id"])  # fresh status from staff
    ack = m["ack"]
    with st.chat_message("assistant", avatar=BOT):
        if m.get("duplicate"):
            st.caption("You already sent this report today — this is the existing record.")
        if m.get("attach_err"):
            st.warning("The photo/clip was not saved: " + m["attach_err"])
        st.markdown(f"{html.escape(ack.get('short') or ack['text'])} {ui.resident_badge(ack['label'])}",
                    unsafe_allow_html=True)
        r1, r2 = st.columns([3, 2], vertical_alignment="bottom")
        with r1:
            st.markdown("<span class='cf-label'>Reference</span>", unsafe_allow_html=True)
            st.code(rep["report_id"], language=None)  # built-in copy button
        r2.button("Track report", key=f"track_{i}_{rep['report_id']}", icon=":material/search:",
                  on_click=go_track, args=(rep["report_id"],), width="stretch")
        st.markdown(status_row(rep), unsafe_allow_html=True)
        if rep["routing_state"] == "urgent_review":
            st.error(core.URGENT_NOTICE, icon=":material/warning:")
        ui.show_attachment(rep.get("attachment_path"))
        with st.expander("How was this routed?"):
            ui.render_routing_explanation(m["routing"], rep, meta, cfg)

# ---------------------------------------------------------------- options: area map picker, photo, unsure
k = st.session_state["form_key"]
sel_key, pick_key = f"loc_{k}", f"_last_pick_{k}"
st.session_state.setdefault(sel_key, "No location")
with st.expander(":material/add_location_alt: Add an area, photo or details (optional)"):
    st.caption("Tap an area on the map or choose it from the list. Area points are approximate.")
    chosen = st.session_state[sel_key]
    pts = [{"name": n, "lat": la, "lon": lo, "color": [14, 140, 135] if n == chosen else [20, 40, 75],
            "dot_m": 260 if n == chosen else 150} for n, (la, lo) in AREAS.items()]
    event = st.pydeck_chart(
        pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", id="areas", data=pts, get_position="[lon, lat]",
                                   get_fill_color="color", get_radius="dot_m", radius_min_pixels=5,
                                   radius_max_pixels=12, pickable=True, opacity=0.85)],
                 initial_view_state=pdk.ViewState(latitude=cfg["map_center"][0], longitude=cfg["map_center"][1],
                                                  zoom=11.3), tooltip={"text": "{name}"}),
        on_select="rerun", selection_mode="single-object", key=f"area_map_{k}", height=260)
    picked = (event.selection.get("objects", {}).get("areas") or [{}])[0].get("name") if event else None
    if picked and picked != st.session_state.get(pick_key):  # a new tap on the map wins over the list
        st.session_state[pick_key] = picked
        st.session_state[sel_key] = picked
        st.rerun()
    c1, c2 = st.columns([3, 2])
    loc_mode = c1.selectbox(":material/location_on: Area", ["No location", *AREAS, "Enter coordinates"], key=sel_key,
                            help=cfg.get("map_presets_note", ""))
    lat = lon = None
    if loc_mode in AREAS:
        lat, lon = AREAS[loc_mode]
    elif loc_mode == "Enter coordinates":
        cc1, cc2 = st.columns(2)
        lat = cc1.number_input("Latitude", value=float(cfg["map_center"][0]), format="%.6f", key=f"lat_{k}")
        lon = cc2.number_input("Longitude", value=float(cfg["map_center"][1]), format="%.6f", key=f"lon_{k}")
    c2.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    unsure = c2.checkbox("Multiple issues / unsure", key=f"unsure_{k}", help="Sends the report to staff review.")
    upload = st.file_uploader(f":material/photo_camera: Photo or short clip (max {core.MAX_ATTACHMENT_MB} MB)",
                              type=sorted(core.ALLOWED_ATTACHMENTS), key=f"up_{k}")
    st.caption("Photos and clips are for staff review only. They aren't analyzed by AI and don't change where "
               "your report goes. Please avoid faces and car plates.")

chips = [f":material/location_on: {loc_mode}" if loc_mode != "No location" else "",
         f":material/photo_camera: {upload.name}" if upload else "",
         ":material/help: multiple issues / unsure" if unsure else ""]
if any(chips):
    st.caption("Will be sent with your next message: " + " · ".join(c for c in chips if c))

prompt = st.chat_input("Describe the issue in English…", max_chars=core.MAX_TEXT)
if prompt:
    ok, clean, err = core.validate_text(prompt)
    if not ok:
        st.error(err)
    else:
        routing = core.route(clean, cfg, model, meta, unsure=unsure)
        rid, is_new = core.save_report(conn, cfg, clean, lat, lon, routing, meta)
        attach_err = None
        if upload is not None and is_new:
            try:
                core.save_attachment(conn, cfg["city_id"], rid, upload.name, upload.getvalue())
            except ValueError as exc:
                attach_err = str(exc)
        rep = core.get_report(conn, cfg["city_id"], rid)
        cache = st.session_state.setdefault("ack_cache", {})
        if rid not in cache:  # one draft per saved report
            with st.spinner("Writing your reply… (the first time, the local AI model loads)"):
                cache[rid] = ai_service.draft_acknowledgment(rep)
        where = None
        if loc_mode in AREAS:
            where = f"{loc_mode} (approximate area)"
        elif loc_mode == "Enter coordinates":
            near = core.nearest_area(lat, lon, cfg)
            where = f"{lat:.5f}, {lon:.5f}" + (f" · near {near}" if near else "")
        history.append({"text": clean, "where": where, "report_id": rid, "routing": routing, "ack": cache[rid],
                        "duplicate": not is_new, "attach_err": attach_err})
        st.session_state["form_key"] += 1  # clear area, photo and checkbox for the next report
        st.rerun()
