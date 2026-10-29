"""Resident-facing page: one clean chat-style form. Submitting routes, saves and replies automatically."""
from __future__ import annotations

import html

import streamlit as st

import ai_service
import core
import ui_common as ui

st.set_page_config(page_title="CityFix · Report an issue", page_icon="🛠️", layout="centered")
ui.apply_style()
model, meta = ui.require_model()
conn = ui.get_conn()
cfg = core.load_cities()["il_tel_aviv"]

ui.header("CityFix", f"{cfg['city_name']} · Report a street issue. "
                     "Demo only — nothing is sent to the municipality.")

chat = st.session_state.setdefault("chat", [])

with st.chat_message("assistant", avatar="🛠️"):
    st.write("Hi! Tell me what's wrong — potholes, garbage, street lights or noise. "
             "I'll record it and propose the right department right away.")


def render_reply(m: dict):
    rep, r, ack = m["report"], m["routing"], m["ack"]
    if m.get("duplicate"):
        st.caption("You already sent this report today — here is the existing record.")
    if rep["routing_state"] == "urgent_review":
        st.error(core.URGENT_NOTICE)
    dept = rep["proposed_department"] if rep["routing_state"] == "routed" else "Pending staff review"
    st.markdown(
        f"<div class='cf-card'>{html.escape(ack['text'])}"
        f"<div class='cf-row'><span><b>Reference</b> {rep['report_id']}</span>"
        f"<span><b>Status</b> {ui.pill(rep['routing_state'])}</span>"
        f"<span><b>Department</b> {html.escape(dept)}</span></div>"
        f"{ui.draft_tag(ack['label'], ack['detail'])}</div>", unsafe_allow_html=True)
    with st.expander("How was this routed?"):
        st.write(f"Predicted category: **{r['predicted_category']}** · model score {r['score']:.2f} · "
                 f"margin {r['margin']:.2f} · location: {rep['location_status']}")
        if r["reasons"]:
            st.write("Sent to staff review because: " + ", ".join(r["reasons"]))
        st.caption("The model score is an uncalibrated estimate. Department labels are illustrative (demo).")


for m in chat:
    with st.chat_message(m["role"], avatar="🙂" if m["role"] == "user" else "🛠️"):
        if m["role"] == "user":
            st.write(m["text"])
            if m.get("where"):
                st.caption(f"📍 {m['where']}")
        else:
            render_reply(m)

with st.expander("📍 Add a location or mark multiple issues (optional)"):
    c1, c2 = st.columns([3, 2])
    loc_mode = c1.selectbox("Location", ["No location", *cfg["map_presets"], "Enter coordinates"])
    lat = lon = None
    if loc_mode in cfg["map_presets"]:
        lat, lon = cfg["map_presets"][loc_mode]
    elif loc_mode == "Enter coordinates":
        cc1, cc2 = st.columns(2)
        lat = cc1.number_input("Latitude", value=float(cfg["map_center"][0]), format="%.6f")
        lon = cc2.number_input("Longitude", value=float(cfg["map_center"][1]), format="%.6f")
    unsure = c2.checkbox("Multiple issues / unsure", help="Sends the report to staff review.")

prompt = st.chat_input("Describe the issue in English…", max_chars=core.MAX_TEXT)
if prompt:
    ok, clean, err = core.validate_text(prompt)
    if not ok:
        st.error(err)
    else:
        routing = core.route(clean, cfg, model, meta, unsure=unsure)
        rid, is_new = core.save_report(conn, cfg, clean, lat, lon, routing, meta)
        rep = core.get_report(conn, cfg["city_id"], rid)
        cache = st.session_state.setdefault("ack_cache", {})
        if rid not in cache:  # one draft per saved report
            cache[rid] = ai_service.draft_acknowledgment(rep)
        where = None if loc_mode == "No location" else (loc_mode if loc_mode in cfg["map_presets"]
                                                        else f"{lat:.5f}, {lon:.5f}")
        chat.append({"role": "user", "text": clean, "where": where})
        chat.append({"role": "assistant", "report": rep, "routing": routing, "ack": cache[rid],
                     "duplicate": not is_new})
        st.rerun()

if chat and st.button("Start a new report conversation"):
    st.session_state["chat"] = []
    st.rerun()
