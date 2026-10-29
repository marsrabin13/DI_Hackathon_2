"""Shared UI helpers: styling, cached model/DB, small render components."""
from __future__ import annotations

import json

import streamlit as st

import ai_service
import core

NAVY, TEAL = "#14284B", "#0E9F9A"
CAT_COLORS = {"roads": [20, 40, 75], "waste": [14, 159, 154], "lighting": [230, 170, 30], "noise": [190, 70, 90]}
STATE_UI = {
    "routed": ("Routed", "#0E9F9A"),
    "review_required": ("Staff review", "#C98A12"),
    "urgent_review": ("Urgent review", "#B4233C"),
}

CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; }}
  h1, h2, h3, h4 {{ color:{NAVY}; }}
  .cf-head {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:2px; }}
  .cf-title {{ font-size:1.7rem; font-weight:700; color:{NAVY}; }}
  .cf-badge {{ background:{TEAL}; color:#fff; padding:2px 10px; border-radius:12px; font-size:.75rem; font-weight:600; }}
  .cf-sub {{ color:#5B6475; font-size:.9rem; margin-bottom:14px; }}
  .cf-pill {{ display:inline-block; color:#fff; padding:2px 10px; border-radius:10px; font-size:.78rem; font-weight:600; }}
  .cf-card {{ border:1px solid #E3E8EF; border-left:4px solid {TEAL}; border-radius:10px; padding:12px 14px;
              background:#FFFFFF; margin-top:4px; }}
  .cf-row {{ display:flex; gap:22px; flex-wrap:wrap; margin-top:8px; font-size:.88rem; color:#2B3445; }}
  .cf-row b {{ color:{NAVY}; }}
  .cf-tag {{ font-size:.72rem; color:#6B7280; margin-top:6px; }}
  .cf-tag .live {{ background:{TEAL}; color:#fff; padding:1px 7px; border-radius:7px; }}
  .cf-tag .fb {{ background:#6B7280; color:#fff; padding:1px 7px; border-radius:7px; }}
  div.stButton > button[kind="primary"], div.stFormSubmitButton > button {{ background:{NAVY}; border:0; }}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str):
    st.markdown(f"<div class='cf-head'><span class='cf-title'>{title}</span>"
                f"<span class='cf-badge'>Simulated Tel Aviv reports</span></div>"
                f"<div class='cf-sub'>{subtitle}</div>", unsafe_allow_html=True)


@st.cache_resource
def load_model_cached():
    return core.load_model()


@st.cache_resource
def get_conn():
    conn = core.connect()
    cities = core.load_cities()
    if not core.DEMO_CSV.exists():
        st.error(f"Missing map fixture file: data/{core.DEMO_CSV.name}")
        st.stop()
    core.import_seed(conn, core.DEMO_CSV, cities["il_tel_aviv"])  # version = file hash
    return conn


def require_model():
    try:
        return load_model_cached()
    except FileNotFoundError:
        st.error("No trained model found. Run `python train.py` first (uses the supplied data/gpt_ files).")
        st.stop()


def pill(state: str) -> str:
    label, color = STATE_UI.get(state, (state, "#6B7280"))
    return f"<span class='cf-pill' style='background:{color}'>{label}</span>"


def draft_tag(label: str, detail: str = "") -> str:
    cls = "live" if label == ai_service.LIVE else "fb"
    extra = f" · {detail}" if detail and label != ai_service.LIVE else ""
    return f"<div class='cf-tag'><span class='{cls}'>{label}</span>{extra}</div>"


# ---------------------------------------------------------------- plain-language routing explanation
CATEGORY_LABEL = {"roads": "Roads & sidewalks", "waste": "Waste & cleanliness",
                  "lighting": "Street lighting", "noise": "Noise"}
LOCATION_LABEL = {"valid": "Inside the demo area", "missing": "Not provided",
                  "outside_bbox": "Outside the demo area (not used on the map)", "invalid": "Not readable"}


def reason_text(reason: str, min_score: float, min_margin: float) -> str:
    if reason == "urgent_phrase_guard":
        return "Possible danger words were detected, so staff should look at it urgently."
    if reason == "no_known_vocabulary":
        return "None of the words matched what the model has learned."
    if reason.startswith("low_score"):
        return f"The best match is too weak to route automatically (needs at least {min_score:.2f})."
    if reason.startswith("low_margin"):
        return f"Two categories are too close to call (the winner needs a lead of at least {min_margin:.2f})."
    if reason == "no_department_mapping":
        return "This city has no department set up for that category."
    if reason == "citizen_marked_multiple_or_unsure":
        return "You marked the report as multiple issues / unsure."
    if reason == "operator_correction":
        return "A staff member set the category."
    return reason


def render_routing_explanation(r: dict, rep: dict, meta: dict, cfg: dict):
    """Readable 'How was this routed?' panel: outcome sentence, score bars, reasons, location."""
    min_score, min_margin = core.thresholds_for(cfg, meta)
    probs = sorted(r["probs"].items(), key=lambda kv: kv[1], reverse=True)
    best, runner = probs[0], probs[1]
    best_lbl, run_lbl = CATEGORY_LABEL.get(best[0], best[0]), CATEGORY_LABEL.get(runner[0], runner[0])
    state = rep["routing_state"]
    if state == "routed" and rep.get("category_source") == "operator_correction":
        st.markdown(f"**Staff set this report to {CATEGORY_LABEL.get(rep['approved_category'])}.**")
    elif state == "routed":
        st.markdown(f"**Sent automatically to {rep['proposed_department']}.** The text best matches "
                    f"*{best_lbl}*, clearly ahead of *{run_lbl}*.")
    elif state == "urgent_review":
        st.markdown("**Flagged for urgent staff attention** instead of automatic routing.")
    else:
        st.markdown(f"**A staff member will check this one.** The closest match was *{best_lbl}*, "
                    f"but the model wasn't sure enough to route it on its own.")

    st.markdown("**How well the text matches each category**")
    for cat, p in probs:
        c1, c2 = st.columns([2, 5])
        c1.write(CATEGORY_LABEL.get(cat, cat) + (" ← best" if cat == best[0] else ""))
        c2.progress(min(max(p, 0.0), 1.0), text=f"{p:.2f}")
    st.caption(f"Automatic routing needs a best match of at least {min_score:.2f} and a lead of at least "
               f"{min_margin:.2f} over the next category. This one: {best[1]:.2f}, lead {best[1] - runner[1]:.2f}.")

    if r["reasons"]:
        st.markdown("**Why it needs a person**")
        for reason in r["reasons"]:
            st.markdown(f"- {reason_text(reason, min_score, min_margin)}")
    st.markdown(f"**Location:** {LOCATION_LABEL.get(rep['location_status'], rep['location_status'])}")
    if rep.get("attachment_path"):
        st.markdown("**Photo / clip:** stored for staff review — not analyzed by AI and not used for routing.")
    st.caption("Match scores come from a text model trained on simulated examples; they are rough "
               "indicators, not probabilities you can rely on. Department names are illustrative (demo).")
