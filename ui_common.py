"""Shared UI helpers: styling, cached model/DB, small render components.

Palette: light-gray page, white cards, navy text, teal primary actions, amber ONLY for cases needing review.
Icons: Material Symbols (outlined) via Streamlit's :material/name: syntax - no emojis.
"""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

import ai_service
import core

NAVY, TEAL, AMBER, RED, GRAY = "#14284B", "#0E8C87", "#C98A12", "#B4233C", "#6B7280"
BLUE, GREEN, SLATE = "#3B6BA5", "#2E7D4F", "#4B5B76"
CAT_COLORS = {"roads": [20, 40, 75], "waste": [14, 140, 135], "lighting": [59, 107, 165], "noise": [150, 80, 140]}

# Routing state = how the report was routed (separate from case progress)
STATE_UI = {"routed": ("Routed automatically", TEAL), "review_required": ("Needs staff review", AMBER),
            "urgent_review": ("Urgent review", RED), "spam": ("Spam", GRAY)}
# Case status = progress toward resolution
CASE_UI = {"open": ("Received", SLATE), "in_progress": ("In progress", BLUE),
           "resolved": ("Resolved", GREEN), "closed": ("Closed – not actionable", GRAY)}

CSS = f"""
<style>
  h1, h2, h3, h4 {{ color:{NAVY}; }}
  .cf-head {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:2px; }}
  .cf-title {{ font-size:1.45rem; font-weight:700; color:{NAVY}; }}
  .cf-brand {{ display:flex; align-items:center; gap:6px; font-weight:700; color:{NAVY}; font-size:1rem; }}
  .cf-brand .fix {{ color:{TEAL}; }}
  .cf-divider {{ width:1px; height:22px; background:#CBD3DD; }}
  .cf-badge {{ border:1px solid {TEAL}; color:{TEAL}; background:#FFFFFF; padding:1px 10px; border-radius:12px;
               font-size:.72rem; font-weight:600; }}
  .cf-sub {{ color:#5B6475; font-size:.88rem; margin-bottom:14px; }}
  .cf-pill {{ display:inline-block; color:#fff; padding:2px 10px; border-radius:10px; font-size:.76rem; font-weight:600; }}
  .cf-card {{ border:1px solid #E1E6ED; border-radius:12px; padding:14px 16px; background:#FFFFFF; margin:4px 0 8px 0;
              box-shadow:0 1px 2px rgba(20,40,75,.05); }}
  .cf-card.review {{ border-left:4px solid {AMBER}; }}
  .cf-row {{ display:flex; gap:22px; flex-wrap:wrap; margin-top:8px; font-size:.86rem; color:#2B3445; }}
  .cf-row b {{ color:{NAVY}; font-weight:600; }}
  .cf-label {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; color:#6B7280; }}
  .cf-ref {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:1.05rem; font-weight:700; color:{NAVY}; }}
  .cf-tag {{ font-size:.74rem; color:#6B7280; margin-top:8px; }}
  .cf-tag .live {{ background:{TEAL}; color:#fff; padding:1px 8px; border-radius:8px; }}
  .cf-tag .fb {{ background:#E5E7EB; color:#374151; padding:1px 8px; border-radius:8px; }}
  .cf-kpi {{ background:#FFFFFF; border:1px solid #E1E6ED; border-radius:12px; padding:12px 16px; }}
  .cf-kpi .v {{ font-size:1.9rem; font-weight:700; color:{NAVY}; line-height:1.2; }}
  .cf-kpi.review {{ border-top:4px solid {AMBER}; }}
  .cf-kpi.review .v {{ color:{AMBER}; }}
  .cf-kpi .s {{ font-size:.75rem; color:#6B7280; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#FFFFFF; border-radius:12px; }}
  div.stButton > button[kind="primary"], div.stFormSubmitButton > button {{
      background:{TEAL} !important; border:0 !important; color:#FFFFFF !important; }}
  div.stButton > button[kind="primary"] *, div.stFormSubmitButton > button * {{ color:#FFFFFF !important; }}
  div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button:hover {{
      background:{NAVY} !important; color:#FFFFFF !important; }}
</style>
"""


def apply_style(max_width_px: int | None = None):
    css = CSS
    if max_width_px:
        css += f"<style>.block-container {{ max-width:{max_width_px}px; }}</style>"
    st.markdown(css, unsafe_allow_html=True)


LOGO_SVG = ("<svg width='26' height='26' viewBox='0 0 40 40' aria-hidden='true'><rect x='4' y='4' width='32' "
            "height='32' rx='8' fill='#0E8C87'/><path d='M20 10.5c-4.4 0-7.5 3.2-7.5 7.3 0 5.4 7.5 12.7 7.5 12.7s7.5-7.3 "
            "7.5-12.7c0-4.1-3.1-7.3-7.5-7.3z' fill='none' stroke='#fff' stroke-width='2.2' stroke-linejoin='round'/>"
            "<circle cx='20' cy='17.8' r='2.7' fill='none' stroke='#fff' stroke-width='2.2'/></svg>")


def header(page_title: str, city_name: str = "Tel Aviv-Yafo"):
    """Small CityFix logo, city name and the single simulated-data badge."""
    st.markdown(f"<div class='cf-head'><span class='cf-brand'>{LOGO_SVG}City<span class='fix'>Fix</span></span>"
                f"<span class='cf-divider'></span><span class='cf-title'>{html.escape(city_name)}</span>"
                f"<span class='cf-badge'>Simulated Tel Aviv reports</span></div>"
                f"<div class='cf-sub'>{html.escape(page_title)}</div>", unsafe_allow_html=True)


def dept(name) -> str:
    """Display name for a department: drop the '(demo)' suffix kept in config and saved history."""
    return str(name or "").replace(" (demo)", "")


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
    label, color = STATE_UI.get(state, (state, GRAY))
    return f"<span class='cf-pill' style='background:{color}'>{label}</span>"


def case_pill(status: str) -> str:
    label, color = CASE_UI.get(status, (status, GRAY))
    return f"<span class='cf-pill' style='background:{color}'>{label}</span>"


def kpi(label: str, value, sub: str = "", review: bool = False) -> str:
    return (f"<div class='cf-kpi{' review' if review else ''}'><div class='cf-label'>{html.escape(label)}</div>"
            f"<div class='v'>{value}</div><div class='s'>{html.escape(sub)}</div></div>")


def draft_tag(label: str, detail: str = "") -> str:
    """Staff-facing label with diagnostic detail (admin only)."""
    cls = "live" if label == ai_service.LIVE else "fb"
    extra = f" · {html.escape(detail)}" if detail else ""
    return f"<div class='cf-tag'><span class='{cls}'>{label}</span>{extra}</div>"


def resident_badge(label: str) -> str:
    """Resident-facing: understandable badge only; diagnostics live under Admin → Model & data."""
    if label == ai_service.LIVE:
        return "<span class='cf-tag'><span class='live'>AI-written reply</span></span>"
    return "<span class='cf-tag'><span class='fb'>Template response</span></span>"


def show_attachment(rel_path: str | None, thumb_px: int = 160):
    """Small thumbnail (use the image's fullscreen button to enlarge); clips open in an expander."""
    if not rel_path:
        return
    path = core.ROOT / rel_path
    if not path.exists():
        st.caption("Attachment file not found.")
        return
    if Path(path).suffix.lower() in (".mp4", ".mov"):
        with st.expander(":material/videocam: View attached clip"):
            st.video(str(path))
    else:
        st.image(str(path), width=thumb_px, caption="Attached photo · hover and use ⤢ to enlarge")


# ---------------------------------------------------------------- plain-language routing explanation
CATEGORY_LABEL = {"roads": "Roads & sidewalks", "waste": "Waste & cleanliness",
                  "lighting": "Street lighting", "noise": "Noise", "spam": "Spam / not a city issue"}
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
        return "Marked by the resident as multiple issues / unsure."
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
        st.markdown(f"**Sent automatically to {dept(rep['proposed_department'])}.** The text best matches "
                    f"*{best_lbl}*, clearly ahead of *{run_lbl}*.")
    elif state == "urgent_review":
        st.markdown("**Flagged for urgent staff attention** instead of automatic routing.")
    else:
        st.markdown(f"**A staff member will check this one.** The closest match was *{best_lbl}*, "
                    f"but the model wasn't sure enough to route it on its own.")

    st.markdown("**How well the text matches each category**")
    for cat, p in probs:
        c1, c2 = st.columns([2, 5])
        c1.write(CATEGORY_LABEL.get(cat, cat) + (" (best)" if cat == best[0] else ""))
        c2.progress(min(max(p, 0.0), 1.0), text=f"{p:.2f}")
    st.caption(f"Automatic routing needs a best match of at least {min_score:.2f} and a lead of at least "
               f"{min_margin:.2f} over the next category. This one: {best[1]:.2f}, lead {best[1] - runner[1]:.2f}.")

    if r["reasons"]:
        st.markdown("**Why it needs a person**")
        for reason in r["reasons"]:
            st.markdown(f"- {reason_text(reason, min_score, min_margin)}")
    st.markdown(f"**Location:** {LOCATION_LABEL.get(rep['location_status'], rep['location_status'])}")
    st.caption("Match scores come from a text model trained on simulated examples; they are rough "
               "indicators, not probabilities you can rely on. Department names are illustrative.")
