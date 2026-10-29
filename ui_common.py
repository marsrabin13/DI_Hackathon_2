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
