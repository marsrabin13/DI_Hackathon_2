"""CityFix entry point: top navigation for the resident page and the admin dashboard.
Run: streamlit run app.py"""
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"  # logo files; the small logo is drawn in each page header

pages = [
    st.Page("citizen_page.py", title="Report an issue", icon=":material/edit_note:", default=True),
    st.Page("admin_page.py", title="Admin dashboard", icon=":material/dashboard:"),
]
st.navigation(pages, position="top").run()
