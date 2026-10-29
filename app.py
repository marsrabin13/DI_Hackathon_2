"""CityFix entry point: resident page + separate admin dashboard. Run: streamlit run app.py"""
import streamlit as st

pages = [
    st.Page("citizen_page.py", title="Report an issue", icon="💬", default=True),
    st.Page("admin_page.py", title="Admin dashboard", icon="📊"),
]
st.navigation(pages, position="top").run()
