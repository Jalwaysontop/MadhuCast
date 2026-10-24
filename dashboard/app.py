"""MadhuTwin clinician dashboard. Run: streamlit run dashboard/app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from dashboard.components import common

st.set_page_config(
    page_title="MadhuTwin", page_icon=str(Path(__file__).resolve().parent / "assets" / "mark.svg"), layout="wide"
)
common.load_css()
common.init_session()

pages = [
    st.Page("pages/alert_queue.py", title="Alert queue", icon=":material/notifications_active:", default=True),
    st.Page("pages/patient_card.py", title="Patient", icon=":material/badge:"),
    st.Page("pages/forecast_view.py", title="Forecast", icon=":material/monitoring:"),
    st.Page("pages/meal_timeline.py", title="Meals", icon=":material/restaurant:"),
    st.Page("pages/whatif_lab.py", title="What-if lab", icon=":material/science:"),
    st.Page("pages/explain_panel.py", title="Why", icon=":material/psychology_alt:"),
    st.Page("pages/trust_page.py", title="Model trust", icon=":material/verified:"),
]
pg = st.navigation(pages)
common.sidebar()
common.header()
pg.run()
common.footer()
