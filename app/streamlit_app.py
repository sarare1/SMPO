"""Smart Manufacturing Process Optimisation - dashboard entry point.

Run:  streamlit run app/streamlit_app.py
"""
import streamlit as st

import common

st.set_page_config(page_title="Smart factory", page_icon=":material/precision_manufacturing:", layout="wide")

common.init_state()

page = st.navigation(
    {
        "Operations": [
            st.Page("app_pages/overview.py", title="Fleet overview", icon=":material/dashboard:", default=True),
            st.Page("app_pages/machine.py", title="Machine health", icon=":material/monitor_heart:"),
            st.Page("app_pages/process.py", title="Process optimizer", icon=":material/tune:"),
            st.Page("app_pages/maintenance.py", title="Maintenance plan", icon=":material/build:"),
        ],
        "AI decision agents": [
            st.Page("app_pages/agents.py", title="Agent action plan", icon=":material/smart_toy:"),
            st.Page("app_pages/assistant.py", title="Assistant", icon=":material/chat:"),
        ],
        "Models": [
            st.Page("app_pages/performance.py", title="Model performance", icon=":material/insights:"),
        ],
    }
)
common.sidebar_clock()
st.title(page.title, anchor=False)
page.run()
