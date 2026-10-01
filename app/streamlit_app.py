"""Smart Manufacturing Process Optimisation - dashboard entry point.

WHAT THIS FILE DOES (plain English)
-----------------------------------
The starting point of the web dashboard. It sets up the browser page, prepares
the shared factory state, builds the navigation menu (the list of pages on the
left), shows the factory clock in the sidebar, and then displays whichever page
the user picked. The pages themselves live in the app_pages/ folder.

Run:  streamlit run app/streamlit_app.py
"""
# --- Imports: tools this file needs -----------------------------------------
import streamlit as st  # Streamlit turns Python scripts into interactive web pages

import common  # shared helpers for all pages (app/common.py)

# Browser tab title and icon; "wide" uses the full screen width.
st.set_page_config(page_title="Smart factory", page_icon=":material/precision_manufacturing:", layout="wide")

# Create the factory clock, toolbox, chat and decision memory on first visit.
common.init_state()

# The navigation menu: page files grouped into three sections.
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
# Sidebar controls shared by every page (factory clock, production cycle, AI engine status).
common.sidebar_clock()
# Page heading, then the selected page's content.
st.title(page.title, anchor=False)
page.run()
