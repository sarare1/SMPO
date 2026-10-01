"""Headless smoke test of every dashboard page.

WHAT THIS FILE CHECKS (plain English)
-------------------------------------
Opens each of the seven dashboard pages in a simulated browser (no real window)
and checks that none of them crashes.
"""
# --- Imports: tools this file needs -----------------------------------------
from pathlib import Path  # file paths

import pytest                             # the test framework
from streamlit.testing.v1 import AppTest  # Streamlit's tool for running an app without a browser

# The dashboard's starting file and the page files to open.
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ["overview", "machine", "process", "maintenance", "agents", "assistant", "performance"]


@pytest.fixture(scope="module")
def app(artifacts_ready):
    """Start the dashboard once for all page tests."""
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    """Switch to the page and confirm it shows without an error."""
    app.switch_page(f"app_pages/{page}.py")
    app.run()
    assert not app.exception, [e.value for e in app.exception]
