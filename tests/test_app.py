"""Headless smoke test of every dashboard page."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ["overview", "machine", "process", "maintenance", "agents", "assistant", "performance"]


@pytest.fixture(scope="module")
def app(artifacts_ready):
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.switch_page(f"app_pages/{page}.py")
    app.run()
    assert not app.exception, [e.value for e in app.exception]
