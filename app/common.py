"""Shared state, cached resources, and chart helpers for the dashboard."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.agents.tools import ToolBox  # noqa: E402
from src.models.predictor import get_fleet_model, get_process_model  # noqa: E402
from src.optimization.kpis import plant_kpis  # noqa: E402
from src.simulation.stream import FactoryState  # noqa: E402

# Validated reference palette (dataviz skill): categorical slots, blue sequential
# ramp, blue<->red diverging pair, reserved status colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING = {"lowers risk": "#2a78d6", "raises risk": "#e34948"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


@st.cache_resource(show_spinner="Loading models ...")
def fleet_model():
    return get_fleet_model()


@st.cache_resource(show_spinner="Loading models ...")
def process_model():
    return get_process_model()


@st.cache_data(max_entries=64, show_spinner=False)
def snapshot(ts: pd.Timestamp) -> pd.DataFrame:
    return fleet_model().snapshot(ts)


@st.cache_data(show_spinner=False)
def kpis() -> dict:
    return plant_kpis()


@st.cache_data(show_spinner=False)
def model_metrics() -> dict:
    path = config.REPORTS_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(ttl=15, show_spinner=False)
def llm_ready() -> bool:
    return config.llm_available()


def init_state() -> None:
    if "factory" not in st.session_state:
        pm = process_model()
        failing = pm.test_cycles.index[pm.test_cycles["machine_failure"] == 1]
        # Start on a failing cycle so the process optimiser has something to do
        st.session_state.factory = FactoryState(ts_index=200, cycle_index=int(failing[0]))
        st.session_state.toolbox = ToolBox(st.session_state.factory)
        st.session_state.agent_run = None
        st.session_state.decisions = {}
        st.session_state.chat = None
        st.session_state.chat_log = []


def state() -> FactoryState:
    return st.session_state.factory


def toolbox() -> ToolBox:
    return st.session_state.toolbox


def risk_level(p: float) -> str:
    if p >= 0.5:
        return ":red[:material/error: critical]"
    if p >= 0.2:
        return ":orange[:material/warning: elevated]"
    return ":green[:material/check_circle: normal]"


def sidebar_clock() -> None:
    """Fleet clock and production-cycle selector shared by every page."""
    fm = fleet_model()
    labels = [ts.strftime("%Y-%m-%d %H:%M") for ts in fm.timestamps]
    s = state()
    with st.sidebar:
        st.subheader("Factory clock")
        st.caption("Replays the held-out test period (Sep 2015 - Jan 2016) in 3-hour steps.")
        choice = st.select_slider("Current time", options=labels, value=labels[s.ts_index])
        s.ts_index = labels.index(choice)
        with st.container(horizontal=True):
            if st.button("+3h", icon=":material/skip_next:"):
                s.advance(1)
                st.rerun()
            if st.button("+1 day", icon=":material/fast_forward:"):
                s.advance(8)
                st.rerun()

        st.subheader("Machining line")
        pm = process_model()
        cycles = pm.test_cycles
        subset = st.segmented_control(
            "Cycles", ["Failure cycles", "All cycles"], default="Failure cycles", key="cycle_subset"
        )
        pool = cycles[cycles["machine_failure"] == 1] if subset == "Failure cycles" else cycles
        options = [int(i) for i in pool.index]
        if s.cycle_index not in options:
            s.cycle_index = options[0]
        s.cycle_index = st.selectbox(
            "Production cycle",
            options,
            index=options.index(s.cycle_index),
            format_func=lambda i: f"#{i} · type {cycles.loc[i, 'type']} · {cycles.loc[i, 'product_id']}",
        )
        if s.process_override:
            st.info("Using custom setpoints from the process optimiser.", icon=":material/tune:")
            if st.button("Reset to recorded cycle"):
                s.process_override = None
                st.rerun()
        st.divider()
        if llm_ready():
            st.caption(f"Decision engine: {config.llm_label()}")
        else:
            st.caption(f"Decision engine: **rule-based** ({config.llm_label()} unavailable)")


def log_decision(action: dict, decision: str, run_ts: str) -> None:
    record = {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "factory_time": run_ts,
        "decision": decision,
        **action,
    }
    with config.DECISIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def shap_chart(drivers: list[dict]) -> alt.Chart:
    df = pd.DataFrame(drivers)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=18)
        .encode(
            x=alt.X("impact_log_odds:Q", title="Contribution to failure risk (log-odds)"),
            y=alt.Y("label:N", sort=alt.EncodingSortField("impact_log_odds", op="max", order="descending"),
                    title=None),
            color=alt.Color("direction:N", scale=alt.Scale(domain=list(DIVERGING), range=list(DIVERGING.values())),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("label:N", title="Feature"), alt.Tooltip("value:Q", title="Value"),
                     alt.Tooltip("impact_log_odds:Q", title="Impact", format="+.2f"), "direction:N"],
        )
        .properties(height=alt.Step(26))
    )
