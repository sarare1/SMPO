"""Fleet overview page.

WHAT THIS PAGE SHOWS (plain English)
------------------------------------
The big picture of the whole fleet at the current factory time:
  * Headline numbers: machines at critical risk, at elevated risk, behaving
    unusually, and the plant's OEE score with and without predictive maintenance.
  * A heat map of the 25 riskiest machines: darker blue = higher chance that a
    component fails within 7 days.
  * The yearly business case for predictive maintenance.
  * A watch-list table of the 15 riskiest machines.
"""
# --- Imports: tools this page needs -----------------------------------------
import altair as alt    # charts
import pandas as pd     # tables of data
import streamlit as st  # the dashboard framework

import common                             # shared dashboard helpers
from src.data.features import COMPONENTS  # the four component names

# Current factory time, every machine's risk at that time, and the plant KPIs.
ts = common.state().timestamp
snap = common.snapshot(ts)
k = common.kpis()
proj = k["pdm_projection"]

st.caption(f"Fleet of {len(snap)} machines at **{ts:%Y-%m-%d %H:%M}**")

# --- Row of headline number tiles -----------------------------------------
with st.container(horizontal=True):
    st.metric("Critical (24h risk > 50%)", int((snap["risk_24h"] > 0.5).sum()), border=True,
              help="Machines with a component likely to fail within 24 hours.")
    st.metric("Elevated (7-day risk > 30%)", int((snap["risk_7d"] > 0.3).sum()), border=True)
    st.metric("Anomalous sensors", int((snap["anomaly_score"] > 1).sum()), border=True,
              help="Sensor pattern beyond the 99th percentile of normal operation.")
    st.metric("OEE (historical)", f"{k['oee']:.1%}", border=True,
              help="Availability x performance x quality over the full year.")
    st.metric("OEE with predictive maintenance", f"{proj['oee_with_pdm']:.1%}",
              f"{(proj['oee_with_pdm'] - k['oee']) * 100:+.1f} pts", border=True)

# Two columns: heat map on the left (wider), business case on the right.
left, right = st.columns([3, 2])

# --- Heat map: 7-day failure risk per machine and component --------------------
with left, st.container(border=True):
    st.subheader("7-day failure risk by component", anchor=False)
    top = snap.head(25)  # the 25 riskiest machines
    # Reshape into one row per (machine, component) for the chart.
    heat = (
        top[[f"p_{c}_7d" for c in COMPONENTS]]
        .rename(columns=lambda c: c.split("_")[1])
        .reset_index()
        .melt(id_vars="machineID", var_name="component", value_name="probability")
    )
    heat["machine"] = "M" + heat["machineID"].astype(str)
    order = ["M" + str(m) for m in top.index]  # keep riskiest machines on the left
    chart = (
        alt.Chart(heat)
        .mark_rect(stroke="white", strokeWidth=2, cornerRadius=3)
        .encode(
            x=alt.X("machine:N", sort=order, title="Machine (riskiest first)"),
            y=alt.Y("component:N", title=None),
            color=alt.Color("probability:Q", scale=alt.Scale(domain=[0, 1], range=common.SEQUENTIAL),
                            legend=alt.Legend(title="P(fail ≤ 7d)", format=".0%")),
            tooltip=["machine:N", "component:N", alt.Tooltip("probability:Q", format=".1%")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart)
    st.caption("Top 25 machines by 7-day risk. Hover a cell for the exact probability.")

# --- Business case: what predictive maintenance is worth per year ---------------
with right, st.container(border=True):
    st.subheader("Predictive maintenance business case", anchor=False)
    st.table(pd.DataFrame(
        {
            "Value": [
                f"{k['unplanned_failures']}",
                f"{k['unplanned_downtime_hours']:,} h",
                f"{proj['failures_prevented_per_year']}",
                f"{proj['downtime_hours_saved_per_year']:,} h",
                f"${proj['cost_saved_per_year_usd']:,.0f}",
                f"{k['availability']:.1%} → {proj['availability_with_pdm']:.1%}",
            ]
        },
        index=[
            "Unplanned failures / year", "Unplanned downtime / year", "Failures prevented / year",
            "Downtime saved / year", "Cost saved / year", "Availability",
        ],
    ))
    st.caption(
        f"Assumes {proj['detection_rate_24h_models']:.0%} detection (24h models) and "
        f"{proj['realization_factor']:.0%} of detections converted to planned work. "
        "Edit cost assumptions in `src/config.py`."
    )

# --- Watch-list: the 15 riskiest machines, with risk bars ---------------------------
with st.container(border=True):
    st.subheader("Watch-list", anchor=False)
    table = snap.head(15).reset_index()[
        ["machineID", "model", "age_years", "top_component", "risk_24h", "risk_7d", "anomaly_score"]
    ]
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "machineID": st.column_config.NumberColumn("Machine"),
            "model": "Model",
            "age_years": st.column_config.NumberColumn("Age (yrs)"),
            "top_component": "Component at risk",
            "risk_24h": st.column_config.ProgressColumn("Risk 24h", min_value=0, max_value=1, format="percent"),
            "risk_7d": st.column_config.ProgressColumn("Risk 7 days", min_value=0, max_value=1, format="percent"),
            "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.2f",
                                                           help="> 1 = beyond normal range"),
        },
    )
