import altair as alt
import pandas as pd
import streamlit as st

import common
from src import config
from src.optimization import scheduler

ts = common.state().timestamp
snap = common.snapshot(ts)

with st.container(horizontal=True, vertical_alignment="bottom"):
    horizon = st.slider("Planning horizon (days)", 3, 14, config.PLANNING_HORIZON_DAYS)
    crew = st.slider("Crew capacity (jobs/day)", 1, 10, config.MAINTENANCE_CREW_PER_DAY)

plan = scheduler.plan(snap, horizon_days=horizon, crew_per_day=crew)
if not plan["jobs"]:
    st.success("No component is at meaningful risk - no maintenance needed.", icon=":material/check_circle:")
    st.stop()

with st.container(horizontal=True):
    st.metric("Jobs scheduled", plan["jobs_scheduled"], border=True)
    st.metric("Expected cost with plan", f"${plan['expected_cost_plan']:,.0f}", border=True)
    st.metric("Expected cost, run-to-failure", f"${plan['expected_cost_run_to_failure']:,.0f}", border=True)
    st.metric("Expected savings", f"${plan['expected_savings']:,.0f}", border=True)

jobs = pd.DataFrame(plan["jobs"])
left, right = st.columns([2, 3])
with left, st.container(border=True):
    st.subheader("Crew load per day", anchor=False)
    per_day = (
        jobs.dropna(subset=["day"]).groupby("day").size()
        .reindex(range(horizon), fill_value=0).rename("jobs").reset_index()
    )
    per_day["load"] = plan["daily_load"]
    bars = alt.Chart(per_day).mark_bar(cornerRadiusEnd=4, color=common.SERIES[0]).encode(
        x=alt.X("day:O", title="Day (0 = today)"),
        y=alt.Y("jobs:Q", title="Jobs", scale=alt.Scale(domain=[0, max(crew, 1)])),
        tooltip=["day:O", "jobs:Q", alt.Tooltip("load:Q", title="Production load", format=".1f")],
    )
    cap = alt.Chart(pd.DataFrame({"cap": [crew]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="cap:Q")
    st.altair_chart((bars + cap).properties(height=240))
    st.caption("Dashed line: crew capacity. Lower-load days are cheaper for planned downtime.")

with right, st.container(border=True):
    st.subheader("Schedule", anchor=False)
    st.dataframe(
        jobs, hide_index=True,
        column_order=["machine_id", "component", "action", "p_fail_24h", "p_fail_7d", "production_load", "expected_cost"],
        column_config={
            "machine_id": st.column_config.NumberColumn("Machine"),
            "component": "Component",
            "action": "Action",
            "p_fail_24h": st.column_config.ProgressColumn("Risk 24h", min_value=0, max_value=1, format="percent"),
            "p_fail_7d": st.column_config.ProgressColumn("Risk 7d", min_value=0, max_value=1, format="percent"),
            "production_load": st.column_config.NumberColumn("Day load", format="%.1f"),
            "expected_cost": st.column_config.NumberColumn("Expected cost", format="$%.0f"),
        },
    )

st.caption(
    f"Solver: OR-Tools CP-SAT ({plan['status']}). Costs: planned job ${config.COST['planned_maintenance']:,.0f} "
    f"+ {config.DOWNTIME_HOURS['planned']:.0f} h downtime; unplanned failure ${config.COST['unplanned_failure']:,.0f} "
    f"+ {config.DOWNTIME_HOURS['unplanned']:.0f} h downtime at ${config.COST['downtime_per_hour']:,.0f}/h."
)
