import altair as alt
import pandas as pd
import streamlit as st

import common
from src.data.features import COMPONENTS, SENSORS

s = common.state()
ts = s.timestamp
snap = common.snapshot(ts)
fm = common.fleet_model()

machine_id = st.selectbox(
    "Machine",
    list(snap.index),
    format_func=lambda m: f"Machine {m} · 7-day risk {snap.loc[m, 'risk_7d']:.0%}",
    help="Sorted by 7-day risk.",
)
detail = fm.machine_detail(int(machine_id), ts)

with st.container(horizontal=True):
    p24 = max(detail["failure_probability_24h"].values())
    p7 = max(detail["failure_probability_7d"].values())
    st.metric("Highest 24h risk", f"{p24:.0%}", border=True)
    st.metric("Highest 7-day risk", f"{p7:.0%}", border=True)
    st.metric("Component at risk", detail["highest_risk_component"], border=True)
    st.metric("Anomaly score", f"{detail['anomaly_score']:.2f}", border=True, help="> 1 = beyond normal range")
    st.metric("Machine", f"{detail['model']} · {detail['age_years']} yrs", border=True)
st.markdown(f"Status: {common.risk_level(p24)}")

left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Failure probability per component", anchor=False)
    comp = pd.DataFrame({
        "Component": COMPONENTS,
        "Within 24h": [detail["failure_probability_24h"][c] for c in COMPONENTS],
        "Within 7 days": [detail["failure_probability_7d"][c] for c in COMPONENTS],
        "Days since replaced": [detail["days_since_replacement"][c] for c in COMPONENTS],
    })
    st.dataframe(
        comp, hide_index=True,
        column_config={
            "Within 24h": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Within 7 days": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Days since replaced": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    errors = detail["errors_last_7d"]
    st.markdown("**Errors, last 7 days:** " + (", ".join(f"{k} ×{v}" for k, v in errors.items()) or "none"))
    st.markdown("**Recent replacements:** " + (
        ", ".join(f"{r['component']} ({r['date']})" for r in detail["last_replacements"]) or "none"))

with right, st.container(border=True):
    st.subheader(f"Why {detail['highest_risk_component']} is at risk", anchor=False)
    st.altair_chart(common.shap_chart(detail["top_risk_drivers_for_highest_component"]))
    st.caption("SHAP contributions to the 7-day failure model for this component.")

with st.container(border=True):
    st.subheader("Sensor telemetry, last 14 days", anchor=False)
    tel = fm.telemetry_window(int(machine_id), ts).reset_index()
    cols = st.columns(2)
    for i, sensor in enumerate(SENSORS):
        df = tel[["datetime", sensor]].rename(columns={sensor: "value"})
        df["24h mean"] = df["value"].rolling(24, min_periods=1).mean()
        base = alt.Chart(df).encode(x=alt.X("datetime:T", title=None))
        raw = base.mark_line(strokeWidth=1, opacity=0.35, color=common.SERIES[0]).encode(
            y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("datetime:T", format="%b %d %H:%M"), alt.Tooltip("value:Q", format=".1f")],
        )
        smooth = base.mark_line(strokeWidth=2, color=common.SERIES[0]).encode(y="24h mean:Q")
        dev = detail["sensor_last_24h_vs_lifetime_mean"][sensor]["deviation_pct"]
        with cols[i % 2]:
            st.markdown(f"**{sensor}** · last 24h {dev:+.1f}% vs. lifetime mean")
            st.altair_chart((raw + smooth).properties(height=160))
    st.caption("Faint line: hourly reading. Solid line: 24-hour rolling mean.")

actual = fm.actual_failures(int(machine_id), ts)
with st.expander("Ground truth (demo only - hidden from the agents)", icon=":material/visibility:"):
    if actual.empty:
        st.write("No recorded failure for this machine in the next 7 days.")
    else:
        st.dataframe(actual, hide_index=True)
