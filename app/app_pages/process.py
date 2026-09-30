import pandas as pd
import streamlit as st

import common
from src.optimization import process_optimizer

s = common.state()
pm = common.process_model()
cycle = s.process_cycle
diag = pm.diagnose(cycle)
p = diag["failure_probability"]

with st.container(horizontal=True):
    st.metric("Failure probability", f"{p:.1%}", border=True)
    st.metric("Speed", f"{cycle['rpm']:.0f} rpm", border=True)
    st.metric("Torque", f"{cycle['torque_nm']:.1f} Nm", border=True)
    st.metric("Tool wear", f"{cycle['tool_wear_min']:.0f} min", border=True)
    st.metric("Power", f"{diag['derived']['power_w'] / 1000:.2f} kW", border=True)
    st.metric("Temp gap", f"{diag['derived']['temp_diff_k']:.1f} K", border=True,
              help="Process minus air temperature. Low values impair heat dissipation.")
st.markdown(f"Status: {common.risk_level(p)} · product type **{cycle['type']}**")

left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Diagnosis", anchor=False)
    modes = pd.DataFrame(
        {"Probability": diag["failure_mode_probabilities"]}
    ).rename_axis("Failure mode").reset_index()
    st.dataframe(modes, hide_index=True, column_config={
        "Probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    flags = diag["physics_rule_flags"]
    if flags:
        for mode, text in flags.items():
            st.warning(text, icon=":material/warning:")
    else:
        st.success("No physics-rule limits breached.", icon=":material/check_circle:")

with right, st.container(border=True):
    st.subheader("Risk drivers", anchor=False)
    st.altair_chart(common.shap_chart(diag["top_risk_drivers"]))

with st.container(border=True):
    st.subheader("Setpoint optimizer", anchor=False)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        min_tp = st.slider("Minimum throughput (speed vs. current)", 0.85, 1.0, 0.9, 0.05, format="%.2f")
        allow_tool = st.toggle("Allow tool change", value=True)
    result = process_optimizer.optimize(pm, cycle, min_throughput_ratio=min_tp, allow_tool_change=allow_tool)
    cands = pd.DataFrame(result["recommended"])
    cands["remaining_physics_flags"] = cands["remaining_physics_flags"].apply(
        lambda f: ", ".join(f) if f else "none")
    st.dataframe(
        cands, hide_index=True,
        column_config={
            "rpm": st.column_config.NumberColumn("Speed (rpm)", format="%.0f"),
            "rpm_change_pct": st.column_config.NumberColumn("Speed Δ", format="%+.0f%%"),
            "torque_nm": st.column_config.NumberColumn("Torque (Nm)", format="%.1f"),
            "torque_change_pct": st.column_config.NumberColumn("Torque Δ", format="%+.0f%%"),
            "replace_tool": st.column_config.CheckboxColumn("New tool"),
            "failure_probability": st.column_config.ProgressColumn(
                "Risk after", min_value=0, max_value=1, format="percent"),
            "risk_reduction": st.column_config.NumberColumn("Risk reduction", format="percent"),
            "throughput_ratio": st.column_config.NumberColumn("Throughput", format="%.2f"),
            "remaining_physics_flags": "Remaining flags",
        },
    )
    best = result["recommended"][0]
    if st.button("Apply best candidate", icon=":material/check:", type="primary", disabled=p < 0.05):
        s.process_override = dict(cycle, rpm=best["rpm"], torque_nm=best["torque_nm"],
                                  tool_wear_min=0.0 if best["replace_tool"] else cycle["tool_wear_min"])
        st.rerun()
    st.caption("Every candidate is re-scored by the failure model. Applying sets custom setpoints "
               "for the line (reset from the sidebar).")

with st.container(border=True):
    st.subheader("What-if simulator", anchor=False)
    with st.form("whatif", border=False):
        c1, c2, c3 = st.columns(3)
        rpm = c1.number_input("Speed (rpm)", 1000.0, 3000.0, float(cycle["rpm"]), 10.0)
        torque = c2.number_input("Torque (Nm)", 3.0, 80.0, float(cycle["torque_nm"]), 0.5)
        tool = c3.toggle("Replace tool")
        submitted = st.form_submit_button("Simulate", icon=":material/science:")
    if submitted:
        sim = process_optimizer.simulate(pm, cycle, rpm=rpm, torque_nm=torque, replace_tool=tool)
        with st.container(horizontal=True):
            st.metric("Risk before", f"{sim['failure_probability_before']:.1%}", border=True)
            st.metric("Risk after", f"{sim['failure_probability_after']:.1%}",
                      f"{-sim['risk_reduction'] * 100:+.1f} pts", delta_color="inverse", border=True)
        if not sim["within_training_range"]:
            st.warning("Setpoints are outside the range seen in training - treat the prediction with caution.")
        for text in sim["remaining_physics_flags"].values():
            st.warning(text, icon=":material/warning:")
