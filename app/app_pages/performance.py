import pandas as pd
import streamlit as st

import common

metrics = common.model_metrics()
if not metrics:
    st.warning("No metrics found - run `python -m src.models.train` first.")
    st.stop()


def frame(block: dict, names: dict) -> pd.DataFrame:
    rows = []
    for key, label in names.items():
        m = block[key]
        rows.append({"Model": label, "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "Brier": m["brier"],
                     "Positives in test": m["positives_in_test"]})
    return pd.DataFrame(rows)


cfg = {c: st.column_config.NumberColumn(format="%.3f") for c in ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"]}
cfg["Brier"] = st.column_config.NumberColumn(format="%.4f", help="Calibration error; lower is better.")

with st.container(border=True):
    st.subheader("Machining line (AI4I 2020) · stratified 20% hold-out", anchor=False)
    st.dataframe(frame(metrics["ai4i"], {
        "machine_failure": "Machine failure", "HDF": "Heat dissipation", "PWF": "Power",
        "OSF": "Overstrain", "TWF": "Tool wear"}), hide_index=True, column_config=cfg)
    st.caption("Tool-wear failures happen at a random wear between 200-240 min, so they are not predictable "
               "from a single cycle; the system covers them with a physics rule (flag from 190 min).")

with st.container(border=True):
    st.subheader("Fleet (Azure PdM) · time split, test from 2015-09-01", anchor=False)
    comps = ["comp1", "comp2", "comp3", "comp4"]
    names = {f"fail_{c}_{h}": f"{c} fails within {'24h' if h == '24h' else '7 days'}"
             for h in ("24h", "7d") for c in comps}
    st.dataframe(frame(metrics["azure_pdm"], names), hide_index=True, column_config=cfg)
    a = metrics["azure_pdm"]["anomaly"]
    st.caption(
        "The 24h models are near-perfect because the Azure dataset is synthetic with strong pre-failure "
        "signals; expect lower scores on real plant data. The 7-day models are ~10x better than the "
        f"base rate. Anomaly detector flags {a['flag_rate_24h_before_failure']:.0%} of readings in the 24h "
        f"before a failure vs. {a['flag_rate_test']:.0%} overall."
    )
