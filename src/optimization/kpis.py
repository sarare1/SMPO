"""Plant KPIs: OEE and the business case for predictive maintenance.

OEE = Availability x Performance x Quality
  Availability: from the Azure PdM year of failures and planned replacements
  Performance:  mean spindle rotation vs. nominal (config.NOMINAL_ROTATION)
  Quality:      share of AI4I production cycles that end without a failure
Downtime hours per event come from config (assumptions, edit for your plant).
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from src import config
from src.data.features import load_azure_raw


@lru_cache(maxsize=1)
def plant_kpis() -> dict:
    raw = load_azure_raw()
    tel, fails, maint = raw["telemetry"], raw["failures"], raw["maint"]
    n_machines = tel["machineID"].nunique()
    hours = (tel["datetime"].max() - tel["datetime"].min()).total_seconds() / 3600
    in_period = maint["datetime"] >= tel["datetime"].min()

    # Replacements that coincide with a failure are unplanned
    keys = ["datetime", "machineID"]
    fail_keys = fails.rename(columns={"failure": "comp"})[[*keys, "comp"]]
    unplanned = maint[in_period].merge(fail_keys, on=[*keys, "comp"], how="inner")
    planned_count = int(in_period.sum() - len(unplanned))
    fail_count = len(fails)

    down_unplanned = fail_count * config.DOWNTIME_HOURS["unplanned"]
    down_planned = planned_count * config.DOWNTIME_HOURS["planned"]
    availability = 1 - (down_unplanned + down_planned) / (n_machines * hours)
    performance = min(1.0, tel["rotate"].mean() / config.NOMINAL_ROTATION)
    ai4i = pd.read_csv(config.DATA_RAW / "ai4i2020.csv", encoding="utf-8-sig")
    quality = 1 - ai4i["Machine failure"].mean()
    oee = availability * performance * quality

    # Projection: failures the 24h models would catch become planned jobs
    metrics_path = config.REPORTS_DIR / "metrics.json"
    recall = 0.0
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())["azure_pdm"]
        recall = sum(m[f"fail_{c}_24h"]["recall"] for c in ("comp1", "comp2", "comp3", "comp4")) / 4
    prevented = fail_count * recall * config.PDM_REALIZATION
    downtime_saved = prevented * (config.DOWNTIME_HOURS["unplanned"] - config.DOWNTIME_HOURS["planned"])
    cost_saved = prevented * (
        config.COST["unplanned_failure"] - config.COST["planned_maintenance"]
    ) + downtime_saved * config.COST["downtime_per_hour"]
    availability_pdm = availability + downtime_saved / (n_machines * hours)

    return {
        "machines": n_machines,
        "period_days": round(hours / 24),
        "unplanned_failures": fail_count,
        "planned_replacements": planned_count,
        "unplanned_downtime_hours": round(down_unplanned),
        "availability": round(availability, 4),
        "performance": round(performance, 4),
        "quality": round(quality, 4),
        "oee": round(oee, 4),
        "pdm_projection": {
            "detection_rate_24h_models": round(recall, 3),
            "realization_factor": config.PDM_REALIZATION,
            "failures_prevented_per_year": round(prevented),
            "downtime_hours_saved_per_year": round(downtime_saved),
            "cost_saved_per_year_usd": round(cost_saved, -3),
            "availability_with_pdm": round(availability_pdm, 4),
            "oee_with_pdm": round(availability_pdm * performance * quality, 4),
        },
        "assumptions": {"downtime_hours": config.DOWNTIME_HOURS, "costs_usd": config.COST},
    }
