"""Inference API over the trained models, with SHAP explanations.

ProcessModel -> AI4I production-cycle failure risk (process / setpoint level)
FleetModel   -> Azure PdM fleet health (machine / component level)
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from src import config
from src.data.features import (
    AI4I_MODES,
    AI4I_SETPOINTS,
    COMPONENTS,
    SENSORS,
    add_ai4i_features,
    load_azure_raw,
    physics_rule_flags,
)

FEATURE_LABELS = {
    "type_code": "product quality variant",
    "air_temp_k": "air temperature",
    "process_temp_k": "process temperature",
    "rpm": "rotational speed",
    "torque_nm": "torque",
    "tool_wear_min": "tool wear",
    "temp_diff_k": "process-air temperature gap",
    "power_w": "mechanical power",
    "strain_minnm": "tool strain (wear x torque)",
    "strain_ratio": "tool strain vs. overstrain limit",
    "model_code": "machine model",
    "age": "machine age (years)",
    **{f"{c}_days_since_repl": f"days since {c} replaced" for c in COMPONENTS},
    **{f"error{i}_count_24h": f"error{i} count (24h)" for i in range(1, 6)},
    **{
        f"{s}_{stat}_{w}h": f"{s} {w}h {'mean' if stat == 'mean' else 'volatility'}"
        for s in SENSORS for stat in ("mean", "std") for w in (3, 24)
    },
}
MODE_NAMES = {
    "TWF": "tool wear failure",
    "HDF": "heat dissipation failure",
    "PWF": "power failure",
    "OSF": "overstrain failure",
}


def _shap_matrix(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older SHAP returns one array per class
        values = values[1]
    values = np.asarray(values)
    return values[..., 1] if values.ndim == 3 else values


def _top_contributions(names, values, row, k: int = 5) -> list[dict]:
    order = np.argsort(-np.abs(values))[:k]
    return [
        {
            "feature": names[i],
            "label": FEATURE_LABELS.get(names[i], names[i]),
            "value": round(float(row[names[i]]), 3),
            "impact_log_odds": round(float(values[i]), 3),
            "direction": "raises risk" if values[i] > 0 else "lowers risk",
        }
        for i in order
    ]


# --------------------------------------------------------------------------
# Process level (AI4I 2020)
# --------------------------------------------------------------------------
class ProcessModel:
    def __init__(self) -> None:
        bundle = joblib.load(config.MODELS_DIR / "ai4i_models.joblib")
        self.failure = bundle["failure"]
        self.modes = bundle["modes"]
        self.threshold = bundle["threshold"]
        self.features = bundle["features"]
        self.explainer = shap.TreeExplainer(self.failure)
        self.test_cycles = pd.read_parquet(config.DATA_PROCESSED / "ai4i_test.parquet")
        ranges = self.test_cycles[AI4I_SETPOINTS]
        self.bounds = {c: (float(ranges[c].min()), float(ranges[c].max())) for c in AI4I_SETPOINTS}

    def _frame(self, rows: pd.DataFrame | dict) -> pd.DataFrame:
        df = pd.DataFrame([rows]) if isinstance(rows, dict) else rows
        return add_ai4i_features(df[["type", *AI4I_SETPOINTS]])

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.failure.predict_proba(self._frame(rows)[self.features])[:, 1]

    def diagnose(self, row: dict) -> dict:
        X = self._frame(row)
        feats = X[self.features]
        p_fail = float(self.failure.predict_proba(feats)[0, 1])
        modes = {m: round(float(self.modes[m].predict_proba(feats)[0, 1]), 4) for m in AI4I_MODES}
        shap_vals = _shap_matrix(self.explainer, feats)[0]
        return {
            "setpoints": {k: row[k] for k in ["type", *AI4I_SETPOINTS]},
            "derived": {
                "temp_diff_k": round(float(X["temp_diff_k"].iloc[0]), 2),
                "power_w": round(float(X["power_w"].iloc[0]), 1),
                "strain_minnm": round(float(X["strain_minnm"].iloc[0]), 1),
            },
            "failure_probability": round(p_fail, 4),
            "alert": bool(p_fail >= self.threshold),
            "alert_threshold": round(self.threshold, 3),
            "failure_mode_probabilities": {MODE_NAMES[m]: p for m, p in modes.items()},
            "physics_rule_flags": physics_rule_flags(row),
            "top_risk_drivers": _top_contributions(self.features, shap_vals, X.iloc[0]),
        }

    def cycle(self, index: int) -> dict:
        row = self.test_cycles.loc[index]
        return {k: (row[k] if k == "type" else float(row[k])) for k in ["type", *AI4I_SETPOINTS]}


# --------------------------------------------------------------------------
# Fleet level (Azure PdM)
# --------------------------------------------------------------------------
class FleetModel:
    def __init__(self) -> None:
        bundle = joblib.load(config.MODELS_DIR / "azure_models.joblib")
        self.components = bundle["components"]  # {horizon: {comp: model}}
        self.anomaly = bundle["anomaly"]
        self.anomaly_ref = bundle["anomaly_ref"]
        self.features = bundle["features"]
        self.anomaly_features = bundle["anomaly_features"]
        self.explainers = {
            tag: {c: shap.TreeExplainer(m) for c, m in models.items()}
            for tag, models in self.components.items()
        }
        feats = pd.read_parquet(config.DATA_PROCESSED / "azure_features.parquet")
        self.data = feats[feats["datetime"] >= config.AZURE_SPLIT_DATE].reset_index(drop=True)
        self.timestamps = pd.DatetimeIndex(sorted(self.data["datetime"].unique()))
        raw = load_azure_raw()
        self.telemetry = raw["telemetry"]
        self.errors = raw["errors"]
        self.maint = raw["maint"]
        self.failures = raw["failures"]
        self.machines = raw["machines"].set_index("machineID")

    def _anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """0 = typical (train median), 1 = the train 99th percentile."""
        raw = -self.anomaly.score_samples(X[self.anomaly_features].fillna(0))
        p50, p99 = self.anomaly_ref
        return np.clip((raw - p50) / (p99 - p50), 0, None)

    def snapshot(self, ts: pd.Timestamp) -> pd.DataFrame:
        """Risk for every machine at time ts (one row per machine)."""
        rows = self.data[self.data["datetime"] == ts].set_index("machineID").sort_index()
        X = rows[self.features]
        out = pd.DataFrame(index=rows.index)
        for tag, models in self.components.items():
            for comp, model in models.items():
                out[f"p_{comp}_{tag}"] = model.predict_proba(X)[:, 1]
        cols_24 = [f"p_{c}_24h" for c in COMPONENTS]
        cols_7d = [f"p_{c}_7d" for c in COMPONENTS]
        out["risk_24h"] = 1 - (1 - out[cols_24]).prod(axis=1)
        out["risk_7d"] = 1 - (1 - out[cols_7d]).prod(axis=1)
        out["top_component"] = out[cols_7d].idxmax(axis=1).str.extract(r"p_(comp\d)")[0]
        out["anomaly_score"] = self._anomaly_score(rows)
        out["model"] = self.machines.loc[out.index, "model"]
        out["age_years"] = self.machines.loc[out.index, "age"]
        return out.sort_values("risk_7d", ascending=False)

    def machine_detail(self, machine_id: int, ts: pd.Timestamp, days: int = 7) -> dict:
        row = self.data[(self.data["datetime"] == ts) & (self.data["machineID"] == machine_id)]
        if row.empty:
            raise ValueError(f"No data for machine {machine_id} at {ts}")
        X = row[self.features]
        risks, drivers = {}, {}
        for tag, models in self.components.items():
            risks[tag] = {c: round(float(m.predict_proba(X)[0, 1]), 4) for c, m in models.items()}
        worst = max(risks["7d"], key=risks["7d"].get)
        drivers = _top_contributions(
            self.features, _shap_matrix(self.explainers["7d"][worst], X)[0], row.iloc[0], k=6
        )
        since = ts - pd.Timedelta(days=days)
        m = self.machines.loc[machine_id]
        tel = self.telemetry[
            (self.telemetry["machineID"] == machine_id) & self.telemetry["datetime"].between(since, ts)
        ]
        recent = tel[SENSORS].tail(24).mean()
        baseline = self.telemetry.loc[self.telemetry["machineID"] == machine_id, SENSORS].mean()
        errors = self.errors[
            (self.errors["machineID"] == machine_id) & self.errors["datetime"].between(since, ts)
        ]
        maint = self.maint[(self.maint["machineID"] == machine_id) & (self.maint["datetime"] <= ts)]
        return {
            "machine_id": int(machine_id),
            "timestamp": str(ts),
            "model": m["model"],
            "age_years": int(m["age"]),
            "failure_probability_24h": risks["24h"],
            "failure_probability_7d": risks["7d"],
            "highest_risk_component": worst,
            "top_risk_drivers_for_highest_component": drivers,
            "anomaly_score": round(float(self._anomaly_score(row)[0]), 3),
            "sensor_last_24h_vs_lifetime_mean": {
                s: {"last_24h": round(float(recent[s]), 2), "lifetime": round(float(baseline[s]), 2),
                    "deviation_pct": round(float((recent[s] / baseline[s] - 1) * 100), 1)}
                for s in SENSORS
            },
            "errors_last_7d": errors["errorID"].value_counts().to_dict(),
            "days_since_replacement": {
                c: round(float(row.iloc[0][f"{c}_days_since_repl"]), 1) for c in COMPONENTS
            },
            "last_replacements": [
                {"date": str(r.datetime.date()), "component": r.comp} for r in maint.tail(5).itertuples()
            ],
        }

    def telemetry_window(self, machine_id: int, ts: pd.Timestamp, days: int = 14) -> pd.DataFrame:
        since = ts - pd.Timedelta(days=days)
        tel = self.telemetry[
            (self.telemetry["machineID"] == machine_id) & self.telemetry["datetime"].between(since, ts)
        ]
        return tel.set_index("datetime")[SENSORS]

    def actual_failures(self, machine_id: int, ts: pd.Timestamp, days: int = 7) -> pd.DataFrame:
        """Ground truth, for evaluating the demo only (never shown to agents)."""
        end = ts + pd.Timedelta(days=days)
        f = self.failures
        return f[(f["machineID"] == machine_id) & (f["datetime"] > ts) & (f["datetime"] <= end)]


@lru_cache(maxsize=1)
def get_process_model() -> ProcessModel:
    return ProcessModel()


@lru_cache(maxsize=1)
def get_fleet_model() -> FleetModel:
    return FleetModel()
