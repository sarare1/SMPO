"""Feature engineering for both datasets.

AI4I 2020  -> one row per production cycle, physics-derived features.
Azure PdM  -> one row per machine per (sampled) hour, rolling telemetry stats,
              error counts, component age, and forward-looking labels.

Usage:  python -m src.data.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# --------------------------------------------------------------------------
# AI4I 2020
# --------------------------------------------------------------------------
AI4I_RENAME = {
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}
AI4I_MODES = ["TWF", "HDF", "PWF", "OSF"]  # RNF is random by construction -> not modelled
AI4I_TYPE_CODE = {"L": 0, "M": 1, "H": 2}
# Overstrain limit (tool wear x torque, min*Nm) per product quality variant.
OSF_LIMIT = {"L": 11_000, "M": 12_000, "H": 13_000}

AI4I_SETPOINTS = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
AI4I_FEATURES = [
    "type_code",
    *AI4I_SETPOINTS,
    "temp_diff_k",
    "power_w",
    "strain_minnm",
    "strain_ratio",
]


def add_ai4i_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive physics features. Works on raw-renamed rows or what-if rows."""
    out = df.copy()
    out["type_code"] = out["type"].map(AI4I_TYPE_CODE).astype(int)
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rpm"] * 2 * np.pi / 60
    out["strain_minnm"] = out["tool_wear_min"] * out["torque_nm"]
    out["strain_ratio"] = out["strain_minnm"] / out["type"].map(OSF_LIMIT)
    return out


def load_ai4i() -> pd.DataFrame:
    raw = pd.read_csv(config.DATA_RAW / "ai4i2020.csv", encoding="utf-8-sig")
    df = raw.rename(columns=AI4I_RENAME).drop(columns=["UDI"])
    return add_ai4i_features(df)


def physics_rule_flags(row: pd.Series | dict) -> dict[str, str]:
    """Known failure mechanisms of the AI4I process (from the dataset paper).

    Returns {mode: explanation} for every mechanism whose threshold is breached
    or nearly breached. These give the agents hard, explainable evidence.
    """
    r = dict(row)
    flags: dict[str, str] = {}
    temp_diff = r["process_temp_k"] - r["air_temp_k"]
    power = r["torque_nm"] * r["rpm"] * 2 * np.pi / 60
    strain = r["tool_wear_min"] * r["torque_nm"]
    limit = OSF_LIMIT[r["type"]]
    if temp_diff < 8.6 and r["rpm"] < 1380:
        flags["HDF"] = (
            f"Heat dissipation: temp diff {temp_diff:.1f} K < 8.6 K and speed "
            f"{r['rpm']:.0f} rpm < 1380 rpm"
        )
    if power < 3500 or power > 9000:
        flags["PWF"] = f"Power {power:.0f} W outside safe band 3500-9000 W"
    elif power < 3800 or power > 8600:
        flags["PWF"] = f"Power {power:.0f} W close to band limit 3500-9000 W"
    if strain > 0.9 * limit:
        flags["OSF"] = (
            f"Overstrain: wear x torque {strain:.0f} min*Nm vs limit {limit} "
            f"for type {r['type']} ({strain / limit:.0%})"
        )
    if r["tool_wear_min"] >= 190:
        flags["TWF"] = f"Tool wear {r['tool_wear_min']:.0f} min - tools fail between 200-240 min"
    return flags


# --------------------------------------------------------------------------
# Azure Predictive Maintenance
# --------------------------------------------------------------------------
SENSORS = ["volt", "rotate", "pressure", "vibration"]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
ERRORS = ["error1", "error2", "error3", "error4", "error5"]
MODEL_CODE = {"model1": 0, "model2": 1, "model3": 2, "model4": 3}


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.DATA_RAW / f"{name}.csv")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_azure_raw() -> dict[str, pd.DataFrame]:
    return {
        "telemetry": _read("PdM_telemetry"),
        "errors": _read("PdM_errors"),
        "maint": _read("PdM_maint"),
        "failures": _read("PdM_failures"),
        "machines": _read("PdM_machines"),
    }


def _rolling(tel: pd.DataFrame, window: int) -> pd.DataFrame:
    g = tel.groupby("machineID")[SENSORS]
    mean = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    std = g.rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
    mean.columns = [f"{s}_mean_{window}h" for s in SENSORS]
    std.columns = [f"{s}_std_{window}h" for s in SENSORS]
    return pd.concat([mean, std], axis=1)


def _asof_per_component(
    grid: pd.DataFrame, events: pd.DataFrame, comp_col: str, direction: str
) -> pd.DataFrame:
    """For each grid row and component, time of nearest event strictly before/after."""
    out = grid[["machineID", "datetime"]].copy()
    left = grid[["machineID", "datetime"]].sort_values("datetime")
    for comp in COMPONENTS:
        ev = (
            events.loc[events[comp_col] == comp, ["machineID", "datetime"]]
            .rename(columns={"datetime": "event_time"})
            .sort_values("event_time")
        )
        merged = pd.merge_asof(
            left,
            ev,
            left_on="datetime",
            right_on="event_time",
            by="machineID",
            direction=direction,
            allow_exact_matches=False,
        )
        out[comp] = merged.set_index(left.index)["event_time"]
    return out


def build_azure_features(sample_every_h: int = config.AZURE_SAMPLE_EVERY_H) -> pd.DataFrame:
    raw = load_azure_raw()
    tel = raw["telemetry"].sort_values(["machineID", "datetime"]).reset_index(drop=True)

    feats = pd.concat([tel, _rolling(tel, 3), _rolling(tel, 24)], axis=1)

    # Error counts over the previous 24h
    err = raw["errors"].copy()
    err_hot = pd.get_dummies(err["errorID"]).reindex(columns=ERRORS, fill_value=0).astype(int)
    err_hot[["machineID", "datetime"]] = err[["machineID", "datetime"]]
    err_hot = err_hot.groupby(["machineID", "datetime"], as_index=False).sum()
    feats = feats.merge(err_hot, on=["machineID", "datetime"], how="left")
    feats[ERRORS] = feats[ERRORS].fillna(0)
    rolled = feats.groupby("machineID")[ERRORS].rolling(24, min_periods=1).sum()
    feats[[f"{e}_count_24h" for e in ERRORS]] = rolled.reset_index(level=0, drop=True).values
    feats = feats.drop(columns=ERRORS)

    # Down-sample (keeps rolling windows computed on full hourly data)
    feats = feats[feats["datetime"].dt.hour % sample_every_h == 0].reset_index(drop=True)

    # Component age: days since last replacement (strictly before t)
    last_rep = _asof_per_component(feats, raw["maint"], "comp", "backward")
    for comp in COMPONENTS:
        age = (feats["datetime"] - last_rep[comp]).dt.total_seconds() / 86400
        feats[f"{comp}_days_since_repl"] = age.fillna(365.0)  # no record -> old component

    # Machine metadata
    machines = raw["machines"].assign(model_code=lambda d: d["model"].map(MODEL_CODE))
    feats = feats.merge(machines[["machineID", "model_code", "age"]], on="machineID", how="left")

    # Labels: component fails within each horizon (strictly after t)
    next_fail = _asof_per_component(feats, raw["failures"], "failure", "forward")
    for comp in COMPONENTS:
        delta = next_fail[comp] - feats["datetime"]
        for tag, hours in config.HORIZONS_H.items():
            feats[f"fail_{comp}_{tag}"] = (delta <= pd.Timedelta(hours=hours)).astype(int)

    float_cols = feats.select_dtypes("float64").columns
    feats[float_cols] = feats[float_cols].astype("float32")
    return feats


AZURE_FEATURES = (
    SENSORS
    + [f"{s}_{stat}_{w}h" for w in (3, 24) for stat in ("mean", "std") for s in SENSORS]
    + [f"{e}_count_24h" for e in ERRORS]
    + [f"{c}_days_since_repl" for c in COMPONENTS]
    + ["model_code", "age"]
)
AZURE_LABELS = {tag: [f"fail_{c}_{tag}" for c in COMPONENTS] for tag in config.HORIZONS_H}
ANOMALY_FEATURES = [f"{s}_{stat}_24h" for stat in ("mean", "std") for s in SENSORS]


def main() -> None:
    ai4i = load_ai4i()
    ai4i.to_parquet(config.DATA_PROCESSED / "ai4i_features.parquet", index=False)
    print(f"AI4I: {ai4i.shape}, failure rate {ai4i['machine_failure'].mean():.2%}")

    azure = build_azure_features()
    azure.to_parquet(config.DATA_PROCESSED / "azure_features.parquet", index=False)
    print(f"Azure PdM: {azure.shape}")
    for labels in AZURE_LABELS.values():
        print(azure[labels].mean().rename("positive rate"))


if __name__ == "__main__":
    main()
