"""Feature engineering for both datasets.

WHAT THIS FILE DOES (plain English)
-----------------------------------
Step 2 of the pipeline. Raw sensor readings are not very informative on their own,
so this file calculates more meaningful measurements ("features") that help the
prediction models spot problems, and marks what actually happened ("labels").

AI4I 2020 (machining line) -> one row per production cycle. Adds physics-based
    values such as the machine's power and the strain on the cutting tool.
Azure PdM (machine fleet) -> one row per machine every 3 hours. Adds averages and
    fluctuation of each sensor over the last 3 and 24 hours, how many error codes
    appeared in the last day, how old each component is, and whether the
    component actually failed within the next 24 hours / 7 days.

It also contains the plant's known "physics rules" - the conditions under which
a production cycle is known to fail - used to explain predictions in plain words.

Usage:  python -m src.data.features
"""
# Lets Python understand modern type hints on all versions.
from __future__ import annotations

# --- Imports: tools this file needs -----------------------------------------
import numpy as np   # fast maths (e.g. the number pi)
import pandas as pd  # tables of data (like Excel sheets inside Python)

from src import config  # project settings: folders, sampling interval, prediction horizons

# --------------------------------------------------------------------------
# AI4I 2020 - machining line
# --------------------------------------------------------------------------
# Rename the dataset's long column titles to short names used in the code.
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
# Failure modes we predict: Tool Wear, Heat Dissipation, Power, OverStrain failure.
AI4I_MODES = ["TWF", "HDF", "PWF", "OSF"]  # RNF is random by construction -> not modelled
# Product quality variants Low / Medium / High, converted to numbers for the models.
AI4I_TYPE_CODE = {"L": 0, "M": 1, "H": 2}
# Overstrain limit (tool wear x torque, min*Nm) per product quality variant.
OSF_LIMIT = {"L": 11_000, "M": 12_000, "H": 13_000}

# The machine settings and readings of one production cycle.
AI4I_SETPOINTS = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
# Everything the AI4I prediction models look at.
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
    # Product variant L/M/H as a number 0/1/2.
    out["type_code"] = out["type"].map(AI4I_TYPE_CODE).astype(int)
    # How much hotter the process is than the surrounding air (cooling headroom).
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    # Mechanical power in watts = torque x rotational speed (converted to radians per second).
    out["power_w"] = out["torque_nm"] * out["rpm"] * 2 * np.pi / 60
    # Strain on the tool = how worn it is x how hard it is pushed.
    out["strain_minnm"] = out["tool_wear_min"] * out["torque_nm"]
    # Strain as a fraction of the overstrain limit for this product variant (1.0 = at the limit).
    out["strain_ratio"] = out["strain_minnm"] / out["type"].map(OSF_LIMIT)
    return out


def load_ai4i() -> pd.DataFrame:
    """Read the AI4I CSV file, tidy the column names and add the physics features."""
    raw = pd.read_csv(config.DATA_RAW / "ai4i2020.csv", encoding="utf-8-sig")
    df = raw.rename(columns=AI4I_RENAME).drop(columns=["UDI"])  # UDI is just a row number
    return add_ai4i_features(df)


def physics_rule_flags(row: pd.Series | dict) -> dict[str, str]:
    """Known failure mechanisms of the AI4I process (from the dataset paper).

    Returns {mode: explanation} for every mechanism whose threshold is breached
    or nearly breached. These give the agents hard, explainable evidence.
    """
    r = dict(row)
    flags: dict[str, str] = {}
    # Recalculate the physics values for this one cycle.
    temp_diff = r["process_temp_k"] - r["air_temp_k"]
    power = r["torque_nm"] * r["rpm"] * 2 * np.pi / 60
    strain = r["tool_wear_min"] * r["torque_nm"]
    limit = OSF_LIMIT[r["type"]]
    # Rule 1 - heat dissipation: too little cooling headroom AND too slow a spindle.
    if temp_diff < 8.6 and r["rpm"] < 1380:
        flags["HDF"] = (
            f"Heat dissipation: temp diff {temp_diff:.1f} K < 8.6 K and speed "
            f"{r['rpm']:.0f} rpm < 1380 rpm"
        )
    # Rule 2 - power: the machine fails when power is outside 3,500-9,000 W
    # (also warn when it is close to either limit).
    if power < 3500 or power > 9000:
        flags["PWF"] = f"Power {power:.0f} W outside safe band 3500-9000 W"
    elif power < 3800 or power > 8600:
        flags["PWF"] = f"Power {power:.0f} W close to band limit 3500-9000 W"
    # Rule 3 - overstrain: tool strain above 90% of the limit for this product variant.
    if strain > 0.9 * limit:
        flags["OSF"] = (
            f"Overstrain: wear x torque {strain:.0f} min*Nm vs limit {limit} "
            f"for type {r['type']} ({strain / limit:.0%})"
        )
    # Rule 4 - tool wear: tools break at a random point between 200 and 240 minutes of use.
    if r["tool_wear_min"] >= 190:
        flags["TWF"] = f"Tool wear {r['tool_wear_min']:.0f} min - tools fail between 200-240 min"
    return flags


# --------------------------------------------------------------------------
# Azure Predictive Maintenance - machine fleet
# --------------------------------------------------------------------------
SENSORS = ["volt", "rotate", "pressure", "vibration"]          # the four sensors on each machine
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]              # the four replaceable components
ERRORS = ["error1", "error2", "error3", "error4", "error5"]     # the five error codes machines report
MODEL_CODE = {"model1": 0, "model2": 1, "model3": 2, "model4": 3}  # machine model as a number


def _read(name: str) -> pd.DataFrame:
    """Read one Azure CSV file and turn its date column into real dates."""
    df = pd.read_csv(config.DATA_RAW / f"{name}.csv")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_azure_raw() -> dict[str, pd.DataFrame]:
    """Read all five Azure files into a dictionary of tables."""
    return {
        "telemetry": _read("PdM_telemetry"),   # hourly sensor readings
        "errors": _read("PdM_errors"),         # error codes raised by machines
        "maint": _read("PdM_maint"),           # component replacements
        "failures": _read("PdM_failures"),     # component failures
        "machines": _read("PdM_machines"),     # machine model and age
    }


def _rolling(tel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Average and fluctuation (standard deviation) of each sensor over the last `window` hours."""
    g = tel.groupby("machineID")[SENSORS]  # calculate separately for each machine
    mean = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    std = g.rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
    # Name the new columns, e.g. "volt_mean_24h", "volt_std_24h".
    mean.columns = [f"{s}_mean_{window}h" for s in SENSORS]
    std.columns = [f"{s}_std_{window}h" for s in SENSORS]
    return pd.concat([mean, std], axis=1)


def _asof_per_component(
    grid: pd.DataFrame, events: pd.DataFrame, comp_col: str, direction: str
) -> pd.DataFrame:
    """For each grid row and component, time of nearest event strictly before/after."""
    # Used twice: "when was this component last replaced?" (backward) and
    # "when does it next fail?" (forward). "Strictly" avoids peeking at the same moment.
    out = grid[["machineID", "datetime"]].copy()
    left = grid[["machineID", "datetime"]].sort_values("datetime")
    for comp in COMPONENTS:
        # Events (replacements or failures) for this one component only.
        ev = (
            events.loc[events[comp_col] == comp, ["machineID", "datetime"]]
            .rename(columns={"datetime": "event_time"})
            .sort_values("event_time")
        )
        # Match every row to the nearest event of the same machine in time.
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
    """Build the full Azure feature table: one row per machine every few hours."""
    raw = load_azure_raw()
    # Put readings in order: machine by machine, oldest first.
    tel = raw["telemetry"].sort_values(["machineID", "datetime"]).reset_index(drop=True)

    # Raw readings + 3-hour and 24-hour averages and fluctuations.
    feats = pd.concat([tel, _rolling(tel, 3), _rolling(tel, 24)], axis=1)

    # Error counts over the previous 24h
    err = raw["errors"].copy()
    # One column per error code with a 1 where that error happened.
    err_hot = pd.get_dummies(err["errorID"]).reindex(columns=ERRORS, fill_value=0).astype(int)
    err_hot[["machineID", "datetime"]] = err[["machineID", "datetime"]]
    err_hot = err_hot.groupby(["machineID", "datetime"], as_index=False).sum()
    feats = feats.merge(err_hot, on=["machineID", "datetime"], how="left")
    feats[ERRORS] = feats[ERRORS].fillna(0)  # no error that hour -> 0
    # Add up each error code over the last 24 hours, per machine.
    rolled = feats.groupby("machineID")[ERRORS].rolling(24, min_periods=1).sum()
    feats[[f"{e}_count_24h" for e in ERRORS]] = rolled.reset_index(level=0, drop=True).values
    feats = feats.drop(columns=ERRORS)

    # Down-sample (keeps rolling windows computed on full hourly data)
    feats = feats[feats["datetime"].dt.hour % sample_every_h == 0].reset_index(drop=True)

    # Component age: days since last replacement (strictly before t)
    last_rep = _asof_per_component(feats, raw["maint"], "comp", "backward")
    for comp in COMPONENTS:
        age = (feats["datetime"] - last_rep[comp]).dt.total_seconds() / 86400  # seconds -> days
        feats[f"{comp}_days_since_repl"] = age.fillna(365.0)  # no record -> old component

    # Machine metadata (model and age in years)
    machines = raw["machines"].assign(model_code=lambda d: d["model"].map(MODEL_CODE))
    feats = feats.merge(machines[["machineID", "model_code", "age"]], on="machineID", how="left")

    # Labels: component fails within each horizon (strictly after t)
    # These are the "answers" the models learn to predict: 1 = failed within 24h / 7 days.
    next_fail = _asof_per_component(feats, raw["failures"], "failure", "forward")
    for comp in COMPONENTS:
        delta = next_fail[comp] - feats["datetime"]  # time until this component's next failure
        for tag, hours in config.HORIZONS_H.items():
            feats[f"fail_{comp}_{tag}"] = (delta <= pd.Timedelta(hours=hours)).astype(int)

    # Use smaller number storage to halve memory use.
    float_cols = feats.select_dtypes("float64").columns
    feats[float_cols] = feats[float_cols].astype("float32")
    return feats


# Everything the Azure prediction models look at: sensors, their averages and
# fluctuations, error counts, component ages, machine model and age.
AZURE_FEATURES = (
    SENSORS
    + [f"{s}_{stat}_{w}h" for w in (3, 24) for stat in ("mean", "std") for s in SENSORS]
    + [f"{e}_count_24h" for e in ERRORS]
    + [f"{c}_days_since_repl" for c in COMPONENTS]
    + ["model_code", "age"]
)
# The answer columns, grouped by horizon: {"24h": [fail_comp1_24h, ...], "7d": [...]}.
AZURE_LABELS = {tag: [f"fail_{c}_{tag}" for c in COMPONENTS] for tag in config.HORIZONS_H}
# The 24-hour sensor statistics used by the anomaly (unusual behaviour) detector.
ANOMALY_FEATURES = [f"{s}_{stat}_24h" for stat in ("mean", "std") for s in SENSORS]


def main() -> None:
    """Build both feature tables and save them to data/processed/."""
    ai4i = load_ai4i()
    ai4i.to_parquet(config.DATA_PROCESSED / "ai4i_features.parquet", index=False)
    print(f"AI4I: {ai4i.shape}, failure rate {ai4i['machine_failure'].mean():.2%}")

    azure = build_azure_features()
    azure.to_parquet(config.DATA_PROCESSED / "azure_features.parquet", index=False)
    print(f"Azure PdM: {azure.shape}")
    # Show how often each failure label is 1 (how rare failures are).
    for labels in AZURE_LABELS.values():
        print(azure[labels].mean().rename("positive rate"))


# Runs only when this file is started directly (python -m src.data.features).
if __name__ == "__main__":
    main()
