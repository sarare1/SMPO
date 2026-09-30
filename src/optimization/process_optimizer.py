"""What-if simulation and setpoint optimisation for a production cycle (AI4I).

Every candidate adjustment is re-scored with the trained failure model, so
recommendations always carry a model-verified risk reduction.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.data.features import physics_rule_flags
from src.models.predictor import ProcessModel

RPM_FACTORS = np.round(np.arange(0.85, 1.30, 0.05), 2)
TORQUE_FACTORS = np.round(np.arange(0.70, 1.51, 0.05), 2)
TOOL_CHANGE_PENALTY = 0.02   # changeover time expressed in risk-equivalent units
THROUGHPUT_WEIGHT = 0.5      # penalty per unit of lost spindle speed (throughput proxy)
CHANGE_WEIGHT = 0.01         # prefer the smallest adjustment that works


def simulate(model: ProcessModel, base: dict, rpm: float | None = None,
             torque_nm: float | None = None, replace_tool: bool = False) -> dict:
    """Score a single what-if scenario against the current cycle."""
    new = dict(base)
    if rpm is not None:
        new["rpm"] = float(rpm)
    if torque_nm is not None:
        new["torque_nm"] = float(torque_nm)
    if replace_tool:
        new["tool_wear_min"] = 0.0
    before, after = model.predict_proba(pd.DataFrame([base, new]))
    return {
        "scenario": {k: new[k] for k in ("rpm", "torque_nm", "tool_wear_min")},
        "tool_replaced": replace_tool,
        "failure_probability_before": round(float(before), 4),
        "failure_probability_after": round(float(after), 4),
        "risk_reduction": round(float(before - after), 4),
        "remaining_physics_flags": physics_rule_flags(new),
        "within_training_range": _in_range(model, new),
    }


def _in_range(model: ProcessModel, row: dict) -> bool:
    return all(lo <= row[k] <= hi for k, (lo, hi) in model.bounds.items() if k in ("rpm", "torque_nm"))


def optimize(model: ProcessModel, base: dict, min_throughput_ratio: float = 0.9,
             allow_tool_change: bool = True, top_k: int = 3) -> dict:
    """Grid search over speed/torque (and optional tool change) minimising
    failure risk + throughput loss + size of change."""
    grid = list(itertools.product(RPM_FACTORS, TORQUE_FACTORS, [False, True] if allow_tool_change else [False]))
    rows = []
    for rf, tf, tool in grid:
        if rf < min_throughput_ratio:
            continue
        r = dict(base, rpm=base["rpm"] * rf, torque_nm=base["torque_nm"] * tf)
        if tool:
            r["tool_wear_min"] = 0.0
        if not _in_range(model, r):
            continue
        rows.append((rf, tf, tool, r))
    frame = pd.DataFrame([r for *_, r in rows])
    risk = model.predict_proba(frame)
    base_risk = float(model.predict_proba(pd.DataFrame([base]))[0])

    scored = []
    for (rf, tf, tool, r), p in zip(rows, risk):
        objective = (
            p
            + THROUGHPUT_WEIGHT * max(0.0, 1 - rf)
            + CHANGE_WEIGHT * (abs(rf - 1) + abs(tf - 1)) * 10
            + (TOOL_CHANGE_PENALTY if tool else 0.0)
        )
        scored.append((objective, p, rf, tf, tool, r))
    scored.sort(key=lambda s: s[0])

    candidates, seen = [], set()
    for objective, p, rf, tf, tool, r in scored:
        key = (round(rf, 2), round(tf, 2), tool)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "rpm": round(r["rpm"], 0),
            "rpm_change_pct": round((rf - 1) * 100, 1),
            "torque_nm": round(r["torque_nm"], 1),
            "torque_change_pct": round((tf - 1) * 100, 1),
            "replace_tool": tool,
            "failure_probability": round(float(p), 4),
            "risk_reduction": round(base_risk - float(p), 4),
            "throughput_ratio": round(float(rf), 2),
            "remaining_physics_flags": physics_rule_flags(r),
        })
        if len(candidates) >= top_k:
            break
    return {
        "current_failure_probability": round(base_risk, 4),
        "current_physics_flags": physics_rule_flags(base),
        "constraints": {"min_throughput_ratio": min_throughput_ratio, "allow_tool_change": allow_tool_change},
        "recommended": candidates,
    }
