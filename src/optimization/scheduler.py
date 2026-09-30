"""Risk- and load-aware maintenance scheduling with OR-Tools CP-SAT.

Each at-risk (machine, component) is a candidate job. Scheduling it on day d
costs planned maintenance + downtime weighted by that day's production load,
plus the expected cost of failing before day d. Leaving it unscheduled costs
the expected failure cost over the horizon. The crew can do a limited number
of jobs per day.
"""
from __future__ import annotations

import pandas as pd
from ortools.sat.python import cp_model

from src import config
from src.data.features import COMPONENTS

UNPLANNED_COST = config.COST["unplanned_failure"] + config.DOWNTIME_HOURS["unplanned"] * config.COST["downtime_per_hour"]


def failure_prob_before_day(p24: float, p7d: float, day: int) -> float:
    """P(failure before maintenance at the start of `day`), day 0 = today.

    Day 1 uses the 24h model; later days extend it with the constant daily
    hazard implied by the 7-day model.
    """
    if day <= 0:
        return 0.0
    daily = 1 - (1 - min(p7d, 0.999)) ** (1 / 7)
    return 1 - (1 - p24) * (1 - daily) ** (day - 1)


def plan(
    snapshot: pd.DataFrame,
    horizon_days: int = config.PLANNING_HORIZON_DAYS,
    crew_per_day: int = config.MAINTENANCE_CREW_PER_DAY,
    daily_load: list[float] | None = None,
    min_risk: float = 0.05,
) -> dict:
    load = (daily_load or config.DAILY_LOAD)[:horizon_days]
    load += [1.0] * (horizon_days - len(load))

    jobs = []
    for machine_id, row in snapshot.iterrows():
        for comp in COMPONENTS:
            p24, p7 = float(row[f"p_{comp}_24h"]), float(row[f"p_{comp}_7d"])
            if max(p24, p7) < min_risk:
                continue
            p_h = failure_prob_before_day(p24, p7, horizon_days)
            day_costs = [
                config.COST["planned_maintenance"]
                + config.DOWNTIME_HOURS["planned"] * config.COST["downtime_per_hour"] * load[d]
                + failure_prob_before_day(p24, p7, d) * UNPLANNED_COST
                for d in range(horizon_days)
            ]
            jobs.append({
                "machine_id": int(machine_id), "component": comp, "p24": p24, "p7d": p7,
                "skip_cost": p_h * UNPLANNED_COST, "day_costs": day_costs,
            })

    if not jobs:
        return {"jobs": [], "expected_cost_plan": 0.0, "expected_cost_run_to_failure": 0.0,
                "expected_savings": 0.0, "status": "no at-risk components"}

    m = cp_model.CpModel()
    x = {(j, d): m.NewBoolVar(f"x_{j}_{d}") for j in range(len(jobs)) for d in range(horizon_days)}
    skip = {j: m.NewBoolVar(f"skip_{j}") for j in range(len(jobs))}
    for j in range(len(jobs)):
        m.AddExactlyOne([x[j, d] for d in range(horizon_days)] + [skip[j]])
    for d in range(horizon_days):
        m.Add(sum(x[j, d] for j in range(len(jobs))) <= crew_per_day)
    m.Minimize(
        sum(int(jobs[j]["day_costs"][d]) * x[j, d] for j in range(len(jobs)) for d in range(horizon_days))
        + sum(int(jobs[j]["skip_cost"]) * skip[j] for j in range(len(jobs)))
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(m)

    scheduled, total = [], 0.0
    for j, job in enumerate(jobs):
        day = next((d for d in range(horizon_days) if solver.Value(x[j, d])), None)
        cost = job["day_costs"][day] if day is not None else job["skip_cost"]
        total += cost
        scheduled.append({
            "machine_id": job["machine_id"],
            "component": job["component"],
            "p_fail_24h": round(job["p24"], 3),
            "p_fail_7d": round(job["p7d"], 3),
            "action": f"replace on day {day}" if day is not None else "monitor (not worth scheduling)",
            "day": day,
            "production_load": load[day] if day is not None else None,
            "expected_cost": round(cost, 0),
        })
    scheduled.sort(key=lambda s: (s["day"] is None, s["day"] if s["day"] is not None else 0, -s["p_fail_7d"]))
    run_to_failure = sum(j["skip_cost"] for j in jobs)
    return {
        "status": solver.StatusName(status),
        "horizon_days": horizon_days,
        "crew_per_day": crew_per_day,
        "daily_load": load,
        "jobs": scheduled,
        "jobs_scheduled": sum(s["day"] is not None for s in scheduled),
        "expected_cost_plan": round(total, 0),
        "expected_cost_run_to_failure": round(run_to_failure, 0),
        "expected_savings": round(run_to_failure - total, 0),
    }
