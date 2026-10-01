"""Risk- and load-aware maintenance scheduling with OR-Tools CP-SAT.

WHAT THIS FILE DOES (plain English)
-----------------------------------
Builds the maintenance plan for the coming week. For every component the models
say is at risk, it decides: replace it today, on a later day, or not at all -
and picks the combination that costs least overall.

The trade-off it balances:
  * Replace too late  -> the part may break first (expensive surprise breakdown).
  * Replace on a busy day -> the stopped machine costs more lost production.
  * The crew can only do a limited number of jobs per day.

Each at-risk (machine, component) is a candidate job. Scheduling it on day d
costs planned maintenance + downtime weighted by that day's production load,
plus the expected cost of failing before day d. Leaving it unscheduled costs
the expected failure cost over the horizon. The crew can do a limited number
of jobs per day.

OR-Tools CP-SAT is Google's free optimisation solver; it searches all possible
plans and returns the cheapest one that respects the crew limit.
"""
# Lets Python understand modern type hints on all versions.
from __future__ import annotations

# --- Imports: tools this file needs -----------------------------------------
import pandas as pd                  # tables of data
from ortools.sat.python import cp_model  # Google OR-Tools optimisation solver

from src import config                   # costs, downtime hours, crew size, daily load
from src.data.features import COMPONENTS  # the four component names

# Full cost of one surprise breakdown: repair cost + lost production while the machine is down.
UNPLANNED_COST = config.COST["unplanned_failure"] + config.DOWNTIME_HOURS["unplanned"] * config.COST["downtime_per_hour"]


def failure_prob_before_day(p24: float, p7d: float, day: int) -> float:
    """P(failure before maintenance at the start of `day`), day 0 = today.

    Day 1 uses the 24h model; later days extend it with the constant daily
    hazard implied by the 7-day model.
    """
    # Maintenance today happens before any failure.
    if day <= 0:
        return 0.0
    # Spread the 7-day risk evenly over 7 days to get a per-day chance of failure.
    daily = 1 - (1 - min(p7d, 0.999)) ** (1 / 7)
    # Chance of surviving day 1 (24h model) and each later day, turned into a chance of failing.
    return 1 - (1 - p24) * (1 - daily) ** (day - 1)


def plan(
    snapshot: pd.DataFrame,
    horizon_days: int = config.PLANNING_HORIZON_DAYS,
    crew_per_day: int = config.MAINTENANCE_CREW_PER_DAY,
    daily_load: list[float] | None = None,
    min_risk: float = 0.05,
) -> dict:
    """Return the cheapest maintenance plan for the fleet risks in `snapshot`."""
    # Production load for each planned day (missing days count as normal load 1.0).
    load = (daily_load or config.DAILY_LOAD)[:horizon_days]
    load += [1.0] * (horizon_days - len(load))

    # Step 1 - list the candidate jobs and what each would cost on each day.
    jobs = []
    for machine_id, row in snapshot.iterrows():
        for comp in COMPONENTS:
            p24, p7 = float(row[f"p_{comp}_24h"]), float(row[f"p_{comp}_7d"])
            # Ignore components with less than 5% risk - not worth planning.
            if max(p24, p7) < min_risk:
                continue
            # Expected cost if we do nothing this week.
            p_h = failure_prob_before_day(p24, p7, horizon_days)
            # Expected cost if we replace it on day d: job cost + downtime on that day's
            # load + the chance it breaks before we get there.
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

    # Nothing at risk -> empty plan.
    if not jobs:
        return {"jobs": [], "expected_cost_plan": 0.0, "expected_cost_run_to_failure": 0.0,
                "expected_savings": 0.0, "status": "no at-risk components"}

    # Step 2 - describe the decision to the solver.
    m = cp_model.CpModel()
    # x[j, d] = yes/no: "do job j on day d"; skip[j] = yes/no: "don't schedule job j".
    x = {(j, d): m.NewBoolVar(f"x_{j}_{d}") for j in range(len(jobs)) for d in range(horizon_days)}
    skip = {j: m.NewBoolVar(f"skip_{j}") for j in range(len(jobs))}
    # Rule: each job is done on exactly one day, or skipped.
    for j in range(len(jobs)):
        m.AddExactlyOne([x[j, d] for d in range(horizon_days)] + [skip[j]])
    # Rule: no more jobs per day than the crew can handle.
    for d in range(horizon_days):
        m.Add(sum(x[j, d] for j in range(len(jobs))) <= crew_per_day)
    # Goal: the lowest total expected cost.
    m.Minimize(
        sum(int(jobs[j]["day_costs"][d]) * x[j, d] for j in range(len(jobs)) for d in range(horizon_days))
        + sum(int(jobs[j]["skip_cost"]) * skip[j] for j in range(len(jobs)))
    )
    # Step 3 - solve (stop after 10 seconds at most).
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(m)

    # Step 4 - read the chosen day for each job and add up the costs.
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
    # Order the plan: earliest day first, riskiest first within a day, skipped jobs last.
    scheduled.sort(key=lambda s: (s["day"] is None, s["day"] if s["day"] is not None else 0, -s["p_fail_7d"]))
    # For comparison: the expected cost of doing nothing at all ("run to failure").
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
