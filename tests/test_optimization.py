import pandas as pd
import pytest

from src.optimization import process_optimizer, scheduler


def test_failure_prob_before_day_is_monotonic():
    probs = [scheduler.failure_prob_before_day(0.1, 0.5, d) for d in range(8)]
    assert probs[0] == 0.0
    assert probs[1] == pytest.approx(0.1)
    assert all(a <= b for a, b in zip(probs, probs[1:]))
    # By day 8 the 7-day hazard has fully accumulated on top of day 1
    assert scheduler.failure_prob_before_day(0.0, 0.5, 8) == pytest.approx(0.5, abs=1e-6)


def _snapshot(rows):
    cols = {}
    for c in ["comp1", "comp2", "comp3", "comp4"]:
        cols[f"p_{c}_24h"] = [r.get(c, (0, 0))[0] for r in rows]
        cols[f"p_{c}_7d"] = [r.get(c, (0, 0))[1] for r in rows]
    return pd.DataFrame(cols, index=range(1, len(rows) + 1))


def test_scheduler_respects_crew_capacity_and_urgency():
    snap = _snapshot([{"comp1": (0.9, 0.95)}, {"comp2": (0.8, 0.9)}, {"comp3": (0.7, 0.9)}, {"comp4": (0.0, 0.3)}])
    plan = scheduler.plan(snap, horizon_days=7, crew_per_day=2)
    days = [j["day"] for j in plan["jobs"] if j["day"] is not None]
    assert max(days.count(d) for d in set(days)) <= 2
    urgent = {(j["machine_id"], j["component"]): j["day"] for j in plan["jobs"]}
    assert urgent[(1, "comp1")] == 0 and urgent[(2, "comp2")] == 0
    assert plan["expected_savings"] > 0


def test_scheduler_skips_low_risk():
    plan = scheduler.plan(_snapshot([{"comp1": (0.0, 0.01)}]))
    assert plan["jobs"] == []


def test_optimizer_resolves_heat_dissipation_failure(process_model):
    tc = process_model.test_cycles
    base = process_model.cycle(int(tc.index[tc["HDF"] == 1][0]))
    result = process_optimizer.optimize(process_model, base, min_throughput_ratio=0.9)
    best = result["recommended"][0]
    assert result["current_failure_probability"] > 0.5
    assert best["failure_probability"] < 0.1
    assert best["throughput_ratio"] >= 0.9
    # Verified by independent re-simulation
    sim = process_optimizer.simulate(process_model, base, rpm=best["rpm"], torque_nm=best["torque_nm"],
                                     replace_tool=best["replace_tool"])
    assert sim["failure_probability_after"] == pytest.approx(best["failure_probability"], abs=0.02)
