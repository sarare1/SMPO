import numpy as np
import pandas as pd

from src.data.features import add_ai4i_features, physics_rule_flags

BASE = {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0, "rpm": 1500.0,
        "torque_nm": 40.0, "tool_wear_min": 100.0}


def test_derived_features():
    out = add_ai4i_features(pd.DataFrame([BASE])).iloc[0]
    assert out["temp_diff_k"] == 10.0
    assert np.isclose(out["power_w"], 40 * 1500 * 2 * np.pi / 60)
    assert out["strain_minnm"] == 4000.0
    assert np.isclose(out["strain_ratio"], 4000 / 12000)
    assert out["type_code"] == 1


def test_healthy_cycle_has_no_flags():
    assert physics_rule_flags(BASE) == {}


def test_heat_dissipation_rule():
    flags = physics_rule_flags(dict(BASE, process_temp_k=308.0, rpm=1300.0))
    assert "HDF" in flags


def test_power_rule_both_sides():
    assert "PWF" in physics_rule_flags(dict(BASE, rpm=2800.0, torque_nm=8.0))    # ~2.3 kW, too low
    assert "PWF" in physics_rule_flags(dict(BASE, rpm=1500.0, torque_nm=65.0))   # ~10.2 kW, too high


def test_overstrain_limit_depends_on_type():
    row = dict(BASE, tool_wear_min=200.0, torque_nm=56.0)  # 11200 min*Nm
    assert "OSF" in physics_rule_flags(dict(row, type="L"))       # limit 11000
    assert "OSF" not in physics_rule_flags(dict(row, type="H"))   # limit 13000 (> 90% = 11700)


def test_tool_wear_rule():
    assert "TWF" in physics_rule_flags(dict(BASE, tool_wear_min=195.0))
