"""Tools the LLM agents can call. Every tool is backed by the trained models,
the optimiser, or the scheduler - the LLM never predicts failures itself.

WHAT THIS FILE DOES (plain English)
-----------------------------------
The AI agents cannot read the factory data directly. Instead they get a fixed
set of seven "tools" - like buttons they can press - and each button runs our
own tested code and returns the facts:

  get_fleet_overview          - which machines are most at risk right now
  get_machine_details         - everything about one machine
  get_process_status          - health check of the current production cycle
  simulate_process_change     - what-if: risk after changing speed/torque/tool
  optimize_process_setpoints  - find the safest small setting change
  plan_maintenance            - build the cheapest maintenance plan for the week
  get_plant_kpis              - OEE and the yearly business case

TOOL_DEFINITIONS describes each tool to the AI (name, purpose, inputs).
ToolBox actually runs a tool when the AI asks for it.
"""
# Lets Python understand modern type hints on all versions.
from __future__ import annotations

# --- Imports: tools this file needs -----------------------------------------
import json                      # turns results into text the AI can read
from typing import Any, Callable  # type hints for "any value" and "a function"

from src import config                                               # default horizon and crew size
from src.models.predictor import get_fleet_model, get_process_model  # the prediction models
from src.optimization import kpis, process_optimizer, scheduler      # KPI, optimiser and scheduler code
from src.simulation.stream import FactoryState                       # the current replay time and cycle


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    """Describe a tool's inputs in the standard JSON-schema format the AI models understand."""
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,  # the AI may not invent extra inputs
    }


# The "menu" of tools shown to the AI: what each one does and which inputs it takes.
TOOL_DEFINITIONS: dict[str, dict] = {
    "get_fleet_overview": {
        "description": (
            "Current health of all 100 machines in the fleet: 24-hour and 7-day failure "
            "probability per component (comp1-comp4) from the predictive models, plus an "
            "anomaly score (0 = typical, >1 = beyond the 99th percentile of normal). "
            "Returns the riskiest machines first."
        ),
        "input_schema": _schema({"top_n": {"type": "integer", "description": "How many machines to return (1-30)."}}),
    },
    "get_machine_details": {
        "description": (
            "Deep dive on one machine: per-component failure probabilities, the SHAP drivers "
            "behind the highest-risk component, sensor readings vs. lifetime mean, recent "
            "error codes, and maintenance history."
        ),
        "input_schema": _schema({"machine_id": {"type": "integer", "description": "Machine ID, 1-100."}}),
    },
    "get_process_status": {
        "description": (
            "Diagnosis of the current production cycle on the machining line: setpoints, "
            "failure probability, probability per failure mode (tool wear, heat dissipation, "
            "power, overstrain), known physics-rule violations, and SHAP risk drivers."
        ),
        "input_schema": _schema({}),
    },
    "simulate_process_change": {
        "description": (
            "What-if: re-score the current production cycle with changed setpoints. Omitted "
            "values keep their current setting. Use to verify a proposed adjustment before "
            "recommending it."
        ),
        "input_schema": _schema(
            {
                "rpm": {"type": ["number", "null"], "description": "New rotational speed in rpm, or null to keep."},
                "torque_nm": {"type": ["number", "null"], "description": "New torque in Nm, or null to keep."},
                "replace_tool": {"type": "boolean", "description": "Fit a new tool (tool wear -> 0)."},
            }
        ),
    },
    "optimize_process_setpoints": {
        "description": (
            "Search speed/torque adjustments (and optionally a tool change) that minimise "
            "failure risk while keeping throughput above a floor. Returns the best "
            "model-verified candidates."
        ),
        "input_schema": _schema(
            {
                "min_throughput_ratio": {
                    "type": "number",
                    "description": "Lowest acceptable spindle speed as a fraction of current (e.g. 0.9).",
                },
                "allow_tool_change": {"type": "boolean", "description": "Whether a tool change is allowed."},
            }
        ),
    },
    "plan_maintenance": {
        "description": (
            "Optimise a maintenance schedule over the coming days with OR-Tools: chooses which "
            "at-risk components to replace and on which day, trading failure risk against "
            "crew capacity and production load. Returns jobs and expected cost vs. run-to-failure."
        ),
        "input_schema": _schema(
            {
                "horizon_days": {"type": "integer", "description": "Planning horizon in days (1-14)."},
                "crew_per_day": {"type": "integer", "description": "Maintenance jobs the crew can do per day."},
            }
        ),
    },
    "get_plant_kpis": {
        "description": (
            "Plant OEE (availability x performance x quality), downtime, failure counts, and "
            "the projected yearly impact of predictive maintenance, with the cost assumptions used."
        ),
        "input_schema": _schema({}),
    },
}


class ToolBox:
    """Executes tool calls against the current factory state."""

    def __init__(self, state: FactoryState) -> None:
        self.state = state  # the current replay time and production cycle
        # Link each tool name to the code that runs it.
        self.handlers: dict[str, Callable[..., Any]] = {
            "get_fleet_overview": self.get_fleet_overview,
            "get_machine_details": self.get_machine_details,
            "get_process_status": self.get_process_status,
            "simulate_process_change": self.simulate_process_change,
            "optimize_process_setpoints": self.optimize_process_setpoints,
            "plan_maintenance": self.plan_maintenance,
            "get_plant_kpis": self.get_plant_kpis,
        }

    def definitions(self, names: list[str] | None = None) -> list[dict]:
        """Tool descriptions to send to the AI: all tools, or only the named ones."""
        names = list(TOOL_DEFINITIONS) if names is None else names
        return [{"name": n, "strict": True, **TOOL_DEFINITIONS[n]} for n in names]

    def run(self, name: str, args: dict) -> str:
        """Run the tool the AI asked for and return its result as compact text (JSON)."""
        if name not in self.handlers:
            raise ValueError(f"Unknown tool: {name}")
        # Drop arguments the tool does not take (small local models sometimes invent some)
        allowed = TOOL_DEFINITIONS[name]["input_schema"]["properties"]
        args = {k: v for k, v in args.items() if k in allowed}
        # Compact JSON keeps the context small, which matters most for local models
        return json.dumps(self.handlers[name](**args), default=str, separators=(",", ":"))

    # --- tool implementations -------------------------------------------
    def get_fleet_overview(self, top_n: int = 10) -> dict:
        """The `top_n` riskiest machines right now, plus fleet-wide counts."""
        top_n = max(1, min(int(top_n), 30))  # keep the request between 1 and 30 machines
        snap = get_fleet_model().snapshot(self.state.timestamp)
        cols = [c for c in snap.columns if c.startswith("p_")]
        machines = []
        for mid, r in snap.head(top_n).iterrows():
            machines.append({
                "machine_id": int(mid),
                "model": r["model"],
                "age_years": int(r["age_years"]),
                "risk_24h": round(float(r["risk_24h"]), 3),
                "risk_7d": round(float(r["risk_7d"]), 3),
                "top_component": r["top_component"],
                "anomaly_score": round(float(r["anomaly_score"]), 2),
                # Only list component risks of at least 1%, to keep the answer short.
                "component_probabilities": {c[2:]: round(float(r[c]), 3) for c in cols if r[c] >= 0.01},
            })
        return {
            "timestamp": str(self.state.timestamp),
            "fleet_size": len(snap),
            "machines_risk_24h_over_50pct": int((snap["risk_24h"] > 0.5).sum()),
            "machines_risk_7d_over_30pct": int((snap["risk_7d"] > 0.3).sum()),
            "machines_anomalous": int((snap["anomaly_score"] > 1).sum()),
            "riskiest_machines": machines,
        }

    def get_machine_details(self, machine_id: int) -> dict:
        """Full details of one machine at the current time."""
        return get_fleet_model().machine_detail(int(machine_id), self.state.timestamp)

    def get_process_status(self) -> dict:
        """Health check of the current production cycle."""
        return get_process_model().diagnose(self.state.process_cycle)

    def simulate_process_change(self, rpm=None, torque_nm=None, replace_tool: bool = False) -> dict:
        """What-if: failure risk of the current cycle after the given changes."""
        return process_optimizer.simulate(
            get_process_model(), self.state.process_cycle, rpm=rpm, torque_nm=torque_nm, replace_tool=replace_tool
        )

    def optimize_process_setpoints(self, min_throughput_ratio: float = 0.9, allow_tool_change: bool = True) -> dict:
        """The best setting changes for the current cycle."""
        return process_optimizer.optimize(
            get_process_model(),
            self.state.process_cycle,
            min_throughput_ratio=float(min_throughput_ratio),
            allow_tool_change=allow_tool_change,
        )

    def plan_maintenance(self, horizon_days: int = config.PLANNING_HORIZON_DAYS,
                         crew_per_day: int = config.MAINTENANCE_CREW_PER_DAY) -> dict:
        """The cheapest maintenance plan for the next `horizon_days` days (1-14)."""
        snap = get_fleet_model().snapshot(self.state.timestamp)
        return scheduler.plan(snap, horizon_days=max(1, min(int(horizon_days), 14)),
                              crew_per_day=max(1, int(crew_per_day)))

    def get_plant_kpis(self) -> dict:
        """Plant OEE and the yearly business case."""
        return kpis.plant_kpis()
