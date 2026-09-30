"""Replay of held-out data as a 'live' factory.

The fleet clock steps through the Azure PdM test period (3-hour resolution);
the process line steps through unseen AI4I production cycles. Agents and the
dashboard only ever see the state at the current clock position.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.models.predictor import get_fleet_model, get_process_model


@dataclass
class FactoryState:
    ts_index: int = 200
    cycle_index: int | None = None
    process_override: dict | None = field(default=None)  # operator-entered setpoints

    @property
    def timestamp(self) -> pd.Timestamp:
        ts = get_fleet_model().timestamps
        return ts[min(max(self.ts_index, 0), len(ts) - 1)]

    @property
    def process_cycle(self) -> dict:
        if self.process_override:
            return dict(self.process_override)
        pm = get_process_model()
        idx = self.cycle_index if self.cycle_index is not None else int(pm.test_cycles.index[0])
        return pm.cycle(idx)

    def advance(self, steps: int = 1) -> None:
        """Move the fleet clock forward by `steps` x 3 hours."""
        self.ts_index = min(self.ts_index + steps, len(get_fleet_model().timestamps) - 1)
