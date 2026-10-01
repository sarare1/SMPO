"""Replay of held-out data as a 'live' factory.

WHAT THIS FILE DOES (plain English)
-----------------------------------
There is no real factory connected, so the dashboard "plays back" historical
data that the models never saw during training, as if it were happening now.

The fleet clock steps through the Azure PdM test period (3-hour resolution);
the process line steps through unseen AI4I production cycles. Agents and the
dashboard only ever see the state at the current clock position.

FactoryState holds "what time is it" and "which production cycle is running".
"""
# Lets Python understand modern type hints on all versions.
from __future__ import annotations

# --- Imports: tools this file needs -----------------------------------------
from dataclasses import dataclass, field  # a compact way to define a small record of values

import pandas as pd  # date/time values

from src.models.predictor import get_fleet_model, get_process_model  # the loaded prediction models


@dataclass
class FactoryState:
    """The current moment in the replayed factory."""

    ts_index: int = 200              # position of the fleet clock (each step = 3 hours)
    cycle_index: int | None = None   # which test production cycle the machining line is running
    process_override: dict | None = field(default=None)  # operator-entered setpoints

    @property
    def timestamp(self) -> pd.Timestamp:
        """The current factory date and time (kept within the available data)."""
        ts = get_fleet_model().timestamps
        return ts[min(max(self.ts_index, 0), len(ts) - 1)]

    @property
    def process_cycle(self) -> dict:
        """Settings of the machining line right now: the operator's custom settings if
        they applied some, otherwise the selected recorded cycle."""
        if self.process_override:
            return dict(self.process_override)
        pm = get_process_model()
        idx = self.cycle_index if self.cycle_index is not None else int(pm.test_cycles.index[0])
        return pm.cycle(idx)

    def advance(self, steps: int = 1) -> None:
        """Move the fleet clock forward by `steps` x 3 hours."""
        # Stop at the last available moment instead of running past the end of the data.
        self.ts_index = min(self.ts_index + steps, len(get_fleet_model().timestamps) - 1)
