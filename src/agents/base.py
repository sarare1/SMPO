"""Types shared by all LLM backends.

WHAT THIS FILE DOES (plain English)
-----------------------------------
Defines two small building blocks used by both AI engines (Claude and Ollama):

  AgentError  - a clear, user-friendly error ("Ollama is not running", "no API
                credits") that the dashboard can show instead of crashing.
  AgentResult - the package an AI agent hands back: its written answer, the
                steps it took (tool calls, reasoning), time taken and tokens used.
"""
# Lets Python understand modern type hints on all versions.
from __future__ import annotations

# --- Imports: tools this file needs -----------------------------------------
from dataclasses import dataclass, field  # a compact way to define a record of values


class AgentError(RuntimeError):
    """User-facing failure of an LLM call."""


@dataclass
class AgentResult:
    """Everything one AI agent produced in one run."""

    agent: str                                         # which agent (e.g. "Monitoring Agent")
    text: str                                          # its final written answer
    trace: list[dict] = field(default_factory=list)    # the steps it took: tool calls and reasoning
    messages: list = field(default_factory=list)       # the full conversation with the AI model
    stop_reason: str | None = None                     # why the model stopped (finished, too long, refused...)
    seconds: float = 0.0                               # how long the run took
    # "Tokens" are word pieces; they measure how much text went into and came out of the AI.
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
