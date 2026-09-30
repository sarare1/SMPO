"""Types shared by all LLM backends."""
from __future__ import annotations

from dataclasses import dataclass, field


class AgentError(RuntimeError):
    """User-facing failure of an LLM call."""


@dataclass
class AgentResult:
    agent: str
    text: str
    trace: list[dict] = field(default_factory=list)
    messages: list = field(default_factory=list)
    stop_reason: str | None = None
    seconds: float = 0.0
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
