"""Central configuration: paths, model settings, and business assumptions."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DECISIONS_LOG = ROOT / "data" / "decisions.jsonl"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset sources -------------------------------------------------------
AI4I_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
# Microsoft's own mirror of the Azure AI Gallery predictive-maintenance sample.
AZURE_PDM_BASE = (
    "https://raw.githubusercontent.com/microsoft/sqlworkshops/master/"
    "SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data"
)
AZURE_PDM_FILES = ["PdM_telemetry", "PdM_errors", "PdM_maint", "PdM_failures", "PdM_machines"]

# --- Modelling ---------------------------------------------------------------
RANDOM_STATE = 42
AZURE_SPLIT_DATE = "2015-09-01"  # time-based split: train before, test after
AZURE_SAMPLE_EVERY_H = 3          # down-sample hourly telemetry for training
HORIZONS_H = {"24h": 24, "7d": 168}  # labels: component fails within next N hours

# --- LLM agents --------------------------------------------------------------
# "ollama" = local model (free, offline); "claude" = Anthropic API (needs credits)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "medium")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "16000"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))  # context window; smaller is faster on CPU
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() == "true"  # reasoning mode: better, much slower on CPU
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "600"))
OLLAMA_MAX_OUTPUT = int(os.getenv("OLLAMA_MAX_OUTPUT", "1024"))  # cap on generated tokens per call

# "tools":    each specialist agent decides which tools to call (multi-round), then
#             a coordinator merges the reports. Best quality; suited to Claude.
# "evidence": the code runs each specialist's tools; the model writes each report in
#             one call, then the coordinator merges them (5 LLM calls).
# "compact":  the code runs all specialists' tools; the model writes the action plan
#             in a single call (needs a GPU or a fast CPU for local models).
# "narrate":  the rule engine builds the grounded actions; the model writes the
#             supervisor briefing (headline, summary, caveats). The default for local
#             models - a CPU-only laptop generates only ~2-7 tokens/s.
AGENT_MODE = os.getenv("AGENT_MODE", "tools" if LLM_PROVIDER == "claude" else "narrate").strip().lower()

AGENT_MAX_TOOL_ROUNDS = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "8" if LLM_PROVIDER == "claude" else "5"))


def llm_model() -> str:
    return CLAUDE_MODEL if LLM_PROVIDER == "claude" else OLLAMA_MODEL


def llm_label() -> str:
    return f"Claude `{CLAUDE_MODEL}`" if LLM_PROVIDER == "claude" else f"Ollama `{OLLAMA_MODEL}` (local)"


def llm_available() -> bool:
    """True when the configured LLM backend can be used."""
    if LLM_PROVIDER == "claude":
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    return ollama_status() == "ready"


def ollama_status() -> str:
    """'ready', 'not_running', or 'model_missing' for the configured Ollama model."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as resp:
            names = {m["name"] for m in json.load(resp).get("models", [])}
    except OSError:
        return "not_running"
    wanted = OLLAMA_MODEL if ":" in OLLAMA_MODEL else f"{OLLAMA_MODEL}:latest"
    return "ready" if wanted in names else "model_missing"


# --- Business assumptions (edit to match your plant) -----------------------
COST = {
    "unplanned_failure": 12_000.0,   # $ per unplanned component failure (repair + scrap + expedite)
    "planned_maintenance": 1_500.0,  # $ per planned component replacement
    "downtime_per_hour": 800.0,      # $ lost margin per hour of machine downtime
}
DOWNTIME_HOURS = {"unplanned": 24.0, "planned": 4.0}
MAINTENANCE_CREW_PER_DAY = 4         # jobs the crew can execute per day
PLANNING_HORIZON_DAYS = 7
# Relative production load for each day of the horizon (1.0 = normal).
# Maintenance on high-load days costs more lost output.
DAILY_LOAD = [1.0, 1.2, 1.2, 1.0, 0.8, 0.5, 0.4]
NOMINAL_ROTATION = 450.0             # Azure PdM rpm treated as 100% performance
# Share of model-detected failures the plant actually converts into planned
# work (crew availability, spare parts, lead time). Keeps projections honest.
PDM_REALIZATION = 0.7
