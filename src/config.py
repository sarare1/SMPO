"""Central settings for the whole project.

WHAT THIS FILE DOES (plain English)
-----------------------------------
This is the project's "control panel". Every other file reads its settings from
here, so you can change how the system behaves in one place:

  * WHERE files live      - folders for raw data, processed data, trained models, reports.
  * WHERE data comes from  - download links for the two public datasets.
  * HOW models are built   - e.g. which date separates training data from test data.
  * WHICH AI model to use  - a free local model (Ollama) or Claude (paid, online).
  * BUSINESS ASSUMPTIONS   - costs of failures, repair times, crew size, workload.

Values that start with os.getenv(...) can be overridden in the ".env" file in
the project folder, without touching any code.
"""
# Lets Python understand modern type hints such as "str | None" on all versions.
from __future__ import annotations

# --- Imports: tools this file needs -----------------------------------------
import os                    # reads settings from the computer's environment / .env file
from pathlib import Path     # builds folder and file paths that work on Windows, Mac and Linux

from dotenv import load_dotenv  # loads the key=value lines from the ".env" file

# --- Folders and files ----------------------------------------------------------
# ROOT = the project folder (two levels up from this file: src/config.py -> SMPO/).
ROOT = Path(__file__).resolve().parents[1]
# Read the private ".env" file (API keys, model choice) so its values are available below.
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"              # original downloaded CSV files
DATA_PROCESSED = ROOT / "data" / "processed"  # cleaned data with calculated features
MODELS_DIR = ROOT / "models"                  # trained prediction models
REPORTS_DIR = ROOT / "reports"                # model accuracy results (metrics.json)
DECISIONS_LOG = ROOT / "data" / "decisions.jsonl"  # record of accepted/rejected recommendations

# Create the folders above if they do not exist yet, so later steps never fail on a missing folder.
for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset sources ----------------------------------------------------------
# AI4I 2020: 10,000 production cycles of a machining line, from the UCI repository.
AI4I_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
# Azure Predictive Maintenance: sensor history of 100 machines.
# Microsoft's own mirror of the Azure AI Gallery predictive-maintenance sample.
AZURE_PDM_BASE = (
    "https://raw.githubusercontent.com/microsoft/sqlworkshops/master/"
    "SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data"
)
# The five Azure files: sensor readings, error codes, maintenance records, failures, machine info.
AZURE_PDM_FILES = ["PdM_telemetry", "PdM_errors", "PdM_maint", "PdM_failures", "PdM_machines"]

# --- Modelling settings ---------------------------------------------------------
RANDOM_STATE = 42                 # fixed "seed" so results are the same every time you rerun
AZURE_SPLIT_DATE = "2015-09-01"  # time-based split: train before, test after
AZURE_SAMPLE_EVERY_H = 3          # down-sample hourly telemetry for training
HORIZONS_H = {"24h": 24, "7d": 168}  # labels: component fails within next N hours

# --- AI (LLM) agent settings ---------------------------------------------------------
# "LLM" = large language model, the AI that writes explanations and plans.
# "ollama" = local model (free, offline); "claude" = Anthropic API (needs credits)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# Claude settings (only used when LLM_PROVIDER=claude).
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "medium")              # how hard the model thinks
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "16000"))  # max length of an answer

# Ollama (local model) settings (only used when LLM_PROVIDER=ollama).
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # address of the Ollama program
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")              # which local model to use
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))  # context window; smaller is faster on CPU
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() == "true"  # reasoning mode: better, much slower on CPU
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "600"))       # give up on a call after this many seconds
OLLAMA_MAX_OUTPUT = int(os.getenv("OLLAMA_MAX_OUTPUT", "1024"))  # cap on generated tokens per call

# How much of the work the AI does (see README section 8.3):
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

# Safety limit: how many times an agent may call tools before it must give its answer.
AGENT_MAX_TOOL_ROUNDS = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "8" if LLM_PROVIDER == "claude" else "5"))


def llm_model() -> str:
    """Name of the AI model currently in use."""
    return CLAUDE_MODEL if LLM_PROVIDER == "claude" else OLLAMA_MODEL


def llm_label() -> str:
    """Short, human-readable description of the AI engine, shown in the dashboard."""
    return f"Claude `{CLAUDE_MODEL}`" if LLM_PROVIDER == "claude" else f"Ollama `{OLLAMA_MODEL}` (local)"


def llm_available() -> bool:
    """True when the configured LLM backend can be used."""
    # For Claude we only need an API key to be present.
    if LLM_PROVIDER == "claude":
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    # For Ollama the program must be running and the model must be downloaded.
    return ollama_status() == "ready"


def ollama_status() -> str:
    """'ready', 'not_running', or 'model_missing' for the configured Ollama model."""
    import json            # reads the JSON reply from Ollama
    import urllib.request  # makes a simple web request to the local Ollama program

    # Ask Ollama for its list of downloaded models (waits at most 2 seconds).
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as resp:
            names = {m["name"] for m in json.load(resp).get("models", [])}
    except OSError:
        # No answer means the Ollama program is not running.
        return "not_running"
    # Model names without a version tag are stored as "<name>:latest".
    wanted = OLLAMA_MODEL if ":" in OLLAMA_MODEL else f"{OLLAMA_MODEL}:latest"
    return "ready" if wanted in names else "model_missing"


# --- Business assumptions (edit to match your plant) -----------------------
# These numbers turn failure probabilities into money and downtime. They are
# estimates; replace them with your plant's real figures.
COST = {
    "unplanned_failure": 12_000.0,   # $ per unplanned component failure (repair + scrap + expedite)
    "planned_maintenance": 1_500.0,  # $ per planned component replacement
    "downtime_per_hour": 800.0,      # $ lost margin per hour of machine downtime
}
# How long a machine stands still: a surprise breakdown vs. a planned replacement.
DOWNTIME_HOURS = {"unplanned": 24.0, "planned": 4.0}
MAINTENANCE_CREW_PER_DAY = 4         # jobs the crew can execute per day
PLANNING_HORIZON_DAYS = 7            # the maintenance plan looks this many days ahead
# Relative production load for each day of the horizon (1.0 = normal).
# Maintenance on high-load days costs more lost output.
DAILY_LOAD = [1.0, 1.2, 1.2, 1.0, 0.8, 0.5, 0.4]
NOMINAL_ROTATION = 450.0             # Azure PdM rpm treated as 100% performance
# Share of model-detected failures the plant actually converts into planned
# work (crew availability, spare parts, lead time). Keeps projections honest.
PDM_REALIZATION = 0.7
