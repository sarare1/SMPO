"""Agent tools and orchestration. The Claude API is mocked - no network or credits needed.

WHAT THIS FILE CHECKS (plain English)
-------------------------------------
That the AI-agent machinery works correctly without calling a real AI:
  * The seven tools are described correctly and all of them run.
  * The rule-based fallback plan has the required layout.
  * The Claude and Ollama loops run tools, report tool errors, stop on refusals,
    and retry a badly formatted answer - tested with fake AI replies ("mocks").
  * When the AI fails, the user still gets a rule-based plan and the chat stays usable.
  * The "evidence" and "narrate" modes run the tools in code and keep the
    model-grounded actions unchanged.
"""
# --- Imports: tools this file needs -----------------------------------------
import json                         # reads tool results
from types import SimpleNamespace   # quick fake objects that imitate AI replies

import jsonschema  # checks answers against the required layout
import pytest      # the test framework

from src.agents import orchestrator, prompts    # the code being tested
from src.agents.tools import TOOL_DEFINITIONS  # the tool menu


def test_tool_schemas_are_strict_compatible():
    """Every tool lists all its inputs as required and forbids extra ones."""
    for name, spec in TOOL_DEFINITIONS.items():
        schema = spec["input_schema"]
        assert schema["additionalProperties"] is False, name
        assert set(schema["required"]) == set(schema["properties"]), name
        jsonschema.Draft202012Validator.check_schema(schema)


def test_specialists_only_reference_known_tools():
    """Each specialist agent is only given tools that actually exist."""
    for spec in prompts.SPECIALISTS.values():
        assert set(spec["tools"]) <= set(TOOL_DEFINITIONS)


def test_every_tool_runs(toolbox):
    """All seven tools run on real data and return a non-empty result."""
    args = {
        "get_fleet_overview": {"top_n": 5},
        "get_machine_details": {"machine_id": 56},
        "get_process_status": {},
        "simulate_process_change": {"rpm": 1500, "torque_nm": None, "replace_tool": False},
        "optimize_process_setpoints": {"min_throughput_ratio": 0.9, "allow_tool_change": True},
        "plan_maintenance": {"horizon_days": 7, "crew_per_day": 4},
        "get_plant_kpis": {},
    }
    for name in TOOL_DEFINITIONS:
        out = json.loads(toolbox.run(name, args[name]))
        assert out, name


def test_rule_based_plan_matches_schema(toolbox):
    """The no-AI fallback plan has the required layout and puts urgent work first."""
    plan = orchestrator.rule_based_plan(toolbox)
    jsonschema.validate(plan["plan"], prompts.ACTION_PLAN_SCHEMA)
    assert plan["plan"]["actions"][0]["urgency"] == "immediate"


# --- mocked Claude tool loop -------------------------------------------------
# Fake Claude replies, so the loop can be tested without the internet or credits.
def _resp(content, stop_reason):
    """A fake Claude reply with the given content and stop reason."""
    return SimpleNamespace(content=content, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def _block(**kw):
    """One fake piece of a Claude reply (text, tool request, reasoning...)."""
    return SimpleNamespace(**kw)


class FakeMessages:
    """Stands in for the Claude client: returns prepared replies in order and records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _runner(toolbox, responses, monkeypatch):
    """A Claude agent runner wired to the fake client."""
    fake = FakeMessages(responses)
    monkeypatch.setattr(orchestrator.anthropic, "Anthropic",
                        lambda: SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    return orchestrator.ClaudeAgentRunner(toolbox), fake


def test_tool_loop_executes_tools_and_returns_text(toolbox, monkeypatch):
    """Claude asks for a tool, gets the real result back, then answers; settings are sent correctly."""
    runner, fake = _runner(toolbox, [
        _resp([_block(type="thinking", thinking="Check the line first."),
               _block(type="tool_use", id="t1", name="get_process_status", input={})], "tool_use"),
        _resp([_block(type="text", text="Raise speed to 1431 rpm.")], "end_turn"),
    ], monkeypatch)
    messages = [{"role": "user", "content": "go"}]
    result = runner.run("Process", "sys", messages, ["get_process_status"])

    assert result.text == "Raise speed to 1431 rpm."
    assert [t["type"] for t in result.trace] == ["thinking", "tool_call"]
    # Tool result fed back in one user message, history is append-only
    tool_msg = messages[2]
    assert tool_msg["role"] == "user" and tool_msg["content"][0]["tool_use_id"] == "t1"
    assert json.loads(tool_msg["content"][0]["content"])["failure_probability"] > 0.5
    first = fake.calls[0]
    assert first["model"] and first["fallbacks"] == "default"
    assert first["thinking"]["type"] == "adaptive"
    assert all(t["strict"] for t in first["tools"])


def test_tool_errors_are_reported_to_model(toolbox, monkeypatch):
    """A failing tool (unknown machine) is reported back to the AI as an error, not a crash."""
    runner, _ = _runner(toolbox, [
        _resp([_block(type="tool_use", id="t1", name="get_machine_details", input={"machine_id": 999})], "tool_use"),
        _resp([_block(type="text", text="Machine 999 does not exist.")], "end_turn"),
    ], monkeypatch)
    messages = [{"role": "user", "content": "go"}]
    runner.run("Mon", "sys", messages, ["get_machine_details"])
    assert messages[2]["content"][0]["is_error"] is True


def test_refusal_stops_loop(toolbox, monkeypatch):
    """If the AI declines, the agent stops and says so."""
    runner, _ = _runner(toolbox, [_resp([], "refusal")], monkeypatch)
    result = runner.run("X", "sys", [{"role": "user", "content": "go"}], None)
    assert result.stop_reason == "refusal" and "declined" in result.text


def test_team_falls_back_to_rules_on_api_error(toolbox, monkeypatch):
    """If the AI call fails (e.g. no credits), a rule-based plan is returned with the reason."""
    monkeypatch.setattr(orchestrator.config, "llm_available", lambda: True)

    def boom(_):
        raise orchestrator.AgentError("credit balance is too low")

    monkeypatch.setattr(orchestrator, "_run_llm_team", boom)
    out = orchestrator.run_agent_team(toolbox)
    assert out["mode"] == "rule_based" and "credit" in out["llm_error"]


def test_chat_rolls_back_failed_turn(toolbox, monkeypatch):
    """A failed chat question is removed from the history, so the next question still works."""
    monkeypatch.setattr(orchestrator.config, "llm_available", lambda: True)

    def fail(*a, **k):
        raise orchestrator.AgentError("down")

    monkeypatch.setattr(orchestrator, "make_runner", lambda tb: SimpleNamespace(run=fail))
    chat = orchestrator.ChatSession(toolbox)
    with pytest.raises(orchestrator.AgentError):
        chat.ask("hello")
    assert chat.messages == []


# --- mocked Ollama tool loop -----------------------------------------------
# Fake local-model replies, so the loop can be tested without Ollama running.
class FakeOllama:
    """Stands in for the Ollama client: returns prepared replies in order and records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _oresp(content="", tool_calls=None, done_reason="stop"):
    """A fake Ollama reply."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, thinking=None)
    return SimpleNamespace(message=msg, done_reason=done_reason, prompt_eval_count=100, eval_count=20)


def _call(name, args):
    """A fake tool request from the local model."""
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))


def test_ollama_loop_runs_tools(toolbox):
    """The local model's tool request is run (ignoring an invented input) and the answer returned."""
    from src.agents.ollama_runner import OllamaAgentRunner
    fake = FakeOllama([
        _oresp(tool_calls=[_call("get_fleet_overview", {"top_n": 3, "bogus": 1})]),
        _oresp(content="Machine 56 needs comp2 and comp3 replaced today."),
    ])
    runner = OllamaAgentRunner(toolbox, client=fake)
    messages = [{"role": "user", "content": "go"}]
    result = runner.run("Monitoring", "sys", messages, ["get_fleet_overview"])

    assert result.text.startswith("Machine 56")
    assert result.usage == {"input_tokens": 200, "output_tokens": 40}
    tool_msg = messages[2]
    assert tool_msg["role"] == "tool" and tool_msg["tool_name"] == "get_fleet_overview"
    assert len(json.loads(tool_msg["content"])["riskiest_machines"]) == 3  # unknown arg dropped
    first = fake.calls[0]
    assert first["messages"][0] == {"role": "system", "content": "sys"}
    assert first["tools"][0]["function"]["name"] == "get_fleet_overview"


def test_ollama_structured_retries_invalid_json(toolbox):
    """A badly formatted answer is retried once and the second, valid answer is used."""
    from src.agents.ollama_runner import OllamaAgentRunner
    good = {"headline": "h", "situation_summary": "s", "actions": [], "risks_and_caveats": []}
    fake = FakeOllama([_oresp(content="{not json"), _oresp(content=json.dumps(good))])
    data, usage = OllamaAgentRunner(toolbox, client=fake).structured("sys", "u", prompts.ACTION_PLAN_SCHEMA)
    assert data == good and len(fake.calls) == 2
    assert fake.calls[0]["format"] == prompts.ACTION_PLAN_SCHEMA


def test_ollama_structured_gives_up_after_retry(toolbox):
    """Two invalid answers in a row produce a clear error instead of a broken plan."""
    from src.agents.ollama_runner import OllamaAgentRunner
    fake = FakeOllama([_oresp(content="{}"), _oresp(content="{}")])
    with pytest.raises(orchestrator.AgentError):
        OllamaAgentRunner(toolbox, client=fake).structured("sys", "u", prompts.ACTION_PLAN_SCHEMA)


def test_definitions_empty_list_means_no_tools(toolbox):
    """Asking for no tools gives no tools; asking for none specified gives all of them."""
    assert toolbox.definitions([]) == []
    assert len(toolbox.definitions()) == len(TOOL_DEFINITIONS)


def test_evidence_mode_prefetches_tools(toolbox, monkeypatch):
    """In "evidence" mode the code runs the tools (including the what-if check) before the AI writes."""
    calls = []

    class FakeRunner:
        """Records what each agent was given and returns canned answers."""

        def run(self, agent, system, messages, tool_names):
            calls.append((agent, messages[0]["content"], tool_names))
            return orchestrator.AgentResult(agent=agent, text=f"{agent} report")

        def structured(self, system, user, schema):
            plan = {"headline": "h", "situation_summary": "s", "actions": [], "risks_and_caveats": []}
            return plan, {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(orchestrator.config, "AGENT_MODE", "evidence")
    monkeypatch.setattr(orchestrator.config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(orchestrator, "make_runner", lambda tb: FakeRunner())
    out = orchestrator._run_llm_team(toolbox)

    assert len(calls) == 4 and all(tools == [] for _, _, tools in calls)
    process_prompt = next(c for a, c, _ in calls if a == "Process Optimisation Agent")
    assert "simulate_process_change" in process_prompt  # verification done by code
    trace = out["reports"]["process"]["trace"]
    assert [t["tool"] for t in trace] == ["get_process_status", "optimize_process_setpoints", "simulate_process_change"]


def test_narrate_mode_keeps_rule_actions(toolbox, monkeypatch):
    """In "narrate" mode the AI writes only the briefing; the actions stay exactly as the models built them."""
    seen = {}

    class FakeRunner:
        """Records the briefing request and returns a canned briefing."""

        def structured(self, system, user, schema):
            seen["user"] = user
            jsonschema.Draft202012Validator.check_schema(schema)
            return {"headline": "Replace machine 56 parts now", "situation_summary": "s",
                    "risks_and_caveats": ["c"]}, {"input_tokens": 5, "output_tokens": 3}

    monkeypatch.setattr(orchestrator.config, "AGENT_MODE", "narrate")
    monkeypatch.setattr(orchestrator, "make_runner", lambda tb: FakeRunner())
    out = orchestrator._run_llm_team(toolbox)
    rules = orchestrator.rule_based_plan(toolbox)["plan"]["actions"]

    assert out["mode"] == "llm" and out["plan"]["headline"] == "Replace machine 56 parts now"
    assert out["plan"]["actions"] == rules  # actions stay model-grounded
    assert "crew capacity" in seen["user"].lower()
    jsonschema.validate(out["plan"], prompts.ACTION_PLAN_SCHEMA)
