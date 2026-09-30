"""Multi-agent orchestration (manual tool-use loop).

Four specialist agents, each with a restricted tool set, report to a
coordinator that merges them into a structured action plan. The LLM backend
is Claude (API) or a local Ollama model, chosen by LLM_PROVIDER in .env.
When no backend is usable, a deterministic rule-based planner produces the
same plan shape.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

from src import config
from src.agents import prompts
from src.agents.base import AgentError, AgentResult
from src.agents.tools import ToolBox
from src.models.predictor import get_fleet_model, get_process_model
from src.optimization import scheduler

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _api_message(e: anthropic.APIStatusError) -> str:
    body = e.body if isinstance(e.body, dict) else {}
    return body.get("error", {}).get("message") or e.message


class ClaudeAgentRunner:
    def __init__(self, toolbox: ToolBox) -> None:
        self.toolbox = toolbox
        self.client = anthropic.Anthropic()

    def _create(self, **kwargs):
        output_config = {"effort": config.CLAUDE_EFFORT, **kwargs.pop("output_config", {})}
        try:
            return self.client.beta.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config=output_config,
                # On a safety-classifier decline, the API retries on a fallback model
                betas=[FALLBACK_BETA],
                fallbacks="default",
                **kwargs,
            )
        except anthropic.AuthenticationError as e:
            raise AgentError("Claude API rejected the credentials - check ANTHROPIC_API_KEY in .env.") from e
        except anthropic.RateLimitError as e:
            raise AgentError("Claude API rate limit reached - wait a moment and retry.") from e
        except anthropic.BadRequestError as e:
            raise AgentError(f"Claude API rejected the request: {_api_message(e)}") from e
        except anthropic.APIStatusError as e:
            raise AgentError(f"Claude API error {e.status_code}: {_api_message(e)}") from e
        except anthropic.APIConnectionError as e:
            raise AgentError("Could not reach the Claude API - check your network connection.") from e

    def run(self, agent: str, system: str, messages: list, tool_names: list[str] | None) -> AgentResult:
        """Tool-use loop. `messages` is extended in place (append-only)."""
        start = time.time()
        tools = self.toolbox.definitions(tool_names)
        result = AgentResult(agent=agent, text="", messages=messages)
        for round_no in range(config.AGENT_MAX_TOOL_ROUNDS + 1):
            extra = {}
            if tools:
                extra["tools"] = tools
                if round_no == config.AGENT_MAX_TOOL_ROUNDS:
                    extra["tool_choice"] = {"type": "none"}
            resp = self._create(system=system, messages=messages, **extra)
            messages.append({"role": "assistant", "content": resp.content})
            result.stop_reason = resp.stop_reason
            result.usage["input_tokens"] += resp.usage.input_tokens
            result.usage["output_tokens"] += resp.usage.output_tokens

            for block in resp.content:
                if block.type == "thinking" and block.thinking:
                    result.trace.append({"agent": agent, "type": "thinking", "text": block.thinking})
                elif block.type == "fallback":
                    result.trace.append({"agent": agent, "type": "fallback",
                                         "text": f"{block.from_.model} declined; {block.to.model} continued"})

            if resp.stop_reason == "refusal":
                result.text = "The model declined this request, so this agent has no report."
                break
            if resp.stop_reason == "pause_turn":
                continue
            if resp.stop_reason != "tool_use":
                result.text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
                if resp.stop_reason == "max_tokens":
                    result.text += "\n\n_(Output truncated at the token limit.)_"
                break

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                try:
                    output = self.toolbox.run(block.name, dict(block.input))
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
                except Exception as e:  # report tool failures back to the model
                    output = f"Tool error: {e}"
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                         "content": output, "is_error": True})
                result.trace.append({"agent": agent, "type": "tool_call", "tool": block.name,
                                     "input": dict(block.input), "output": output})
            messages.append({"role": "user", "content": tool_results})
        result.seconds = round(time.time() - start, 1)
        return result

    def structured(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        resp = self._create(
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if resp.stop_reason == "refusal":
            raise AgentError("The coordinator model declined to produce a plan.")
        text = next(b.text for b in resp.content if b.type == "text")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return json.loads(text), usage


def make_runner(toolbox: ToolBox):
    if config.LLM_PROVIDER == "claude":
        return ClaudeAgentRunner(toolbox)
    from src.agents.ollama_runner import OllamaAgentRunner
    return OllamaAgentRunner(toolbox)


def unavailable_reason() -> str:
    if config.LLM_PROVIDER == "claude":
        return "Add `ANTHROPIC_API_KEY` to the `.env` file in the project root and restart the app."
    status = config.ollama_status()
    if status == "not_running":
        return f"Ollama is not running at {config.OLLAMA_HOST} - start the Ollama app."
    return f"Ollama model `{config.OLLAMA_MODEL}` is not installed - run `ollama pull {config.OLLAMA_MODEL}`."


def _warm_models() -> None:
    # Load models once before worker threads start (lru_cache is not thread-safe on first call)
    get_fleet_model()
    get_process_model()


def gather_evidence(toolbox: ToolBox, key: str) -> list[tuple[str, dict, str]]:
    """Run a specialist's tools in code (evidence mode). Returns (tool, args, json_output)."""
    calls: list[tuple[str, dict]] = []
    fleet = toolbox.get_fleet_overview(top_n=8)
    top = [m["machine_id"] for m in fleet["riskiest_machines"][:2]]
    if key == "monitoring":
        calls = [("get_fleet_overview", {"top_n": 8})] + [("get_machine_details", {"machine_id": m}) for m in top]
    elif key == "diagnosis":
        calls = [("get_process_status", {})] + [("get_machine_details", {"machine_id": m}) for m in top]
    elif key == "maintenance":
        calls = [("plan_maintenance", {"horizon_days": config.PLANNING_HORIZON_DAYS,
                                       "crew_per_day": config.MAINTENANCE_CREW_PER_DAY}),
                 ("get_plant_kpis", {})]
    elif key == "process":
        opt = toolbox.optimize_process_setpoints(0.9, True)
        best = opt["recommended"][0]
        calls = [("get_process_status", {}),
                 ("optimize_process_setpoints", {"min_throughput_ratio": 0.9, "allow_tool_change": True}),
                 # Verify the top candidate, so the report can cite a real simulation
                 ("simulate_process_change", {"rpm": best["rpm"], "torque_nm": best["torque_nm"],
                                              "replace_tool": best["replace_tool"]})]
    evidence = []
    for name, args in calls:
        output = toolbox.run(name, args)
        if name == "plan_maintenance":  # keep the prompt small: scheduled jobs only
            plan = json.loads(output)
            plan["jobs"] = [j for j in plan["jobs"] if j["day"] is not None][:12]
            output = json.dumps(plan, separators=(",", ":"))
        evidence.append((name, args, output))
    return evidence


def run_agent_team(toolbox: ToolBox) -> dict:
    """Run all specialists, then the coordinator.

    Falls back to the rule-based planner when the LLM backend is unavailable or
    a call fails, and reports why in `llm_error`.
    """
    if not config.llm_available():
        plan = rule_based_plan(toolbox)
        plan["llm_error"] = unavailable_reason()
        return plan
    try:
        return _run_llm_team(toolbox)
    except AgentError as e:
        plan = rule_based_plan(toolbox)
        plan["llm_error"] = str(e)
        return plan


def _run_compact(toolbox: ToolBox, runner) -> dict:
    """One LLM call: all specialists' evidence in, structured action plan out."""
    start = time.time()
    sections, reports = [], {}
    # Each tool output appears once; skip calls another area already made
    skip = {
        "monitoring": set(),
        "diagnosis": {"get_machine_details"},  # already under monitoring
        "maintenance": {"get_plant_kpis"},      # not needed for today's decisions
        "process": {"get_process_status"},      # already under diagnosis
    }
    for key, spec in prompts.SPECIALISTS.items():
        evidence = [e for e in gather_evidence(toolbox, key) if e[0] not in skip[key]]
        body = "\n".join(f"{name}({json.dumps(args)}): {out}" for name, args, out in evidence)
        sections.append(f"## {spec['title'].replace(' Agent', '')}\n{body}")
        reports[key] = {
            "title": spec["title"],
            "text": "_Compact mode: this area's tools were run by code and passed straight to the decision model._",
            "trace": [{"agent": spec["title"], "type": "tool_call", "tool": n, "input": a, "output": o}
                      for n, a, o in evidence],
            "seconds": 0.0,
        }
    plan, usage = runner.structured(
        prompts.COORDINATOR_COMPACT,
        f"Current time: {toolbox.state.timestamp}.\n\nEvidence:\n\n" + "\n\n".join(sections),
        prompts.ACTION_PLAN_SCHEMA,
    )
    return {
        "mode": "llm",
        "model": config.llm_label(),
        "timestamp": str(toolbox.state.timestamp),
        "plan": plan,
        "reports": reports,
        "usage": usage,
        "seconds": round(time.time() - start, 1),
    }


def _run_narrate(toolbox: ToolBox, runner) -> dict:
    """Rule engine builds the actions; the LLM writes the supervisor briefing."""
    start = time.time()
    base = rule_based_plan(toolbox)
    actions = base["plan"]["actions"]
    lines = [
        f"{a['priority']}. [{a['urgency']}] {a['target']}: {a['action']}. {a['rationale']} {a['expected_impact']}"
        for a in actions
    ]
    brief, usage = runner.structured(
        prompts.NARRATOR,
        f"Current time: {toolbox.state.timestamp}. Maintenance crew capacity: "
        f"{config.MAINTENANCE_CREW_PER_DAY} jobs per day.\n{base['plan']['headline']} "
        f"{base['plan']['situation_summary']}\n\nAction list:\n" + "\n".join(lines),
        prompts.NARRATIVE_SCHEMA,
    )
    plan = {
        "headline": brief["headline"],
        "situation_summary": brief["situation_summary"],
        "actions": actions,
        "risks_and_caveats": brief["risks_and_caveats"]
        + ["Actions come from the models, optimiser and scheduler; the briefing was written by the local LLM."],
    }
    return {
        "mode": "llm",
        "model": config.llm_label(),
        "timestamp": str(toolbox.state.timestamp),
        "plan": plan,
        "reports": {},
        "usage": usage,
        "seconds": round(time.time() - start, 1),
    }


def _run_llm_team(toolbox: ToolBox) -> dict:
    _warm_models()
    runner = make_runner(toolbox)
    if config.AGENT_MODE == "compact":
        return _run_compact(toolbox, runner)
    if config.AGENT_MODE == "narrate":
        return _run_narrate(toolbox, runner)
    context = (
        f"Current time: {toolbox.state.timestamp}. The machining line is running the "
        f"current production cycle. {{task}}"
    )

    def _run(key: str) -> AgentResult:
        spec = prompts.SPECIALISTS[key]
        task = context.format(task=spec["task"])
        if config.AGENT_MODE != "evidence":
            return runner.run(spec["title"], spec["system"], [{"role": "user", "content": task}], spec["tools"])
        evidence = gather_evidence(toolbox, key)
        blocks = "\n\n".join(f"### {name}({json.dumps(args)})\n{out}" for name, args, out in evidence)
        msgs = [{"role": "user", "content": (
            f"{task}\n\nThe plant tools have already been run for you. Evidence:\n\n{blocks}\n\n"
            "Write your report using only this evidence, in under 200 words.")}]
        result = runner.run(spec["title"], spec["system"], msgs, [])
        result.trace[:0] = [{"agent": spec["title"], "type": "tool_call", "tool": name, "input": args,
                             "output": out} for name, args, out in evidence]
        return result

    if config.LLM_PROVIDER == "claude":
        with ThreadPoolExecutor(max_workers=len(prompts.SPECIALISTS)) as pool:
            results = dict(zip(prompts.SPECIALISTS, pool.map(_run, prompts.SPECIALISTS)))
    else:
        # A local model serves one request at a time; parallel calls would only queue
        results = {key: _run(key) for key in prompts.SPECIALISTS}

    reports = "\n\n".join(
        f"## {prompts.SPECIALISTS[k]['title']}\n{r.text}" for k, r in results.items()
    )
    plan, usage = runner.structured(
        prompts.COORDINATOR,
        f"Current time: {toolbox.state.timestamp}.\n\nSpecialist reports:\n\n{reports}",
        prompts.ACTION_PLAN_SCHEMA,
    )
    total = {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}
    for r in results.values():
        total["input_tokens"] += r.usage["input_tokens"]
        total["output_tokens"] += r.usage["output_tokens"]
    return {
        "mode": "llm",
        "model": config.llm_label(),
        "timestamp": str(toolbox.state.timestamp),
        "plan": plan,
        "reports": {k: {"title": prompts.SPECIALISTS[k]["title"], "text": r.text,
                        "trace": r.trace, "seconds": r.seconds} for k, r in results.items()},
        "usage": total,
    }


class ChatSession:
    """Multi-turn assistant with access to every tool. History is append-only."""

    def __init__(self, toolbox: ToolBox) -> None:
        self.toolbox = toolbox
        self.messages: list = []

    def ask(self, question: str) -> AgentResult:
        if not config.llm_available():
            return AgentResult(agent="assistant", text=unavailable_reason())
        _warm_models()
        checkpoint = len(self.messages)
        self.messages.append({
            "role": "user",
            "content": f"[Current time: {self.toolbox.state.timestamp}]\n{question}",
        })
        try:
            return make_runner(self.toolbox).run("Assistant", prompts.CHAT, self.messages, None)
        except AgentError:
            del self.messages[checkpoint:]  # drop the failed turn; keeps history valid
            raise


# --------------------------------------------------------------------------
# Offline fallback: deterministic rules over the same tools
# --------------------------------------------------------------------------
def rule_based_plan(toolbox: ToolBox) -> dict:
    fleet = toolbox.get_fleet_overview(top_n=15)
    process = toolbox.get_process_status()
    schedule = toolbox.plan_maintenance()
    actions = []

    for job in [j for j in schedule["jobs"] if j["day"] is not None]:
        urgent = job["p_fail_24h"] >= 0.5
        actions.append({
            "category": "maintenance",
            "urgency": "immediate" if urgent else ("today" if job["day"] == 0 else "this_week"),
            "target": f"Machine {job['machine_id']} / {job['component']}",
            "action": f"Replace {job['component']} ({job['action']})",
            "rationale": f"Failure probability 24h {job['p_fail_24h']:.0%}, 7d {job['p_fail_7d']:.0%}.",
            "expected_impact": (
                f"Planned replacement costs ~${job['expected_cost']:,.0f}; an unplanned failure would cost "
                f"~${scheduler.UNPLANNED_COST:,.0f}."
            ),
            "source_agents": ["Maintenance Planning Agent"],
        })

    if process["alert"]:
        opt = toolbox.optimize_process_setpoints(0.9, True)
        best = opt["recommended"][0]
        change = [f"speed {best['rpm_change_pct']:+.0f}% -> {best['rpm']:.0f} rpm",
                  f"torque {best['torque_change_pct']:+.0f}% -> {best['torque_nm']:.1f} Nm"]
        if best["replace_tool"]:
            change.append("replace tool")
        modes = ", ".join(process["physics_rule_flags"].values()) or "model-detected risk"
        actions.append({
            "category": "process_adjustment",
            "urgency": "immediate",
            "target": "Machining line (current cycle)",
            "action": "Adjust setpoints: " + "; ".join(change),
            "rationale": f"Failure probability {process['failure_probability']:.0%}. {modes}.",
            "expected_impact": f"Risk {opt['current_failure_probability']:.0%} -> {best['failure_probability']:.1%} (model-verified).",
            "source_agents": ["Process Optimisation Agent"],
        })

    for m in fleet["riskiest_machines"]:
        if m["anomaly_score"] > 1 and m["risk_7d"] < 0.3:
            actions.append({
                "category": "inspection",
                "urgency": "this_week",
                "target": f"Machine {m['machine_id']}",
                "action": "Inspect - abnormal sensor pattern without a predicted failure",
                "rationale": f"Anomaly score {m['anomaly_score']:.1f} (>1 = beyond normal range).",
                "expected_impact": "Catch failure patterns the supervised models were not trained on.",
                "source_agents": ["Monitoring Agent"],
            })

    order = {"immediate": 0, "today": 1, "this_week": 2, "monitor": 3}
    actions.sort(key=lambda a: order[a["urgency"]])
    for i, a in enumerate(actions, 1):
        a["priority"] = i
    return {
        "mode": "rule_based",
        "model": None,
        "timestamp": str(toolbox.state.timestamp),
        "plan": {
            "headline": (
                f"{fleet['machines_risk_24h_over_50pct']} machine(s) at high 24h risk, "
                f"{schedule['jobs_scheduled']} maintenance jobs planned this week."
            ),
            "situation_summary": (
                f"Plan saves an expected ${schedule['expected_savings']:,.0f} vs. run-to-failure. "
                f"Current production cycle failure probability {process['failure_probability']:.0%}."
            ),
            "actions": [{"priority": a.pop("priority"), **a} for a in actions],
            "risks_and_caveats": [
                "Rule-based mode: no LLM reasoning was used for this plan.",
            ],
        },
        "reports": {},
        "usage": None,
    }
