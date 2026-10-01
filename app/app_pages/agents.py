"""Agent action plan page.

WHAT THIS PAGE SHOWS (plain English)
------------------------------------
The AI decision agents' prioritised action plan for the current factory time.
  * "Run agent analysis" starts the agents (or the rule-based fallback if no AI
    engine is available - the page says why).
  * The plan: a headline, a short situation summary, and a list of actions, each
    with its urgency, what to do, why, and the expected impact.
  * Accept / Reject buttons for every action; each choice is saved to
    data/decisions.jsonl so there is a record of what was decided.
  * Risks and caveats, and (in the heavier modes) each specialist agent's report
    with its reasoning trace: which tools it used and what it concluded.
"""
# --- Imports: tools this page needs -----------------------------------------
import json  # displays tool inputs and outputs in the reasoning trace

import streamlit as st  # the dashboard framework

import common                                                         # shared dashboard helpers
from src import config                                                # AI engine settings
from src.agents.orchestrator import run_agent_team, unavailable_reason  # runs the agents

# Colour and icon for each urgency level shown on the action cards.
URGENCY_BADGE = {
    "immediate": ("red", ":material/error:"),
    "today": ("orange", ":material/schedule:"),
    "this_week": ("blue", ":material/event:"),
    "monitor": ("gray", ":material/visibility:"),
}


# Show an agent's steps as a timeline: tools it called (inputs and results),
# its reasoning notes, and any switch to a fallback model.
def render_trace(trace: list[dict]) -> None:
    for step in trace:
        if step["type"] == "tool_call":
            with st.expander(f"Tool · {step['tool']}", type="step", icon=":material/build:"):
                st.code(json.dumps(step["input"]), language="json")
                try:
                    st.json(json.loads(step["output"]), expanded=False)
                except (ValueError, TypeError):
                    st.write(step["output"])
        elif step["type"] == "thinking":
            with st.expander("Reasoning", type="step", icon=":material/psychology:"):
                st.markdown(step["text"])
        else:
            with st.expander("Model fallback", type="step", icon=":material/swap_horiz:"):
                st.write(step["text"])


s = common.state()
# --- Explain what will happen (or why the AI is unavailable) ---------------------
if common.llm_ready():
    MODE_TEXT = {
        "tools": "Four specialist agents choose and call their own tools; a coordinator merges their reports.",
        "evidence": "The code runs each specialist's tools; the model writes four reports and a coordinator merges them.",
        "compact": "The code runs all specialists' tools; the model writes the action plan in one call.",
        "narrate": "The models, optimiser and scheduler build the actions; the model writes the supervisor briefing.",
    }
    # Rough run time: a local model on a CPU is much slower than Claude.
    local = config.LLM_PROVIDER != "claude"
    duration = {"narrate": "about 2 minutes", "compact": "10+ minutes", "evidence": "20+ minutes",
                "tools": "20+ minutes"}.get(config.AGENT_MODE) if local else "roughly 1-3 minutes"
    st.caption(f"{MODE_TEXT.get(config.AGENT_MODE, '')} Engine: {config.llm_label()}, mode `{config.AGENT_MODE}`. "
               f"Takes {duration}{' on this CPU' if local else ''}.")
else:
    st.info(f"{unavailable_reason()} Until then the plan comes from the rule-based engine.",
            icon=":material/info:")

# --- Run button: start the agents and keep the result for this session -------------
if st.button("Run agent analysis", type="primary", icon=":material/play_arrow:"):
    with st.status(f":shimmer[Agents analysing the factory at {s.timestamp:%Y-%m-%d %H:%M}]") as status:
        st.write("Monitoring, diagnosis, maintenance planning and process optimisation agents working ...")
        st.session_state.agent_run = run_agent_team(common.toolbox())
        status.update(label="Analysis complete", state="complete")

# Nothing to show until a plan has been generated.
run = st.session_state.agent_run
if run is None:
    st.stop()

# Notices: the AI fell back to rules, or the clock has moved since the plan was made.
if run.get("llm_error"):
    st.warning(f"LLM agents unavailable - showing the rule-based plan instead. {run['llm_error']}",
               icon=":material/warning:")
if run["timestamp"] != str(s.timestamp):
    st.info(f"This plan was generated for {run['timestamp']}. The clock has moved - re-run for the current time.",
            icon=":material/history:")

# --- Plan summary --------------------------------------------------------------
plan = run["plan"]
with st.container(border=True):
    st.subheader(plan["headline"], anchor=False)
    st.write(plan["situation_summary"])
    meta = "Rule-based engine" if run["mode"] == "rule_based" else run["model"]
    if run.get("usage"):
        meta += f" · {run['usage']['input_tokens']:,} input / {run['usage']['output_tokens']:,} output tokens"
    st.caption(meta)

# --- Action cards with Accept / Reject ------------------------------------------
st.subheader("Actions", anchor=False)
for action in plan["actions"]:
    key = f"{run['timestamp']}|{action['priority']}|{action['target']}"  # unique id for this action
    color, icon = URGENCY_BADGE[action["urgency"]]
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(action["urgency"].replace("_", " "), color=color, icon=icon)
            st.badge(action["category"].replace("_", " "), color="gray")
            st.markdown(f"**#{action['priority']} · {action['target']}**")
        st.markdown(action["action"])
        st.caption(f"Why: {action['rationale']}  \nImpact: {action['expected_impact']}  \n"
                   f"From: {', '.join(action['source_agents'])}")
        # Show the decision if one was made; otherwise offer the two buttons.
        decision = st.session_state.decisions.get(key)
        if decision:
            st.markdown(f":green[:material/check: Accepted]" if decision == "accepted"
                        else f":gray[:material/close: Rejected]")
        else:
            with st.container(horizontal=True):
                if st.button("Accept", key=f"acc-{key}", icon=":material/check:"):
                    st.session_state.decisions[key] = "accepted"
                    common.log_decision(action, "accepted", run["timestamp"])
                    st.rerun()
                if st.button("Reject", key=f"rej-{key}", icon=":material/close:"):
                    st.session_state.decisions[key] = "rejected"
                    common.log_decision(action, "rejected", run["timestamp"])
                    st.rerun()

# --- Risks and caveats --------------------------------------------------------
if plan["risks_and_caveats"]:
    with st.container(border=True):
        st.markdown("**Risks and caveats**")
        for item in plan["risks_and_caveats"]:
            st.markdown(f"- {item}")

# --- Specialist reports (only in modes that produce them), one tab per agent --------
if run["reports"]:
    st.subheader("Specialist reports", anchor=False)
    tabs = st.tabs([r["title"] for r in run["reports"].values()])
    for tab, report in zip(tabs, run["reports"].values()):
        with tab:
            st.caption(f"{report['seconds']} s · {sum(t['type'] == 'tool_call' for t in report['trace'])} tool calls")
            st.markdown(report["text"])
            with st.expander("Reasoning trace", type="compact"):
                render_trace(report["trace"])

st.caption(f"Accept/reject decisions are logged to `{config.DECISIONS_LOG.relative_to(config.ROOT)}`.")
