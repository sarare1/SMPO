import json

import streamlit as st

import common
from src import config
from src.agents.orchestrator import run_agent_team, unavailable_reason

URGENCY_BADGE = {
    "immediate": ("red", ":material/error:"),
    "today": ("orange", ":material/schedule:"),
    "this_week": ("blue", ":material/event:"),
    "monitor": ("gray", ":material/visibility:"),
}


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
if common.llm_ready():
    MODE_TEXT = {
        "tools": "Four specialist agents choose and call their own tools; a coordinator merges their reports.",
        "evidence": "The code runs each specialist's tools; the model writes four reports and a coordinator merges them.",
        "compact": "The code runs all specialists' tools; the model writes the action plan in one call.",
        "narrate": "The models, optimiser and scheduler build the actions; the model writes the supervisor briefing.",
    }
    local = config.LLM_PROVIDER != "claude"
    duration = {"narrate": "about 2 minutes", "compact": "10+ minutes", "evidence": "20+ minutes",
                "tools": "20+ minutes"}.get(config.AGENT_MODE) if local else "roughly 1-3 minutes"
    st.caption(f"{MODE_TEXT.get(config.AGENT_MODE, '')} Engine: {config.llm_label()}, mode `{config.AGENT_MODE}`. "
               f"Takes {duration}{' on this CPU' if local else ''}.")
else:
    st.info(f"{unavailable_reason()} Until then the plan comes from the rule-based engine.",
            icon=":material/info:")

if st.button("Run agent analysis", type="primary", icon=":material/play_arrow:"):
    with st.status(f":shimmer[Agents analysing the factory at {s.timestamp:%Y-%m-%d %H:%M}]") as status:
        st.write("Monitoring, diagnosis, maintenance planning and process optimisation agents working ...")
        st.session_state.agent_run = run_agent_team(common.toolbox())
        status.update(label="Analysis complete", state="complete")

run = st.session_state.agent_run
if run is None:
    st.stop()

if run.get("llm_error"):
    st.warning(f"LLM agents unavailable - showing the rule-based plan instead. {run['llm_error']}",
               icon=":material/warning:")
if run["timestamp"] != str(s.timestamp):
    st.info(f"This plan was generated for {run['timestamp']}. The clock has moved - re-run for the current time.",
            icon=":material/history:")

plan = run["plan"]
with st.container(border=True):
    st.subheader(plan["headline"], anchor=False)
    st.write(plan["situation_summary"])
    meta = "Rule-based engine" if run["mode"] == "rule_based" else run["model"]
    if run.get("usage"):
        meta += f" · {run['usage']['input_tokens']:,} input / {run['usage']['output_tokens']:,} output tokens"
    st.caption(meta)

st.subheader("Actions", anchor=False)
for action in plan["actions"]:
    key = f"{run['timestamp']}|{action['priority']}|{action['target']}"
    color, icon = URGENCY_BADGE[action["urgency"]]
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(action["urgency"].replace("_", " "), color=color, icon=icon)
            st.badge(action["category"].replace("_", " "), color="gray")
            st.markdown(f"**#{action['priority']} · {action['target']}**")
        st.markdown(action["action"])
        st.caption(f"Why: {action['rationale']}  \nImpact: {action['expected_impact']}  \n"
                   f"From: {', '.join(action['source_agents'])}")
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

if plan["risks_and_caveats"]:
    with st.container(border=True):
        st.markdown("**Risks and caveats**")
        for item in plan["risks_and_caveats"]:
            st.markdown(f"- {item}")

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
