import json

import streamlit as st

import common
from src import config
from src.agents.orchestrator import AgentError, ChatSession, unavailable_reason

SUGGESTIONS = {
    ":material/warning: Which machines need attention today?": "Which machines need attention today, and why?",
    ":material/tune: Fix the current production cycle": (
        "Why is the current production cycle at risk, and what setpoint change would you make?"),
    ":material/build: What if we only have 2 technicians?": (
        "Plan this week's maintenance with only 2 jobs per day. What do we lose vs. the standard crew?"),
}


def show_trace(trace: list[dict]) -> None:
    calls = [t for t in trace if t["type"] == "tool_call"]
    if not trace:
        return
    with st.expander(f"Used {len(calls)} tool call(s)", type="compact"):
        for step in trace:
            if step["type"] == "tool_call":
                with st.expander(f"{step['tool']}", type="step", icon=":material/build:"):
                    st.code(json.dumps(step["input"]), language="json")
            elif step["type"] == "thinking":
                with st.expander("Reasoning", type="step", icon=":material/psychology:"):
                    st.markdown(step["text"])


if st.session_state.chat is None:
    st.session_state.chat = ChatSession(common.toolbox())
log = st.session_state.chat_log

if common.llm_ready():
    st.caption(f"Answering with {config.llm_label()}.")
else:
    st.info(unavailable_reason(), icon=":material/key:")

for msg in log:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            show_trace(msg.get("trace", []))
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about machines, maintenance or setpoints", submit_mode="disable")
if not log:
    picked = st.pills("Try asking", list(SUGGESTIONS), label_visibility="collapsed")
    if picked:
        prompt = SUGGESTIONS[picked]

if prompt:
    log.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking the plant ... (a local model on CPU can take a few minutes)"):
                result = st.session_state.chat.ask(prompt)
            show_trace(result.trace)
            st.markdown(result.text)
            log.append({"role": "assistant", "content": result.text, "trace": result.trace})
        except AgentError as e:
            log.pop()  # the turn was rolled back in the session too
            st.error(str(e), icon=":material/error:")

if log and st.button("Clear conversation", icon=":material/delete:"):
    st.session_state.chat = None
    st.session_state.chat_log = []
    st.rerun()
