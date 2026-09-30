"""Agent runner for a local model served by Ollama (tool-calling loop).

Same interface as ClaudeAgentRunner: `run()` for a tool-using agent and
`structured()` for a JSON-schema-constrained answer.
"""
from __future__ import annotations

import json
import time

import httpx
import jsonschema
import ollama

from src import config
from src.agents.base import AgentError, AgentResult
from src.agents.tools import ToolBox


class OllamaAgentRunner:
    def __init__(self, toolbox: ToolBox, client: ollama.Client | None = None) -> None:
        self.toolbox = toolbox
        self.client = client or ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT_S)

    def _chat(self, **kwargs):
        model = config.OLLAMA_MODEL
        try:
            return self.client.chat(
                model=model,
                think=config.OLLAMA_THINK,
                keep_alive="30m",  # keep the model loaded between agent calls
                options={"num_ctx": config.OLLAMA_NUM_CTX, "num_predict": config.OLLAMA_MAX_OUTPUT,
                         "temperature": 0.2},
                **kwargs,
            )
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise AgentError(f"Ollama model '{model}' is not installed - run `ollama pull {model}`.") from e
            raise AgentError(f"Ollama error: {e.error}") from e
        except httpx.TimeoutException as e:
            raise AgentError(
                f"Ollama took longer than {config.OLLAMA_TIMEOUT_S:.0f}s to answer - try a smaller model "
                "or raise OLLAMA_TIMEOUT_S in .env."
            ) from e
        except (ConnectionError, httpx.ConnectError) as e:
            raise AgentError(f"Ollama is not running at {config.OLLAMA_HOST} - start the Ollama app.") from e

    def _tools(self, names: list[str] | None) -> list[dict]:
        return [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in self.toolbox.definitions(names)
        ]

    def run(self, agent: str, system: str, messages: list, tool_names: list[str] | None) -> AgentResult:
        """Tool-calling loop. `messages` (without the system prompt) is extended in place."""
        start = time.time()
        tools = self._tools(tool_names)
        result = AgentResult(agent=agent, text="", messages=messages)
        system_msg = {"role": "system", "content": system}
        for round_no in range(config.AGENT_MAX_TOOL_ROUNDS + 1):
            last_round = round_no == config.AGENT_MAX_TOOL_ROUNDS
            resp = self._chat(messages=[system_msg, *messages], tools=None if last_round or not tools else tools)
            msg = resp.message
            messages.append(msg)
            result.stop_reason = resp.done_reason
            result.usage["input_tokens"] += resp.prompt_eval_count or 0
            result.usage["output_tokens"] += resp.eval_count or 0
            if msg.thinking:
                result.trace.append({"agent": agent, "type": "thinking", "text": msg.thinking})

            if not msg.tool_calls:
                result.text = (msg.content or "").strip()
                if resp.done_reason == "length":
                    result.text += "\n\n_(Output truncated at the context limit.)_"
                break

            for call in msg.tool_calls:
                name, args = call.function.name, dict(call.function.arguments or {})
                try:
                    output = self.toolbox.run(name, args)
                except Exception as e:  # small models sometimes send bad arguments
                    output = f"Tool error: {e}"
                messages.append({"role": "tool", "content": output, "tool_name": name})
                result.trace.append({"agent": agent, "type": "tool_call", "tool": name, "input": args,
                                     "output": output})
        result.seconds = round(time.time() - start, 1)
        return result

    def structured(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        usage = {"input_tokens": 0, "output_tokens": 0}
        error = None
        for _ in range(2):  # small models occasionally break the schema; retry once
            resp = self._chat(messages=messages, format=schema)
            usage["input_tokens"] += resp.prompt_eval_count or 0
            usage["output_tokens"] += resp.eval_count or 0
            try:
                data = json.loads(resp.message.content)
                jsonschema.validate(data, schema)
                return data, usage
            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                error = e
        raise AgentError(f"The local model did not produce a valid action plan: {error}")
