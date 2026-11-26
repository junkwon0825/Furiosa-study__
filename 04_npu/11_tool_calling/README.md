# 11. Tool Calling & Agent

This folder preserves the progression from a single LangChain tool definition to a multi-step CLI Agent backed by an RNGD-served LLM.

| Files | Focus |
|---|---|
| `f31_tool_test.py` | Register a tool and inspect the model response |
| `f32_tool_calls.py` | Inspect tool names, arguments, and call IDs |
| `f33_tool_invoke.py` | Invoke a selected Python tool |
| `f34_multi_tools.py` | Register and dispatch multiple arithmetic tools |
| `f35_agent01.py`, `f35_agent02.py` | Introduce `create_agent` and inspect agent messages |
| `f36_agent_tool.py` | Connect four arithmetic tools to an Agent |
| `f37_config.py` | Add multi-step control with `recursion_limit` |
| `f38_agent_error.py` | Handle tool/agent failures |
| `f39_agent_project.py` | Interactive CLI Agent with tool-call tracing |
| `kingkeras.py` | Local RNGD LLM helper and tool-choice serving configuration |

Implemented tools: `add`, `subtract`, `multiply`, and `divide`.

```text
User → RNGD-served LLM → Agent → Tool selection → Python execution
     ← Final response  ← Tool result ←─────────────────────────────
```

The scripts reflect classroom experiments and do not embed API keys or model weights.
