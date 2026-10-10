"""
Trace - prints what each agent is doing, live, as it happens.

This is purely for LEARNING: it lets you watch an agent think. When an agent
runs, it produces a stream of messages:
  - 🤔 it decides to CALL A TOOL (you see the tool name + the arguments)
  - 🔧 the TOOL RETURNS a result
  - 💬 the agent SAYS something (its reasoning / final answer)

`run_streaming` runs an agent with `.stream(...)` (not `.invoke(...)`) so we can
print each step the moment it happens - including the steps of sub-agents that
run inside a tool - giving you a readable, correctly-ordered story of the run.
"""
from __future__ import annotations

VERBOSE = True   # set False (or pass --quiet) to hide the reasoning


def banner(title: str) -> None:
    if not VERBOSE:
        return
    print("\n" + "=" * 72)
    print(f"  🤖 {title}")
    print("=" * 72)


def _short(value, limit: int = 300) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + " ..."


def show_message(m) -> None:
    """Print one message from an agent's stream in a human-friendly way."""
    if not VERBOSE:
        return
    cls = m.__class__.__name__
    content = (getattr(m, "content", "") or "").strip()

    if cls == "HumanMessage":
        print(f"\n👤 input: {_short(content, 200)}")

    elif cls == "AIMessage":
        # An AI message may contain tool calls, free-text reasoning, or both.
        for tc in (getattr(m, "tool_calls", None) or []):
            print(f"\n🤔 decides to call tool → {tc.get('name')}({_short(tc.get('args'), 200)})")
        if content:
            print(f"\n💬 {_short(content, 1500)}")

    elif cls == "ToolMessage":
        print(f"\n🔧 tool '{getattr(m, 'name', 'tool')}' returned → {_short(content, 500)}")


def run_streaming(agent, prompt: str, title: str) -> list:
    """Run an agent, printing every step live, and return its full message list.

    We use stream_mode='values': each step yields the FULL conversation so far,
    so we just print whatever is new since the last step.
    """
    banner(title)
    printed = 0
    messages: list = []
    for state in agent.stream({"messages": [("user", prompt)]}, stream_mode="values"):
        messages = state["messages"]
        for m in messages[printed:]:
            show_message(m)
        printed = len(messages)
    return messages
