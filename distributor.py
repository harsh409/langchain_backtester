"""
THE DISTRIBUTOR - the top agent that routes work to the other agents.

This is the multi-agent part. The Distributor is itself an agent, but its TOOLS
are the other two agents:

    Distributor (agent)
      ├─ tool: backtest_agent_tool  -> runs the Backtest agent (produces results)
      └─ tool: tester_agent_tool    -> runs the Tester agent (verifies results)

So "the distributor distributes to sub-agents" literally: it calls them the same
way any agent calls a tool. This is the classic LangChain "agents-as-tools"
pattern. The Distributor itself writes no backtest and runs no checks - it only
decides who does what and in what order, then reports the outcome.
"""
from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain_core.tools import tool

from llm import build_llm
from trace import run_streaming


# --- the two sub-agents, wrapped as tools the Distributor can call ---

@tool
def backtest_agent_tool(request: str) -> str:
    """Hand a full natural-language backtest request to the Backtest agent, which
    derives the strategy's legs and runs the backtest on real data. Pass the
    user's request through unchanged. Returns JSON including the results-file path
    and a summary.
    """
    from agent import run_and_extract            # imported here to avoid import cycles
    r = run_and_extract(request)
    return json.dumps({
        "results_file": r.get("results_file"),
        "summary": r.get("summary"),
        "strategy": r.get("strategy"),
        "answer": r.get("answer"),
    })


@tool
def tester_agent_tool(results_file: str = "") -> str:
    """Hand a results-file path to the Tester agent, which verifies the backtest
    and returns a PASS/FAIL sign-off. Use the results_file returned by
    backtest_agent_tool.
    """
    from tester_agent import verify
    return verify(results_file)


DISTRIBUTOR_TOOLS = [backtest_agent_tool, tester_agent_tool]

SYSTEM_PROMPT = """You are the Distributor. You do NOT do the work yourself - you
route it to your sub-agents (your tools) and report the outcome.

For a backtest request (e.g. "butterfly nifty 50 from ... using 15 minutes
interval for daily basis"), do exactly this:

1. Call backtest_agent_tool with the user's full request, unchanged. It returns
   JSON that includes "results_file" and "summary".
2. Take that "results_file" and call tester_agent_tool with it, so the Tester
   verifies the backtest.
3. Give the user a final report with:
   - the backtest summary (strategy, number of trades, net P&L, win rate, file), and
   - the Tester's sign-off (PASS or FAIL and why).

Rules:
- Always run the backtest FIRST, then the tester. Never skip the tester.
- Never invent numbers; only report what the tools return.
"""


def build_distributor():
    return create_agent(model=build_llm(), tools=DISTRIBUTOR_TOOLS,
                        system_prompt=SYSTEM_PROMPT)


def run(prompt: str) -> str:
    """Run the whole 3-agent flow and return the Distributor's final report."""
    agent = build_distributor()
    messages = run_streaming(agent, prompt, "DISTRIBUTOR (routes to sub-agents)")
    return getattr(messages[-1], "content", "")
