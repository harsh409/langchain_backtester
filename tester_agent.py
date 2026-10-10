"""
THE TESTER AGENT - checks the backtest result and signs off.

This is the third agent. Its only job: look at the results file the Backtest
agent produced and decide whether it is trustworthy, then say PASS or FAIL.

It has ONE tool, `inspect_results`, which does deterministic Python checks on the
CSV (does it exist? any error rows? does profit - loss = net_pnl? what do the
orders look like?). The agent (the LLM) then READS those findings and JUDGES:
  - is the file valid, and
  - do the orders actually match the strategy's logic?
and signs off. (Checks = Python; judgement = LLM.)
"""
from __future__ import annotations

import csv
import json
import os

from langchain.agents import create_agent
from langchain_core.tools import tool

from llm import build_llm
from trace import run_streaming

EXPECTED_COLUMNS = {"date", "orders_placed", "profit", "loss", "net_pnl"}


@tool
def inspect_results(results_file: str) -> str:
    """Run mechanical checks on a backtest results CSV and report the findings.
    Pass the full path to the results file. Returns JSON with: whether the file
    exists, the row/trade count, any error rows, whether profit-loss=net_pnl is
    consistent, the net P&L total, a few sample order lines, and the strategy
    name taken from the filename. Call this first, then judge the findings.
    """
    if not results_file or not os.path.exists(results_file):
        return json.dumps({"exists": False, "error": f"no such file: {results_file}"})

    with open(results_file, newline="") as f:
        rows = list(csv.DictReader(f))

    columns_ok = EXPECTED_COLUMNS.issubset(set(rows[0].keys())) if rows else False
    error_rows = [r for r in rows if str(r.get("orders_placed", "")).startswith("Error")]

    # Check the P&L arithmetic per row: net_pnl should equal profit - loss.
    inconsistent = 0
    net_sum = 0.0
    for r in rows:
        try:
            profit = float(r.get("profit", 0) or 0)
            loss = float(r.get("loss", 0) or 0)
            net = float(r.get("net_pnl", 0) or 0)
            net_sum += net
            if round(profit - loss, 2) != round(net, 2):
                inconsistent += 1
        except ValueError:
            inconsistent += 1

    sample_orders = [r.get("orders_placed", "") for r in rows[:3]]
    strategy_from_filename = os.path.basename(results_file).replace("_results.csv", "")

    return json.dumps({
        "exists": True,
        "total_trades": len(rows),
        "columns_ok": columns_ok,
        "error_rows": len(error_rows),
        "pnl_rows_inconsistent": inconsistent,
        "net_pnl_total": round(net_sum, 2),
        "sample_orders": sample_orders,
        "strategy_from_filename": strategy_from_filename,
    })


TESTER_TOOLS = [inspect_results]

SYSTEM_PROMPT = """You are the Test Engineer. You verify a backtest results file
and sign off on it. You do NOT edit anything.

Steps:
1. Call inspect_results with the results file path.
2. Judge the findings:
   - The file must exist, have the expected columns, and contain trades.
   - There must be NO error rows, and the P&L arithmetic must be consistent
     (profit - loss = net_pnl for every row -> pnl_rows_inconsistent must be 0).
   - Look at sample_orders and the strategy name: do the orders make sense for
     that strategy? (e.g. a butterfly should show 4 call/put legs with two at the
     same middle strike; a straddle a call and a put at the same strike.)
3. Sign off. Start your final answer with exactly "PASS" or "FAIL", then one or
   two short sentences giving the reason. Be strict but fair: only FAIL for a
   real problem (missing file, error rows, broken arithmetic, orders that clearly
   don't match the strategy).
"""


def build_tester_agent():
    return create_agent(model=build_llm(), tools=TESTER_TOOLS, system_prompt=SYSTEM_PROMPT)


def verify(results_file: str = "") -> str:
    """Verify one results file and return the tester's PASS/FAIL sign-off."""
    # If no path was given, fall back to the newest file in results/ - so the
    # tester can still check "the backtest we just ran".
    if not results_file or not os.path.exists(results_file):
        from engine import OUTPUT_ROOT
        try:
            files = [os.path.join(OUTPUT_ROOT, f) for f in os.listdir(OUTPUT_ROOT)
                     if f.endswith("_results.csv")]
            results_file = max(files, key=os.path.getmtime) if files else results_file
        except FileNotFoundError:
            pass

    agent = build_tester_agent()
    prompt = f"Verify this backtest results file and sign off: {results_file}"
    messages = run_streaming(agent, prompt, "TESTER AGENT")
    return getattr(messages[-1], "content", "")
