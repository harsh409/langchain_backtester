#!/usr/bin/env python3
"""
Command-line entry point.

Run it like:
    python3 backtest.py --prompt "butterfly nifty 50 from 2026-03-24 to 2026-09-15 using 15 minutes interval for daily basis"

Everything interesting happens in agent.py (the LLM) and tools.py (the actions).
This file just reads the --prompt text and hands it to the agent.
"""
import argparse
import os
import sys

# Make sure Python can import our sibling files (llm.py, agent.py, ...) no matter
# where you run this from - we add THIS file's folder to the import path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import trace                       # the live-reasoning printer (verbose by default)
import engine                      # so we can set the entry time before running
from distributor import run        # the top agent that routes to the sub-agents


def main():
    parser = argparse.ArgumentParser(
        description="LangChain Options Backtester (3 agents: Distributor -> Backtest -> Tester)")
    parser.add_argument("--prompt", "-p", type=str,
                        help="natural-language backtest request")
    parser.add_argument("--entry-time", "-e", type=str, default=None,
                        help="time of day to enter trades, e.g. 09:30 "
                             "(default: the 09:15 open). Trades enter on the first "
                             "candle at/after this time; if none exists you get an error.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="hide the agents' live reasoning (show only the final report)")
    args = parser.parse_args()

    if not args.prompt:
        parser.print_help()
        return

    # Set the entry time (a mechanical setting the engine reads). We validate it
    # here so a bad time fails immediately, before any LLM calls.
    if args.entry_time:
        try:
            engine.ENTRY_TIME = engine.normalize_time(args.entry_time)
        except ValueError as exc:
            print(f"Error: {exc}")
            return
        print(f"(entering trades at/after {engine.ENTRY_TIME})")

    trace.VERBOSE = not args.quiet   # verbose ON by default: watch the agents think

    print(f"\nRequest: {args.prompt}")
    print("=" * 60)
    answer = run(args.prompt)        # <- the Distributor agent runs the whole flow
    print("\n" + "=" * 72)
    print("  ✅ FINAL REPORT")
    print("=" * 72)
    print("\n" + answer)


if __name__ == "__main__":
    main()
