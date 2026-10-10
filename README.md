# LangChain Options Backtester (3-agent)

A backtesting engine where **LLM agents run the show** (using LangChain), and
plain Python does the number-crunching. It's a learning-friendly cousin of the
`backtesting_engine` project - same idea, but built properly on LangChain as a
small **multi-agent system** so you can learn the framework.

> **The one rule to remember:** the LLM *decides* (which strategy, which legs,
> which dates), and Python *computes* (the actual profit/loss on real data). An
> LLM can't be trusted to do arithmetic over millions of rows, so it doesn't.

## How to run

```bash
python3 backtest.py --prompt "butterfly nifty 50 from 2026-03-24 to 2026-09-15 using 15 minutes interval for daily basis"
python3 backtest.py --prompt "..." --quiet   # hide the live reasoning
```

By default it is **verbose**: you watch each agent think - the tools it calls,
the arguments, what came back, and its reasoning. Use `--quiet` for just the
final report.


## The three agents

```
            You type a prompt
                   │
                   ▼
        ┌──────────────────────┐
        │  DISTRIBUTOR agent    │  distributor.py - routes the work
        └───────┬──────────┬────┘
    its tools   │          │   (the sub-agents ARE its tools)
                ▼          ▼
     ┌────────────────┐  ┌──────────────────┐
     │ BACKTEST agent │  │  TESTER agent     │
     │ agent.py       │  │  tester_agent.py  │
     │ derives legs + │  │ checks the result │
     │ runs backtest  │  │ file, signs PASS/ │
     └────────────────┘  │ FAIL              │
                         └──────────────────┘
```

Flow: the **Distributor** calls the **Backtest** agent (which produces a results
file), then calls the **Tester** agent on that file (which signs off PASS/FAIL),
then reports both. The Distributor writes no code and runs no checks itself - it
only *routes*. This is the LangChain **"agents-as-tools"** pattern: one agent's
tools are other agents.

## The files (and the LangChain idea each one teaches)

| File | What it is | LangChain concept |
|------|-----------|-------------------|
| `llm.py` | Builds the chat model pointed at LLMAPI | **The model** (`ChatOpenAI`) |
| `tools.py` | The backtest agent's two actions | **Tools** (`@tool`) + **structured output** (Pydantic) |
| `agent.py` | Backtest agent: derive strategy + run it | **An agent** (`create_agent`) |
| `tester_agent.py` | Tester agent: verify the result file | **An agent** with its own tool |
| `distributor.py` | Top agent that routes to the other two | **Multi-agent** (agents-as-tools) |
| `engine.py` | The real backtest math (no LLM) | *(plain Python - the "computes" half)* |
| `trace.py` | Prints each agent's steps live | *(streaming for learning)* |
| `backtest.py` | The command-line entry point | *(reads `--prompt`, calls the Distributor)* |

## The LangChain concepts you're learning here

1. **The model** (`llm.py`) - `ChatOpenAI(base_url=...)` turns any
   OpenAI-compatible endpoint into a LangChain model you can `.invoke()`.
2. **Tools** (`tools.py`, `tester_agent.py`) - a normal Python function + `@tool`
   + a good docstring becomes something the LLM can call. The docstring is how
   the LLM knows what the tool does, so write it for the model.
3. **Structured output** (`tools.py`) - `llm.with_structured_output(Leg)` makes
   the model return a filled-in Pydantic object instead of loose text, so we get
   clean data with no manual JSON parsing.
4. **An agent** (`agent.py`, `tester_agent.py`) - `create_agent(model, tools,
   system_prompt)` gives you an LLM that loops (think → call a tool → think)
   until it's done.
5. **Multi-agent / agents-as-tools** (`distributor.py`) - wrap a whole agent in
   an `@tool` and give it to another agent; now one agent can delegate to
   others. That's all a "distributor/supervisor" is.
6. **Streaming** (`trace.py`) - `agent.stream(..., stream_mode="values")` lets
   you print each step as it happens, so you can watch the agents think.

## What it can and can't backtest

Works: any **single-expiry, static** strategy expressed as legs of
calls/puts/underlying at `Spot ± k×ATR` - butterfly, condor, iron condor,
vertical spreads, straddle, strangle, etc. Unknown ones are learned from the web.

Doesn't work: **multi-expiry** strategies (calendar/diagonal spreads - the engine
runs one expiry per trade) and **dynamic** ones (stop-losses, rolling,
adjustments). The derived legs are printed so you can sanity-check them - a free
model can still get an exotic strategy wrong.

## Data

Reads the same historical option candles as the other project:
`/Desktop/options_data/<index>/options_<interval>_<expiry>.csv`.
Results are written to `results/`.
