"""
TOOLS - the actions the LLM is allowed to take.

LangChain concept #2: A TOOL.
-----------------------------
An LLM can only produce text. A *tool* is a normal Python function that we let
the LLM call by name. We put `@tool` on top of a function and write a clear
docstring; LangChain shows that docstring to the LLM so it knows what the tool
does and when to use it. The LLM says "call get_strategy with 'butterfly'",
LangChain runs our function, and hands the result back to the LLM.

So: the LLM is the brain that DECIDES; these tools are the hands that DO.

We give the agent two tools:
  1. get_strategy   - work out the buy/sell legs of a strategy (web + LLM)
  2. run_backtest   - actually run the trades on real data (plain Python math)
"""
from __future__ import annotations

import json
from typing import List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    from ddgs import DDGS            # web search library (newer name)
except ImportError:
    from duckduckgo_search import DDGS

from engine import coerce_legs, run_backtest_core
from llm import build_llm


# ---------------------------------------------------------------------------
# LangChain concept #3: STRUCTURED OUTPUT (Pydantic).
# Instead of asking the LLM for text and parsing it ourselves, we describe the
# exact shape we want as Pydantic classes. LangChain then makes the model fill
# that shape in and hands us back real Python objects - no manual JSON parsing.
# ---------------------------------------------------------------------------

class Leg(BaseModel):
    """One order in a strategy."""
    side: str = Field(description="BUY or SELL")
    instrument: str = Field(description="CE for a call, PE for a put, FUT for the underlying")
    atr_offset: float = Field(
        description="strike = Spot + atr_offset * ATR. 0 = at-the-money, "
                    "positive = above spot (calls/upper wings), negative = below (puts/lower wings)")


class StrategyDefinition(BaseModel):
    """A whole strategy: a short description plus its list of legs."""
    description: str = Field(description="one sentence: what this strategy is")
    legs: List[Leg] = Field(description="every leg of the strategy, in order")
    risk_profile: str = Field(description="one sentence on the risk/reward")


def _web_snippets(strategy_name: str, n: int = 6) -> str:
    """Fetch a few web snippets about how a strategy is built (plain text)."""
    try:
        with DDGS() as ddgs:
            hits = ddgs.text(
                f"{strategy_name} options strategy legs buy sell call put strikes",
                max_results=n)
        return "\n".join(f"- {h.get('title','')}: {h.get('body','')[:400]}" for h in hits)
    except Exception as exc:
        return f"(web search unavailable: {exc})"


@tool
def get_strategy(strategy_name: str) -> str:
    """Work out the exact option legs (buy/sell, call/put, strike offset) for an
    options strategy by name, e.g. 'iron condor', 'butterfly', 'straddle'.
    Use this FIRST, before running a backtest, to learn how the strategy is built.
    Returns a JSON object with the strategy's description and its legs.
    """
    # 1. Read a bit of the web so even an unfamiliar strategy is grounded in facts.
    research = _web_snippets(strategy_name)

    # 2. Ask the LLM to fill in the StrategyDefinition shape (structured output).
    #    `with_structured_output` = "answer as this Pydantic class, not free text".
    llm = build_llm()
    structured = llm.with_structured_output(StrategyDefinition)

    guidance = (
        "You convert an options strategy into its legs.\n"
        "Rules:\n"
        "- A call above the money uses a POSITIVE atr_offset; a put below the money NEGATIVE.\n"
        "- Include EVERY leg. A butterfly = 4 legs (buy 1 lower, SELL 2 middle, buy 1 upper); "
        "if a strike is traded twice, list that leg twice.\n"
        "- An iron condor = 4 legs. A vertical spread = 2 legs.\n"
        "- Use FUT only if the strategy literally holds the underlying.\n\n"
        f"Strategy to define: \"{strategy_name}\"\n\n"
        f"Web research about it:\n{research}"
    )
    try:
        result: StrategyDefinition = structured.invoke(guidance)
    except Exception as exc:
        return json.dumps({"error": f"could not derive '{strategy_name}': {exc}"})

    # 3. Convert to the simple [side, instrument, offset] the engine expects, and
    #    validate it before handing it on.
    legs = coerce_legs([[l.side, l.instrument, l.atr_offset] for l in result.legs])
    if not legs:
        return json.dumps({"error": f"the derived legs for '{strategy_name}' were invalid"})
    return json.dumps({"strategy": strategy_name, "description": result.description,
                       "legs": legs, "risk_profile": result.risk_profile})


@tool
def run_backtest(strategy_name: str, legs_json: str, index: str,
                 start_date: str, end_date: str,
                 interval: str = "15min", frequency: str = "daily") -> str:
    """Run the backtest for a strategy over a date range and return the results.
    Call get_strategy FIRST, then pass its 'legs' here as `legs_json`.

    Args:
        strategy_name: e.g. 'butterfly' (used only for labelling)
        legs_json: the legs list from get_strategy, e.g. '[["BUY","CE",-1.0],["SELL","CE",0.0]]'
        index: one of 'nifty_50', 'bank_nifty', 'sensex'
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        interval: candle size like '5min','15min','30min','60min','day' (default 15min)
        frequency: 'daily' or 'weekly' (default daily)
    Returns a JSON summary (trades, net P&L, win rate) and the results file path.
    """
    # The legs come in as a JSON string; turn it back into a Python list.
    try:
        legs = coerce_legs(json.loads(legs_json))
    except Exception:
        legs = None
    # Safety net: if the agent forgot/garbled the legs, derive them here.
    if not legs:
        try:
            derived = json.loads(get_strategy.invoke({"strategy_name": strategy_name}))
            legs = coerce_legs(derived.get("legs"))
        except Exception:
            legs = None
    if not legs:
        return json.dumps({"status": "error",
                           "message": f"no valid legs for '{strategy_name}'"})

    result = run_backtest_core(index=index, start_date=start_date, end_date=end_date,
                               interval=interval, frequency=frequency, legs_def=legs, strategy_name=strategy_name)
    # Hand back only the compact summary (not thousands of trade rows).
    if result.get("status") == "completed":
        return json.dumps({"status": "completed", "strategy": strategy_name,
                           "legs": legs, "summary": result["summary"],
                           "results_file": result["results_file"]})
    return json.dumps(result)


# The list the agent is given.
TOOLS = [get_strategy, run_backtest]
