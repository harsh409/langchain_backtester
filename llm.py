"""
The LLM for the LangChain backtester.

LangChain concept #1: THE MODEL.
--------------------------------
LangChain talks to any chat model through a common interface. For an
OpenAI-compatible endpoint (which the freeLLMAPI gateway is) that interface is
`ChatOpenAI` from `langchain_openai`. Point it at a `base_url` and it behaves
like any LangChain model - you can `.invoke()` it, bind tools to it, or ask it
for structured output.

Config is the SAME freeLLMAPI setup the backtesting_engine project uses
(env vars FREELLM_BASE_URL / FREELLM_API_KEY / FREELLM_MODEL, same fallbacks),
so both projects hit the same local gateway.
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

# --- picked from the backtesting_engine project's llm.py (identical values) ---
BASE_URL = os.getenv("your base url")
API_KEY = os.getenv(
    "your model api key"
)
MODEL = os.getenv("your model")


def build_llm(temperature: float = 0.1, max_tokens: int = 1400) -> ChatOpenAI:
    """Return a LangChain chat model wired to the freeLLMAPI gateway.

    `temperature` low = consistent/deterministic answers (good for parsing and
    strategy derivation). This one object is reused everywhere in the app.
    """
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )
