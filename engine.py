"""
The deterministic backtest engine - the "Python computes" half.

There is NO LLM in this file. The LLM (via the LangChain agent) decides WHAT to
run - which strategy, which legs, which dates - but the actual money math (spot
from put-call parity, ATR, opening/closing every leg, P&L, drawdown) has to be
exact, so it lives here in plain pandas. Feeding millions of option rows to an
LLM to "compute P&L" would just produce confident nonsense.

A strategy is a list of legs: (side, instrument, atr_offset)
    side       : "BUY" or "SELL"
    instrument : "CE" (call), "PE" (put), "FUT" (the underlying)
    atr_offset : strike is placed at Spot + atr_offset * ATR (0 = ATM)
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DATA_ROOT = "/home/harsh/Desktop/options_data"
OUTPUT_ROOT = "/home/harsh/Desktop/langchain_backtester/results"
ATR_PERIOD = 14
MIN_ATR_DAYS = 3

# Entry time of day.
#   None  -> enter on the FIRST bar of the day (the 09:15 open) - the default,
#            i.e. exactly how it worked before.
#   "HH:MM" (e.g. "09:30", set via the CLI --entry-time) -> enter on the first
#            bar AT OR AFTER that time. If a day has no bar at/after it, that day
#            cannot be entered, and if NO day can, the run returns an error.
ENTRY_TIME: Optional[str] = None


def normalize_time(text: str) -> str:
    """Turn '9:30' / '09:30' / '09:30:00' into 'HH:MM:SS'. Raises ValueError if
    the text is not a valid clock time."""
    parts = text.strip().split(":")
    if len(parts) == 2:
        h, m, s = parts[0], parts[1], "0"
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError(f"not a valid time: {text!r} (use HH:MM, e.g. 09:30)")
    h, m, s = int(h), int(m), int(s)
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"not a valid time: {text!r}")
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Leg validation (shared with the tools layer)
# ---------------------------------------------------------------------------

def coerce_legs(raw_legs) -> Optional[List[list]]:
    """Validate/normalize legs into [side, instrument, float] triples, or None."""
    if not isinstance(raw_legs, list) or not raw_legs:
        return None
    out: List[list] = []
    for leg in raw_legs:
        if not isinstance(leg, (list, tuple)) or len(leg) != 3:
            return None
        side, itype, mult = leg
        side = str(side).strip().upper()
        itype = str(itype).strip().upper()
        itype = {"CALL": "CE", "PUT": "PE", "UNDERLYING": "FUT",
                 "FUTURE": "FUT", "STOCK": "FUT"}.get(itype, itype)
        if side not in ("BUY", "SELL") or itype not in ("CE", "PE", "FUT"):
            return None
        try:
            mult = float(mult)
        except (TypeError, ValueError):
            return None
        out.append([side, itype, mult])
    return out


# ---------------------------------------------------------------------------
# Data loading + availability
# ---------------------------------------------------------------------------

def load_csv_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    for col in ["strike_price", "open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["strike_price", "close"]).sort_values("timestamp").reset_index(drop=True)


def validate_data_availability(index: str, start_date: Optional[str],
                               end_date: Optional[str], interval: str,
                               frequency: str) -> Dict[str, Any]:
    """Find the expiry CSVs that cover the requested range."""
    data_path = os.path.join(DATA_ROOT, index)
    if not os.path.exists(data_path):
        return {"error": f"Data directory not found: {data_path}"}
    expiries_path = os.path.join(data_path, "expiries.json")
    if not os.path.exists(expiries_path):
        return {"error": f"Expiries file not found: {expiries_path}"}

    with open(expiries_path) as f:
        all_expiries = sorted(json.load(f).get("expiries", []))

    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    relevant = [e for e in all_expiries
                if (start_dt is None or datetime.strptime(e, "%Y-%m-%d") >= start_dt)
                and datetime.strptime(e, "%Y-%m-%d") <= end_dt]
    if frequency == "daily":
        later = [e for e in all_expiries if datetime.strptime(e, "%Y-%m-%d") > end_dt]
        if later:
            relevant.append(later[0])

    found, missing = [], []
    for expiry in relevant:
        csv_file = os.path.join(data_path, f"options_{interval}_{expiry}.csv")
        (found if os.path.exists(csv_file) else missing).append(csv_file if os.path.exists(csv_file) else expiry)

    return {
        "all_expiries": all_expiries,
        "relevant_expiries": relevant,
        "found_files": found,
        "missing_dates": missing,
        "data_available": len(found) > 0,
    }


# ---------------------------------------------------------------------------
# Market math
# ---------------------------------------------------------------------------

def synthetic_spot_series(df: pd.DataFrame, n_strikes: int = 3) -> pd.Series:
    """Spot per timestamp via put-call parity (S ~= K + C - P), median of the
    n strikes nearest ATM."""
    calls = df[df["instrument_type"] == "CE"].pivot_table(
        index="timestamp", columns="strike_price", values="close", aggfunc="last")
    puts = df[df["instrument_type"] == "PE"].pivot_table(
        index="timestamp", columns="strike_price", values="close", aggfunc="last")
    common = calls.columns.intersection(puts.columns)
    diff = (calls[common] - puts[common]).dropna(how="all")
    implied = diff.add(pd.Series(common, index=common), axis=1)
    nearest = diff.abs().rank(axis=1, method="first") <= n_strikes
    return implied.where(nearest).median(axis=1).dropna().sort_index()


def daily_atr(spot: pd.Series, before_date: str, period: int = ATR_PERIOD) -> Optional[float]:
    if spot.empty:
        return None
    days = spot.groupby(spot.index.strftime("%Y-%m-%d")).agg(["max", "min", "last"])
    days = days[days.index < before_date]
    if len(days) < MIN_ATR_DAYS:
        return None
    prev_close = days["last"].shift(1)
    tr = pd.concat([days["max"] - days["min"],
                    (days["max"] - prev_close).abs(),
                    (days["min"] - prev_close).abs()], axis=1).max(axis=1).dropna()
    return float(tr.tail(period).mean()) if not tr.empty else None


def straddle_implied_move(bar: pd.DataFrame, spot: float, days_to_expiry: int) -> float:
    atm = pick_strike(spot, sorted(bar["strike_price"].unique()))
    c = bar[(bar["strike_price"] == atm) & (bar["instrument_type"] == "CE")]["close"]
    p = bar[(bar["strike_price"] == atm) & (bar["instrument_type"] == "PE")]["close"]
    if c.empty or p.empty:
        return spot * 0.01
    return float(c.iloc[0] + p.iloc[0]) * 0.8 / max(days_to_expiry, 1) ** 0.5


def pick_strike(target: float, strikes: List[float]) -> float:
    return min(strikes, key=lambda s: (abs(s - target), s))


def select_legs(legs_def: List[Tuple[str, str, float]], spot: float, atr: float,
                entry_bar: pd.DataFrame) -> List[Dict[str, Any]]:
    legs: List[Dict[str, Any]] = []
    for side, itype, mult in legs_def:
        if itype == "FUT":
            legs.append({"side": side, "type": "FUT", "strike": None, "entry": spot})
            continue
        chain = entry_bar[entry_bar["instrument_type"] == itype].set_index("strike_price")["close"]
        strikes = sorted(chain.index)
        if not strikes:
            continue
        strike = pick_strike(spot + mult * atr, strikes)
        for other in legs:
            if other["type"] == itype and other["side"] != side and other["strike"] == strike:
                outward = [s for s in strikes if (s > strike if mult > 0 else s < strike)]
                if outward:
                    strike = min(outward) if mult > 0 else max(outward)
        legs.append({"side": side, "type": itype, "strike": strike, "entry": float(chain[strike])})
    return legs


def exit_price(df: pd.DataFrame, leg: Dict[str, Any], exit_ts, exit_spot: float) -> float:
    if leg["type"] == "FUT":
        return exit_spot
    exit_day = exit_ts.strftime("%Y-%m-%d")
    rows = df[(df["instrument_type"] == leg["type"]) & (df["strike_price"] == leg["strike"])
              & (df["date"] == exit_day) & (df["timestamp"] <= exit_ts)]
    if not rows.empty:
        return float(rows["close"].iloc[-1])
    if leg["type"] == "CE":
        return max(exit_spot - leg["strike"], 0.0)
    return max(leg["strike"] - exit_spot, 0.0)


def first_and_last_bar(spot: pd.Series, day: str):
    day_ts = spot.index[spot.index.strftime("%Y-%m-%d") == day]
    return (day_ts[0], day_ts[-1]) if len(day_ts) else (None, None)


def entry_bar(spot: pd.Series, day: str, entry_time: Optional[str]):
    """Pick the entry timestamp for a day.

    entry_time None  -> the first bar of the day (09:15 open) - the old default.
    entry_time set    -> the first bar AT OR AFTER that time; None if the day has
                         no such bar (e.g. the time is after the close).
    """
    day_ts = spot.index[spot.index.strftime("%Y-%m-%d") == day]
    if len(day_ts) == 0:
        return None
    if not entry_time:
        return day_ts[0]
    for ts in day_ts:
        if ts.strftime("%H:%M:%S") >= entry_time:
            return ts
    return None


def run_trade(df, spot, legs_def, entry_ts, exit_ts, expiry, qty, lots) -> Optional[Dict[str, Any]]:
    if entry_ts not in spot.index or exit_ts not in spot.index:
        return None
    spot_entry, spot_exit = spot[entry_ts], spot[exit_ts]
    entry_bar = df[df["timestamp"] == entry_ts]
    entry_day = entry_ts.strftime("%Y-%m-%d")

    atr = daily_atr(spot, entry_day)
    atr_source = "atr14"
    if atr is None:
        dte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.strptime(entry_day, "%Y-%m-%d")).days
        atr = straddle_implied_move(entry_bar, spot_entry, dte)
        atr_source = "straddle"

    legs = select_legs(legs_def, spot_entry, atr, entry_bar)
    if not legs:
        return None

    pnl, orders = 0.0, []
    for leg in legs:
        leg["exit"] = exit_price(df, leg, exit_ts, spot_exit)
        sign = 1 if leg["side"] == "BUY" else -1
        pnl += sign * (leg["exit"] - leg["entry"]) * qty
        name = {"CE": "CALL", "PE": "PUT", "FUT": "UNDERLYING"}[leg["type"]]
        strike = f" {int(leg['strike'])}" if leg["strike"] is not None else ""
        orders.append(f"{leg['side']} {name}{strike} @ {leg['entry']:.2f} -> {leg['exit']:.2f}")

    pnl = round(pnl, 2)
    return {
        "date": entry_day if entry_day == exit_ts.strftime("%Y-%m-%d") else expiry,
        "expiry": expiry,
        "orders_opened_time": entry_ts.strftime("%Y-%m-%d %H:%M:%S%z"),
        "orders_closed_time": exit_ts.strftime("%Y-%m-%d %H:%M:%S%z"),
        "spot_at_entry": round(spot_entry, 2), "spot_at_exit": round(spot_exit, 2),
        "atr": round(atr, 2), "atr_source": atr_source,
        "orders_placed": "; ".join(orders), "lot": lots, "qty": qty,
        "profit": pnl if pnl > 0 else 0, "loss": -pnl if pnl < 0 else 0, "net_pnl": pnl,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnls = [r.get("net_pnl", 0) for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    equity = peak = max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "total_trades": len(results),
        "total_profit": round(sum(wins), 2),
        "total_loss": round(-sum(losses), 2),
        "net_pnl": round(sum(pnls), 2),
        "win_rate": round(len(wins) / max(len(results), 1) * 100, 1),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
        "max_drawdown": round(max_dd, 2),
    }


def save_results(results: List[Dict], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fields = ["date", "expiry", "orders_opened_time", "orders_closed_time", "spot_at_entry",
              "spot_at_exit", "atr", "orders_placed", "lot", "qty", "profit", "loss", "net_pnl"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: v for k, v in row.items() if k in fields})
    return output_path


def run_backtest_core(index: str, start_date: str, end_date: str, interval: str,
                      frequency: str, legs_def: List[list], strategy_name: str, lots: int = 1) -> Dict[str, Any]:
    """Run the whole backtest for one strategy over one date range. Pure Python."""
    validation = validate_data_availability(index, start_date, end_date, interval, frequency)
    if "error" in validation:
        return {"status": "error", "message": validation["error"]}
    found_files = validation.get("found_files", [])
    all_expiries = validation.get("all_expiries", [])
    if not found_files:
        return {"status": "error", "message": "No historical data for that index/interval/range"}

    start = start_date or "0000-00-00"
    end = end_date or "9999-99-99"
    entry_time = ENTRY_TIME               # None = 09:15 open (default)
    results: List[Dict[str, Any]] = []
    traded_days = set()
    had_trading_days = False              # did any day have bars at all?
    entry_infeasible = False              # a day had bars but none at/after entry_time

    for csv_file in found_files:
        expiry = os.path.basename(csv_file).replace(f"options_{interval}_", "").replace(".csv", "")
        try:
            df = load_csv_data(csv_file)
            if df.empty:
                continue
            qty = int(df["lot_size"].iloc[0]) * lots
            spot = synthetic_spot_series(df)
            if spot.empty:
                continue
            prev = [e for e in all_expiries if e < expiry]
            prev_expiry = prev[-1] if prev else (
                datetime.strptime(expiry, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
            trading_days = sorted(set(spot.index.strftime("%Y-%m-%d")))

            if frequency == "weekly":
                week = [d for d in trading_days if prev_expiry < d <= expiry]
                if not week or expiry not in trading_days:
                    continue
                had_trading_days = True
                entry_ts = entry_bar(spot, week[0], entry_time)   # at/after entry_time
                _, exit_ts = first_and_last_bar(spot, expiry)     # exit = last bar
                if entry_ts is None:
                    entry_infeasible = True
                    continue
                trade = run_trade(df, spot, legs_def, entry_ts, exit_ts, expiry, qty, lots)
                if trade:
                    results.append(trade)
            else:
                for day in trading_days:
                    if not (prev_expiry < day <= expiry) or not (start <= day <= end) or day in traded_days:
                        continue
                    had_trading_days = True
                    entry_ts = entry_bar(spot, day, entry_time)   # at/after entry_time
                    _, exit_ts = first_and_last_bar(spot, day)    # exit = last bar
                    if entry_ts is None:
                        entry_infeasible = True
                        continue
                    if entry_ts == exit_ts:
                        continue
                    trade = run_trade(df, spot, legs_def, entry_ts, exit_ts, expiry, qty, lots)
                    if trade:
                        results.append(trade)
                        traded_days.add(day)
        except Exception as e:
            results.append({"date": expiry, "expiry": expiry,
                            "orders_placed": f"Error: {type(e).__name__}: {e}",
                            "lot": lots, "profit": 0, "loss": 0, "net_pnl": 0})

    # Requested a specific entry time but no day had a bar at/after it -> error,
    # so the user knows the time was out of session rather than getting empty data.
    if entry_time and not results and had_trading_days and entry_infeasible:
        return {"status": "error",
                "message": (f"No entry possible at {entry_time}: no candle at or "
                            f"after that time on any trading day (the session runs "
                            f"~09:15 to 15:15). Pick an earlier entry time.")}

    results.sort(key=lambda r: r.get("orders_opened_time", r["date"]))
    summary = summarize(results)
    out_path = os.path.join(OUTPUT_ROOT, f"{strategy_name}_{index}_{interval}_{frequency}_results.csv")
    save_results(results, out_path)
    return {"status": "completed", "summary": summary, "results": results,
            "results_file": out_path}
