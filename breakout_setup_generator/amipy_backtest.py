#!/usr/bin/env python3
"""
amipy_backtest.py
Sprint-2: reuse the user's own `amipy.py` engine (the one they said "works
fast and good") as a SECOND-OPINION backtest for the strategies that make
it to the top of the leaderboard, instead of only trusting `backtest.py`'s
custom loop.

Why not replace backtest.py outright: amipy's `run()` does a per-entry
forward scan (see its source -- for every entry it walks forward bar by
bar looking for the matching exit), which is the right level of fidelity
for a handful of finalists but too slow to be the DEFAULT engine for
generating/screening hundreds of candidates. So the split is:

    backtest.py   -- fast, custom, used for the bulk generate/screen pass
    amipy_backtest.py -- the trusted, fuller-featured engine (real
                          position sizing, margin, commission, Sortino,
                          per-trade ledger), used as a cross-check on
                          survivors via `--amipy-check-top N`

This module translates a StrategySpec into the exact inputs amipy.Amipy
expects (buy/short/sell/cover boolean signals + fill-price series), using
grammar-consistent stop/target sizing (ATR-based, or POI-anchored),
via amipy's own `apply_stops_sell_rq` / `apply_stops_cover_rq` (its
built-in per-bar/"rolling quantile" stop-and-target mechanism) rather than
reinventing that logic.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import amipy
from amipy import Amipy
import grammar as g
import indicators as ind
from generator import StrategySpec
from backtest import build_signal


class Context:
    """ Mirrors strategy_bollinger_cmf_bulk.py's Context: tick_size=0.01 +
    tick_value=0.01 makes $ P&L move 1:1 with price, so risk-based position
    sizing (risk=0.1 -> use 10% of equity per trade) behaves sensibly
    regardless of the asset's price scale. """
    def __init__(self, symbol, starting_equity=100_000.0, margin_required=0.0,
                 tick_size=0.01, tick_value=0.01, risk=0.1, warmup_bars=60,
                 commission=0.0):
        self.symbol = symbol
        self.starting_equity = starting_equity
        self.margin_required = margin_required
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.risk = risk
        self.warmup_bars = warmup_bars
        self.commission = commission


def _first_of_run(sig: pd.Series) -> pd.Series:
    """ Our breakout condition (close > level) often stays True for several
    consecutive bars once triggered, but backtest.py's state machine only
    opens ONE trade per such run (it waits for an actual exit before
    re-arming). amipy's apply_stops_*_rq scans forward from EVERY True bar
    independently, so feeding it the raw, un-collapsed signal computes a
    separate (and overlapping/misleading) exit for each of those redundant
    bars. Collapsing each run to its first bar before computing stops keeps
    the two engines' entry semantics aligned. """
    arr = sig.values
    out = np.zeros(len(arr), dtype=bool)
    prev = False
    for i, v in enumerate(arr):
        out[i] = bool(v) and not prev
        prev = bool(v)
    return pd.Series(out, index=sig.index)


def _time_exit_signal(entry: pd.Series, hold_days: int) -> pd.Series:
    """ Build the amipy-style exit signal directly: True exactly `hold_days`
    bars after each (already-deduped) entry -- the vectorized equivalent
    of backtest.py's `bars_held >= max_hold` time exit. """
    n = len(entry)
    out = np.zeros(n, dtype=bool)
    idx = np.where(entry.values)[0]
    for i in idx:
        j = i + hold_days
        if j < n:
            out[j] = True
    return pd.Series(out, index=entry.index)


def _eod_exit_signal(entry: pd.Series, eod_flag: pd.Series) -> pd.Series:
    """ True at the first end-of-session bar at or after each entry --
    vectorized equivalent of backtest.py's eod_exit trigger. """
    n = len(entry)
    out = np.zeros(n, dtype=bool)
    idx = np.where(entry.values)[0]
    eod_arr = eod_flag.values
    for i in idx:
        j = i
        while j < n and not eod_arr[j]:
            j += 1
        if j < n:
            out[j] = True
    return pd.Series(out, index=entry.index)


def run_amipy_backtest(df: pd.DataFrame, spec: StrategySpec, slippage: float = 0.0005,
                        context_kwargs: dict | None = None) -> dict:
    """ Runs one strategy on one asset through amipy.Amipy and returns stats
    normalized to the same keys backtest.run_backtest() produces, so
    score.py and the CSV/report code can treat either engine interchangeably. """
    context_kwargs = context_kwargs or {}
    ohlc = df[["open", "high", "low", "close", "volume"]]
    ctx = Context(symbol=df.attrs.get("symbol", "UNKNOWN"), **context_kwargs)
    engine = Amipy(ctx, ohlc)

    entry_signal, atr, poi = build_signal(df, spec)
    entry_signal = _first_of_run(entry_signal)
    zeros = pd.Series(False, index=df.index)

    buyprice = ohlc.open * (1 + slippage)
    shortprice = ohlc.open * (1 - slippage)
    sellprice = ohlc.open * (1 - slippage)
    coverprice = ohlc.open * (1 + slippage)

    if spec.exit_type in ("atr_stop_target", "poi_anchored_stop"):
        if spec.exit_type == "atr_stop_target":
            stop_dist = spec.stop_mult * atr
        else:  # poi_anchored_stop: distance from entry to the POI level itself
            if spec.direction == "long":
                stop_dist = (buyprice - poi).clip(lower=0.0)
            else:
                stop_dist = (poi - shortprice).clip(lower=0.0)
        target_dist = spec.target_mult * atr

        stop_ticks = (stop_dist / ctx.tick_size).fillna(0.0).values
        target_ticks = (target_dist / ctx.tick_size).fillna(0.0).values

        if spec.direction == "long":
            buy = entry_signal
            msell = engine.apply_stops_sell_rq(buy, zeros, buyprice, stop_ticks, target_ticks)
            sell = pd.Series(msell.astype(bool), index=df.index)
            buy = amipy.ex_rem(buy, sell, 1)
            short = zeros
            cover = zeros
        else:
            short = entry_signal
            mcover = engine.apply_stops_cover_rq(zeros, short, shortprice, stop_ticks, target_ticks)
            cover = pd.Series(mcover.astype(bool), index=df.index)
            short = amipy.ex_rem(short, cover, 1)
            buy = zeros
            sell = zeros
    elif spec.exit_type == "time_exit":
        if spec.direction == "long":
            raw_buy = entry_signal
            sell = _time_exit_signal(raw_buy, spec.hold_days)
            buy = amipy.ex_rem(raw_buy, sell, 1)
            short = zeros
            cover = zeros
        else:
            raw_short = entry_signal
            cover = _time_exit_signal(raw_short, spec.hold_days)
            short = amipy.ex_rem(raw_short, cover, 1)
            buy = zeros
            sell = zeros
    else:  # eod_exit
        eod_flag = ind.is_last_bar_of_session(df)
        if spec.direction == "long":
            raw_buy = entry_signal
            sell = _eod_exit_signal(raw_buy, eod_flag)
            buy = amipy.ex_rem(raw_buy, sell, 1)
            short = zeros
            cover = zeros
        else:
            raw_short = entry_signal
            cover = _eod_exit_signal(raw_short, eod_flag)
            short = amipy.ex_rem(raw_short, cover, 1)
            buy = zeros
            sell = zeros

    engine.run(buy, short, sell, cover, buyprice, shortprice, sellprice, coverprice)

    if engine.trades is None or len(engine.trades) == 0:
        return {"trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
                "total_return_pct": 0.0, "cagr": 0.0, "sharpe": np.nan,
                "sortino": np.nan, "max_drawdown_pct": 0.0}

    engine.analyze_results_silent(rfr=0.0)
    stats = engine.stats
    closing_trades = engine.trades[engine.trades["direction"].isin(["sell", "cover"])]

    return {
        "trades": int(len(closing_trades)),
        "win_rate": round(float((closing_trades["value"] > 0).mean()), 3) if len(closing_trades) else np.nan,
        "profit_factor": stats.get("pfr", np.nan),
        "total_return_pct": round(stats.get("total_ret", 0.0) * 100, 1),
        "cagr": round(stats.get("cagr", 0.0) * 100, 1),
        "sharpe": stats.get("sharpe", np.nan),
        "sortino": stats.get("sortino", np.nan),
        "max_drawdown_pct": round(stats.get("maxdd", 0.0) * 100, 1),
    }
