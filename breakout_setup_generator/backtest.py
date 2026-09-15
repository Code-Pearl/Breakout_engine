#!/usr/bin/env python3
"""
backtest.py
Turns one StrategySpec + one OHLCV DataFrame into a trade list and a
stats dict. Single-position-at-a-time state machine (no pyramiding, no
overlapping trades) so trade-level % returns compound cleanly into an
equity curve without needing a position-sizing model, UNLESS
position_sizing="fixed_fractional" is requested (see below).

Fill assumptions (kept simple and stated up front, not hidden):
- Breakout level built from bar i's causal indicators; entry checked
  against bar i's close; fill happens at bar i+1's OPEN (next-bar fill,
  no same-bar lookahead).
- Once in a trade, stop/target are checked against that bar's low/high.
  If both stop and target would technically be touched on the same bar,
  the stop is assumed to hit first (conservative).
- time_exit strategies exit at the close of the N-th bar held.
- atr_stop_target strategies ALSO get a hard MAX_HOLD_DAYS_FALLBACK time
  stop so a trade can never run open forever if price just drifts sideways
  between stop and target.

Position sizing (position_sizing="full_compounding" | "fixed_fractional"):
- "full_compounding" (default, unchanged from before): every trade
  compounds its raw % return into 100% of equity. Simple, but can produce
  unrealistic extreme numbers on a real edge over a big historical move
  (documented in the README's Known Limitations and "Real data" sections
  -- a real BTC run showed strategies with returns in the tens of millions
  of percent under this assumption).
- "fixed_fractional": each trade risks a fixed % of CURRENT equity
  (risk_pct, default 10%), sized from the trade's actual stop distance
  (or the entry-bar ATR as a fallback risk proxy for exit types with no
  hard stop, like time_exit/eod_exit). This is the standard "risk-based"
  position sizing model and is immune to the full-compounding blowup --
  a real edge still compounds, just realistically, bounded by how much of
  each trade's equity was actually at risk.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

import grammar as g
import indicators as ind
from generator import StrategySpec


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl_pct: float
    # only populated when position_sizing="fixed_fractional"
    shares: float | None = None
    risk_distance: float | None = None
    equity_before: float | None = None
    pnl_dollars: float | None = None
    equity_return_pct: float | None = None


def build_signal(df: pd.DataFrame, spec: StrategySpec):
    poi_fn = g.POI[spec.poi]["fn"]
    poi = poi_fn(df)
    atr = ind.atr(df, spec.atr_period)
    dist = atr * spec.distance_mult

    if spec.direction == "long":
        level = poi + dist
        signal = df.close > level
    else:
        level = poi - dist
        signal = df.close < level

    if spec.filter_type != "none":
        filt_fn = g.FILTERS[spec.filter_type]["fn"]
        filt = filt_fn(df, spec.filter_param)
        signal = signal & filt.fillna(False)

    return signal.fillna(False), atr, poi


def run_backtest(df: pd.DataFrame, spec: StrategySpec,
                  position_sizing: str = "full_compounding",
                  risk_pct: float = 0.10, starting_equity: float = 100_000.0) -> dict:
    signal, atr, poi = build_signal(df, spec)
    o = df.open.values
    h = df.high.values
    l = df.low.values
    c = df.close.values
    a = atr.values
    p = poi.values
    sig = signal.values
    dates = df.index
    n = len(df)

    uses_stop_target = spec.exit_type in ("atr_stop_target", "poi_anchored_stop")
    fixed_fractional = position_sizing == "fixed_fractional"
    trades: list[Trade] = []
    in_pos = False
    entry_price = stop = target = entry_idx = 0.0
    # time_exit is the only type with a real, user-set hold length; every
    # other exit type still gets the same hard safety-valve time stop
    max_hold = spec.hold_days if spec.exit_type == "time_exit" else g.MAX_HOLD_DAYS_FALLBACK
    buffer_pct = g.POI_STOP_BUFFER_PCT
    eod_flag = ind.is_last_bar_of_session(df).values if spec.exit_type == "eod_exit" else None

    equity = starting_equity   # only meaningfully used when fixed_fractional
    ruined = False             # equity hit the floor -- stop sizing new trades

    i = 1
    while i < n - 1:
        if not in_pos:
            if sig[i] and not np.isnan(a[i]):
                entry_price = o[i + 1]
                entry_idx = i + 1
                if entry_price <= 0 or np.isnan(entry_price):
                    i += 1
                    continue
                if spec.exit_type == "atr_stop_target":
                    if spec.direction == "long":
                        stop = entry_price - spec.stop_mult * a[i]
                        target = entry_price + spec.target_mult * a[i]
                    else:
                        stop = entry_price + spec.stop_mult * a[i]
                        target = entry_price - spec.target_mult * a[i]
                elif spec.exit_type == "poi_anchored_stop":
                    if np.isnan(p[i]):
                        i += 1
                        continue
                    if spec.direction == "long":
                        stop = p[i] * (1 - buffer_pct)
                        target = entry_price + spec.target_mult * a[i]
                        if not (stop < entry_price):
                            i += 1
                            continue  # POI isn't below entry -- not a valid support level here, skip
                    else:
                        stop = p[i] * (1 + buffer_pct)
                        target = entry_price - spec.target_mult * a[i]
                        if not (stop > entry_price):
                            i += 1
                            continue  # POI isn't above entry -- not a valid resistance level here, skip
                in_pos = True
                i += 1
                continue
        else:
            bars_held = i - entry_idx
            exited, exit_price, reason = False, None, None
            if uses_stop_target:
                if spec.direction == "long":
                    if l[i] <= stop:
                        exited, exit_price, reason = True, stop, "stop"
                    elif h[i] >= target:
                        exited, exit_price, reason = True, target, "target"
                else:
                    if h[i] >= stop:
                        exited, exit_price, reason = True, stop, "stop"
                    elif l[i] <= target:
                        exited, exit_price, reason = True, target, "target"
            if not exited and eod_flag is not None and eod_flag[i]:
                exited, exit_price, reason = True, c[i], "eod"
            if not exited and bars_held >= max_hold:
                exited, exit_price, reason = True, c[i], "time"

            if exited:
                if spec.direction == "long":
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                trade_kwargs = dict(
                    entry_idx=entry_idx, exit_idx=i,
                    entry_date=dates[entry_idx], exit_date=dates[i],
                    direction=spec.direction, entry_price=entry_price,
                    exit_price=exit_price, exit_reason=reason, pnl_pct=pnl_pct,
                )

                if fixed_fractional:
                    # risk distance: the actual stop if this exit type has one,
                    # else the entry-bar ATR as a fallback risk proxy
                    if uses_stop_target:
                        risk_distance = abs(entry_price - stop)
                    else:
                        risk_distance = a[entry_idx - 1] if entry_idx - 1 < len(a) and not np.isnan(a[entry_idx - 1]) else None

                    equity_before = equity
                    shares = 0.0
                    pnl_dollars = 0.0
                    if not ruined and risk_distance and risk_distance > 0:
                        shares = (equity * risk_pct) / risk_distance
                        pnl_dollars = shares * (exit_price - entry_price) * (1 if spec.direction == "long" else -1)
                        # exit types with a real stop order (atr_stop_target/poi_anchored_stop)
                        # exit AT the stop price, so their loss is mathematically capped at
                        # exactly risk_pct already. Exit types with no real stop (eod_exit/
                        # time_exit) size off the ATR as a RISK PROXY, not an actual protective
                        # order -- price can move adversely far past that proxy before the
                        # eod/time trigger fires, so cap the loss at all the equity that trade
                        # was allocated (can't lose more than you have) rather than letting it
                        # go arbitrarily negative.
                        pnl_dollars = max(pnl_dollars, -equity_before)
                        equity += pnl_dollars
                        if equity < 1.0:
                            equity = 1.0
                            ruined = True
                    equity_return_pct = pnl_dollars / equity_before if equity_before > 0 else 0.0

                    trade_kwargs.update(shares=shares, risk_distance=risk_distance,
                                        equity_before=equity_before, pnl_dollars=pnl_dollars,
                                        equity_return_pct=equity_return_pct)

                trades.append(Trade(**trade_kwargs))
                in_pos = False
        i += 1

    return summarize_trades(trades, df.index, position_sizing=position_sizing, starting_equity=starting_equity)


def summarize_trades(trades: list[Trade], date_index: pd.DatetimeIndex,
                      position_sizing: str = "full_compounding",
                      starting_equity: float = 100_000.0) -> dict:
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "trades": 0, "win_rate": np.nan, "avg_win_pct": np.nan,
            "avg_loss_pct": np.nan, "profit_factor": np.nan,
            "total_return_pct": 0.0, "cagr": 0.0, "sharpe": np.nan,
            "max_drawdown_pct": 0.0, "trades_list": [],
        }

    # win_rate / avg_win / avg_loss / profit_factor describe the underlying
    # EDGE and stay based on raw price pnl_pct in both sizing modes, so
    # they're comparable across modes -- only the compounding-sensitive
    # stats (total_return_pct/cagr/max_dd/sharpe) change with sizing.
    pnl = np.array([t.pnl_pct for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = len(wins) / n_trades
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    if position_sizing == "fixed_fractional":
        # each trade's ACTUAL equity-fraction change, given risk-based sizing --
        # feeding this into the same cumprod machinery as full_compounding
        # gives a real, bounded equity curve instead of raw-price compounding
        returns_for_curve = np.array([t.equity_return_pct for t in trades])
    else:
        returns_for_curve = pnl

    equity = np.cumprod(1 + returns_for_curve)
    total_return_pct = (equity[-1] - 1) * 100

    years = max((date_index[-1] - date_index[0]).days / 365.25, 0.5)
    cagr = (equity[-1] ** (1 / years) - 1) * 100 if equity[-1] > 0 else -100.0

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min() * 100

    # trade-level sharpe (not bar-level): annualize by avg trades/year
    trades_per_year = n_trades / years
    if returns_for_curve.std() > 0:
        sharpe = (returns_for_curve.mean() / returns_for_curve.std()) * np.sqrt(max(trades_per_year, 1e-6))
    else:
        sharpe = np.nan

    result = {
        "trades": n_trades, "win_rate": win_rate, "avg_win_pct": avg_win * 100,
        "avg_loss_pct": avg_loss * 100, "profit_factor": profit_factor,
        "total_return_pct": total_return_pct, "cagr": cagr, "sharpe": sharpe,
        "max_drawdown_pct": max_dd, "trades_list": trades,
    }
    if position_sizing == "fixed_fractional":
        result["final_equity"] = starting_equity * equity[-1]
    return result
