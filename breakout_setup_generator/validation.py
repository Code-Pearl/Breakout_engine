#!/usr/bin/env python3
"""
validation.py
Deeper robustness tests, run AFTER the cheap cross-asset screen in
score.py has already narrowed the field down. These are too expensive to
run on every generated strategy (each one re-backtests several variants
across the whole universe) -- they're meant for the leaderboard survivors
only, via `main.py --validate-top N`.

Three checks, each answering a different "is this real, or did it just get
lucky" question:

1. WALK-FORWARD: split each asset's history into contiguous windows and
   backtest each window separately. A real edge shows up as "profitable in
   most windows", not "one huge window carrying the whole result".
2. MONTE CARLO: bootstrap-resample the actual trade sequence (with
   replacement) thousands of times to see the RANGE of outcomes the
   strategy could plausibly have produced -- not just the one path that
   happened to occur -- especially how bad the worst-case drawdown gets.
3. NEIGHBOR STABILITY: nudge the strategy's numeric parameters to the next
   value up/down in the grammar's own option list and re-backtest. Straight
   from the workbook: "we want... as many neighbor values with similar
   results [as possible]" -- if performance falls off a cliff one step
   away, that's overfitting, not an edge.
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd

import grammar as g
from generator import StrategySpec
from backtest import run_backtest


def walk_forward(df: pd.DataFrame, spec: StrategySpec, n_windows: int = 4) -> dict:
    n = len(df)
    edges = np.linspace(0, n, n_windows + 1).astype(int)
    window_results = []
    for i in range(n_windows):
        window = df.iloc[edges[i]:edges[i + 1]]
        if len(window) < 100:
            continue
        window_results.append(run_backtest(window, spec))

    n_valid = len(window_results)
    profitable_windows = sum(1 for r in window_results if r["trades"] > 0 and r["total_return_pct"] > 0)
    return {
        "n_windows": n_valid,
        "profitable_windows": profitable_windows,
        "window_consistency": round(profitable_windows / n_valid, 2) if n_valid else 0.0,
        "window_returns_pct": [round(r["total_return_pct"], 1) for r in window_results],
    }


def monte_carlo(trade_pnls: np.ndarray, n_sims: int = 1000, seed: int | None = None) -> dict:
    if len(trade_pnls) < 10:
        return {"n_trades": int(len(trade_pnls)), "insufficient_trades": True}

    rng = np.random.default_rng(seed)
    n_trades = len(trade_pnls)
    finals = np.empty(n_sims)
    max_dds = np.empty(n_sims)
    for s in range(n_sims):
        sample = rng.choice(trade_pnls, size=n_trades, replace=True)
        equity = np.cumprod(1 + sample)
        finals[s] = equity[-1]
        running_max = np.maximum.accumulate(equity)
        max_dds[s] = ((equity - running_max) / running_max).min()

    return {
        "n_trades": int(n_trades),
        "insufficient_trades": False,
        "return_p5_pct": round((np.percentile(finals, 5) - 1) * 100, 1),
        "return_p50_pct": round((np.percentile(finals, 50) - 1) * 100, 1),
        "return_p95_pct": round((np.percentile(finals, 95) - 1) * 100, 1),
        "prob_losing_money": round(float((finals < 1.0).mean()), 3),
        "worst_case_dd_pct": round(float(np.percentile(max_dds, 5)) * 100, 1),   # 5th pct = deep tail
        "median_max_dd_pct": round(float(np.percentile(max_dds, 50)) * 100, 1),
    }


def _neighbors(value, options):
    if value not in options:
        return []
    idx = options.index(value)
    out = []
    if idx > 0:
        out.append(options[idx - 1])
    if idx < len(options) - 1:
        out.append(options[idx + 1])
    return out


def neighbor_stability(universe: dict, spec: StrategySpec, min_trades_per_asset: int = 15) -> dict:
    """ Perturb every tunable numeric param one grid-step at a time and
    re-backtest across the whole universe. Returns how many neighbors
    stayed viable and how much the average profit factor swung. """
    from score import score_strategy  # local import, avoids a circular import at module load

    candidates = []
    for new_val in _neighbors(spec.distance_mult, g.DISTANCE_MULTS):
        s = copy.deepcopy(spec); s.distance_mult = new_val; candidates.append(s)
    for new_val in _neighbors(spec.atr_period, g.ATR_PERIODS):
        s = copy.deepcopy(spec); s.atr_period = new_val; candidates.append(s)
    if spec.exit_type == "atr_stop_target":
        for new_val in _neighbors(spec.stop_mult, g.STOP_MULTS):
            s = copy.deepcopy(spec); s.stop_mult = new_val; candidates.append(s)
        for new_val in _neighbors(spec.target_mult, g.TARGET_MULTS):
            s = copy.deepcopy(spec); s.target_mult = new_val; candidates.append(s)
    elif spec.exit_type == "poi_anchored_stop":
        for new_val in _neighbors(spec.target_mult, g.TARGET_MULTS):
            s = copy.deepcopy(spec); s.target_mult = new_val; candidates.append(s)

    if not candidates:
        return {"n_neighbors": 0, "neighbor_viable_rate": None, "neighbor_avg_pf_mean": None,
                "neighbor_avg_pf_std": None}

    pfs, viable_flags = [], []
    for cand in candidates:
        per_asset = {sym: run_backtest(df, cand) for sym, df in universe.items()}
        scored = score_strategy(per_asset, min_trades_per_asset=min_trades_per_asset)
        pfs.append(scored["avg_profit_factor"])
        viable_flags.append(scored["n_profitable_assets"] >= 2)

    return {
        "n_neighbors": len(candidates),
        "neighbor_viable_rate": round(sum(viable_flags) / len(viable_flags), 2),
        "neighbor_avg_pf_mean": round(float(np.mean(pfs)), 2),
        "neighbor_avg_pf_std": round(float(np.std(pfs)), 2),
    }


def full_validation_report(universe: dict, spec: StrategySpec, primary_symbol: str | None = None,
                            min_trades_per_asset: int = 15) -> dict:
    """ Runs all three checks.
    - Walk-forward runs per-asset (cheap: same backtest, just sliced).
    - Monte Carlo uses one representative asset's actual trade sequence
      (defaults to the first asset in the universe).
    - Neighbor stability runs once across the whole universe.
    """
    wf_by_asset = {sym: walk_forward(df, spec) for sym, df in universe.items()}
    wf_consistencies = [w["window_consistency"] for w in wf_by_asset.values() if w["n_windows"] > 0]
    avg_wf_consistency = round(float(np.mean(wf_consistencies)), 2) if wf_consistencies else None

    mc_symbol = primary_symbol or next(iter(universe))
    mc_backtest = run_backtest(universe[mc_symbol], spec)
    pnls = np.array([t.pnl_pct for t in mc_backtest["trades_list"]])
    mc = monte_carlo(pnls, seed=0)

    nb = neighbor_stability(universe, spec, min_trades_per_asset=min_trades_per_asset)

    verdict = _verdict(avg_wf_consistency, mc, nb)

    return {
        "walk_forward_by_asset": wf_by_asset,
        "avg_walk_forward_consistency": avg_wf_consistency,
        "monte_carlo": mc,
        "monte_carlo_symbol": mc_symbol,
        "neighbor_stability": nb,
        "verdict": verdict,
    }


def _verdict(avg_wf_consistency, mc: dict, nb: dict) -> str:
    """ Cheap, transparent traffic light -- NOT a substitute for reading the
    actual numbers, just a quick triage signal for a big batch. """
    reasons = []
    if avg_wf_consistency is None or avg_wf_consistency < 0.5:
        reasons.append("inconsistent across time windows")
    if mc.get("insufficient_trades"):
        reasons.append("too few trades for a reliable Monte Carlo read")
    elif mc.get("prob_losing_money", 1.0) > 0.35:
        reasons.append("meaningful chance of losing money over the sample")
    if nb.get("neighbor_viable_rate") is not None and nb["neighbor_viable_rate"] < 0.5:
        reasons.append("falls apart on neighboring parameter values")

    if not reasons:
        return "PASS -- consistent across time, neighbors, and simulated re-draws"
    if len(reasons) == 1:
        return f"CAUTION -- {reasons[0]}"
    return "FAIL -- " + "; ".join(reasons)
