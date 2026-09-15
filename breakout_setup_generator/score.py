#!/usr/bin/env python3
"""
score.py
Turns "per-asset backtest results for one strategy" into a single
robustness score. This is the anti-bias step: a strategy that only
works on one asset class (or got lucky on one symbol) should score
much lower than one that's mediocre-but-consistent everywhere.

MIN_TRADES_PER_ASSET's default (15) is intentionally low -- it was set
for the daily-bar synthetic MVP, where 10 years of daily bars naturally
produces far fewer signals than the workbook's own "200-400+ trades"
target expects. Real data and --intraday runs routinely produce hundreds
to thousands of trades per strategy (see a real run's numbers in the
README's "Real data" section) -- raise this via the min_trades_per_asset
parameter (or main.py/evolve.py/ml_rank.py's --min-trades flag) once
you're holding a higher-frequency run to a higher bar, rather than
grading it on the low bar meant for sparse daily signals.
"""
from __future__ import annotations
import numpy as np

MIN_TRADES_PER_ASSET = 15
MIN_PROFITABLE_ASSETS = 2  # out of however many assets are in the universe


def score_strategy(per_asset_results: dict[str, dict], min_trades_per_asset: int = MIN_TRADES_PER_ASSET) -> dict:
    n_assets = len(per_asset_results)
    trades_counts = [r["trades"] for r in per_asset_results.values()]
    total_trades = sum(trades_counts)
    min_trades = min(trades_counts) if trades_counts else 0

    profitable = [r for r in per_asset_results.values() if r["trades"] > 0 and r["total_return_pct"] > 0]
    n_profitable = len(profitable)

    pfs = [r["profit_factor"] for r in per_asset_results.values()
           if r["trades"] >= min_trades_per_asset and np.isfinite(r["profit_factor"])]
    sharpes = [r["sharpe"] for r in per_asset_results.values()
               if r["trades"] >= min_trades_per_asset and np.isfinite(r["sharpe"])]

    avg_pf = float(np.mean(pfs)) if pfs else 0.0
    avg_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
    std_sharpe = float(np.std(sharpes)) if len(sharpes) > 1 else 0.0

    eligible = min_trades >= min_trades_per_asset and n_profitable >= MIN_PROFITABLE_ASSETS

    # --- components, each 0-100, then weighted -------------------------
    breadth_score = min(n_profitable / n_assets, 1.0) * 100
    pf_score = min(avg_pf / 2.0, 1.0) * 100          # PF of 2.0+ maxes this out
    sharpe_score = min(max(avg_sharpe, 0) / 1.5, 1.0) * 100
    consistency_score = max(0.0, 100 - std_sharpe * 60)  # penalize wild cross-asset variance
    # "full credit" sample size scales with whatever bar was actually set,
    # not a number tuned only for the daily-bar default
    sample_score = min(total_trades / (n_assets * max(min_trades_per_asset, 1) * 4), 1.0) * 100

    robustness = (
        0.30 * breadth_score +
        0.25 * pf_score +
        0.20 * sharpe_score +
        0.15 * consistency_score +
        0.10 * sample_score
    )
    if not eligible:
        robustness *= 0.4  # heavy penalty, but don't hard-delete -- still visible for debugging

    return {
        "robustness_score": round(robustness, 1),
        "eligible": eligible,
        "n_profitable_assets": n_profitable,
        "n_assets": n_assets,
        "total_trades": total_trades,
        "min_trades_per_asset": min_trades,
        "avg_profit_factor": round(avg_pf, 2),
        "avg_sharpe": round(avg_sharpe, 2),
        "cross_asset_sharpe_std": round(std_sharpe, 2),
    }
