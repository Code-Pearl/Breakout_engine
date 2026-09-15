#!/usr/bin/env python3
"""
plot_equity.py
One JPG per strategy: per-asset equity curves on the left, the full
formula + every tunable variable + headline stats written on the right --
so each output file stands alone when you're comparing strategies.

    python3 main.py --data-dir ./data_cache --num 400 --plot-top 10
"""
from __future__ import annotations
import os
import textwrap

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np


def formula_lines(spec) -> list[str]:
    """ The strategy as an explicit formula, one tunable per line. """
    sign = "+" if spec.direction == "long" else "-"
    lines = [
        f"DIRECTION = {spec.direction.upper()}",
        f"LEVEL  = {spec.poi} {sign} (ATR({spec.atr_period}) x {spec.distance_mult})",
        f"ENTRY  = close breaks LEVEL  (next-bar open fill)",
    ]
    if spec.filter_type == "none":
        lines.append("FILTER = none")
    elif spec.filter_param is None:
        lines.append(f"FILTER = {spec.filter_type}")
    else:
        lines.append(f"FILTER = {spec.filter_type}({spec.filter_param})")
    if spec.exit_type == "atr_stop_target":
        lines.append(f"EXIT   = stop {spec.stop_mult}xATR / target {spec.target_mult}xATR")
    elif spec.exit_type == "poi_anchored_stop":
        lines.append(f"EXIT   = stop @POI level / target {spec.target_mult}xATR")
    elif spec.exit_type == "time_exit":
        lines.append(f"EXIT   = close after {spec.hold_days} bars")
    else:
        lines.append("EXIT   = end of session (EOD)")
    lines += [f"DNA    = {spec.dna}", f"PARAMS = {spec.n_params} tunable numbers"]
    return lines


def _curve_from_trades(result: dict, fixed_fractional: bool):
    trades = result.get("trades_list") or []
    if not trades:
        return None, None
    rets = np.array([t.equity_return_pct if fixed_fractional else t.pnl_pct for t in trades],
                    dtype=float)
    dates = [t.exit_date for t in trades]
    return dates, np.cumprod(1 + rets)


def export_top_plots(results: list[dict], universe: dict, n: int, out_dir: str,
                     position_sizing: str = "full_compounding",
                     risk_pct: float = 0.10) -> list[str]:
    """ Re-backtest the top N eligible strategies (to recover per-trade
    equity) and save one annotated JPG each. Returns the file paths. """
    from backtest import run_backtest
    from generator import StrategySpec

    candidates = [r for r in results if r["eligible"]][:n]
    if not candidates:
        print("No eligible strategy -- nothing to plot.")
        return []
    os.makedirs(out_dir, exist_ok=True)
    fixed = position_sizing == "fixed_fractional"

    paths = []
    for rank, r in enumerate(candidates, 1):
        spec = StrategySpec(**r["_spec"])
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_axes([0.05, 0.10, 0.55, 0.80])
        for symbol, df in universe.items():
            res = run_backtest(df, spec, position_sizing=position_sizing, risk_pct=risk_pct)
            dates, curve = _curve_from_trades(res, fixed)
            if curve is None:
                continue
            total = (curve[-1] - 1) * 100
            ax.plot(dates, curve, label=f"{symbol} ({res['trades']}t, {total:+.0f}%)")
        ax.axhline(1.0, color="k", linewidth=0.8)
        ax.set_title(f"#{rank} {spec.strategy_id}  score={r['robustness_score']}  "
                     f"{'ELIGIBLE' if r['eligible'] else 'rejected'}", fontsize=11)
        ax.set_ylabel("equity (start = 1.0)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        per_asset = "\n".join(
            f"  {s}: {r.get(f'{s}_trades')}t  {r.get(f'{s}_return_pct'):+}%  "
            f"PF {r.get(f'{s}_pf')}  Sharpe {r.get(f'{s}_sharpe')}"
            for s in universe)
        text = "\n".join([
            spec.label(), "",
            *formula_lines(spec), "",
            f"robustness={r['robustness_score']}  trades={r['total_trades']}  "
            f"PF={r['avg_profit_factor']}  Sharpe={r['avg_sharpe']}",
            f"sizing={position_sizing}" + (f" risk={risk_pct:.0%}" if fixed else ""), "",
            "PER-ASSET:", per_asset,
        ])
        text = "\n".join(textwrap.fill(line, width=52, break_long_words=False,
                                   break_on_hyphens=False) for line in text.split("\n"))
        fig.text(0.63, 0.95, text, fontsize=8, family="monospace",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
        path = os.path.join(out_dir, f"{rank:02d}_{spec.strategy_id}.jpg")
        fig.savefig(path, format="jpg", dpi=90)
        plt.close(fig)
        paths.append(path)
    print(f"  plots: {len(paths)} file(s) -> {out_dir}/")
    return paths
