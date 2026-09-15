#!/usr/bin/env python3
"""
main.py
Breakout Setup Generator (BSG) -- MVP

    python3 main.py                          # generate 150 ideas, synthetic 5-asset universe
    python3 main.py --num 500 --top 20
    python3 main.py --data-dir ./my_csvs      # use your own CSVs instead of synthetic data
    python3 main.py --seed 7 --num 300 --csv-out output/latest_run.csv
    python3 main.py --num 300 --validate-top 10       # + walk-forward/Monte Carlo/neighbor checks on the top 10
    python3 main.py --num 300 --export-top 5          # + EasyLanguage-style code for the top 5
    python3 main.py --num 300 --amipy-check-top 5     # + cross-check the top 5 against your amipy.py engine

What it does, in order:
  1. Load a multi-asset universe (5 asset classes by default).
  2. Generate N candidate breakout strategies from the grammar, biased
     away from ideas already heavily explored in previous runs.
  3. Backtest every strategy on EVERY asset independently.
  4. Score each strategy for cross-asset robustness (not just profit).
  5. Rank, print a leaderboard, save full results + persist memory.
  6. (optional, --validate-top) Run deeper robustness tests on the top N
     eligible strategies: walk-forward windows, Monte Carlo trade
     resampling, neighbor-parameter stability.
  7. (optional, --export-top) Export the top N eligible strategies as
     EasyLanguage-style starting-point code.
"""
from __future__ import annotations
import argparse
import time
import warnings

import pandas as pd

import data as data_mod
import database as db_mod
from generator import StrategyGenerator, StrategySpec
from backtest import run_backtest
from score import score_strategy
import validation as validation_mod
import export_easylanguage as el_mod
import export_afl as afl_mod
import export_pinescript as pine_mod
import amipy_backtest

warnings.filterwarnings("ignore")


def run(args):
    t0 = time.time()

    universe = data_mod.load_universe_from_args(
        data_dir=args.data_dir, pattern=args.pattern, intraday=args.intraday,
        days=args.days, bar_minutes=args.bar_minutes, seed=args.seed)

    db = db_mod.load()
    print(f"Research memory: run #{db.get('runs', 0)}, "
          f"{len(db.get('component_memory', {}))} distinct ideas seen before, "
          f"{len(db.get('leaderboard', []))} strategies on the all-time leaderboard\n")

    gen = StrategyGenerator(component_memory=db.get("component_memory", {}), seed=args.seed)
    specs = gen.generate(args.num)
    print(f"Generated {len(specs)} candidate strategies. Backtesting across "
          f"{len(universe)} assets each ({len(specs) * len(universe)} backtests total)...")

    results = []
    for spec in specs:
        per_asset = {}
        for symbol, df in universe.items():
            per_asset[symbol] = run_backtest(df, spec, position_sizing=args.position_sizing, risk_pct=args.risk_pct)
        scored = score_strategy(per_asset, min_trades_per_asset=args.min_trades)
        row = {
            "strategy_id": spec.strategy_id,
            "label": spec.label(),
            "dna": spec.dna,
            "n_params": spec.n_params,
            **scored,
        }
        # keep per-asset headline numbers for the CSV / detail view
        for symbol, r in per_asset.items():
            row[f"{symbol}_trades"] = r["trades"]
            row[f"{symbol}_return_pct"] = round(r["total_return_pct"], 1)
            row[f"{symbol}_pf"] = round(r["profit_factor"], 2) if r["profit_factor"] not in (float("inf"),) else 999.0
            row[f"{symbol}_sharpe"] = round(r["sharpe"], 2) if pd.notna(r["sharpe"]) else None
        row["_spec"] = spec.to_dict()
        results.append(row)

    results.sort(key=lambda r: r["robustness_score"], reverse=True)

    db_mod.update_component_memory(db, specs)
    db_mod.update_leaderboard(db, results)
    db_mod.save(db)
    n_logged = db_mod.append_training_log(results)

    elapsed = time.time() - t0
    n_eligible = sum(1 for r in results if r["eligible"])
    print(f"\nDone in {elapsed:.1f}s. {n_eligible}/{len(results)} strategies passed the "
          f"minimum-trades / cross-asset-profitability bar.")
    print(f"Appended {n_logged} rows to the training log (for ml_rank.py): {db_mod.DEFAULT_TRAINING_LOG}\n")

    print_leaderboard(results[: args.top])

    out_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_spec"} for r in results])
    csv_path = args.csv_out
    out_df.to_csv(csv_path, index=False)
    print(f"\nFull results ({len(out_df)} strategies) saved to: {csv_path}")
    print(f"All-time leaderboard persisted to: {db_mod.DEFAULT_PATH}")

    if args.validate_top > 0:
        run_validation(results, universe, args.validate_top, args.validation_out, args.min_trades)

    if args.export_top > 0:
        run_export(results, args.export_top, args.export_dir, args.market, args.timeframe, args.export_format)

    if args.amipy_check_top > 0:
        run_amipy_check(results, universe, args.amipy_check_top)

    if args.plot_top > 0:
        run_plots(results, universe, args.plot_top, args.plot_dir,
                  args.position_sizing, args.risk_pct)


def run_amipy_check(results, universe, n):
    print(f"\nCross-checking the top {n} eligible strategies against the amipy.py engine "
          f"(real position sizing/margin/commission, Sortino) as a second opinion...")
    candidates = [r for r in results if r["eligible"]][:n]
    if not candidates:
        print("  No strategy cleared the eligibility bar -- nothing to cross-check.")
        return

    print("-" * 100)
    print(f"{'STRATEGY':<12}{'ASSET':<18}{'CUSTOM trades/ret%':<22}{'AMIPY trades/ret%/sortino'}")
    print("-" * 100)
    for r in candidates:
        spec = StrategySpec(**r["_spec"])
        print(spec.label())
        for symbol, df in universe.items():
            custom_trades = r.get(f"{symbol}_trades")
            custom_ret = r.get(f"{symbol}_return_pct")
            a = amipy_backtest.run_amipy_backtest(df, spec)
            sortino = a.get("sortino")
            print(f"  {symbol:<16}custom: {custom_trades}t / {custom_ret}%      "
                  f"amipy: {a['trades']}t / {a['total_return_pct']}% / sortino={sortino}")
        print("-" * 100)
    print("Large disagreements between the two engines on the same strategy/asset are a caution "
          "signal (fragility to fill/timing assumptions), not just noise -- see README.")


def run_validation(results, universe, n, out_path, min_trades):
    print(f"\nRunning deeper robustness validation (walk-forward / Monte Carlo / "
          f"neighbor stability) on the top {n} eligible strategies...")
    candidates = [r for r in results if r["eligible"]][:n]
    if not candidates:
        print("  No strategy cleared the eligibility bar -- nothing to validate. "
              "Try a bigger --num, or a lower --min-trades.")
        return

    reports = {}
    print("-" * 100)
    for r in candidates:
        spec = StrategySpec(**r["_spec"])
        report = validation_mod.full_validation_report(universe, spec, min_trades_per_asset=min_trades)
        reports[spec.strategy_id] = report
        print(f"{spec.strategy_id}  score={r['robustness_score']:<6} {spec.label()}")
        print(f"    walk-forward consistency: {report['avg_walk_forward_consistency']}   "
              f"|  MC prob. losing money: {report['monte_carlo'].get('prob_losing_money')}   "
              f"|  neighbor viable rate: {report['neighbor_stability'].get('neighbor_viable_rate')}")
        print(f"    verdict: {report['verdict']}")
        print("-" * 100)

    import json
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"Full validation report saved to: {out_path}")


def run_export(results, n, out_dir, market, timeframe, formats):
    candidates = [r for r in results if r["eligible"]][:n]
    if not candidates:
        print("\nNo strategy cleared the eligibility bar -- nothing to export.")
        return
    specs = [StrategySpec(**r["_spec"]) for r in candidates]

    exporters = {
        "easylanguage": (el_mod, "easylanguage"),
        "afl": (afl_mod, "afl"),
        "pinescript": (pine_mod, "pinescript"),
    }
    chosen = list(exporters.keys()) if formats == "all" else [f.strip() for f in formats.split(",")]

    print(f"\nExporting {len(specs)} strategy/strategies as: {', '.join(chosen)}")
    for fmt in chosen:
        if fmt not in exporters:
            print(f"  skipping unknown format '{fmt}' (choices: easylanguage, afl, pinescript, all)")
            continue
        mod, subdir = exporters[fmt]
        paths = mod.export_top_strategies(specs, f"{out_dir}/{subdir}", market=market, timeframe=timeframe)
        print(f"  {fmt}: {len(paths)} file(s) -> {out_dir}/{subdir}/")


def run_plots(results, universe, n, out_dir, position_sizing, risk_pct):
    print(f"\nSaving equity-curve JPGs (with annotated formulas) for the top {n} eligible strategies...")
    try:
        import plot_equity as plot_mod
    except ImportError:
        print("  matplotlib not installed -- run `pip install matplotlib` to enable --plot-top.")
        return
    plot_mod.export_top_plots(results, universe, n, out_dir, position_sizing, risk_pct)


def print_leaderboard(rows):
    print("=" * 100)
    print(f"{'RANK':<5}{'SCORE':<7}{'ELIG':<6}{'TRADES':<8}{'PF':<6}{'SHARPE':<8}{'SETUP'}")
    print("-" * 100)
    for i, r in enumerate(rows, 1):
        elig = "Y" if r["eligible"] else "n"
        print(f"{i:<5}{r['robustness_score']:<7}{elig:<6}{r['total_trades']:<8}"
              f"{r['avg_profit_factor']:<6}{r['avg_sharpe']:<8}{r['label']}")
    print("=" * 100)


def build_argparser():
    p = argparse.ArgumentParser(description="Breakout Setup Generator (MVP)")
    p.add_argument("--num", type=int, default=150, help="how many strategies to generate")
    p.add_argument("--top", type=int, default=15, help="how many to print in the leaderboard")
    p.add_argument("--min-trades", type=int, default=15,
                    help="minimum trades per asset to count as eligible (default 15, the daily-bar MVP bar -- "
                         "raise this for --intraday or --data-dir runs, which produce far more signals)")
    p.add_argument("--position-sizing", type=str, default="full_compounding",
                    choices=["full_compounding", "fixed_fractional"],
                    help="full_compounding (default, simple, can produce unrealistic extreme returns on a real "
                         "edge over a big historical move) or fixed_fractional (risk a fixed %% of equity per "
                         "trade, sized from the actual stop distance -- more realistic, still bounded)")
    p.add_argument("--risk-pct", type=float, default=0.10,
                    help="with --position-sizing fixed_fractional, %% of equity risked per trade (default 0.10 = 10%%)")
    p.add_argument("--seed", type=int, default=None, help="random seed (also seeds synthetic data if used)")
    p.add_argument("--days", type=int, default=None,
                    help="bars of synthetic history per asset (default: 2600 daily / 650 intraday sessions)")
    p.add_argument("--intraday", action="store_true",
                    help="use the synthetic intraday universe (30min bars, RTH session) instead of daily bars -- "
                         "unlocks hod_running/lod_running POIs, time_window_* filters, and eod_exit")
    p.add_argument("--bar-minutes", type=int, default=30, help="intraday bar size in minutes (with --intraday)")
    p.add_argument("--data-dir", type=str, default=None, help="folder of real OHLCV CSVs to use instead of synthetic data")
    p.add_argument("--pattern", type=str, default="*.csv", help="glob pattern inside --data-dir")
    p.add_argument("--csv-out", type=str, default="output/latest_run.csv", help="where to save full results")
    p.add_argument("--validate-top", type=int, default=0,
                    help="run walk-forward / Monte Carlo / neighbor-stability checks on the top N eligible strategies (0 = skip)")
    p.add_argument("--validation-out", type=str, default="output/validation_report.json",
                    help="where to save the validation report")
    p.add_argument("--export-top", type=int, default=0,
                    help="export the top N eligible strategies as starting-point code (0 = skip)")
    p.add_argument("--export-format", type=str, default="all",
                    help="'all', or a comma-separated subset of: easylanguage,afl,pinescript")
    p.add_argument("--export-dir", type=str, default="output/code",
                    help="folder to write exported code files to (one subfolder per format)")
    p.add_argument("--market", type=str, default="YOUR_MARKET", help="market name to put in exported code headers")
    p.add_argument("--timeframe", type=str, default="YOUR_TIMEFRAME", help="timeframe to put in exported code headers")
    p.add_argument("--amipy-check-top", type=int, default=0,
                     help="cross-check the top N eligible strategies against the amipy.py engine as a second opinion (0 = skip)")
    p.add_argument("--plot-top", type=int, default=0,
                     help="save equity-curve JPGs with annotated formulas for the top N eligible strategies (0 = skip)")
    p.add_argument("--plot-dir", type=str, default="output/plots",
                     help="folder to write equity-curve JPGs to")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
