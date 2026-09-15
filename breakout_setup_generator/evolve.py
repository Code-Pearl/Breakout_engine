#!/usr/bin/env python3
"""
evolve.py
Sprint-4: genetic evolution instead of pure weighted-random generation.

main.py's StrategyGenerator draws each strategy from scratch every time --
good for broad coverage, but it never combines two DIFFERENT ideas that
each showed partial promise. This module runs an actual generational loop:

    population -> evaluate fitness (robustness_score across the 5-asset
    universe, from score.py) -> select parents -> crossover + mutate ->
    next generation

Crossover is uniform, gene-by-gene (POI / atr_period / distance_mult /
filter / exit), each gene independently inherited from parent A or B, with
a validity check afterward (e.g. a POI valid for a "long" parent isn't
always in the "short" pool, so an invalid inherited gene gets re-rolled
rather than silently producing a broken strategy).

Usage:
    python3 evolve.py --generations 15 --population 60 --seed 1
    python3 evolve.py --generations 20 --population 80 --data-dir ./my_csvs
"""
from __future__ import annotations
import argparse
import random
import time
import warnings

import numpy as np
import pandas as pd

import grammar as g
import data as data_mod
import database as db_mod
from generator import StrategySpec, random_poi, random_filter, random_exit, new_strategy_id
from backtest import run_backtest
from score import score_strategy

warnings.filterwarnings("ignore")


def _pick_matching(parent_a, parent_b, exit_type, field_name, options, rng):
    """ For a numeric exit-param gene: prefer inheriting the actual value
    from whichever parent already has this exit_type, else roll fresh. """
    candidates = []
    if parent_a.exit_type == exit_type and getattr(parent_a, field_name) is not None:
        candidates.append(getattr(parent_a, field_name))
    if parent_b.exit_type == exit_type and getattr(parent_b, field_name) is not None:
        candidates.append(getattr(parent_b, field_name))
    return rng.choice(candidates) if candidates else rng.choice(options)


def crossover(parent_a: StrategySpec, parent_b: StrategySpec, rng: random.Random) -> StrategySpec:
    direction = rng.choice([parent_a.direction, parent_b.direction])
    poi_pool = g.POI_LONG if direction == "long" else g.POI_SHORT
    filter_pool = g.FILTERS_LONG if direction == "long" else g.FILTERS_SHORT

    poi_candidate = parent_a.poi if rng.random() < 0.5 else parent_b.poi
    poi = poi_candidate if poi_candidate in poi_pool else rng.choice(poi_pool)

    atr_period = rng.choice([parent_a.atr_period, parent_b.atr_period])
    distance_mult = rng.choice([parent_a.distance_mult, parent_b.distance_mult])

    filter_candidate = parent_a.filter_type if rng.random() < 0.5 else parent_b.filter_type
    if filter_candidate in filter_pool:
        filter_type = filter_candidate
        # inherit the matching param if either parent already used this filter, else re-roll
        param_candidates = [p.filter_param for p in (parent_a, parent_b) if p.filter_type == filter_type]
        filter_param = rng.choice(param_candidates) if param_candidates else rng.choice(g.FILTERS[filter_type]["params"])
    else:
        filter_type = rng.choice(filter_pool)
        filter_param = rng.choice(g.FILTERS[filter_type]["params"])

    exit_type = parent_a.exit_type if rng.random() < 0.5 else parent_b.exit_type
    stop_mult = target_mult = hold_days = None
    if exit_type == "atr_stop_target":
        stop_mult = _pick_matching(parent_a, parent_b, exit_type, "stop_mult", g.STOP_MULTS, rng)
        target_mult = _pick_matching(parent_a, parent_b, exit_type, "target_mult", g.TARGET_MULTS, rng)
    elif exit_type == "poi_anchored_stop":
        target_mult = _pick_matching(parent_a, parent_b, exit_type, "target_mult", g.TARGET_MULTS, rng)
    else:
        hold_days = _pick_matching(parent_a, parent_b, exit_type, "hold_days", g.HOLD_DAYS, rng)

    return StrategySpec(
        strategy_id=new_strategy_id(), direction=direction, poi=poi,
        atr_period=atr_period, distance_mult=distance_mult,
        filter_type=filter_type, filter_param=filter_param,
        exit_type=exit_type, stop_mult=stop_mult, target_mult=target_mult,
        hold_days=hold_days,
    )


def mutate(spec: StrategySpec, rng: random.Random, rate: float = 0.15) -> StrategySpec:
    """ Independently re-roll each gene with probability `rate`. Direction
    mutating is rare-ish (it reshuffles POI/filter pools too) but allowed --
    that's how a long-only lineage can occasionally discover a short idea. """
    direction = spec.direction
    if rng.random() < rate:
        direction = rng.choice(g.DIRECTIONS)

    poi = random_poi(direction) if rng.random() < rate else spec.poi
    if poi not in (g.POI_LONG if direction == "long" else g.POI_SHORT):
        poi = random_poi(direction)

    atr_period = rng.choice(g.ATR_PERIODS) if rng.random() < rate else spec.atr_period
    distance_mult = rng.choice(g.DISTANCE_MULTS) if rng.random() < rate else spec.distance_mult

    if rng.random() < rate:
        filter_type, filter_param = random_filter(direction)
    else:
        filter_pool = g.FILTERS_LONG if direction == "long" else g.FILTERS_SHORT
        if spec.filter_type in filter_pool:
            filter_type, filter_param = spec.filter_type, spec.filter_param
        else:
            filter_type, filter_param = random_filter(direction)

    if rng.random() < rate:
        exit_type, stop_mult, target_mult, hold_days = random_exit()
    else:
        exit_type, stop_mult, target_mult, hold_days = (
            spec.exit_type, spec.stop_mult, spec.target_mult, spec.hold_days)

    return StrategySpec(
        strategy_id=new_strategy_id(), direction=direction, poi=poi,
        atr_period=atr_period, distance_mult=distance_mult,
        filter_type=filter_type, filter_param=filter_param,
        exit_type=exit_type, stop_mult=stop_mult, target_mult=target_mult,
        hold_days=hold_days,
    )


def random_individual(rng: random.Random) -> StrategySpec:
    direction = rng.choice(g.DIRECTIONS)
    filter_type, filter_param = random_filter(direction)
    exit_type, stop_mult, target_mult, hold_days = random_exit()
    return StrategySpec(
        strategy_id=new_strategy_id(), direction=direction, poi=random_poi(direction),
        atr_period=rng.choice(g.ATR_PERIODS), distance_mult=rng.choice(g.DISTANCE_MULTS),
        filter_type=filter_type, filter_param=filter_param,
        exit_type=exit_type, stop_mult=stop_mult, target_mult=target_mult, hold_days=hold_days,
    )


def tournament_select(population, fitness, k, rng):
    contenders = rng.sample(range(len(population)), min(k, len(population)))
    best_idx = max(contenders, key=lambda i: fitness[i])
    return population[best_idx]


class FitnessCache:
    """ GA populations naturally re-explore the same DNA + params repeatedly
    (elitism keeps winners, crossover of two winners often reproduces one of
    them) -- cache by full param key so we don't re-backtest identical
    strategies across generations. Also keeps one representative spec per
    key so the full evaluated set (winners AND rejects) can be dumped for
    ML training, not just the elites that make the final leaderboard. """

    def __init__(self, universe, min_trades_per_asset=15, position_sizing="full_compounding", risk_pct=0.10):
        self.universe = universe
        self.min_trades_per_asset = min_trades_per_asset
        self.position_sizing = position_sizing
        self.risk_pct = risk_pct
        self._cache: dict[tuple, dict] = {}
        self._spec_by_key: dict[tuple, StrategySpec] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, spec: StrategySpec) -> tuple:
        d = spec.to_dict()
        d.pop("strategy_id", None)  # id is random, not a real gene
        return tuple(sorted(d.items()))

    def evaluate(self, spec: StrategySpec) -> dict:
        key = self._key(spec)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        per_asset = {sym: run_backtest(df, spec, position_sizing=self.position_sizing, risk_pct=self.risk_pct)
                     for sym, df in self.universe.items()}
        scored = score_strategy(per_asset, min_trades_per_asset=self.min_trades_per_asset)
        self._cache[key] = scored
        self._spec_by_key[key] = spec
        return scored

    def all_evaluated(self):
        """ Every distinct genotype evaluated this run, (spec, score) pairs --
        for feeding the training log, not just the final leaderboard. """
        return [(self._spec_by_key[k], s) for k, s in self._cache.items()]


def run_evolution(universe, generations, population_size, elite_frac=0.15,
                   mutation_rate=0.15, tournament_k=3, seed=None, min_trades_per_asset=15,
                   position_sizing="full_compounding", risk_pct=0.10,
                   ml_pipe=None, offspring_pool_mult=3):
    """ ml_pipe: an optional trained ml_rank.py model. If provided, each
    generation's offspring aren't generated 1:1 -- a LARGER pool
    (offspring_pool_mult x the number of non-elite slots) is produced via
    crossover/mutation, the model predicts each candidate's promise WITHOUT
    backtesting (cheap), and only the top-predicted subset actually enters
    the next generation's backtest budget. Same number of backtests per
    generation either way; this spends extra (cheap) crossover/mutation
    calls to pick BETTER candidates before spending the backtest budget on
    them, rather than backtesting whatever crossover/mutation happened to
    produce first. """
    rng = random.Random(seed)
    cache = FitnessCache(universe, min_trades_per_asset=min_trades_per_asset,
                          position_sizing=position_sizing, risk_pct=risk_pct)
    n_elite = max(1, int(population_size * elite_frac))
    ml_prescreen = ml_pipe is not None

    population = [random_individual(rng) for _ in range(population_size)]
    history = []
    all_time_best = []

    for gen in range(1, generations + 1):
        scored = [(spec, cache.evaluate(spec)) for spec in population]
        scored.sort(key=lambda t: t[1]["robustness_score"], reverse=True)
        fitness = [s["robustness_score"] for _, s in scored]

        best_spec, best_score = scored[0][0], fitness[0]
        avg_score = float(np.mean(fitness))
        n_eligible = sum(1 for _, s in scored if s["eligible"])
        history.append({"generation": gen, "best_score": best_score, "avg_score": avg_score,
                         "n_eligible": n_eligible, "best_label": best_spec.label()})
        print(f"gen {gen:>3}/{generations}  best={best_score:<6} avg={round(avg_score, 1):<6} "
              f"eligible={n_eligible}/{population_size}   best: {best_spec.label()}")

        all_time_best.extend((spec, s) for spec, s in scored[:n_elite])

        # --- build next generation ------------------------------------
        ranked_pop = [spec for spec, _ in scored]
        next_gen = ranked_pop[:n_elite]  # elitism: carry the best forward unchanged
        n_needed = population_size - n_elite

        if ml_prescreen:
            pool_size = max(n_needed * offspring_pool_mult, n_needed)
            candidate_pool = []
            while len(candidate_pool) < pool_size:
                parent_a = tournament_select(ranked_pop, fitness, tournament_k, rng)
                parent_b = tournament_select(ranked_pop, fitness, tournament_k, rng)
                child = crossover(parent_a, parent_b, rng)
                child = mutate(child, rng, rate=mutation_rate)
                candidate_pool.append(child)
            import ml_rank  # local import: keeps evolve.py usable even without sklearn installed, unless --ml-prescreen is actually used
            promise = ml_rank.predict_promise(ml_pipe, candidate_pool)
            order = np.argsort(-promise)[:n_needed]
            print(f"    ml-prescreen: generated {pool_size} candidates, backtesting only the "
                  f"top {n_needed} by predicted promise (avg predicted promise of chosen: "
                  f"{promise[order].mean():.2f}, of the full pool: {promise.mean():.2f})")
            next_gen += [candidate_pool[i] for i in order]
        else:
            while len(next_gen) < population_size:
                parent_a = tournament_select(ranked_pop, fitness, tournament_k, rng)
                parent_b = tournament_select(ranked_pop, fitness, tournament_k, rng)
                child = crossover(parent_a, parent_b, rng)
                child = mutate(child, rng, rate=mutation_rate)
                next_gen.append(child)
        population = next_gen

    # final generation's fitness for every individual still in play
    final_scored = [(spec, cache.evaluate(spec)) for spec in population]
    all_time_best.extend(final_scored)

    # dedupe all-time-best by param key, keep the best score for each
    best_by_key = {}
    for spec, s in all_time_best:
        key = cache._key(spec)
        if key not in best_by_key or s["robustness_score"] > best_by_key[key][1]["robustness_score"]:
            best_by_key[key] = (spec, s)
    ranked_best = sorted(best_by_key.values(), key=lambda t: t[1]["robustness_score"], reverse=True)

    print(f"\nFitness cache: {cache.hits} hits / {cache.misses} misses "
          f"({cache.hits / max(cache.hits + cache.misses, 1):.0%} reused across generations)")

    return ranked_best, pd.DataFrame(history), cache


def build_argparser():
    p = argparse.ArgumentParser(description="Breakout Setup Generator -- genetic evolution mode")
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--population", type=int, default=60)
    p.add_argument("--elite-frac", type=float, default=0.15)
    p.add_argument("--mutation-rate", type=float, default=0.15)
    p.add_argument("--tournament-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--days", type=int, default=None,
                    help="bars of synthetic history per asset (default: 2600 daily / 650 intraday sessions)")
    p.add_argument("--intraday", action="store_true",
                    help="use the synthetic intraday universe instead of daily bars")
    p.add_argument("--bar-minutes", type=int, default=30, help="intraday bar size in minutes (with --intraday)")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--pattern", type=str, default="*.csv")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--min-trades", type=int, default=15,
                    help="minimum trades per asset to count as eligible -- raise for --intraday/--data-dir runs")
    p.add_argument("--position-sizing", type=str, default="full_compounding",
                    choices=["full_compounding", "fixed_fractional"])
    p.add_argument("--risk-pct", type=float, default=0.10)
    p.add_argument("--ml-prescreen", action="store_true",
                    help="use ml_rank.py's trained model to screen each generation's offspring before "
                         "backtesting them (auto-trains a model first if none exists yet)")
    p.add_argument("--offspring-pool-mult", type=int, default=3,
                    help="with --ml-prescreen, generate this many x more candidates than needed per "
                         "generation and keep only the top-predicted subset to backtest")
    p.add_argument("--csv-out", type=str, default="output/evolution_run.csv")
    p.add_argument("--history-out", type=str, default="output/evolution_history.csv")
    return p


def main():
    args = build_argparser().parse_args()
    t0 = time.time()

    universe = data_mod.load_universe_from_args(
        data_dir=args.data_dir, pattern=args.pattern, intraday=args.intraday,
        days=args.days, bar_minutes=args.bar_minutes, seed=args.seed)

    ml_pipe = None
    if args.ml_prescreen:
        import os
        import ml_rank
        if os.path.exists(ml_rank.MODEL_PATH):
            print(f"Loading existing ML model from {ml_rank.MODEL_PATH} for offspring prescreening...")
            ml_pipe = ml_rank.load_model()
        else:
            print("No trained ML model found -- auto-training one first "
                  "(this bootstraps output/training_log.jsonl if it's thin)...")
            ml_pipe, report = ml_rank.train(universe, min_rows=500, min_trades_per_asset=args.min_trades)
            print(f"  trained on {report['n_rows']} rows, test accuracy {report['test_accuracy']:.0%} "
                  f"(base rate {report['base_rate']:.0%})")

    print(f"\nEvolving {args.population} strategies over {args.generations} generations "
          f"(elite={args.elite_frac:.0%}, mutation={args.mutation_rate:.0%}"
          f"{', ML-prescreened offspring' if ml_pipe is not None else ''})...\n")

    ranked_best, history_df, cache = run_evolution(
        universe, args.generations, args.population, args.elite_frac,
        args.mutation_rate, args.tournament_k, args.seed, min_trades_per_asset=args.min_trades,
        position_sizing=args.position_sizing, risk_pct=args.risk_pct,
        ml_pipe=ml_pipe, offspring_pool_mult=args.offspring_pool_mult)

    print(f"\nDone in {time.time() - t0:.1f}s.\n")
    print("=" * 100)
    print(f"{'RANK':<5}{'SCORE':<7}{'ELIG':<6}{'TRADES':<8}{'PF':<6}{'SHARPE':<8}{'SETUP'}")
    print("-" * 100)
    for i, (spec, s) in enumerate(ranked_best[: args.top], 1):
        elig = "Y" if s["eligible"] else "n"
        print(f"{i:<5}{s['robustness_score']:<7}{elig:<6}{s['total_trades']:<8}"
              f"{s['avg_profit_factor']:<6}{s['avg_sharpe']:<8}{spec.label()}")
    print("=" * 100)

    rows = []
    for spec, s in ranked_best:
        rows.append({"strategy_id": spec.strategy_id, "label": spec.label(), "dna": spec.dna,
                     "n_params": spec.n_params, "_spec": spec.to_dict(), **s})
    pd.DataFrame([{k: v for k, v in r.items() if k != "_spec"} for r in rows]).to_csv(args.csv_out, index=False)
    history_df.to_csv(args.history_out, index=False)
    print(f"\nFull evolved population saved to: {args.csv_out}")
    print(f"Per-generation history saved to: {args.history_out}")

    db = db_mod.load()
    db_mod.update_component_memory(db, [spec for spec, _ in ranked_best])
    db_mod.update_leaderboard(db, rows)
    db_mod.save(db)
    print(f"All-time leaderboard updated: {db_mod.DEFAULT_PATH}")

    # log EVERY genotype the GA actually evaluated this run (winners and
    # rejects alike) -- this is the diversity ml_rank.py's classifier needs,
    # not just the elites that made ranked_best
    all_eval_rows = [{"strategy_id": spec.strategy_id, "label": spec.label(), "dna": spec.dna,
                       "n_params": spec.n_params, "_spec": spec.to_dict(), **s}
                      for spec, s in cache.all_evaluated()]
    n_logged = db_mod.append_training_log(all_eval_rows)
    print(f"Appended {n_logged} rows (full evaluated population) to the training log: "
          f"{db_mod.DEFAULT_TRAINING_LOG}")


if __name__ == "__main__":
    main()
