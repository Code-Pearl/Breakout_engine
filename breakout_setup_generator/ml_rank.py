#!/usr/bin/env python3
"""
ml_rank.py
Sprint-5: use accumulated history (database.py's append-only training log)
to predict which freshly-generated strategies are worth backtesting at
all, before actually backtesting them.

This only becomes useful once there's enough history -- a few hundred
rows minimum, ideally a few thousand -- which is why every mode here can
self-bootstrap the training log by generating+backtesting more strategies
if it's too thin. Ranking survivors is the job for ML here, not inventing
strategies from scratch: the model never proposes a new POI/filter/exit
combination, it only scores combinations the grammar already knows how to
produce.

Modes:
    python3 ml_rank.py --bootstrap 500          # top up the training log to >=500 rows
    python3 ml_rank.py --train                    # train + save a model from the current log
    python3 ml_rank.py --evaluate 300 50           # honest recall@K test vs a random baseline
    python3 ml_rank.py --prescreen 500 --keep 50   # generate 500, backtest only the model's top 50
"""
from __future__ import annotations
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import data as data_mod
import database as db_mod
from generator import StrategyGenerator, StrategySpec
from backtest import run_backtest
from score import score_strategy

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "output", "ml_model.joblib")

CATEGORICAL_GENES = ["direction", "poi", "filter_type", "exit_type"]
NUMERIC_GENES = ["atr_period", "distance_mult", "filter_param", "stop_mult", "target_mult", "hold_days"]


def spec_to_features(spec_dict: dict) -> dict:
    """ Flatten one strategy's genes into a feature row. Optional numeric
    genes (filter_param/stop_mult/target_mult/hold_days) are None for many
    exit/filter types -- fill with a sentinel AND keep an explicit
    "has_<gene>" flag, since "missing because not applicable" is different
    information than "missing at zero". """
    row = {g: spec_dict.get(g) for g in CATEGORICAL_GENES}
    for g in NUMERIC_GENES:
        val = spec_dict.get(g)
        row[f"has_{g}"] = val is not None
        row[g] = val if val is not None else -1.0
    return row


def build_dataset(rows: list[dict]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    feats = [spec_to_features(r["spec"]) for r in rows]
    X = pd.DataFrame(feats)
    y_eligible = pd.Series([bool(r["eligible"]) for r in rows], name="eligible")
    y_score = pd.Series([float(r["robustness_score"]) for r in rows], name="robustness_score")
    return X, y_eligible, y_score


def build_pipeline() -> Pipeline:
    numeric_cols = [g for g in NUMERIC_GENES] + [f"has_{g}" for g in NUMERIC_GENES]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_GENES),
        ("num", "passthrough", numeric_cols),
    ])
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                  class_weight="balanced", random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


def bootstrap_training_log(min_rows: int, universe: dict, batch_size: int = 200, seed: int | None = None,
                            min_trades_per_asset: int = 15) -> int:
    """ Top up the training log to at least min_rows by generating and
    backtesting fresh (never-before-tried, thanks to the shared component
    memory) strategies, purely so ml_rank.py works out of the box without
    requiring the person to have already run main.py a bunch of times. """
    current = len(db_mod.load_training_log())
    added = 0
    if current >= min_rows:
        return 0
    print(f"Training log has {current} rows, topping up to {min_rows}...")
    db = db_mod.load()
    while current + added < min_rows:
        gen = StrategyGenerator(component_memory=db.get("component_memory", {}), seed=seed)
        specs = gen.generate(min(batch_size, min_rows - (current + added)))
        rows = []
        for spec in specs:
            per_asset = {sym: run_backtest(df, spec) for sym, df in universe.items()}
            scored = score_strategy(per_asset, min_trades_per_asset=min_trades_per_asset)
            rows.append({"strategy_id": spec.strategy_id, "label": spec.label(), "dna": spec.dna,
                         "n_params": spec.n_params, "_spec": spec.to_dict(), **scored})
        db_mod.update_component_memory(db, specs)
        db_mod.update_leaderboard(db, rows)
        added += db_mod.append_training_log(rows)
        seed = None if seed is None else seed + 1  # avoid regenerating an identical batch
    db_mod.save(db)
    print(f"Training log now has {current + added} rows.")
    return added


def train(universe: dict | None = None, min_rows: int = 500, save: bool = True,
          min_trades_per_asset: int = 15) -> tuple[Pipeline, dict]:
    if universe is not None:
        bootstrap_training_log(min_rows, universe, min_trades_per_asset=min_trades_per_asset)
    rows = db_mod.load_training_log()
    if len(rows) < 50:
        raise RuntimeError(f"Only {len(rows)} rows in the training log -- run with a universe to "
                            f"auto-bootstrap, or run main.py/evolve.py a few times first.")

    X, y_eligible, y_score = build_dataset(rows)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_eligible, test_size=0.25, random_state=0, stratify=y_eligible if y_eligible.nunique() > 1 else None)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)

    # feature importance, mapped back through the one-hot encoder to human labels
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_labels = list(ohe.get_feature_names_out(CATEGORICAL_GENES))
    numeric_cols = NUMERIC_GENES + [f"has_{g}" for g in NUMERIC_GENES]
    all_labels = cat_labels + numeric_cols
    importances = pipe.named_steps["clf"].feature_importances_
    top_features = sorted(zip(all_labels, importances), key=lambda t: t[1], reverse=True)[:10]

    report = {
        "n_rows": len(rows), "n_eligible": int(y_eligible.sum()),
        "train_accuracy": round(train_acc, 3), "test_accuracy": round(test_acc, 3),
        "base_rate": round(float(y_eligible.mean()), 3),  # accuracy a "always predict majority class" model gets
        "top_features": [(name, round(float(imp), 3)) for name, imp in top_features],
    }
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
    return pipe, report


def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model at {MODEL_PATH} -- run `python3 ml_rank.py --train` first.")
    return joblib.load(MODEL_PATH)


def predict_promise(pipe: Pipeline, specs: list[StrategySpec]) -> np.ndarray:
    """ Probability each spec will clear the eligibility bar, predicted from
    its genes alone -- no backtest run. """
    X = pd.DataFrame([spec_to_features(s.to_dict()) for s in specs])
    return pipe.predict_proba(X)[:, 1]


def evaluate_recall_at_k(pipe: Pipeline, universe: dict, n_candidates: int, k: int, seed: int | None = None,
                          min_trades_per_asset: int = 15):
    """ Honest evaluation, not a victory lap: generate n_candidates fresh
    strategies, predict their promise WITHOUT backtesting, then actually
    backtest all of them anyway (evaluation-only cost) to get ground truth.
    Compare: of the true top-K by real robustness_score, how many did the
    model's predicted top-K actually contain, vs. a random draw of K? """
    gen = StrategyGenerator(seed=seed)
    specs = gen.generate(n_candidates)

    promise = predict_promise(pipe, specs)
    model_top_k_idx = set(np.argsort(-promise)[:k])

    rng = np.random.default_rng(seed)
    random_top_k_idx = set(rng.choice(n_candidates, size=k, replace=False))

    true_scores = []
    for spec in specs:
        per_asset = {sym: run_backtest(df, spec) for sym, df in universe.items()}
        true_scores.append(score_strategy(per_asset, min_trades_per_asset=min_trades_per_asset)["robustness_score"])
    true_scores = np.array(true_scores)
    true_top_k_idx = set(np.argsort(-true_scores)[:k])

    model_recall = len(model_top_k_idx & true_top_k_idx) / k
    random_recall = len(random_top_k_idx & true_top_k_idx) / k

    return {
        "n_candidates": n_candidates, "k": k,
        "model_recall_at_k": round(model_recall, 2),
        "random_baseline_recall_at_k": round(random_recall, 2),
        "backtests_needed_with_prescreen": k,
        "backtests_needed_without": n_candidates,
    }


def prescreen_and_backtest(pipe: Pipeline, universe: dict, n_candidates: int, keep: int, seed: int | None = None,
                            min_trades_per_asset: int = 15):
    """ The actual production use: generate n_candidates, predict promise
    for all of them (cheap), backtest only the top `keep` (expensive). """
    gen = StrategyGenerator(seed=seed)
    specs = gen.generate(n_candidates)
    promise = predict_promise(pipe, specs)
    order = np.argsort(-promise)[:keep]
    chosen = [specs[i] for i in order]

    results = []
    for spec, p in zip(chosen, promise[order]):
        per_asset = {sym: run_backtest(df, spec) for sym, df in universe.items()}
        scored = score_strategy(per_asset, min_trades_per_asset=min_trades_per_asset)
        results.append({"strategy_id": spec.strategy_id, "label": spec.label(),
                         "predicted_promise": round(float(p), 3), **scored, "_spec": spec.to_dict()})
    results.sort(key=lambda r: r["robustness_score"], reverse=True)
    return results


def build_argparser():
    p = argparse.ArgumentParser(description="ML pre-screener for the Breakout Setup Generator")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--days", type=int, default=None,
                    help="bars of synthetic history per asset (default: 2600 daily / 650 intraday sessions)")
    p.add_argument("--intraday", action="store_true",
                    help="use the synthetic intraday universe instead of daily bars")
    p.add_argument("--bar-minutes", type=int, default=30, help="intraday bar size in minutes (with --intraday)")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--pattern", type=str, default="*.csv")
    p.add_argument("--bootstrap", type=int, default=0, metavar="MIN_ROWS",
                    help="top up the training log to at least MIN_ROWS rows, then exit")
    p.add_argument("--train", action="store_true", help="train (auto-bootstrapping if needed) and save a model")
    p.add_argument("--evaluate", type=int, nargs=2, metavar=("N", "K"),
                    help="honest recall@K test: generate N, compare model's top-K picks to ground truth")
    p.add_argument("--prescreen", type=int, default=0, metavar="N", help="generate N candidates")
    p.add_argument("--keep", type=int, default=50, help="with --prescreen, only backtest the model's top KEEP")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--min-trades", type=int, default=15,
                    help="minimum trades per asset to count as eligible -- raise for --intraday/--data-dir runs")
    return p


def _load_universe(args):
    return data_mod.load_universe_from_args(
        data_dir=args.data_dir, pattern=args.pattern, intraday=args.intraday,
        days=args.days, bar_minutes=args.bar_minutes, seed=args.seed)


def main():
    args = build_argparser().parse_args()

    if args.bootstrap:
        universe = _load_universe(args)
        bootstrap_training_log(args.bootstrap, universe, seed=args.seed, min_trades_per_asset=args.min_trades)
        return

    if args.train:
        universe = _load_universe(args)
        t0 = time.time()
        pipe, report = train(universe, min_rows=500, min_trades_per_asset=args.min_trades)
        print(f"Trained in {time.time() - t0:.1f}s on {report['n_rows']} rows "
              f"({report['n_eligible']} eligible, base rate {report['base_rate']:.0%}).")
        print(f"Train accuracy: {report['train_accuracy']:.0%}  |  "
              f"Test accuracy: {report['test_accuracy']:.0%}  |  "
              f"(a model that always guessed the majority class would get ~{max(report['base_rate'], 1 - report['base_rate']):.0%})")
        print("Top features:")
        for name, imp in report["top_features"]:
            print(f"  {name:<30} {imp}")
        print(f"\nModel saved to: {MODEL_PATH}")
        return

    if args.evaluate:
        n, k = args.evaluate
        universe = _load_universe(args)
        pipe = load_model() if os.path.exists(MODEL_PATH) else train(universe, min_rows=500, min_trades_per_asset=args.min_trades)[0]
        result = evaluate_recall_at_k(pipe, universe, n, k, seed=args.seed, min_trades_per_asset=args.min_trades)
        print(f"Generated {n} fresh candidates, evaluated the model's top {k} vs. ground truth:")
        print(f"  model recall@{k}:  {result['model_recall_at_k']:.0%}  "
              f"(fraction of the TRUE top {k} the model's predicted top {k} actually contains)")
        print(f"  random recall@{k}: {result['random_baseline_recall_at_k']:.0%}  (a random draw of {k}, for comparison)")
        print(f"  -> pre-screening would have needed {k} backtests instead of {n} to find those strategies "
              f"({k / n:.0%} of the compute)")
        return

    if args.prescreen:
        universe = _load_universe(args)
        pipe = load_model() if os.path.exists(MODEL_PATH) else train(universe, min_rows=500, min_trades_per_asset=args.min_trades)[0]
        results = prescreen_and_backtest(pipe, universe, args.prescreen, args.keep, seed=args.seed, min_trades_per_asset=args.min_trades)
        n_eligible = sum(1 for r in results if r["eligible"])
        print(f"Pre-screened {args.prescreen} candidates, backtested only the model's top {args.keep} "
              f"({args.keep / args.prescreen:.0%} of the compute a full run would need).")
        print(f"{n_eligible}/{len(results)} of those backtested strategies cleared the eligibility bar.\n")
        print("=" * 100)
        print(f"{'RANK':<5}{'SCORE':<7}{'PROMISE':<9}{'ELIG':<6}{'TRADES':<8}{'SETUP'}")
        print("-" * 100)
        for i, r in enumerate(results[: args.top], 1):
            elig = "Y" if r["eligible"] else "n"
            print(f"{i:<5}{r['robustness_score']:<7}{r['predicted_promise']:<9}{elig:<6}"
                  f"{r['total_trades']:<8}{r['label']}")
        print("=" * 100)

        db = db_mod.load()
        db_mod.update_leaderboard(db, results)
        db_mod.save(db)
        db_mod.append_training_log(results)
        return

    print("Nothing to do -- pass one of --bootstrap / --train / --evaluate / --prescreen. See --help.")


if __name__ == "__main__":
    main()
