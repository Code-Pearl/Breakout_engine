#!/usr/bin/env python3
"""
database.py
Tiny JSON-file "database" (no server needed) that persists across runs:

- component_memory: dna -> how many times that idea has been generated,
  so the generator can bias away from over-explored territory next run.
- leaderboard: the best strategies ever seen, so a mediocre run doesn't
  erase yesterday's discoveries.

This is intentionally simple. When this grows into "millions of
strategies," swap this file for sqlite/parquet -- the interface
(load/save/update) stays the same.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "output", "research_db.json")


DEFAULT_TRAINING_LOG = os.path.join(os.path.dirname(__file__), "output", "training_log.jsonl")


def load(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {"component_memory": {}, "leaderboard": [], "runs": 0}
    with open(path, "r") as f:
        return json.load(f)


def save(db: dict, path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(db, f, indent=2, default=str)


def update_component_memory(db: dict, specs: list) -> None:
    mem = db.setdefault("component_memory", {})
    for spec in specs:
        mem[spec.dna] = mem.get(spec.dna, 0) + 1


def update_leaderboard(db: dict, results: list[dict], keep_top: int = 50) -> None:
    """ `results` rows MUST keep their `_spec` field (renamed to `spec` here)
    -- the leaderboard is also ml_rank.py's fallback training source if the
    fuller append-only training log doesn't exist yet, so it needs to carry
    every gene value, not just the summary/label strings. """
    board = db.setdefault("leaderboard", [])
    for r in results:
        entry = {k: v for k, v in r.items() if k != "_spec"}
        if "_spec" in r:
            entry["spec"] = r["_spec"]
        board.append(entry)
    board.sort(key=lambda r: r["robustness_score"], reverse=True)
    # dedupe by strategy_id, keep first (highest scoring) occurrence
    seen = set()
    deduped = []
    for r in board:
        if r["strategy_id"] in seen:
            continue
        seen.add(r["strategy_id"])
        deduped.append(r)
    db["leaderboard"] = deduped[:keep_top]
    db["runs"] = db.get("runs", 0) + 1
    db["last_run_utc"] = datetime.now(timezone.utc).isoformat()


def append_training_log(results: list[dict], path: str = DEFAULT_TRAINING_LOG) -> int:
    """ Append every evaluated strategy (eligible or not -- the ML
    pre-screener in ml_rank.py needs BOTH classes to learn anything) to a
    growing, newline-delimited JSON log. Unlike the leaderboard, this is
    never trimmed, so it accumulates real training data across every run
    of main.py / evolve.py. Returns how many rows were written. """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "a") as f:
        for r in results:
            spec = r.get("_spec") or r.get("spec")
            if spec is None:
                continue
            row = {k: v for k, v in r.items() if k not in ("_spec", "spec") and not k.endswith("_pf")
                   and not k.endswith("_sharpe")}
            row["spec"] = spec
            f.write(json.dumps(row, default=str) + "\n")
            n += 1
    return n


def load_training_log(path: str = DEFAULT_TRAINING_LOG) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
