#!/usr/bin/env python3
"""
generator.py
Assembles valid breakout strategy specs from the grammar in grammar.py.

Two things keep this from being "dumb random":
1. Rule-of-six-ish param cap: every generated strategy only ever has
   ~5-6 tunable numbers (poi, atr_period, distance_mult, filter_param,
   stop_mult/target_mult or hold_days) -- matches the Mr. Breakouts
   Formula's own "3-6 optimization inputs max" rule, just applied to
   generation instead of optimization.
2. DNA-based novelty: every strategy reduces to a short DNA string
   (POI-FILTER-EXIT-DIRECTION). We track how many times each DNA has
   been generated (across this run AND prior runs, via database.py)
   and bias selection away from over-explored regions using weighted
   random choice -- exactly the "component memory" idea. This isn't
   pure random.choice(), it's weighted random over a grammar.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import grammar as g


def _weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]


def random_poi(direction: str) -> str:
    pool = g.POI_LONG if direction == "long" else g.POI_SHORT
    return random.choice(pool)


def random_filter(direction: str) -> tuple[str, "float | None"]:
    pool = g.FILTERS_LONG if direction == "long" else g.FILTERS_SHORT
    # bias toward "none" a bit less than an even split so filtered setups are
    # the common case, but pure breakouts still appear
    weights = [2.0 if f == "none" else 1.0 for f in pool]
    filter_type = _weighted_choice(pool, weights)
    filter_param = random.choice(g.FILTERS[filter_type]["params"])
    return filter_type, filter_param


def random_exit() -> tuple[str, "float | None", "float | None", "int | None"]:
    exit_type = random.choice(g.EXIT_TYPES)
    stop_mult = target_mult = hold_days = None
    if exit_type == "atr_stop_target":
        stop_mult = random.choice(g.STOP_MULTS)
        target_mult = random.choice(g.TARGET_MULTS)
    elif exit_type == "poi_anchored_stop":
        target_mult = random.choice(g.TARGET_MULTS)
    elif exit_type == "time_exit":
        hold_days = random.choice(g.HOLD_DAYS)
    # eod_exit: no extra params at all
    return exit_type, stop_mult, target_mult, hold_days


def new_strategy_id() -> str:
    return f"S{random.randint(0, 9_999_999):07d}"


@dataclass
class StrategySpec:
    strategy_id: str
    direction: str
    poi: str
    atr_period: int
    distance_mult: float
    filter_type: str
    filter_param: Optional[float]
    exit_type: str
    stop_mult: Optional[float] = None
    target_mult: Optional[float] = None
    hold_days: Optional[int] = None
    dna: str = field(default="")
    n_params: int = field(default=0)

    def __post_init__(self):
        self.dna = f"{self.poi}|{self.filter_type}|{self.exit_type}|{self.direction}"
        n = 3  # poi, atr_period, distance_mult
        if self.filter_type != "none":
            n += 1
        if self.exit_type == "atr_stop_target":
            n += 2
        elif self.exit_type == "poi_anchored_stop":
            n += 1  # only target_mult is tunable -- the stop is derived from the POI, not optimized
        elif self.exit_type == "time_exit":
            n += 1
        # eod_exit adds zero extra tunables
        self.n_params = n

    def label(self) -> str:
        d = "LONG" if self.direction == "long" else "SHORT"
        parts = [f"{d}: {self.poi} + {self.distance_mult}xATR({self.atr_period})"]
        if self.filter_type != "none":
            fp = f"({self.filter_param})" if self.filter_param is not None else ""
            parts.append(f"filter={self.filter_type}{fp}")
        if self.exit_type == "atr_stop_target":
            parts.append(f"exit=ATR stop {self.stop_mult}x / target {self.target_mult}x")
        elif self.exit_type == "poi_anchored_stop":
            parts.append(f"exit=stop@POI / target {self.target_mult}xATR")
        elif self.exit_type == "eod_exit":
            parts.append("exit=EOD (close at end of session)")
        else:
            parts.append(f"exit=time({self.hold_days} bars)")
        return "  |  ".join(parts)

    def to_dict(self):
        return asdict(self)


class StrategyGenerator:
    """
    component_memory: dict of dna -> times generated across all runs
    (persisted by database.py). Used to bias generation away from
    ideas that have already been explored heavily.
    """

    def __init__(self, component_memory: Optional[dict] = None, seed: Optional[int] = None):
        self.component_memory = component_memory or {}
        if seed is not None:
            random.seed(seed)

    def _novelty_weight(self, dna: str) -> float:
        seen = self.component_memory.get(dna, 0)
        # weight decays as a component gets explored more; never hits zero
        return 1.0 / (1.0 + seen) ** 0.5

    def generate_one(self, max_tries: int = 25) -> StrategySpec:
        """ Try a few times to land on a not-too-stale DNA before giving up
        and accepting whatever weighted draw comes out (keeps the loop bounded). """
        best_spec, best_weight = None, -1.0
        for _ in range(max_tries):
            spec = self._draw()
            w = self._novelty_weight(spec.dna)
            # accept immediately if it's fresh; otherwise keep the least-stale draw seen
            if w > 0.5 or w > best_weight:
                best_spec, best_weight = spec, w
            if w > 0.5:
                break
        return best_spec

    def _draw(self) -> StrategySpec:
        direction = random.choice(g.DIRECTIONS)
        poi = random_poi(direction)
        atr_period = random.choice(g.ATR_PERIODS)
        distance_mult = random.choice(g.DISTANCE_MULTS)
        filter_type, filter_param = random_filter(direction)
        exit_type, stop_mult, target_mult, hold_days = random_exit()

        return StrategySpec(
            strategy_id=new_strategy_id(), direction=direction, poi=poi,
            atr_period=atr_period, distance_mult=distance_mult,
            filter_type=filter_type, filter_param=filter_param,
            exit_type=exit_type, stop_mult=stop_mult, target_mult=target_mult,
            hold_days=hold_days,
        )

    def generate(self, n: int, dedupe_exact: bool = True, max_per_dna: int = 4) -> list[StrategySpec]:
        """ Generate n strategy specs. dedupe_exact drops strategies whose
        FULL parameter set already appeared in this batch. max_per_dna caps
        how many variants of the same DNA (idea) can appear in one batch,
        so the batch doesn't become 40 near-clones of one idea. """
        out: list[StrategySpec] = []
        seen_exact = set()
        dna_counts: dict[str, int] = {}
        tries = 0
        max_total_tries = n * 20
        while len(out) < n and tries < max_total_tries:
            tries += 1
            spec = self.generate_one()
            key = tuple(sorted(spec.to_dict().items()))
            if dedupe_exact and key in seen_exact:
                continue
            if dna_counts.get(spec.dna, 0) >= max_per_dna:
                continue
            seen_exact.add(key)
            dna_counts[spec.dna] = dna_counts.get(spec.dna, 0) + 1
            out.append(spec)
        return out
