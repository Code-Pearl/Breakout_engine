#!/usr/bin/env python3
"""
grammar.py
The "language" the generator assembles strategies from, based on the
Mr. Breakouts Formula shape:

    BREAKOUT_LEVEL = POI +/- (ATR x FRACT) + FILTER + EXIT

Every POI and FILTER function is CAUSAL: it is only allowed to use data
up to and including bar i-1 relative to the breakout check on bar i, so
the generated strategies can't cheat with lookahead. That's why every
POI lambda ends in .shift(1) even for things (like an SMA) that are
already backward-looking -- the extra shift makes sure the value used
to build "today's breakout level" was fully known at the START of today.
"""
from __future__ import annotations
import indicators as ind

# ---------------------------------------------------------------------
# POI (Point of Initiation) -- category is used for the "rule of six /
# don't build a strategy that's the same idea three times" check.
# ---------------------------------------------------------------------
POI = {
    "prev_close":        {"category": "price", "fn": lambda df: df.close.shift(1)},
    "prev_high":         {"category": "price", "fn": lambda df: df.high.shift(1)},
    "prev_low":          {"category": "price", "fn": lambda df: df.low.shift(1)},
    "session_open":      {"category": "price", "fn": lambda df: df.open},
    "sma20":             {"category": "trend", "fn": lambda df: ind.sma(df.close, 20).shift(1)},
    "sma50":             {"category": "trend", "fn": lambda df: ind.sma(df.close, 50).shift(1)},
    "donchian_high20":   {"category": "range", "fn": lambda df: ind.donchian_high(df, 20).shift(1)},
    "donchian_low20":    {"category": "range", "fn": lambda df: ind.donchian_low(df, 20).shift(1)},
    "highest_high_10":   {"category": "range", "fn": lambda df: df.high.rolling(10).max().shift(1)},
    "lowest_low_10":     {"category": "range", "fn": lambda df: df.low.rolling(10).min().shift(1)},
    # true intraday HOD/LOD: the running high/low SO FAR TODAY, resetting at
    # the start of every session. On daily bars this collapses to NaN (no
    # prior bar in a one-bar "session"), so it's a harmless no-op there --
    # it only does something real on intraday data
    # (data.load_synthetic_intraday_universe / real intraday CSVs).
    "hod_running":       {"category": "session", "fn": lambda df: ind.running_session_high(df)},
    "lod_running":       {"category": "session", "fn": lambda df: ind.running_session_low(df)},
}

# valid POIs for LONG breakouts vs SHORT breakdowns. Per the workbook's
# explicit "always experiment -- swap the HOD/LOD POIs mutually" advice
# (LONG = HOD + ATR*X is one valid model, but LONG = LOD + ATR*X is a
# SECOND, equally valid one), both the "matching" and the "opposite"
# side of the range are included for each direction, not just the
# obvious pairing.
POI_LONG = ["prev_close", "prev_high", "prev_low", "session_open", "sma20", "sma50",
            "donchian_high20", "donchian_low20", "highest_high_10", "lowest_low_10",
            "hod_running", "lod_running"]
POI_SHORT = ["prev_close", "prev_high", "prev_low", "session_open", "sma20", "sma50",
             "donchian_high20", "donchian_low20", "highest_high_10", "lowest_low_10",
             "hod_running", "lod_running"]

ATR_PERIODS = [10, 14, 20, 30, 40]
DISTANCE_MULTS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

# ---------------------------------------------------------------------
# FILTERS -- market-state / confirmation conditions. "none" is a valid
# choice too (a pure, unfiltered breakout) and is weighted so it still
# gets generated sometimes as a baseline.
# ---------------------------------------------------------------------
FILTERS = {
    "none": {
        "category": "none", "params": [None],
        "fn": None,
    },
    "adx_trend": {
        "category": "trend", "params": [15, 20, 25, 30],
        "fn": lambda df, p: ind.adx(df, 14) > p,
    },
    # mirrors breakout_samples.md CODE_1/CODE_2: FILTER = ADX(25) < Filter_Period.
    # counter-intuitive vs adx_trend on purpose -- only take the breakout while
    # the trend is NOT yet overextended.
    "adx_below": {
        "category": "trend", "params": [30, 35, 40, 45],
        "fn": lambda df, p: ind.adx(df, 25) < p,
    },
    # the workbook's "ADX MASTER FILTER": one fixed, non-optimized rule
    # (ADX_PERIOD=15, ADX_LEVEL=20) meant to be reused unchanged across every
    # strategy/market in a group instead of being tuned per idea -- so unlike
    # every other filter here it deliberately has a single param value, not a
    # range, and generate() should never vary it.
    "adx_master_filter": {
        "category": "trend", "params": [None],
        "fn": lambda df, p: ind.adx(df, 15) > 20,
    },
    "close_above_sma100": {
        "category": "trend", "params": [None],
        "fn": lambda df, p: df.close > ind.sma(df.close, 100),
    },
    "close_below_sma100": {
        "category": "trend", "params": [None],
        "fn": lambda df, p: df.close < ind.sma(df.close, 100),
    },
    "atr_expansion": {
        "category": "volatility", "params": [1.0, 1.1, 1.2, 1.3],
        "fn": lambda df, p: ind.atr(df, 10) > ind.atr(df, 50) * p,
    },
    "atr_contraction": {
        "category": "volatility", "params": [0.7, 0.8, 0.9],
        "fn": lambda df, p: ind.atr(df, 10) < ind.atr(df, 50) * p,
    },
    "volume_spike": {
        "category": "volume", "params": [1.2, 1.5, 2.0],
        "fn": lambda df, p: df.volume > ind.volume_sma(df, 20) * p,
    },
    # approximates the DowBreaker sample's higher/main-timeframe pullback
    # conditions (EntryCond1long / EntryCond2long: distance from a recent
    # high, scaled by volatility) collapsed onto one timeframe: only take the
    # breakout if price hasn't already pulled back too far from its recent
    # high/low, i.e. the move is still "fresh".
    "shallow_pullback_long": {
        "category": "pullback", "params": [0.5, 0.75, 1.0, 1.5],
        "fn": lambda df, p: (ind.donchian_high(df, 10).shift(1) - df.close.shift(1)) < p * ind.atr(df, 14),
    },
    "shallow_pullback_short": {
        "category": "pullback", "params": [0.5, 0.75, 1.0, 1.5],
        "fn": lambda df, p: (df.close.shift(1) - ind.donchian_low(df, 10).shift(1)) < p * ind.atr(df, 14),
    },
    # mirrors breakout_samples.md CODE_2's TimeFilter (Test_Hour buckets):
    # only trade during a specific part of the session. Expressed as
    # session-relative thirds instead of fixed clock times so it works
    # whatever session hours the intraday universe uses. On daily bars
    # bar_of_day_fraction is always 0, so only "time_window_open" ever
    # fires there -- a harmless degenerate case, not a bug.
    "time_window_open": {
        "category": "time_of_day", "params": [None],
        "fn": lambda df, p: ind.bar_of_day_fraction(df) < 0.34,
    },
    "time_window_mid": {
        "category": "time_of_day", "params": [None],
        "fn": lambda df, p: (ind.bar_of_day_fraction(df) >= 0.34) & (ind.bar_of_day_fraction(df) < 0.67),
    },
    "time_window_close": {
        "category": "time_of_day", "params": [None],
        "fn": lambda df, p: ind.bar_of_day_fraction(df) >= 0.67,
    },
}

# a filter is "trend-direction-aware": close_above/shallow_pullback_long work
# for longs, close_below/shallow_pullback_short for shorts. Everything else
# (including the fixed adx_master_filter) applies to both directions.
FILTERS_LONG = ["none", "adx_trend", "adx_below", "adx_master_filter", "close_above_sma100",
                "atr_expansion", "atr_contraction", "volume_spike", "shallow_pullback_long",
                "time_window_open", "time_window_mid", "time_window_close"]
FILTERS_SHORT = ["none", "adx_trend", "adx_below", "adx_master_filter", "close_below_sma100",
                  "atr_expansion", "atr_contraction", "volume_spike", "shallow_pullback_short",
                  "time_window_open", "time_window_mid", "time_window_close"]

# ---------------------------------------------------------------------
# EXITS
# ---------------------------------------------------------------------
# poi_anchored_stop is the workbook's HOD/LOD "secret sauce": instead of an
# ATR-multiple stop, the stop sits right at the POI level itself (one small
# buffer beyond it), since a real key level (prior high/low, Donchian edge...)
# is already a logical support/resistance line. Only a distance-based target
# is still tunable -- the stop itself isn't optimized at all, which is exactly
# the point (near-zero overfitting risk on the risk-management side).
EXIT_TYPES = ["atr_stop_target", "time_exit", "poi_anchored_stop", "eod_exit"]
STOP_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
TARGET_MULTS = [1.5, 2.0, 3.0, 4.0, 5.0]
HOLD_DAYS = [3, 5, 10, 15, 20]
MAX_HOLD_DAYS_FALLBACK = 30  # safety valve so a stop/target trade can't run forever
POI_STOP_BUFFER_PCT = 0.0005  # ~"one tick": small buffer beyond the POI level
# eod_exit mirrors the uploaded samples' literal
# "if marketposition <> 0 then setexitonclose" / "if time = 1500 then sell...
# at close" -- close out at the end of the entry day, no stop/target
# params at all (zero extra tunables, the simplest possible exit). On
# daily bars every bar IS the last bar of its one-bar "session", so this
# degenerates to a 1-bar hold there -- meaningful mainly on intraday data.

DIRECTIONS = ["long", "short"]
