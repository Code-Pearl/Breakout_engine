#!/usr/bin/env python3
"""
data.py
Provides the multi-asset universe the generator backtests every idea
against, so a strategy has to work across DIFFERENT ASSET CLASSES
before it's trusted -- not just curve-fit to one symbol.

Two sources, same shape (DataFrame indexed by date, columns
open/high/low/close/volume, lowercase):

1. SYNTHETIC (default, always available, no network needed): five
   regime-based random-walk markets built to mimic the *character* of
   five different asset classes (equity index, commodity, crypto, fx,
   bond). These are NOT real prices -- they exist so the engine has
   something to run against out of the box and so you can sanity-check
   that a "strategy" isn't just fitting noise. Swap in real data as
   soon as you have it.

2. REAL CSVs: point --data-dir at a folder of CSVs shaped like
   csv_loader.py already expects (Date,Open,High,Low,Close,Volume).
   Every csv in the folder is loaded and used as one asset in the
   universe, symbol name taken from the filename.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

# (name, asset_class, annual_drift, annual_vol, mean_reversion_strength,
#  vol_of_vol, jump_prob, jump_scale)
# mean_reversion_strength = 0 -> pure trending random walk;
# higher -> pulls price back toward a slow-moving anchor (range-y market)
SYNTHETIC_UNIVERSE = [
    ("SIM_EQUITY_INDEX", "equity_index", 0.08, 0.16, 0.02, 0.10, 0.002, 4.0),
    ("SIM_COMMODITY",    "commodity",    0.02, 0.28, 0.10, 0.25, 0.006, 6.0),
    ("SIM_CRYPTO",       "crypto",       0.15, 0.65, 0.03, 0.40, 0.010, 10.0),
    ("SIM_FX",           "fx",           0.00, 0.08, 0.15, 0.08, 0.001, 2.0),
    ("SIM_BOND",         "rates",        0.03, 0.06, 0.20, 0.06, 0.0005, 1.5),
]


def _make_synthetic_asset(name, asset_class, drift, vol, mr_strength,
                           vol_of_vol, jump_prob, jump_scale,
                           dates, bars_per_year, start_price=100.0, seed=0):
    """ dates: the exact DatetimeIndex to generate bars for (daily business
    days, or intraday session timestamps -- see the two loader functions
    below). bars_per_year: how many bars make up one trading year at this
    frequency (252 for daily; bars_per_day * ~252 for intraday), used to
    size each bar's dt so vol/drift stay correctly annualized regardless
    of bar frequency. """
    n = len(dates)
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year

    # stochastic (regime-shifting) volatility so ATR/ADX-style filters
    # have something real to key off, instead of constant-vol noise
    vol_path = np.empty(n)
    v = vol
    for i in range(n):
        v = max(0.02, v + rng.normal(0, vol_of_vol * dt) - 0.5 * (v - vol) * dt)
        vol_path[i] = v

    anchor = start_price
    price = start_price
    closes = np.empty(n)
    for i in range(n):
        shock = rng.normal(0, 1)
        ret = drift * dt + vol_path[i] * np.sqrt(dt) * shock
        # mean reversion pulls log-price back toward a slowly drifting anchor
        anchor *= (1 + drift * dt * 0.5)
        ret += mr_strength * (np.log(anchor) - np.log(price)) * dt
        if rng.random() < jump_prob:
            ret += rng.normal(0, 1) * jump_scale * vol_path[i] * np.sqrt(dt)
        price *= np.exp(ret)
        closes[i] = price

    closes = pd.Series(closes)
    opens = closes.shift(1).fillna(closes.iloc[0])
    intraday_range = closes * (vol_path * np.sqrt(dt)) * rng.uniform(0.4, 1.2, n)
    highs = np.maximum(opens, closes) + np.abs(intraday_range) * rng.uniform(0.3, 1.0, n)
    lows = np.minimum(opens, closes) - np.abs(intraday_range) * rng.uniform(0.3, 1.0, n)
    volume = rng.lognormal(mean=np.log(1_000_000), sigma=0.35, size=n) * (
        1.0 + 2.0 * (np.abs(closes.pct_change().fillna(0))))

    df = pd.DataFrame({
        "open": opens.values, "high": np.asarray(highs), "low": np.asarray(lows),
        "close": closes.values, "volume": np.asarray(volume),
    }, index=dates)
    df.index.name = "datetime"
    df.attrs["symbol"] = name
    df.attrs["asset_class"] = asset_class
    return df


def _daily_index(n_days: int) -> pd.DatetimeIndex:
    # bdate_range(end=..., periods=n) can return n-1 dates when `end` itself
    # isn't a business day (discovered because "today" happened to be a
    # Saturday while testing) -- over-request and take the tail to
    # guarantee exactly n_days regardless of which weekday "today" is.
    return pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days + 7)[-n_days:]


def _intraday_index(n_days: int, bar_minutes: int, session_start: str, session_end: str) -> pd.DatetimeIndex:
    """ One DatetimeIndex spanning n_days trading sessions, with bars every
    bar_minutes between session_start and session_end on each business day
    (weekends skipped -- no overnight/weekend bars, matching the "Regular
    Trading Hours" template in the uploaded EasyLanguage samples). """
    days = _daily_index(n_days)
    per_day_times = pd.date_range(session_start, session_end, freq=f"{bar_minutes}min").time
    stamps = [pd.Timestamp.combine(d.date(), t) for d in days for t in per_day_times]
    return pd.DatetimeIndex(stamps)


def load_synthetic_universe(n_days: int = 2600, seed: int = 42) -> dict[str, pd.DataFrame]:
    """ Returns {symbol: ohlcv_df} for the 5-asset-class synthetic universe,
    one bar per trading day. """
    dates = _daily_index(n_days)
    universe = {}
    for i, spec in enumerate(SYNTHETIC_UNIVERSE):
        name = spec[0]
        universe[name] = _make_synthetic_asset(*spec, dates=dates, bars_per_year=252, seed=seed + i)
    return universe


def load_synthetic_intraday_universe(n_days: int = 650, bar_minutes: int = 30,
                                       session_start: str = "09:30", session_end: str = "15:30",
                                       seed: int = 42) -> dict[str, pd.DataFrame]:
    """ Returns {symbol: ohlcv_df} for the same 5-asset-class universe, but
    at intraday bar frequency within regular trading hours -- unlocks the
    session-aware components (running HOD/LOD, time-of-day filters, EOD
    exit) that only make sense within a single trading day. Defaults to
    ~2.5 years (650 sessions) rather than 10, since the state-machine
    backtester is a Python loop and this is bars_per_day x more bars per
    asset per strategy; raise --days once you've confirmed things work. """
    dates = _intraday_index(n_days, bar_minutes, session_start, session_end)
    bars_per_day = len(pd.date_range(session_start, session_end, freq=f"{bar_minutes}min"))
    bars_per_year = bars_per_day * 252
    universe = {}
    for i, spec in enumerate(SYNTHETIC_UNIVERSE):
        name = spec[0]
        universe[name] = _make_synthetic_asset(*spec, dates=dates, bars_per_year=bars_per_year, seed=seed + i)
        universe[name].attrs["bars_per_day"] = bars_per_day
    return universe


def load_csv_universe(csv_dir: str, pattern: str = "*.csv") -> dict[str, pd.DataFrame]:
    """ Load every csv in csv_dir (same shape csv_loader.py expects) as one
    asset each. Symbol taken from the filename. """
    paths = sorted(glob.glob(os.path.join(csv_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSVs matching {pattern!r} found in {csv_dir}")

    universe = {}
    for path in paths:
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if c in ("date", "datetime", "time", "timestamp")), None)
        if date_col is None:
            raise ValueError(f"{path}: couldn't find a date column")
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}")
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        df.index.name = "datetime"
        df = df[~df.index.duplicated(keep="last")][REQUIRED_COLS].dropna(how="any")
        symbol = os.path.splitext(os.path.basename(path))[0]
        df.attrs["symbol"] = symbol
        df.attrs["asset_class"] = "user_supplied"
        universe[symbol] = df
    return universe


def load_universe_from_args(data_dir: str | None = None, pattern: str = "*.csv",
                             intraday: bool = False, days: int | None = None,
                             bar_minutes: int = 30, session_start: str = "09:30",
                             session_end: str = "15:30", seed: int | None = None) -> dict[str, pd.DataFrame]:
    """ One shared entry point for "which universe should this run use",
    so main.py / evolve.py / ml_rank.py don't each reimplement the same
    data-dir vs. synthetic vs. synthetic-intraday branching. Prints a short
    note about what it loaded (real data vs. illustrative synthetic). """
    if data_dir:
        universe = load_csv_universe(data_dir, pattern)
        print(f"Loaded {len(universe)} asset(s) from {data_dir}")
        return universe

    if intraday:
        universe = load_synthetic_intraday_universe(
            n_days=days or 650, bar_minutes=bar_minutes,
            session_start=session_start, session_end=session_end, seed=seed or 42)
        bars_per_day = next(iter(universe.values())).attrs.get("bars_per_day")
        print(f"Using synthetic INTRADAY {len(universe)}-asset-class universe "
              f"({', '.join(universe.keys())}), {bar_minutes}min bars, "
              f"{session_start}-{session_end} session ({bars_per_day} bars/day)")
        print("  (synthetic = illustrative, NOT real prices -- swap in --data-dir for real data)")
        return universe

    universe = load_synthetic_universe(n_days=days or 2600, seed=seed or 42)
    print(f"Using synthetic {len(universe)}-asset-class universe ({', '.join(universe.keys())})")
    print("  (synthetic = illustrative, NOT real prices -- swap in --data-dir for real data)")
    return universe
