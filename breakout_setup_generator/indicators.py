#!/usr/bin/env python3
"""
indicators.py
Small, dependency-free (pandas/numpy only) indicator library.
Every function is causal (only looks backward) so it's safe to use
directly in signal generation without an extra shift, UNLESS noted.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def stddev(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).std(ddof=0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - prev_close).abs(),
        (df.low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ Wilder's ATR (EMA of True Range). """
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ Wilder's ADX (trend strength, 0-100). """
    up_move = df.high.diff()
    down_move = -df.low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def donchian_high(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df.high.rolling(period, min_periods=period).max()


def donchian_low(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df.low.rolling(period, min_periods=period).min()


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df.volume.rolling(period, min_periods=period).mean()


# --- session-aware indicators (meaningful on intraday bars; harmless
# near-no-ops on daily bars, since each daily bar is its own one-bar
# "session" so there's never a prior bar in the same session to reference) ---

def _session_groups(df: pd.DataFrame):
    return df.groupby(df.index.date)


def running_session_high(df: pd.DataFrame) -> pd.Series:
    """ The running high-of-day (HOD), known only up to the PRIOR bar within
    the same session -- resets at the start of every session instead of
    carrying yesterday's high into today's first bar. This is the causal,
    intraday-bar version of the workbook's HOD POI (true same-day HOD isn't
    knowable until the session ends, so this is "HOD so far today"). """
    return _session_groups(df).high.cummax().groupby(df.index.date).shift(1)


def running_session_low(df: pd.DataFrame) -> pd.Series:
    """ The running low-of-day (LOD), same causality rule as running_session_high. """
    return _session_groups(df).low.cummin().groupby(df.index.date).shift(1)


def bar_of_day_fraction(df: pd.DataFrame) -> pd.Series:
    """ Each bar's position within its session, as a fraction from 0
    (first bar) to ~1 (last bar). On daily data every session has exactly
    one bar, so this is always 0 -- harmless, just means time-of-day
    filters built on this never distinguish anything on daily bars. """
    groups = _session_groups(df)
    rank = groups.cumcount()
    size = groups.high.transform("size")
    denom = (size - 1).replace(0, 1)  # avoid divide-by-zero for one-bar sessions
    return (rank / denom).astype(float)


def is_last_bar_of_session(df: pd.DataFrame) -> pd.Series:
    """ True on the last bar of each trading session (or the very last bar
    of the whole dataset) -- the EOD-exit trigger. On daily data this is
    True on every bar (each day IS its own one-bar session), which makes
    an `eod_exit` strategy on daily data equivalent to a 1-bar hold; that's
    an expected degenerate case, not a bug. """
    dates = pd.Series(df.index.date, index=df.index)
    is_last = dates.ne(dates.shift(-1))
    is_last.iloc[-1] = True
    return is_last
