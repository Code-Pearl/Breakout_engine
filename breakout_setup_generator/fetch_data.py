#!/usr/bin/env python3
"""
fetch_data.py
Download real OHLCV data (via yfinance) into a local cache folder, in the
exact Date,Open,High,Low,Close,Volume shape data.py's load_csv_universe
expects -- so `main.py --data-dir ./data_cache` just works.

Caching: a symbol is only re-downloaded if its cached CSV is older than
--max-age-days (default 1). Otherwise the cached file is reused untouched.

    python3 fetch_data.py                        # 5-asset universe, daily
    python3 fetch_data.py --refresh              # force re-download all
    python3 fetch_data.py --interval 1h          # hourly bars (Yahoo caps at ~2y)
    python3 fetch_data.py --symbols SPY BTC-USD  # custom set
"""
from __future__ import annotations
import argparse
import os
import re
import time

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

# one real market per asset class, mirroring real_data/'s layout
DEFAULT_SYMBOLS = ["SPY", "GLD", "BTC-USD", "EURUSD=X", "^TNX"]


def safe_name(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", symbol)


def cache_path(symbol: str, interval: str, cache_dir: str) -> str:
    suffix = "hourly" if interval == "1h" else "daily"
    return os.path.join(cache_dir, f"{safe_name(symbol)}_{suffix}.csv")


def is_fresh(path: str, max_age_days: float) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_days * 86400


def download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf  # local import: only needed when actually downloading
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):  # yfinance >= 0.12 nests Ticker level
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    out = pd.DataFrame({
        "Date": df.index.tz_localize(None) if df.index.tz is not None else df.index,
        "Open": df["Open"], "High": df["High"], "Low": df["Low"],
        "Close": df["Close"], "Volume": df["Volume"].fillna(0),
    }).dropna(subset=["Open", "High", "Low", "Close"])
    return out


def fetch(symbols, interval="1d", period=None, cache_dir=CACHE_DIR,
          max_age_days=1.0, refresh=False) -> dict:
    period = period or ("2y" if interval == "1h" else "max")
    os.makedirs(cache_dir, exist_ok=True)
    result = {}
    for symbol in symbols:
        path = cache_path(symbol, interval, cache_dir)
        if not refresh and is_fresh(path, max_age_days):
            result[symbol] = path
            print(f"  {symbol:<10} cached (fresh) -> {path}")
            continue
        try:
            df = download(symbol, interval, period)
            df.to_csv(path, index=False)
            result[symbol] = path
            print(f"  {symbol:<10} downloaded {len(df)} bars "
                  f"({df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}) -> {path}")
        except Exception as e:  # keep what we have; one dead symbol shouldn't kill the rest
            if os.path.exists(path):
                result[symbol] = path
                print(f"  {symbol:<10} FAILED ({e}) -- using stale cache -> {path}")
            else:
                print(f"  {symbol:<10} FAILED ({e}) -- no cache, skipped")
    return result


def main():
    p = argparse.ArgumentParser(description="Download + cache real OHLCV data via yfinance")
    p.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    p.add_argument("--interval", default="1d", choices=["1d", "1h"])
    p.add_argument("--period", default=None, help="yfinance period (default: max for daily, 2y for hourly)")
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--max-age-days", type=float, default=1.0,
                   help="reuse cached CSV if newer than this (default 1 day)")
    p.add_argument("--refresh", action="store_true", help="force re-download everything")
    args = p.parse_args()
    print(f"Caching {len(args.symbols)} symbol(s) [{args.interval}] into {args.cache_dir} ...")
    got = fetch(args.symbols, args.interval, args.period,
                args.cache_dir, args.max_age_days, args.refresh)
    print(f"\nDone: {len(got)}/{len(args.symbols)} available. "
          f"Use with:  python main.py --data-dir {args.cache_dir} --num 400")


if __name__ == "__main__":
    main()
