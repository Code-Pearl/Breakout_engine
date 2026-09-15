#!/usr/bin/env python3
"""
ui_server.py
Local web UI for the Breakout Setup Generator -- same engine as the .bat
menu (main.py / fetch_data.py / backtest.py), exposed over HTTP +
Socket.IO so the ui/ pages can run backtests, stream the log, browse the
leaderboard, and chart strategies with trade markers.

    python ui_server.py              # http://127.0.0.1:5000
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import threading
import time

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO

HERE = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(HERE, "ui")
OUTPUT_DIR = os.path.join(HERE, "output")

app = Flask(__name__, static_folder=UI_DIR, static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_run_lock = threading.Lock()
_run_state = {"running": False, "started": None, "label": ""}

HISTORY_PATH = os.path.join(OUTPUT_DIR, "ui_history.json")
_dl_state: dict = {}  # symbol -> {status, bars, file, msg, updated}


def _load_history() -> list:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _log_history(entry: dict) -> None:
    hist = _load_history()
    hist.insert(0, {"ts": time.strftime("%Y-%m-%d %H:%M"), **entry})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist[:100], f, indent=1)


def _emit_dl(symbol: str) -> None:
    socketio.emit("dl_status", {"symbol": symbol, **_dl_state[symbol]})


# --- log streaming: main.py only prints, so fan print() out to Socket.IO ---

class _SocketWriter(io.TextIOBase):
    def __init__(self):
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                socketio.emit("log", {"ts": time.strftime("%H:%M:%S"),
                                      "msg": line.strip(), "level": "info"})
        return len(s)


def _emit(level, msg):
    socketio.emit("log", {"ts": time.strftime("%H:%M:%S"), "msg": msg, "level": level})


# --- pages / static ---

@app.get("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.get("/strategy.html")
def strategy():
    return send_from_directory(UI_DIR, "strategy.html")


# --- status / data sources ---

@app.get("/api/status")
def status():
    import database as db_mod
    db = db_mod.load()
    import glob as _glob
    cached = sorted(os.path.basename(p) for p in _glob.glob(
        os.path.join(HERE, "data_cache", "*.csv")))
    return jsonify({
        "runs": db.get("runs", 0),
        "ideas_seen": len(db.get("component_memory", {})),
        "leaderboard": len(db.get("leaderboard", [])),
        "training_rows": len(db_mod.load_training_log()),
        "cached_symbols": cached,
        "running": _run_state,
    })


@app.post("/api/fetch")
def fetch():
    import fetch_data as fd
    body = request.get_json(force=True, silent=True) or {}
    interval = body.get("interval", "1d")
    refresh = bool(body.get("refresh", False))
    custom_symbols = [s.strip() for s in body.get("symbols","").split(",") if s.strip()] or None
    if _run_lock.locked():
        return jsonify({"error": "a run is already in progress"}), 409

    def _work():
        import fetch_data as fd
        cache_dir = os.path.join(HERE, "data_cache")
        os.makedirs(cache_dir, exist_ok=True)
        syms = custom_symbols or fd.DEFAULT_SYMBOLS
        period = "2y" if interval == "1h" else "max"
        with _run_lock:
            t0 = time.time()
            _run_state.update(running=True, started=time.strftime("%H:%M:%S"),
                              label=f"fetch {interval} ({len(syms)} symbols)")
            socketio.emit("run_state", _run_state)
            ok, failed = 0, []
            try:
                with contextlib.redirect_stdout(_SocketWriter()):
                    for symbol in syms:
                        path = fd.cache_path(symbol, interval, cache_dir)
                        _dl_state[symbol] = {"status": "downloading", "bars": None,
                                             "file": os.path.basename(path),
                                             "msg": "", "updated": time.strftime("%H:%M:%S")}
                        _emit_dl(symbol)
                        try:
                            if not refresh and fd.is_fresh(path, 1.0):
                                import pandas as _pd
                                n = len(_pd.read_csv(path))
                                _dl_state[symbol].update(status="cached", bars=n,
                                                         msg="fresh cache reused")
                                ok += 1
                            else:
                                df = fd.download(symbol, interval, period)
                                df.to_csv(path, index=False)
                                _dl_state[symbol].update(
                                    status="success", bars=len(df),
                                    msg=f"{df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}")
                                ok += 1
                        except Exception as e:  # noqa: BLE001 -- one symbol must not kill the rest
                            if os.path.exists(path):
                                _dl_state[symbol].update(status="cached",
                                                         msg=f"FAILED, stale cache kept: {e}")
                                ok += 1
                            else:
                                _dl_state[symbol].update(status="error", msg=str(e))
                                failed.append(symbol)
                        _dl_state[symbol]["updated"] = time.strftime("%H:%M:%S")
                        _emit_dl(symbol)
                        st = _dl_state[symbol]
                        print(f"  {symbol}: {st['status']} "
                              f"{st['bars'] or ''} {st['msg']}".strip())
                _log_history({"type": "fetch", "label": interval,
                              "ok": not failed, "duration_s": round(time.time() - t0, 1),
                              "error": f"failed: {', '.join(failed)}" if failed else ""})
                _emit("success", f"fetch done: {ok}/{len(syms)} in data_cache/")
                socketio.emit("run_done", {"ok": not failed})
            except Exception as e:  # noqa: BLE001
                _emit("error", f"fetch failed: {e}")
                socketio.emit("run_done", {"ok": False, "error": str(e)})
            finally:
                _run_state.update(running=False, label="")
                socketio.emit("run_state", _run_state)

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"ok": True})


# --- run a backtest (mirrors the .bat menu options) ---

_DATA_SOURCES = {
    "cache": os.path.join(HERE, "data_cache"),
    "real": os.path.join(HERE, "real_data"),
    "intraday": os.path.join(HERE, "real_intraday_data"),
}


@app.post("/api/run")
def run():
    import main as main_mod
    body = request.get_json(force=True, silent=True) or {}
    if _run_lock.locked():
        return jsonify({"error": "a run is already in progress"}), 409

    src = body.get("data_source", "cache")
    data_dir = _DATA_SOURCES.get(src, _DATA_SOURCES["cache"])
    if src == "custom":
        data_dir = body.get("custom_dir") or os.path.join(HERE, "my_csvs")
    sizing = body.get("position_sizing", "full_compounding")

    ns = argparse.Namespace(
        num=int(body.get("num", 150)), top=int(body.get("top", 15)),
        min_trades=int(body.get("min_trades", 15)),
        position_sizing=sizing, risk_pct=float(body.get("risk_pct", 0.10)),
        seed=body.get("seed"),
        days=None, intraday=False, bar_minutes=30,
        data_dir=data_dir, pattern="*.csv",
        csv_out=os.path.join(OUTPUT_DIR, "latest_run.csv"),
        validate_top=int(body.get("validate_top", 0)),
        validation_out=os.path.join(OUTPUT_DIR, "validation_report.json"),
        export_top=int(body.get("export_top", 0)),
        export_format=body.get("export_format", "all"),
        export_dir=os.path.join(OUTPUT_DIR, "code"),
        market=body.get("market", "YOUR_MARKET"),
        timeframe=body.get("timeframe", "YOUR_TIMEFRAME"),
        amipy_check_top=int(body.get("amipy_check_top", 0)),
        plot_top=int(body.get("plot_top", 10)),
        plot_dir=os.path.join(OUTPUT_DIR, "plots"),
    )

    def _work():
        with _run_lock:
            t0 = time.time()
            label = f"run {ns.num} ideas on {os.path.basename(data_dir)}"
            _run_state.update(running=True, started=time.strftime("%H:%M:%S"), label=label)
            socketio.emit("run_state", _run_state)
            try:
                with contextlib.redirect_stdout(_SocketWriter()):
                    main_mod.run(ns)
                _log_history({"type": "run", "label": label, "ok": True,
                              "duration_s": round(time.time() - t0, 1), "error": ""})
                _emit("success", "run finished -- leaderboard + plots updated")
                socketio.emit("run_done", {"ok": True})
            except Exception as e:  # noqa: BLE001
                _log_history({"type": "run", "label": label, "ok": False,
                              "duration_s": round(time.time() - t0, 1), "error": str(e)})
                _emit("error", f"run failed: {e}")
                socketio.emit("run_done", {"ok": False, "error": str(e)})
            finally:
                _run_state.update(running=False, label="")
                socketio.emit("run_state", _run_state)

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"ok": True})


# --- results ---

def _latest_df():
    path = os.path.join(OUTPUT_DIR, "latest_run.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _find_spec(sid):
    """ Strategy genes for any id: all-time leaderboard first, then the
    never-trimmed training log (so ids pushed off the top-50 board still
    resolve). """
    import database as db_mod
    for entry in db_mod.load().get("leaderboard", []):
        if entry.get("strategy_id") == sid:
            sd = entry.get("spec") or entry.get("_spec")
            if sd:
                return sd
    for entry in reversed(db_mod.load_training_log()):
        if entry.get("strategy_id") == sid:
            sd = entry.get("spec") or entry.get("_spec")
            if sd:
                return sd
    return None


@app.get("/api/history")
def history():
    return jsonify(_load_history())


@app.get("/api/dl_state")
def dl_state():
    import fetch_data as fd
    cache_dir = os.path.join(HERE, "data_cache")
    state = {}
    for s in fd.DEFAULT_SYMBOLS:
        if s in _dl_state:
            state[s] = _dl_state[s]
            continue
        path = fd.cache_path(s, "1d", cache_dir)
        if os.path.exists(path):
            try:
                import pandas as _pd
                n = len(_pd.read_csv(path))
            except (OSError, ValueError):
                n = None
            state[s] = {"status": "cached", "bars": n,
                        "file": os.path.basename(path),
                        "msg": "on disk from an earlier session",
                        "updated": time.strftime("%Y-%m-%d %H:%M",
                                                 time.localtime(os.path.getmtime(path)))}
        else:
            state[s] = {"status": "waiting", "bars": None,
                        "file": os.path.basename(path), "msg": "", "updated": ""}
    return jsonify(state)


@app.get("/api/results")
def results():
    df = _latest_df()
    if df is None:
        return jsonify({"rows": [], "note": "no run yet"})
    cols = ["strategy_id", "label", "dna", "n_params", "robustness_score",
            "eligible", "total_trades", "avg_profit_factor", "avg_sharpe"]
    cols = [c for c in cols if c in df.columns]
    df = df.sort_values("robustness_score", ascending=False)
    return jsonify({"rows": df[cols].head(200).to_dict("records"),
                    "total": len(df)})


@app.get("/api/strategy/<sid>")
def strategy_detail(sid):
    from generator import StrategySpec
    import plot_equity as plot_mod
    df = _latest_df()
    if df is None:
        return jsonify({"error": "no run yet"}), 404
    hit = df[df["strategy_id"] == sid]
    row = None
    if not hit.empty:
        row = hit.iloc[0].to_dict()
        for k, v in list(row.items()):
            if isinstance(v, float) and (pd.isna(v) or v == float("inf")):
                row[k] = None
    else:
        # id from an older run: all-time leaderboard first, then the
        # never-trimmed training log (per-asset PF/Sharpe absent there)
        import database as db_mod
        for entry in db_mod.load().get("leaderboard", []):
            if entry.get("strategy_id") == sid:
                row = {k: v for k, v in entry.items() if k != "spec"}
                break
        if row is None:
            for entry in reversed(db_mod.load_training_log()):
                if entry.get("strategy_id") == sid:
                    row = {k: v for k, v in entry.items() if k != "spec"}
                    break
    if row is None:
        return jsonify({"error": "unknown strategy_id"}), 404
    # full spec lives in research_db leaderboard / training log, not the CSV;
    # re-derive the formula panel from the label + dna instead (no re-run).
    spec = None
    sd = _find_spec(sid)
    if sd:
        spec = StrategySpec(**sd)
    formula = plot_mod.formula_lines(spec) if spec else []
    per_asset = {c[:-len("_trades")]: {
        "trades": row.get(f"{c[:-len('_trades')]}_trades"),
        "return_pct": row.get(f"{c[:-len('_trades')]}_return_pct"),
        "pf": row.get(f"{c[:-len('_trades')]}_pf"),
        "sharpe": row.get(f"{c[:-len('_trades')]}_sharpe"),
    } for c in df.columns if c.endswith("_trades") and c != "total_trades"}
    return jsonify({"row": row, "formula": formula, "per_asset": per_asset})


# --- chart data: bars + trade markers + equity, per strategy + symbol ---

def _universe_for(symbols_dir: str):
    import data as data_mod
    return data_mod.load_csv_universe(symbols_dir)


def _epoch(t) -> int:
    """ LWC wants a unique, ascending time per point. Unix seconds survive both
    daily and intraday data; date-only strings collapse an intraday bar set
    into duplicate timestamps and make the renderer throw. """
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


@app.get("/api/symbols")
def symbols():
    src = request.args.get("src", "cache")
    d = _DATA_SOURCES.get(src, _DATA_SOURCES["cache"])
    try:
        uni = _universe_for(d)
        return jsonify({"symbols": sorted(uni), "src": src})
    except FileNotFoundError as e:
        return jsonify({"symbols": [], "error": str(e)})


@app.get("/api/bars")
def bars():
    src = request.args.get("src", "cache")
    symbol = request.args.get("symbol", "")
    limit = int(request.args.get("limit", 500))
    try:
        uni = _universe_for(_DATA_SOURCES.get(src, _DATA_SOURCES["cache"]))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    if symbol not in uni:
        return jsonify({"error": f"unknown symbol {symbol}"}), 404
    df = uni[symbol] if limit <= 0 else uni[symbol].tail(limit)
    intraday = bool((df.index != df.index.normalize()).any())
    out = [{"time": _epoch(ts), "open": round(float(r.open), 4),
            "high": round(float(r.high), 4), "low": round(float(r.low), 4),
            "close": round(float(r.close), 4),
            "volume": int(r.volume) if pd.notna(r.volume) else 0}
           for ts, r in df.iterrows()]
    return jsonify({"symbol": symbol, "bars": out, "intraday": intraday})


@app.get("/api/trades")
def trades():
    """ Trade markers + equity curve for one strategy on one symbol.

    strategy=<id>&symbol=X&src=cache -- re-backtests a single strategy
    (milliseconds-to-seconds), so markers always match the live data. """
    from backtest import run_backtest
    from generator import StrategySpec
    sid = request.args.get("strategy", "")
    symbol = request.args.get("symbol", "")
    src = request.args.get("src", "cache")
    spec_dict = _find_spec(sid)
    if spec_dict is None:
        return jsonify({"error": "strategy spec not found (run it first)"}), 404
    try:
        uni = _universe_for(_DATA_SOURCES.get(src, _DATA_SOURCES["cache"]))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    if symbol not in uni:
        return jsonify({"error": f"unknown symbol {symbol}"}), 404
    spec = StrategySpec(**spec_dict)
    res = run_backtest(uni[symbol], spec)
    markers = []
    for t in res.get("trades_list") or []:
        side = "long" if t.direction == "long" else "short"
        markers.append({
            "time": _epoch(t.entry_date),
            "position": "belowBar" if side == "long" else "aboveBar",
            "shape": "arrowUp" if side == "long" else "arrowDown",
            "color": "#22c55e" if side == "long" else "#ef4444",
            "text": f"{side} @{round(t.entry_price, 2)}",
        })
        markers.append({
            "time": _epoch(t.exit_date),
            "position": "aboveBar" if side == "long" else "belowBar",
            "shape": "circle",
            "color": "#facc15",
            "text": f"exit {t.exit_reason} {t.pnl_pct:+.2%}",
        })
    markers.sort(key=lambda m: m["time"])
    equity = []
    curve = 1.0
    for t in res.get("trades_list") or []:
        curve *= 1 + t.pnl_pct
        equity.append({"time": _epoch(t.exit_date), "value": round(curve, 4)})
    # several trades can close on the same bar -> collapse to one point per bar
    last_by_time = {p["time"]: p["value"] for p in equity}
    equity = [{"time": k, "value": v} for k, v in sorted(last_by_time.items())]
    return jsonify({"symbol": symbol, "strategy_id": sid,
                    "trades": res["trades"], "markers": markers, "equity": equity})


# --- output files + plots ---

@app.get("/api/files")
def files():
    rows = []
    for root, _, fns in os.walk(OUTPUT_DIR):
        for fn in sorted(fns):
            p = os.path.join(root, fn)
            rows.append({"name": os.path.relpath(p, OUTPUT_DIR),
                         "size_kb": round(os.path.getsize(p) / 1024, 1),
                         "modified": time.strftime("%Y-%m-%d %H:%M",
                                                   time.localtime(os.path.getmtime(p)))})
    return jsonify(rows)


@app.get("/api/files/download/<path:name>")
def file_download(name):
    return send_from_directory(OUTPUT_DIR, name, as_attachment=True)


@app.get("/api/plots/<path:name>")
def plot_file(name):
    return send_from_directory(os.path.join(OUTPUT_DIR, "plots"), name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    print(f"Breakout Generator UI on http://127.0.0.1:{args.port}")
    socketio.run(app, host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
