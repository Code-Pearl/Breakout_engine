# Breakout Setup Generator (BSG) — v2

A working, small research tool that:

1. **Generates** breakout strategy ideas from a grammar (POI + ATR distance +
   optional filter + exit), inspired by the `BREAKOUT_LEVEL = POI ± (ATR×FRACT) + FILTER + EXIT`
   shape in your uploaded material.
2. **Backtests** every idea across **5 different asset classes at once**
   (equity index, commodity, crypto, fx, rates) so a "strategy" has to
   survive multiple, structurally different markets before it's trusted —
   not just one symbol it accidentally fits.
3. **Scores** each idea for cross-asset robustness (not raw profit) and
   ranks them.
4. **Remembers.** A small JSON file (`output/research_db.json`) tracks
   which ideas have already been explored and keeps an all-time
   leaderboard, so every run builds on the last one instead of starting
   from zero.

**v2 adds a full local web interface** (`ui_server.py` + `ui/`) over the same
engine, plus live Yahoo Finance data downloads (`fetch_data.py`). You can run a
backtest, watch it stream through real-time logs, browse the all-time
leaderboard, drill into any strategy's interactive candlestick + equity chart
(lightweight-charts v5), and download end-to-end data per symbol — all from a
browser, no terminal needed.

This is intentionally a small research tool at its core: the web layer is a
thin UI over the same engine, so everything below applies unchanged — the CLI
pipeline (`main.py`, `evolve.py`, ...) still works exactly as documented, and
the interface just exposes it (plus live data) in a browser.

## Quick start

### Web interface (v2) — recommended

```bash
cd breakout_setup_generator
python3 ui_server.py                      # serves the UI on http://127.0.0.1:5000
```

On Windows you can just double-click **`RUN_UI.bat`** — it finds Python,
picks the first free port 5000–5009, waits for the server, then opens your
browser. (Start more than one instance of the tool side-by-side on different
ports if you like.)

What the interface gives you:

- **Live Data** page — download real OHLCV from Yahoo Finance (daily full
  history or hourly ~2y) for the default 5-asset universe, **or add any
  ticker you like** (`AAPL, MSFT, GC=F, ...`) straight from the page. Each
  symbol streams its own real-time status (downloading / cached / error) into
  the download backlog table.
- **Runs** (the dashboard) — configure and launch a backtest (ideas, top,
  min-trades, position sizing, risk %, data source, validation/export/plot
  counts), watch stdout stream live via Socket.IO, then browse the all-time
  leaderboard below it — one row per strategy; click any row to open its
  **interactive chart** (candles, entry/exit markers, and the cumulative
  equity curve in a synced top pane).
- **Backlog / Output Files** pages — every fetch/run logged to
  `output/ui_history.json`, every generated file/plot downloadable.

CLI still works the same as ever:

```bash
python3 main.py                       # 150 ideas, synthetic 5-asset universe
python3 main.py --num 500 --top 25    # bigger batch
python3 main.py --seed 7              # reproducible run
```

Output:
- A ranked leaderboard printed to the terminal
- `output/latest_run.csv` — every strategy generated this run, with
  per-asset trade counts, returns, profit factor, and Sharpe
- `output/research_db.json` — persistent memory across runs

## Using your own data instead of the synthetic universe

The synthetic data (`SIM_EQUITY_INDEX`, `SIM_COMMODITY`, `SIM_CRYPTO`,
`SIM_FX`, `SIM_BOND`) is regime-based random-walk data built to have the
*character* of five asset classes (different vol, drift, mean-reversion,
jump risk) — it exists purely so the tool works out of the box and you can
sanity-check the engine. **It is not real price history.**

Drop real OHLCV CSVs (same `Date,Open,High,Low,Close,Volume` shape your
`csv_loader.py` already uses) into a folder — ideally 5 real markets from
different asset classes (e.g. an index future, a commodity, a crypto pair,
an FX pair, and a single stock or bond ETF) — and point at it:

```bash
python3 main.py --data-dir ./my_real_data --num 300
```

Every CSV in that folder becomes one asset in the universe. The symbol is the
**full filename stem**: `AAPL_daily.csv` loads as `AAPL_daily` and
`AAPL_hourly.csv` as `AAPL_hourly` — keep the `_daily` / `_hourly` suffixes and
both timeframes load as two independent assets instead of overwriting each other
(`data.py` deliberately does *not* strip the suffix).

## How a strategy is built (the grammar)

```
BREAKOUT_LEVEL = POI  ±  (ATR(period) × distance_mult)
ENTRY          = close breaks through BREAKOUT_LEVEL  AND  optional FILTER
EXIT           = ATR stop + ATR target   OR   fixed time exit
```

See `grammar.py` for the full component lists (POIs, filters, exit types).
Every generated strategy has ~5–6 tunable numbers total, matching the
"3–6 optimization inputs max" rule in your `AI_STRATEGY_GENERATOR_PROMPT.md`
— just applied to *generation* instead of to optimizing one fixed idea.

## How ideas avoid becoming repetitive

Every strategy reduces to a short **DNA** string:
`POI|FILTER|EXIT_TYPE|DIRECTION`. The generator:
- caps how many variants of the same DNA can appear in one batch
  (`max_per_dna` in `generator.py`)
- weights *future* generation away from DNA strings that have already
  been produced many times across all previous runs (`component_memory`
  in `output/research_db.json`)

So over time the tool naturally drifts toward unexplored combinations
instead of grinding out 40 near-clones of "SMA20 breakout with an ADX
filter."

## How robustness scoring avoids bias toward one market

`score.py` never ranks on profit alone. A strategy's robustness score is:

- 30% — how many of the 5 assets it was profitable on (breadth)
- 25% — average profit factor (only counted on assets with enough trades)
- 20% — average trade-level Sharpe
- 15% — **consistency**: penalizes a strategy that crushes it on one
  asset and is a disaster everywhere else (cross-asset Sharpe std-dev)
- 10% — sample size (total trades across all assets)

A strategy also has to clear a minimum bar (≥15 trades on its worst asset,
profitable on ≥2 of 5 assets) to be marked `eligible` — otherwise its
score is heavily discounted rather than deleted, so you can still see
*why* it failed in the CSV.

## Files

```
breakout_setup_generator/
    grammar.py             # every building block (POI, distance, filter, exit)
    generator.py           # weighted-random assembly + novelty/dedupe rules
    evolve.py                # genetic evolution: crossover + mutation on strategy DNA, optional --ml-prescreen
    ml_rank.py                # ML pre-screener: predict promise before backtesting (Sprint 5)
    indicators.py              # ATR, ADX, SMA, Donchian, volume SMA (pandas only)
    data.py                     # synthetic 5-asset-class universe + real CSV loader
    backtest.py                  # state-machine backtester -> trades + stats (default, fast engine)
    amipy.py                      # your own engine, vendored in as-is
    amipy_backtest.py              # adapter: StrategySpec -> amipy signals, for --amipy-check-top
    score.py                        # cross-asset robustness scoring
    validation.py                    # walk-forward / Monte Carlo / neighbor stability (--validate-top)
    export_easylanguage.py            # EasyLanguage-style code export
    export_afl.py                      # AmiBroker AFL code export
    export_pinescript.py                # TradingView Pine Script v5 code export
    database.py                          # persistent component memory + leaderboard + training log (JSON/JSONL)
    main.py                               # CLI that runs the random-generation pipeline

    # v2: live data + web interface
    fetch_data.py                        # Yahoo Finance OHLCV downloader → data_cache/
    ui_server.py                         # Flask+SocketIO server: endpoints + live log streaming
    ui/                                  # web frontend
      index.html                           # main dashboard: Runs, Leaderboard, Backlog, Live Data, Output Files
      strategy.html                        # interactive strategy chart: candles, equity pane, trade markers
      lightweight-charts.js                # vendored lightweight-charts v5.2.1 (charting engine)
      breakoutLOGO.svg                     # Investing Compass logo
    RUN_UI.bat                          # double-click: find python, pick port, open browser
    RUN_BACKTEST_MENU.bat                # double-click: interactive terminal menu (main.py + evolve.py)

    data_cache/                          # Yahoo CSVs land here (SYM_daily.csv / SYM_hourly.csv)
    output/                              # latest_run.csv, research_db.json, training_log.jsonl,
                                         # validation_report.json, evolution_run.csv, ml_model.joblib,
                                         # code/<format>/*.*, plots/*.* land here
```

## Exporting to your own platform (EasyLanguage / AFL / Pine Script)
strategy into starting-point code in **your choice of three formats** —
`easylanguage` (default alongside the other two, in the same
`Input:`/`vars:`/`POI`/`SPACE`/`FILTER` shape as your uploaded samples),
`afl` (AmiBroker Formula Language), and `pinescript` (TradingView Pine
Script v5):

```bash
python3 main.py --num 300 --export-top 5                                  # all three formats
python3 main.py --num 300 --export-top 5 --export-format pinescript       # just one
python3 main.py --num 300 --export-top 5 --export-format easylanguage,afl # a subset
```

All three are first drafts to save you the translation work, not
compilers — verify each one actually compiles and behaves as intended on
its own platform before trusting it with money. Same disclaimer header as
the source material's own code samples.

## Position sizing (`--position-sizing fixed_fractional`)

```bash
python3 main.py --num 300 --position-sizing fixed_fractional --risk-pct 0.10
python3 evolve.py --generations 15 --position-sizing fixed_fractional --risk-pct 0.05
```

Default (`full_compounding`) is unchanged from earlier in this README —
100% of equity compounds through every trade's raw % return. Switch to
`fixed_fractional` to risk a fixed % of equity per trade instead, sized
from the trade's actual stop distance (ATR-based fallback for exit types
with no hard stop). See "Known Limitations" below for exactly what this
does and doesn't fix, including a real bug caught and fixed while
building it.

## Cross-checking against your own amipy.py engine

`main.py --amipy-check-top N` re-runs the top N eligible strategies through
your own `amipy.py` (vendored into this project) as a second opinion,
using its real position-sizing/margin/commission model and Sortino ratio
instead of the simplified compounding-% model `backtest.py` uses:

```bash
python3 main.py --num 300 --amipy-check-top 10
```

`amipy_backtest.py` translates a `StrategySpec` into the boolean
buy/short/sell/cover signal matrices `amipy.Amipy.run()` expects, reusing
its own `apply_stops_sell_rq` / `apply_stops_cover_rq` (per-bar ATR-sized
stop/target) machinery rather than reinventing that logic, and
`amipy.ex_rem()` for signal deduplication (the same helper the
`strategy_bollinger_cmf_bulk.py` sample already uses).

**Read the numbers as a second opinion, not a duplicate.** In testing, the
two engines agree closely (often exactly) on trade count and direction for
lower-frequency strategies, but diverge more on very high-signal-density
ones -- expected for two reasons, one structural and one a known rough
edge:

- **Structural, not a bug**: `amipy`'s default `risk=0.1` sizes each trade
  to ~10% of equity via share count, while `backtest.py` compounds 100% of
  equity through each trade's raw % return. amipy's total-return numbers
  will generally run lower for that reason alone -- that's the more
  realistic assumption, and part of why it's worth checking against.
- **Known rough edge**: `amipy`'s stop/target scanner
  (`apply_stops_*_rq`) walks forward independently from every entry bar,
  so very dense signals (a strategy re-triggering on many consecutive
  bars) can produce more divergence than sparser ones. `amipy_backtest.py`
  collapses same-day repeat signals before computing stops to mostly
  avoid this, but it isn't a byte-for-byte reproduction of
  `backtest.py`'s state machine.

If a strategy's numbers wildly diverge between the two engines, treat that
as a caution flag about fragility to fill/timing assumptions -- worth
knowing before trusting the strategy, not something to paper over.

## Genetic evolution mode (instead of pure random generation)

`main.py` draws every strategy fresh, independently. `evolve.py` runs an
actual generational loop on top of the same grammar/backtest/score:

```bash
python3 evolve.py --generations 15 --population 60 --seed 1
python3 evolve.py --generations 25 --population 100 --data-dir ./my_csvs --top 20
```

Each generation:
1. **Evaluate** every individual's robustness score (same `score.py` used
   by `main.py`) — cached by exact parameter set, since crossover/elitism
   naturally re-produce the same strategy across generations.
2. **Select** parents via tournament selection (pick `k` random
   individuals, keep the fittest).
3. **Crossover**: build a child gene-by-gene (direction / POI / ATR period
   / distance / filter / exit), each gene independently inherited from
   parent A or B — with a validity check afterward, so e.g. inheriting a
   `long`-only POI onto a `short` child gets re-rolled instead of silently
   producing a broken strategy.
4. **Mutate**: each gene independently has a chance (`--mutation-rate`,
   default 15%) to get re-rolled from scratch, keeping some fresh
   exploration alive alongside the crossover-driven exploitation.
5. **Elitism**: the top `--elite-frac` (default 15%) carries forward
   unchanged, so a strong idea can't be crossed/mutated out of existence.

A real run (10 generations, population 50) took average fitness from
~32 to ~69 and best fitness from ~68 to ~82, converging on a genuine
*family* of related winners (same POI, neighboring ATR periods) rather
than one lucky outlier — which is what you want to see; a GA that
converges on one single point instead of a neighborhood is a red flag for
overfitting, not a sign it "solved" the problem.

Writes to the same shared `output/research_db.json` as `main.py`, so
ideas discovered by evolution feed the same novelty memory and leaderboard
random search draws from — the two modes reinforce each other over time.

### ML-prescreened offspring (`--ml-prescreen`)

The GA and the ML pre-screener (`ml_rank.py`) used to run as two
completely independent tools. `--ml-prescreen` merges them: instead of
generating exactly as many children as the next generation needs,
`evolve.py` generates a **larger pool** (`--offspring-pool-mult`, default
3x) via the same crossover/mutation, has the trained model predict every
candidate's promise **without backtesting them**, and only the
top-predicted subset actually spends a real backtest:

```bash
python3 evolve.py --generations 15 --population 60 --ml-prescreen
python3 evolve.py --data-dir ./real_data --generations 15 --ml-prescreen --offspring-pool-mult 5
```

No trained model yet? It auto-trains one first (bootstrapping the
training log if needed) rather than failing. **Same number of real
backtests per generation either way** — this spends extra, cheap
crossover/mutation+prediction calls to pick *better* candidates before
spending the backtest budget on them, not to reduce the budget itself.

**An honest, measured comparison, not just an assertion this helps** —
identical seed, population, and generation count, with vs. without:

| | avg score (gen 5) | best score (gen 5) | eligible/30 (gen 5) | wall time |
|---|---|---|---|---|
| without `--ml-prescreen` | 65.2 | 77.4 | 27/30 | 3.1s |
| with `--ml-prescreen` | **71.2** | **78.1** | **30/30** | 17.3s |

Real, measurable improvement in population quality per backtest spent —
at the real cost of more wall-clock time (generating and scoring the
larger candidate pool each generation), not less. The printed
per-generation `ml-prescreen:` line shows the actual predicted-promise
gap between the chosen subset and the full pool, so you can see the
filter doing real work rather than trusting it blindly.

## Digging deeper: robustness validation

Two more stages, both opt-in (off by default because they're more
expensive than the base screen), meant to run on the leaderboard survivors
only:

```bash
# generate, screen, then stress-test the top 10 eligible strategies
python3 main.py --num 300 --validate-top 10

# generate, screen, then export the top 5 as EasyLanguage-style code
python3 main.py --num 300 --export-top 5 --market "@ES" --timeframe "60min"

# do both in one run
python3 main.py --num 300 --validate-top 10 --export-top 5
```

**`--validate-top N`** (writes `output/validation_report.json`) runs three
checks straight from the "kill 99% of ideas" philosophy in your source
material, per strategy:

- **Walk-forward** — splits each asset's history into 4 contiguous windows
  and backtests each separately. A real edge is profitable in *most*
  windows, not carried by one lucky stretch.
- **Monte Carlo** — bootstrap-resamples the actual trade sequence 1,000
  times to see the plausible *range* of outcomes (5th/50th/95th percentile
  return, probability of losing money, worst-case drawdown) instead of
  trusting the one path that happened to occur.
- **Neighbor stability** — nudges every tunable number to its neighbor in
  the grammar's own option list (e.g. `distance_mult` 1.5 → 1.0 or 2.0) and
  re-backtests. Straight from the workbook: *"we want... as many neighbor
  values with similar results [as possible]"* — a strategy that collapses
  one step away from its exact parameters is overfit, not edge.

Each strategy gets a plain-language verdict (`PASS` / `CAUTION` / `FAIL`)
as a quick triage signal — read the actual numbers in the JSON before
trusting the label.

**`--export-top N`** exports as EasyLanguage, AFL, and Pine Script — see
"Exporting to your own platform" below for details.

## Intraday bars (unlocks HOD/LOD, time-of-day filters, EOD exit)

Everything above defaults to one bar per trading day. Add `--intraday` to
any of the three tools (`main.py`, `evolve.py`, `ml_rank.py`) to switch to
a synthetic intraday universe (30-min bars, 09:30-15:30 session by
default) instead:

```bash
python3 main.py --intraday --num 300
python3 main.py --intraday --bar-minutes 15 --days 500   # 15min bars, ~2 years
python3 evolve.py --intraday --generations 15
```

This unlocks three grammar components that are only meaningful within a
single trading session and are harmless no-ops on daily bars (each daily
bar is its own one-bar "session", so these components simply never have
a prior bar to reference and quietly never fire there):

- **`hod_running` / `lod_running` POIs** — the TRUE running high/low
  *so far today*, causally correct (only knows what's happened up to the
  previous bar, resets at the start of every session) instead of the
  daily-bar workaround (`prev_high`/`donchian_high20`/etc.) needed
  earlier in this README.
- **`time_window_open` / `_mid` / `_close` filters** — mirrors
  `breakout_samples.md` CODE_2's literal `Test_Hour` bucket filter,
  expressed as session-relative thirds instead of hardcoded clock times
  so it works with whatever session hours you configure.
- **`eod_exit`** — closes the position at the end of the entry session,
  the exact `SetExitOnClose` / `if time = 1500 then sell ... at close`
  pattern from the uploaded samples. Zero extra tunable parameters.

`--days` defaults to 2600 (daily) or 650 (`--intraday`) sessions; the
state-machine backtester is a Python loop, so intraday runs process
`bars_per_day`x more bars per asset per strategy — start smaller than the
daily default and raise it once you've confirmed timing is acceptable.

## Real data: an actual run, with actual results

`real_data/` ships with this project: five REAL market histories, not
synthetic, one per asset class, pulled from public GitHub-hosted datasets:

| Symbol | Asset class | Source | Range | Bars |
|---|---|---|---|---|
| `BTC` | Crypto | Kaggle/GitHub (nileshiq) daily BTC-USD | 2010-07 to 2024-04 | 5,021 |
| `EURUSD` | FX | komo135/forex-historical-data (broker feed) | 2012-12 to 2022-03 | 2,400 |
| `GOLD` | Commodity | komo135/forex-historical-data XAUUSD (broker feed) | 2012-11 to 2022-03 | 2,400 |
| `SPY` | Equity index | willhjw/big_movers daily SPY | 2000-01 to 2026-03 | 6,593 |
| `UST10Y` | Rates | epogrebnyak/data-ust, US 10Y Treasury yield | 1990-01 to 2022-03 | 8,055 |

**Two honest caveats about this data, not glossed over:**
- The EURUSD/GOLD broker feed stores prices as scaled integers (pipette
  precision); converted back to real prices by dividing by 100,000 and 100
  respectively — verified against known real EURUSD/gold prices on
  specific dates before trusting the conversion.
- `UST10Y` is a **yield level, not a traded price** — there's no real
  open/high/low/volume for a benchmark yield, so `Open=High=Low=Close`
  = the yield and `Volume=0`. That's a legitimate way to backtest "trading
  the level" but it's not a real OHLCV instrument, and it visibly breaks
  down in the `amipy` cross-check below (near-zero High-Low range confuses
  its ATR-based stop sizing) — a good demonstration of exactly the kind of
  cross-engine disagreement flagged as a caution signal earlier in this
  README, not a data pipeline bug.

**A real run** (`main.py --data-dir ./real_data --num 400`) found 99/400
generated strategies eligible. The top 5 by robustness score all passed
full validation (walk-forward, Monte Carlo, neighbor stability) with
`PASS` verdicts — genuinely, not cherry-picked, all 5 passed. All were
long-only trend/breakout setups (`sma20`/`sma50`/`prev_close`/`prev_low`
POIs with `adx_below` or `atr_contraction` filters), profitable on at
least 4 of 5 real assets. **`evolve.py` on the same data converged even
more strongly**: 10 generations landed the population on a coherent
family — `sma50` breakout + `atr_contraction(0.9)` (only trade after a
volatility squeeze) — scoring 82-84 versus random search's best of ~74.

**The compounding-blowup caveat from Known Limitations is not
hypothetical — it showed up immediately on real data.** BTC's genuine
2010-2024 bull run (real prices, real dates) combined with `backtest.py`'s
100%-compounding assumption produced total returns in the
**tens of millions of percent** for some long-only strategies. That's the
math of "a real edge, compounded fully, on an asset that went up
~1000x+" — not a bug, but a vivid illustration of why the amipy
cross-check's realistic position sizing (`--amipy-check-top`) is the
number to trust, not `backtest.py`'s raw compounding: the SAME strategies
through `amipy` show BTC returns of 36%-120%, still excellent, but
believable.

Try it yourself:
```bash
python3 main.py --data-dir ./real_data --num 400 --validate-top 10 --amipy-check-top 5 --export-top 5
python3 evolve.py --data-dir ./real_data --generations 15 --population 80
```

### Real intraday data too

`real_intraday_data/` ships alongside `real_data/`: real HOURLY bars for
3 of the 5 asset classes (crypto, FX, commodity) — real equity-index and
rates intraday data wasn't findable through the same free, publicly-hosted
route in the time available, so this is honestly a 3-asset universe, not
padded out with a weak substitute:

| Symbol | Asset class | Source | Range | Bars |
|---|---|---|---|---|
| `BTC` | Crypto | zengqiang041 (Bitfinex hourly) | 2013-04 to 2018-03 | 42,825 |
| `EURUSD` | FX | komo135/forex-historical-data hourly | 2012-11 to 2022-03 | 57,600 |
| `GOLD` | Commodity | komo135/forex-historical-data XAUUSD hourly | 2012-05 to 2022-03 | 57,600 |

This is where `--intraday`'s session-aware components (`hod_running`/
`lod_running`, `time_window_*`, `eod_exit`) stop being a synthetic-only
demonstration: a real run (`main.py --data-dir ./real_intraday_data
--num 400`) put **`hod_running`- and `lod_running`-based strategies
directly in the top 15 by robustness score**, alongside more conventional
`sma20`/`highest_high_10` setups — genuine evidence the true intraday POIs
add value on real intraday bars, not just a feature that runs without
crashing. 4 of the top 5 passed full validation; the 5th was correctly
flagged `CAUTION` rather than rubber-stamped.

Note this run also needed `--min-trades` raised well above the daily-bar
default of 15 — see below.

```bash
python3 main.py --data-dir ./real_intraday_data --num 400 --min-trades 100 --validate-top 5
```

### Raising the sample-size bar (`--min-trades`)

`score.py`'s `MIN_TRADES_PER_ASSET` used to be a hardcoded module constant
(15) — fine for the sparse daily-bar synthetic MVP, wrong for real or
intraday data where hundreds-to-thousands of trades per strategy is
normal (see the real-intraday numbers above: single strategies with
1,000-9,000+ trades). It's now a parameter, exposed as `--min-trades` on
`main.py`, `evolve.py`, and `ml_rank.py`:

```bash
python3 main.py --data-dir ./real_intraday_data --min-trades 200   # hold a high-frequency run to a real bar
python3 main.py --num 300                                           # unset = still 15, right for daily bars
```

Nothing changes the default — daily-bar runs are unaffected — but any
higher-frequency run (real data, `--intraday`, or both) should raise this
to something closer to the workbook's own "200-400+ trades" standard
instead of being graded on a bar sized for a different sampling frequency.

## Cross-checked against the EasyLanguage samples / workbook

After you uploaded `breakout_samples.md` (the two `BREAKOUT_TRADING_REVOLUTION_CODE`
snippets + the DowBreaker strategy) and `breakouts_workbookk_v3_0.md` (the
HOD/LOD "Key Levels" walkthrough + the ADX MASTER FILTER derivation), I
checked the grammar/generator/backtester against them line by line. Result:
the core shape already matched (`POI + FRACT×ATR + FILTER + EXIT`, one trade
at a time, cross-market validation before trusting anything), but a few
concrete techniques from those docs weren't in the grammar yet. Added:

- **POI-swapping.** The workbook is explicit: `LONG = HOD + ATR×X` is one
  valid model, but `LONG = LOD + ATR×X` is a second, equally valid one —
  "swap the POIs mutually and see what happens." `POI_LONG`/`POI_SHORT` in
  `grammar.py` now include both sides of the range for both directions
  instead of only the "obvious" pairing.
- **`adx_below` filter** — mirrors `CODE_1`/`CODE_2`'s exact
  `FILTER = ADX(25) < Filter_Period` (only breakout while the trend isn't
  already overextended), the inverse of the existing `adx_trend` filter.
- **`adx_master_filter`** — the workbook's derived-through-multi-market-
  optimization universal rule, `ADX(15) > 20`, hard-coded as a single
  non-optimized choice (not a param range) exactly as the source insists:
  *"the filter should stay the same for all strategies... don't optimize
  the input values any further."*
- **`shallow_pullback_long` / `_short`** — approximates the DowBreaker
  sample's higher/main-timeframe pullback conditions (distance from a
  recent high/low, scaled by volatility) collapsed onto one timeframe.
- **`poi_anchored_stop` exit** — the workbook's HOD/LOD "secret sauce" risk
  management: instead of an ATR-multiple stop, the stop sits right at the
  POI level itself (a real support/resistance line), and only the profit
  target is still tunable. If the POI isn't actually on the correct side of
  price for that trade, the entry is skipped rather than faked.

All three were regression-tested: 400 generated strategies × 5 assets with
every new filter/exit type appearing and backtesting cleanly, then a full
`main.py` run confirmed nothing else broke (see the `LONG: prev_low + ...`
entries near the top of a real leaderboard run — that's the POI-swap advice
showing up as a genuinely strong idea, not just a compliance checkbox).

**Now shipped** (previously listed here as deliberately out of scope —
see "Intraday bars" above):
- **True HOD/LOD as the *current* day's running high/low** — `hod_running`
  / `lod_running`, causally correct, via `--intraday`.
- **Time-of-day filters** — `time_window_open`/`_mid`/`_close`, via `--intraday`.
- **The literal EOD exit** (`SetExitOnClose` / `sell at 1500 close`) — `eod_exit`.

**Still NOT ported** (structural mismatches worth knowing about):
- **Fixed-dollar stop-loss** (`Setstoploss(1500)`) and **conditional position
  sizing** (2 contracts vs 1 based on yesterday's direction) — both need a
  point-value/contract model this MVP doesn't have yet (it works in trade-%
  terms, see Limitations below).
- **The 400+ trades / 10 years sample-size bar** the workbook uses is for
  20/15-minute intraday strategies with one EOD exit per day. `--intraday`
  gets you much closer to that signal rate (see the real trade counts in
  the "Intraday bars" section above), but `score.py`'s
  `MIN_TRADES_PER_ASSET = 15` hasn't been raised to match it yet — that's
  a one-line change once you're ready to hold intraday runs to a higher bar.

## ML pre-screening (predict promise before backtesting)

`ml_rank.py` uses accumulated history to predict which freshly-generated
strategies are worth backtesting at all, *before* actually backtesting
them — ranking survivors, not inventing new ideas (it never proposes a
POI/filter/exit combo the grammar doesn't already know how to produce).

```bash
python3 ml_rank.py --bootstrap 1000          # top up the training log to >=1000 rows
python3 ml_rank.py --train                     # train + save a model (auto-bootstraps if needed)
python3 ml_rank.py --evaluate 300 50            # honest recall@K test vs. a random baseline
python3 ml_rank.py --prescreen 500 --keep 50    # generate 500, backtest ONLY the model's top 50
```

**How the training data works**: every strategy `main.py` and `evolve.py`
ever evaluate — winners AND rejects alike — gets appended to
`output/training_log.jsonl` (unlike the leaderboard, this is never
trimmed). A classifier needs both classes to learn anything; a log of
only winners would have nothing to contrast them against. `--bootstrap`
tops this log up on its own if it's too thin (a few hundred rows minimum,
more is better) so `ml_rank.py` works standalone, without requiring you to
have already run the other tools first.

**`--train`** fits a `RandomForestClassifier` (via scikit-learn) on the
strategy's genes (POI, filter, exit type, and their numeric parameters,
one-hot encoded where categorical) to predict `eligible`, and reports test
accuracy against the naive "always guess the majority class" baseline —
so you can tell if the model learned something real. In one test run: 500
rows, 29% base eligibility rate, 80% test accuracy.

**`--evaluate N K`** is the honest check, not a victory lap: generates N
fresh candidates, has the model rank its predicted top K *without*
backtesting anything, then actually backtests all N anyway (evaluation-
only cost) to get ground truth, and reports **recall@K** — what fraction
of the TRUE top K by real robustness score the model's predicted top K
actually contains — against a random-K-of-N baseline for comparison. One
run: 57% recall@30 vs. 10% for random.

**`--prescreen N --keep K`** is the actual production use: generate N,
predict promise for all of them (cheap — no backtest), backtest only the
top K (expensive). One run: pre-screening the top 40 out of 300 candidates
found a batch with a 92.5% eligibility rate, vs. the ~29% base rate in the
training data — using only 13% of the backtest compute a full run of 300
would need.

## Known limitations (this is the MVP, on purpose)

- **No position sizing by default, full % compounding.** `backtest.py`
  defaults to treating every trade as 100%-of-equity compounding of its
  raw % return, with no leverage cap. On a long-only strategy with a real
  edge over hundreds of trades on a high-volatility asset (crypto in
  particular), this can compound into absurd, unrealistic total-return
  numbers (a real example from testing: **171,182% total return on
  SIM_CRYPTO** for one strategy) — that's the compounding math working
  exactly as coded, not a bug, but it's not a realistic assumption either.
  **Fix**: pass `--position-sizing fixed_fractional` (all three tools) to
  risk a fixed % of equity per trade (`--risk-pct`, default 10%), sized
  from the trade's actual stop distance (or the entry-bar ATR as a
  fallback risk proxy for exit types with no hard stop). This is
  immune to the single-trade blowup that caused the number above — I
  found and fixed a real bug while building it, where an unprotected
  exit (no real stop order) could lose *more* than its intended risk
  fraction and drive equity negative; losses are now capped at all the
  equity a trade was allocated. **What `fixed_fractional` does NOT fix**:
  a real edge, compounded over enough winning trades, is still
  mathematically capable of huge cumulative numbers over a long enough
  real-data history (thousands of trades on BTC 2010-2024, for instance)
  — that's genuine compounding, not a flaw, and CAGR (also reported) is
  the more meaningful annualized number to read on very long backtests
  rather than the raw multi-year total.
- **Position sizing is scoped to `main.py` and `evolve.py`'s core
  backtest/fitness loops only** — `validation.py`'s walk-forward/neighbor
  checks and `ml_rank.py`'s bootstrap/prescreen still use the
  `full_compounding` default internally, since those are about relative
  comparison and eligibility, not final realistic dollar figures. Not
  wired through every call site by design, not an oversight.
- **Synthetic data is illustrative only** — plug in real data via
  `--data-dir` before trusting any number this produces.
- **Sample-size bar is intentionally low for daily bars**
  (`MIN_TRADES_PER_ASSET = 15` in `score.py`) — nowhere near the
  workbook's own "400+ trades / 10 years" expectation, which is calibrated
  for intraday bars. `--intraday` gets you much closer (a real test run:
  438-1275+ trades per strategy on ~80 sessions of 30-min bars) but the
  bar in `score.py` hasn't been raised to match yet — do that yourself if
  you move to intraday as the default.
- **Intraday mode is new and lightly tested.** The session-aware pieces
  (`hod_running`/`lod_running`, `time_window_*`, `eod_exit`) were verified
  to run error-free across hundreds of generated strategies and all three
  code exporters, but haven't been validated against real intraday data
  or checked for subtle timezone/session-boundary edge cases a real
  exchange calendar would surface.
- **Fixed-$ stops and conditional position sizing** (both present in the
  DowBreaker sample) aren't modeled anywhere in this codebase yet.

## Where this goes next (from your own research notes)

Straight out of `breakout_idea_generator.md`'s sprint plan -- all five
original sprints now have at least an MVP:

- **Done**: cross-asset generation + scoring (`main.py`, the base pipeline).
- **Done**: reused your own `amipy.py` engine (`amipy_backtest.py`,
  `--amipy-check-top`) as a second-opinion check on the leaderboard
  survivors -- real position sizing, margin, commission, Sortino. See
  "Cross-checking against your own amipy.py engine" above for its honestly
  -documented limitations vs. a byte-for-byte replacement.
- **Done**: walk-forward, Monte Carlo, and neighbor-parameter stability
  testing (`validation.py`, `--validate-top`).
- **Done**: genetic evolution (`evolve.py`) -- crossover + mutation on
  strategy DNA instead of pure weighted-random draws, sharing the same
  novelty memory and leaderboard as `main.py`.
- **Done**: ML pre-screening (`ml_rank.py`) -- predicts which strategies
  are worth backtesting before spending compute on them, trained on the
  growing `output/training_log.jsonl`.
- **Done**: code export to EasyLanguage, AmiBroker AFL, and Pine Script
  v5 (`export_easylanguage.py`, `export_afl.py`, `export_pinescript.py`).
- **Done**: intraday bars (`--intraday`) — unlocks true `hod_running`/
  `lod_running` POIs, `time_window_*` filters, and the literal `eod_exit`
  pattern from the uploaded samples, across all three tools.

- **Done**: real data (`real_data/`, see "Real data: an actual run,
  with actual results" above) -- 5 real markets, one per asset class, run
  through the full pipeline with genuinely-passing validated results, not
  just synthetic illustrations.

- **Done**: real intraday data (`real_intraday_data/`, see "Real
  intraday data too" above) -- real hourly bars for crypto/FX/commodity,
  with `hod_running`/`lod_running` genuinely landing in the top 15 by
  robustness score on real data, not just running error-free on synthetic.

- **Done**: configurable sample-size bar (`--min-trades`, see "Raising
  the sample-size bar" above) -- no longer a hardcoded constant sized
  only for daily bars.

- **Done**: real, risk-based position sizing (`--position-sizing
  fixed_fractional`, see "Position sizing" above) -- `backtest.py` no
  longer only supports naive 100%-compounding; risking a fixed % of
  equity per trade is now a first-class option in `main.py` and
  `evolve.py`, and a real single-trade-blowup bug was caught and fixed
  while building it (see Known Limitations).

- **Done**: a real GA x ML loop (`evolve.py --ml-prescreen`, see
  "ML-prescreened offspring" above) -- `evolve.py` and `ml_rank.py` no
  longer run independently; the trained model screens each generation's
  offspring before backtesting them, with a measured (not just asserted)
  improvement in population quality per backtest spent.

What's genuinely still open, if you want to keep pushing:
- **Real intraday equity-index and rates data.** Only found free,
  easily-downloadable real intraday history for 3 of 5 asset classes
  (crypto, FX, commodity) in the time available -- equity index and
  rates intraday data would complete the set.
- **Fixed-dollar stops and conditional position sizing.** The DowBreaker
  sample's `Setstoploss(1500)` (a flat dollar amount, not ATR-scaled) and
  its 1-vs-2-contract sizing based on yesterday's direction still aren't
  modeled -- `fixed_fractional` sizing is risk-based, not those specific
  patterns.

## License

**Personal use only — not for commercial use.** This project is licensed
for strictly personal, non-commercial purposes: private research, learning,
and personal trading analysis. Selling, reselling, hosted/SaaS deployment,
or any use of the software (or its output) for commercial gain is
prohibited without written consent. Modified versions are subject to the
same terms and may not be redistributed. See [LICENSE](LICENSE) for the
full terms.

The software is provided as-is, without warranty. It is a research tool for
personal experimentation and does not constitute financial advice —
trading involves substantial risk of loss.
