# Pairs Trading Engine

Cointegration-based statistical arbitrage research and backtesting engine.
Screens a candidate universe of tickers for pairs whose price relationship is
statistically mean-reverting (Engle-Granger two-step test), sizes and times
entries/exits off a rolling z-score of the spread, backtests the resulting
portfolio with transaction costs and a concurrency cap, and produces a daily
signal report. **Backtesting and signal generation only — this engine never
places an order or talks to a broker.**

## Layout

```
data/loader.py              yfinance fetch (auto_adjust close) + local CSV cache in data_cache/
screening/universe.py       default sector-grouped ticker universe + candidate pair generation
screening/cointegration.py  Engle-Granger two-step test (OLS hedge ratio -> ADF on residual),
                             AR(1)/OU half-life estimation, full-universe screen
signals/spread.py           spread construction, causal rolling z-score, entry/exit/stop-loss
                             state machine, HedgeRatioModel interface (StaticOLSHedgeRatio for v1)
backtest/simulator.py       PairBacktester — costs/slippage, dollar-neutral sizing, max-concurrent-
                             pairs cap, periodic rolling re-cointegration checks per pair
backtest/metrics.py         Sharpe, max drawdown, win rate, profit factor, trade stats
reporting/daily_report.py   "what would today's signal be" — pure computation, no execution
visualization/plots.py      spread/z-score, equity curve, cointegration p-value heatmap
tests/                      pytest suite on synthetic/seeded fixtures — no network
run_demo.py                 deterministic, no-network demo of the full pipeline
run_screen.py                live yfinance screen -> backtest -> daily signal report
```

## Setup

```
py -m pip install -r requirements.txt
```

## Usage

Deterministic demo (no network, exit 0 = every stage verified):

```
py run_demo.py
```

Live screen against real market data (fetches prices, screens the default
universe, backtests the top surviving pairs, prints today's signal report):

```
py run_screen.py
```

Run the tests:

```
py -m pytest
```

## Conventions

- Spreads are built on **log prices** by default (`use_log_prices=True`
  throughout), which makes the OLS hedge ratio scale-invariant.
- Entry at `|z| > 2.0`, exit at `|z| <= 0.5`, stop-loss at `|z| >= 3.75`
  (`SignalConfig` defaults). Position convention: `+1` = long the spread
  (long ticker A, short `hedge_ratio` * ticker B), `-1` = short the spread.
- Sharpe ratio uses a risk-free rate of 0 and 252 trading days/year.
- `PairBacktester` re-estimates each pair's hedge ratio and re-runs the
  cointegration test every `recheck_freq_days` (default 60) from the trailing
  `recheck_window_days` (default 252) — always causal, never using data past
  the estimation date. A failed recheck disables *new* entries into that pair
  until the next successful one; an already-open position still exits/stops
  normally.
- Position sizing is **dollar-neutral per leg** (`capital_per_pair` split
  evenly, e.g. $5k long / $5k short), not hedge-ratio-weighted dollar sizing.
- The equity curve reflects **realized P&L at each trade's close**, not daily
  mark-to-market of open positions.

## Known limitations

- **Multiple-hypothesis testing**: screening ~60-250 pairs at p < 0.05 implies
  several expected false positives by chance alone; not corrected for (no
  Bonferroni/Benjamini-Hochberg adjustment) in v1.
- **Survivorship bias**: the default universe is currently-listed, liquid
  tickers only.
- **Static per-regime hedge ratio**: the hedge ratio is re-estimated
  periodically (see above) but held fixed *within* each regime — a
  rolling/Kalman-filter continuously-time-varying hedge ratio is a natural
  future extension (`HedgeRatioModel` in `signals/spread.py` is already an
  interface for this) but isn't implemented.
- **Window-sensitive results**: `run_screen.py`'s screen and
  `reporting/daily_report.py`'s daily report use different lookback windows
  (the screen's full fetch window vs. the report's `lookback_days`, default
  400), so a pair's `is_cointegrated` flag can legitimately differ between the
  two outputs in the same run — this reflects real sensitivity of the ADF
  test to sample length, not a bug.
- `reporting/daily_report.py` replays the full signal state machine from the
  start of its lookback window on every call; it does not persist an actual
  held position across separate runs. Treat it as a monitoring/research tool,
  not a live position tracker.
