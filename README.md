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
- The equity curve is **daily mark-to-market**: realized P&L from closed
  trades plus the unrealized gain/loss on any currently open position, valued
  at that day's prices. Any position still open at the end of the backtest
  window is force-liquidated at the final date's price (logged with
  `exit_reason="END_OF_SAMPLE"`) so total return is always fully realized.
- `screening.cointegration.screen_universe(..., apply_multiple_testing_correction=True)`
  additionally requires each pair's ADF p-value survive a Benjamini-Hochberg
  false-discovery-rate correction across every pair tested (adds a
  `bh_significant` column). Off by default for the library function; enabled
  by default in `run_screen.py`, which reports both the raw count and the
  FDR-corrected count so the caveat is visible without silently gating the
  demo to zero output when nothing survives correction.
- `PairBacktestConfig` has three named risk-profile presets:
  `.conservative()` (smaller size per pair, fewer concurrent pairs, tighter
  stop-loss — targets ~5% max drawdown), `.moderate()` (the plain defaults),
  and `.aggressive()` (larger size, more concurrent pairs, looser entry/stop).
  `run_screen.py`'s `RISK_PROFILE` constant selects which one it backtests
  with.
- `screening.cointegration.screen_universe(..., require_out_of_sample_validation=True)`
  additionally requires each pair survive `validate_out_of_sample()`: split
  the series into an earlier formation window and a later held-out validation
  window (`formation_fraction`, default 0.7), fit the hedge ratio and test
  cointegration on formation data only, then apply that *same* hedge ratio
  (never re-estimated) to the validation window and ADF-test whether the
  spread is still stationary there. Adds `oos_formation_pvalue` /
  `oos_validation_pvalue` / `oos_validated` columns; `tradeable` then also
  requires `oos_validated`. This is a stronger filter than FDR correction
  alone — FDR controls false discoveries across pairs, but says nothing about
  whether any single "significant" pair will hold up going forward. Off by
  default for the library function; enabled by default in `run_screen.py`,
  which backtests the out-of-sample survivors when there are any, and falls
  back to the raw p<0.05 pool (clearly labeled as unvalidated/exploratory)
  when there aren't, rather than silently going empty.

## Known limitations

- **Multiple-hypothesis testing**: with `apply_multiple_testing_correction`
  off (the library default), several expected false positives by chance alone
  are not corrected for. Turn the flag on for a stricter, FDR-controlled read
  — in practice, on a ~60-pair screen this can (and did, in an actual live
  run) flag *zero* survivors even when several pairs pass the raw p<0.05
  threshold, which is the honest result of testing many hypotheses at once,
  not a bug.
- **Current default universe (as of 2026-07-16) has no validated edge.** With
  both `apply_multiple_testing_correction` and `require_out_of_sample_validation`
  on, a live screen of the 29-ticker/6-sector default universe returned zero
  pairs surviving either filter — the 8 pairs that pass the raw p<0.05 +
  half-life screen all had out-of-sample validation p-values between 0.11 and
  0.79 (nowhere close to significant): they looked cointegrated in-sample and
  did not hold up on held-out data. This isn't a code bug; it's the honest
  result of testing a fairly small, correlated (mostly same-sector) universe.
  Widening the universe (more sectors, more tickers, less within-sector
  correlation among candidates) is the natural next step before trusting any
  pair enough to size real capital into it.
- **Survivorship bias**: the default universe is currently-listed, liquid
  tickers only.
- **Static per-regime hedge ratio**: the hedge ratio is re-estimated
  periodically (see above) but held fixed *within* each regime — a
  rolling/Kalman-filter continuously-time-varying hedge ratio is a natural
  future extension (`HedgeRatioModel` in `signals/spread.py` is already an
  interface for this) but isn't implemented.
- **ADF is window-length sensitive by nature**: `run_screen.py` now passes its
  own `LOOKBACK_DAYS` through to `generate_daily_signal_report(...,
  lookback_days=LOOKBACK_DAYS)`, so the screen and the daily report always
  agree within one run of that script. Calling `generate_daily_signal_report`
  directly with a different `lookback_days` than whatever screen you're
  comparing it to will still legitimately produce a different `is_cointegrated`
  read — that's real sensitivity of the ADF test to sample length, not a bug,
  just something to keep the window consistent for when comparing by hand.
- `reporting/daily_report.py` replays the full signal state machine from the
  start of its lookback window on every call; it does not persist an actual
  held position across separate runs. Treat it as a monitoring/research tool,
  not a live position tracker.
