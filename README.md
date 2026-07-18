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
screening/universe.py       73-ticker / 13-sector universe + candidate pair generation
screening/cointegration.py  Engle-Granger two-step test (OLS hedge ratio -> ADF on residual),
                             AR(1)/OU half-life estimation, BH FDR correction, out-of-sample
                             validation, full-universe screen
screening/regime.py         macro regime overlay: VIX/GVZ/10y/dollar/oil panel, causal rolling
                             stress-percentile mask (entries-allowed), spread-vs-macro diagnostics
signals/spread.py           spread construction, causal rolling z-score, entry/exit/stop-loss/
                             time-exit state machine; hedge ratio models: StaticOLS + Kalman
                             filter (per-bar beta AND one-step-ahead innovation z-scores)
backtest/simulator.py       PairBacktester — costs/slippage, dollar-neutral sizing, max-concurrent-
                             pairs cap, rolling re-cointegration gating, daily mark-to-market;
                             hedge modes: "regime" | "kalman" | "kalman_innovation"
backtest/metrics.py         Sharpe, max drawdown, win rate, profit factor, trade stats
montecarlo/simulator.py     OU fit + 1000-path simulation + per-path strategy P&L,
                             net of transaction costs, slippage, and short-leg borrow
reporting/daily_report.py   "what would today's signal be" — pure computation, no execution
reporting/journal.py        forward-test journal: idempotent daily signal log + outcome grader
visualization/plots.py      static matplotlib figures (spread/z, equity, heatmap, MC fan PNG)
visualization/interactive.py dark-theme Plotly dashboards (fan chart, P&L, spread/z, equity)
tests/                      105-test pytest suite on synthetic/seeded fixtures — no network
run_demo.py                 deterministic, no-network demo of the full pipeline
run_screen.py                live screen -> regime overlay -> backtest -> daily signal report
run_pair_study.py            one-pair diagnostics + 3-mode backtest comparison
run_montecarlo.py            1000-path OU Monte Carlo -> interactive dashboards, gross AND net P&L
run_journal.py               forward-test journal runner (outputs/signal_journal.csv, git-tracked)
run_live_monitor.py          REAL-TIME monitor: Yahoo websocket streaming (default) with polling
                             fallback, live z-score + signal state, auto-refreshing dashboard
docs/                        published interactive dashboards (GitHub Pages)
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

Deep-dive one pair (diagnostics + regime-OLS vs Kalman hedge backtest):

```
py run_pair_study.py GDX GLD
```

Real-time intraday monitor (streams live ticks over Yahoo's websocket by
default, falling back to polling automatically; writes an auto-refreshing
dashboard to `outputs/live_monitor.html` — open it in a browser; Ctrl+C to stop):

```
py run_live_monitor.py GDX GLD                # stream mode (default)
py run_live_monitor.py GDX GLD --mode poll    # polling fallback, --poll 30
```

Forward-test journal (run once a day; grades entries after their 10-bar
outcome window closes):

```
py run_journal.py GDX GLD
```

## Interactive dashboards (GitHub Pages)

The `docs/` folder contains the published interactive Plotly dashboards —
the 1,000-path Monte Carlo fan chart, net-of-costs P&L distribution,
spread/z-score with signal bands, backtest equity curve, and a live-monitor
snapshot — with an index page linking them all. To serve them as live pages,
enable GitHub Pages once in the repo settings (Settings -> Pages -> Deploy
from a branch -> `main` / `docs`); they then render at
`https://ryanakratzer-cpu.github.io/pairs-trading-engine/`. Regenerate the
dashboards any time with `py run_montecarlo.py GDX GLD` and copy the
refreshed `outputs/interactive_*.html` files into `docs/`.

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
- `KalmanHedgeRatio` (`signals/spread.py`) provides a strictly causal,
  per-bar time-varying hedge ratio via a 2-state (intercept, beta) random-walk
  Kalman filter, plus `innovation_series()` — the filter's standardized
  one-step-ahead surprises. Three backtest hedge modes:
  `"regime"` (piecewise OLS, re-fit at each rolling recheck),
  `"kalman"` (per-bar beta, rolling z of the Kalman spread — documented to
  under-trade because the adaptive beta absorbs divergences), and
  `"kalman_innovation"` (trades the innovation z-score directly, the
  canonical fix). Calibration matters for the innovation mode: the filter's
  observation-variance sets the z-score's noise floor, so
  `PairBacktestConfig.kalman_innovation_obs_variance` (default 1e-4) should
  be in the ballpark of the pair's actual daily noise variance or the
  z-scores are structurally under-dispersed and never reach the entry band.
- `screening/regime.py` overlays macro regime awareness: causal rolling
  percentile ranks of VIX and gold-vol (GVZ) produce a stress mask
  (entries-allowed) compatible with `generate_signals`' `tradeable`
  parameter, plus spread-vs-macro correlation diagnostics (10y yield, dollar,
  oil). Currently informational in `run_screen.py` — it does not gate the
  backtest yet. Missing macro data defaults to calm so a data outage cannot
  silently halt trading.
- `SignalConfig.max_holding_bars` adds a time-based exit (`TIME_EXIT`): if a
  position hasn't converged within ~2-3x the pair's half-life, the
  mean-reversion thesis has failed — exit rather than sit through further
  divergence waiting for the price stop.
- `run_live_monitor.py` fits its hedge ratio on the trailing 252 daily bars
  (the same trailing-regime convention the backtester trades on), then polls
  1-minute intraday bars and re-computes the live z-score each poll. Yahoo
  intraday data can lag real-time by up to ~15 minutes depending on exchange;
  the poll floor is 10s to avoid rate-limiting. The dashboard flags stale
  data / market-closed explicitly.
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
