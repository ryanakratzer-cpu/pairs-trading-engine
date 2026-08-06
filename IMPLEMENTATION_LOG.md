---
tags: [quant-research, pairs-trading, cointegration, log]
created: 2026-07-16
---

# Pairs Trading Engine Implementation Log

Build history for the engine in `quant_research/pairs-trading-engine/`. Hub: [[Pairs Trading Engine (MOC)]].

## 2026-07-16 — Full implementation (single session)

- Scaffolded `data/`, `screening/`, `signals/`, `backtest/`, `reporting/`,
  `visualization/` — the first project in `quant_research/` to need real
  market data (the convertible-bond and options-pricing engines both run on
  synthetic data only), so `data/loader.py` (yfinance + local CSV cache) is
  new infrastructure, not a port of an existing pattern.
- Verified `yfinance` installs and works cleanly on this machine's Python
  3.14.5 before building around it — confirmed as risk #1 in the plan;
  `threads=False` was needed on `yf.download()` to avoid an intermittent
  sqlite cache-lock error when downloading multiple tickers concurrently.
- **`screening/cointegration.py`**: manual two-step Engle-Granger (OLS for the
  hedge ratio, then `statsmodels.tsa.stattools.adfuller` on the OLS residual)
  rather than the black-box `coint()`, so the hedge ratio used downstream is
  the exact one the ADF test validated. Half-life via AR(1)/OU fit
  (`ln(2)/-slope`), filtered to a 5–30 trading-day tradeable band.
- **`signals/spread.py`**: causal rolling z-score (verified via a dedicated
  test that truncating the series doesn't change the last computed z-score —
  proves no look-ahead), and a path-dependent entry/exit/stop state machine
  with a `tradeable` gate so a failed rolling re-cointegration check can
  disable *new* entries without disturbing an already-open position's
  exit/stop logic.
- **`backtest/simulator.py`**: `PairBacktester` re-estimates each pair's hedge
  ratio and re-tests cointegration every `recheck_freq_days` from a trailing
  `recheck_window_days` window (causal — never touches future data), producing
  a piecewise-constant hedge ratio and a `tradeable` flag per date. Portfolio-
  level day-by-day loop enforces `max_concurrent_pairs`, ranking competing
  same-day entries by `|z|`-strength. Equity curve reflects realized P&L at
  trade close (documented simplification, not daily mark-to-market).
- Live end-to-end run (`run_screen.py`) against real yfinance data (29
  tickers, ~2.5 years): 8/66 candidate pairs passed the cointegration +
  half-life filter (e.g. COP/SLB, BAC/GS, GS/MS), backtest and daily signal
  report both ran cleanly with the no-execution disclaimer intact.
- 22/22 tests green, all on synthetic/seeded fixtures — no network calls in
  the test suite.

### Bugs caught during build

1. **pytest collected a production function as a test.** `tests/test_cointegration.py`
   did `from screening.cointegration import test_pair_cointegration` — the
   bare import binds a module-level name starting with `test_`, which pytest's
   default `python_functions = test_*` collection rule then tried to run as a
   test (with no fixtures, since it's a real function needing `price_a`/`price_b`
   args), producing a fixture-not-found collection error. Fixed by importing it
   under an alias (`as engle_granger_test`) in the test file — the production
   name in `screening/cointegration.py` itself is unaffected.
2. **Concurrency test double-counted a same-day exit/entry handoff.** The
   `max_concurrent_pairs` enforcement test computed "open" as
   `entry_date <= day <= exit_date` for every trade; when one pair's exit and
   another's entry land on the same day (exit is processed before new entries
   each day in the simulator, so this is legitimate — the freed slot is reused
   same-day), the inclusive exit boundary counted both trades as open on the
   handoff day, tripping a false capacity-violation assertion. Fixed by making
   `exit_date` exclusive in the test's own occupancy count — the simulator's
   actual behavior was correct throughout; only the test's measurement was off.

## 2026-07-16 (overnight) — Mark-to-market equity + FDR correction

User paused for the night before picking stocks; continued hardening the two
stock-agnostic open threads from the same-day build that most affected how
trustworthy the numbers are, so the engine is in better shape before capital
gets attached to specific tickers.

- **Daily mark-to-market equity curve.** Previously the equity curve only
  moved when a trade closed, hiding drawdown while a position was open.
  `PairBacktester.run()` now marks every open position to that day's prices
  each iteration (`_mark_to_market`, refactored out of `_close_position` so
  both share one gross-P&L formula) and adds it to realized P&L for that day's
  equity value. Any position still open when the price series ends is now
  force-liquidated at the final date's price (`exit_reason="END_OF_SAMPLE"`),
  so total return is always fully realized rather than leaving a dangling
  unrealized tail. Effect on the synthetic demo: Sharpe dropped from 1.43 to
  0.87 and max drawdown widened from -0.17% to -0.77% for the *same* trades —
  the earlier numbers were real P&L but an optimistic risk picture.
- **Benjamini-Hochberg FDR correction.** Added `_benjamini_hochberg()` and an
  opt-in `apply_multiple_testing_correction` flag on `screen_universe()` (off
  by default in the library, on by default in `run_screen.py`). Live result
  on the actual 66-pair default universe: 8 pairs passed the raw p<0.05
  cointegration + half-life filter, but **zero** survived FDR correction —
  the honest consequence of testing that many hypotheses at once with p-values
  that were all in the 0.002-0.05 range (none small enough to survive
  correction for 66 comparisons). `run_screen.py` was adjusted to keep
  backtesting/reporting on the pre-correction survivors (clearly labeled) so
  the demo doesn't go silently empty, while printing both counts plus an
  explicit "these are leads, not confirmed edges" note when correction wipes
  out the raw hits — worth internalizing before treating any of the 8 flagged
  pairs above as more than a starting point for further investigation.
- Added 5 new tests (mark-to-market vs. close consistency, forced end-of-
  sample liquidation, BH correction against a hand-worked textbook p-value
  array, BH flagging nothing when all inputs fail, and BH-correction-can-only-
  shrink-not-grow-the-tradeable-set on the synthetic universe fixture).
  27/27 tests green.
- **Aligned the screen/report lookback windows.** `run_screen.py` was
  independently calling `generate_daily_signal_report(...)` with the
  function's own 400-day default instead of the screen's 900-day
  `LOOKBACK_DAYS`, which is exactly the "window-sensitive results" limitation
  the same-day README already flagged — and it wasn't hypothetical: the first
  overnight live run showed COP/SLB and BAC/GS as cointegrated in the screen
  table but `is_cointegrated=False` in the report table two sections later,
  same run, same pairs. Passing `lookback_days=LOOKBACK_DAYS` through fixed it;
  re-ran live and confirmed `hedge_ratio`/`is_cointegrated`/`half_life_days`
  now match exactly between the two tables.

### Bugs caught during this pass

3. **Force-liquidation test assumed a fixed holding period.** First attempt
   truncated the price series at `entry_date + 2 bars` assuming the trade
   would still be open — but with `entry_z=1.0`/`exit_z=0.3` (the tight
   thresholds used for fast test convergence), the first trade in the fixture
   exited naturally after only 1 bar, so the assumption was already false.
   Fixed by scanning `full_result["trade_log"]` for a trade actually held 2+
   bars before picking a truncation point, rather than assuming any fixed
   offset would land mid-trade.

## 2026-07-16 (later) — Risk-profile presets, first real backtest read on the 8 candidates

User's back and gave direction: start with the 8 raw-screen candidates
(COP/SLB, BAC/GS, BAC/MS, PG/WMT, GS/MS, GDX/GLD, PG/XLP, C/GS) rather than
widening the universe first, keep capital hypothetical, target a conservative
(~5% max drawdown) risk profile, and push to GitHub.

- Added `PairBacktestConfig.conservative()/.moderate()/.aggressive()` preset
  classmethods (smaller size + fewer concurrent pairs + tighter stop for
  conservative; the plain dataclass defaults are `.moderate()`). `run_screen.py`
  now selects via a `RISK_PROFILE` constant instead of a one-off hardcoded
  config, and `TOP_N_TO_BACKTEST` raised from 5 to 10 so all 8 current
  candidates get backtested rather than being cut off at 5.
- One new test asserting the three presets are internally valid (z-threshold
  ordering) and monotonically ordered by exposure (capital_per_pair,
  max_concurrent_pairs, stop_z all conservative <= moderate <= aggressive).
  28/28 tests green.
- **First live conservative-profile read on the 8 candidates** (~2.5 years,
  same yfinance pull): 58 trades, total_return=-1.09%, Sharpe=-0.80,
  max_drawdown=-1.32%. The drawdown target was met with room to spare, but
  the strategy was slightly unprofitable over this window for this exact pair
  set — consistent with the earlier finding that none of these 8 survived FDR
  correction. Read as: risk controls are working as designed, but the edge
  itself is weak/negative for this specific pair set and window, which is
  exactly why "start with these 8, widen later" was the plan rather than
  betting real capital on them as-is.
- GitHub push requested but blocked: the target repo
  (github.com/ryanakratzer-cpu/pairs-trading-engine) doesn't exist yet and
  `gh` CLI still isn't installed on this machine — confirmed the existing git
  credential manager can already authenticate to GitHub (`git ls-remote` on
  convertible-bond-engine's origin succeeds non-interactively), so once the
  empty repo exists on GitHub a plain `git push` should work without needing
  `gh` at all.

Pushed to GitHub once the user created the empty repo:
[github.com/ryanakratzer-cpu/pairs-trading-engine](https://github.com/ryanakratzer-cpu/pairs-trading-engine).
Confirmed the credential-manager auth theory from above — plain `git remote
add` + `git push -u origin main` worked with no prompts.

## 2026-07-16 (latest) — Out-of-sample validation

User asked how to improve price prediction / find real signals. Recommended
out-of-sample validation as the highest-leverage next step over widening the
universe, on the reasoning that a single in-sample cointegration test (even
FDR-corrected across pairs) says nothing about whether one specific
"significant" pair would have held up on data it wasn't fit to — that's a
different failure mode than the multiple-comparisons problem FDR addresses.
User said "do it."

- Added `validate_out_of_sample()` to `screening/cointegration.py`: splits a
  pair's history into an earlier formation window and later held-out
  validation window (`formation_fraction`, default 0.7), fits the hedge ratio
  and tests cointegration on formation data only, then applies that *same*
  hedge ratio (deliberately not re-estimated — re-fitting on validation data
  would defeat the purpose, since any two series can be made to look
  mean-reverting with a fitted coefficient) to the validation window and
  ADF-tests whether the spread is still stationary there.
- Wired into `screen_universe(..., require_out_of_sample_validation=True)`:
  adds `oos_formation_pvalue`/`oos_validation_pvalue`/`oos_validated` columns,
  `tradeable` additionally requires `oos_validated`. Pairs too short to split
  (< 2×`MIN_OBSERVATIONS` per side) are marked `oos_validated=False` rather
  than raising, so a mixed-length universe screen doesn't crash on one thin
  pair.
- `run_screen.py` now runs both corrections and backtests the out-of-sample
  survivors when there are any, falling back to the raw p<0.05 pool (clearly
  labeled "NONE passed out-of-sample validation, treat as exploratory") when
  there aren't — same "don't silently gate to empty output" pattern as the
  FDR integration.
- 5 new tests: validates a strong persistent synthetic relationship (needed a
  purpose-built stronger fixture than the shared `cointegrated_pair_prices` —
  see bug note below), rejects a relationship that's cointegrated in
  formation but permanently drifts in validation, short-circuits correctly
  when formation itself fails, raises on too-little data to split, and
  confirms the `screen_universe` wiring is opt-in and can only shrink
  `tradeable`, never grow it. 33/33 tests green.
- **Live result on the 8 raw-screen candidates: zero survived.** Every one of
  the 8 pairs that passed the raw p<0.05 + half-life filter had an
  out-of-sample validation p-value between 0.11 and 0.79 — nowhere close to
  significant. This is a much sharper answer than the FDR correction alone
  (which just said "not confident enough") — it directly shows these specific
  relationships were fit to formation-window noise and would not have been
  tradeable going forward. Confirms widening the universe is the right next
  move, not tuning risk parameters on this set further.

### Bugs caught during this pass

4. **First out-of-sample fixture wasn't strong enough to reliably validate.**
   Reused the shared `cointegrated_pair_prices` fixture (theta=0.1, tuned for
   the full-sample detection test's tolerances) for the "validates a
   persistent relationship" test — it failed intermittently because applying
   the *formation-only* hedge ratio to a smaller held-out window is a
   materially stricter test than the full-sample Engle-Granger test the
   fixture was built for; that run's validation p-value landed at 0.073, just
   above 0.05. Fixed by building a purpose-specific, stronger series in the
   test itself (theta=0.2, lower noise, 600 days instead of 500) rather than
   reusing a fixture tuned for a different test's power requirements.

## 2026-07-16 (night) — Wide universe, Kalman hedge, time exits, live intraday monitor

User asked for: the best ways to aid cointegration, focus on the two most
cointegrated stocks right now, a professional-grade max-return/min-risk
model, and a dynamic intraday view updating as fast as possible.

- **Universe widened** from 29 tickers/6 sectors to 73 tickers/13 sectors
  (added tech megacap, semis, healthcare, utilities, telecom, refiners, broad
  index ETFs) plus structural near-twins in `KNOWN_PAIRS` (GOOG/GOOGL,
  SPY/IVV/VOO, GLD/IAU, AMAT/LRCX, CAT/DE) — 197 candidate pairs screened.
- **Wide-screen results**: 27/197 raw hits, 3 survive FDR (SPY/VOO, IVV/VOO,
  QCOM/SMH), 1 survives out-of-sample validation (SO/XLU). The index-fund
  twins are the "most cointegrated" on paper (p as low as 1.5e-27) but have
  sub-day half-lives — no divergence to trade; the half-life band filter
  correctly rejected them.
- **Focus pair chosen: GDX/GLD** (gold miners vs gold bullion). Reasoning:
  full-window cointegrated (p=0.035) with in-band 24.9d half-life; the recent
  validation window is strongly stationary (p=0.014) and the trailing-252d
  regime fit is even stronger (p=0.0095, half-life 8.7d) — the relationship
  is strengthening, unlike DUK/SO which decayed; and it's the most
  economically grounded pair on the board (miners are a leveraged claim on
  gold). SO/XLU technically passed the formal OOS gate but is borderline
  everywhere else (full-window p=0.067) and produced 0 trades under the
  conservative regime gating — the risk controls correctly refusing a weak
  edge, noted as such.
- **KalmanHedgeRatio** (`signals/spread.py`): strictly causal 2-state
  (intercept, beta) random-walk Kalman filter, per-bar beta series +
  `hedge_ratio_mode="kalman"` in the backtester. Tests: recovers static beta
  (matches OLS), tracks a mid-sample structural break where full-window OLS
  lands at -0.85 vs true 0.6→1.1 (wildly wrong for both regimes), and beta
  causality (appending future bars doesn't change past betas).
- **Time-based exit**: `SignalConfig.max_holding_bars` → `TIME_EXIT` event;
  positions that haven't converged within ~2.5x half-life close instead of
  waiting for the price stop. Simulator treats it as a closing event.
- **Honest negative finding**: signals from a rolling z-score of the *Kalman*
  spread under-trade badly (0 trades at every delta tried, 1e-5 → 1e-8) —
  the adaptive beta absorbs exactly the divergences the strategy wants to
  trade, and the diffuse-prior warmup locks early betas onto a degenerate
  alpha/beta split. The canonical fix is trading Kalman *innovations*
  directly (Chan); flagged as future work. Regime-OLS mode is the
  recommended trading configuration; Kalman beta kept as a drift diagnostic.
- **`run_pair_study.py`**: single-pair deep dive (diagnostics + tuned
  regime-vs-Kalman comparison). On GDX/GLD it surfaced that the pair is in an
  **active divergence right now** — the tuned regime backtest's only entries
  were short-spread in the final week, all stopped out as z blew past +3.
  Live monitor shows z back at -0.76 on the trailing-regime baseline.
- **`run_live_monitor.py`**: intraday monitor polling 1-minute bars (default
  30s, floor 10s to respect the data source), hedge ratio fit on the trailing
  252d (same convention the backtester trades on — deliberately, so the live
  z is the quantity the backtest validated), auto-refreshing dark-theme HTML
  dashboard at `outputs/live_monitor.html` with entry/exit/stop bands, stale/
  market-closed detection with gentler off-hours polling, and the standing
  no-execution disclaimer. Verified live at 21:26 (market closed, stale
  handling confirmed working).
- 38/38 tests green.

## 2026-07-16 (late night) — 1000-path OU Monte Carlo + interactive Plotly visuals (subagent build)

User asked for professional, interactive visuals — specifically a Monte Carlo
showing all ~1000 trials — and explicitly requested a subagent for the task.
Delegated with a detailed brief covering house conventions (py launcher, no
network in tests, no em dashes in prints, disclaimer pattern, gitignore for
`interactive_*.html` matching the convertible-bond engine); verified its work
independently before committing.

- **`montecarlo/simulator.py`**: `fit_ou()` estimates theta/mu/sigma from the
  same AR(1) regression convention as `compute_half_life` (implied half-life
  matches exactly — asserted in a test); `simulate_spread_paths()` runs 1000
  seeded, vectorized OU trajectories anchored at the current spread;
  `simulate_strategy_pnl()` replays the real signal state machine on each
  path for a per-path P&L distribution.
- **`visualization/interactive.py`**: dark-theme Plotly dashboards saved as
  self-contained HTML — the 1000-path fan chart (all paths drawn via a single
  None-separated scatter trace for render performance), equity + drawdown,
  spread/z-score with signal bands and range slider, P&L histogram with
  percentile lines. Every chart carries a "research only, not investment
  advice" watermark. Static 200-path-subsample PNG fallback for vault embeds
  (matplotlib alpha-stacks 1000 raster lines into an unreadable block, so the
  PNG subsamples; the HTML keeps all 1000).
- **`run_montecarlo.py`**: real-data runner using the trailing-252d hedge
  convention shared with the backtester and live monitor.
- **GDX/GLD result** (seed 42, $10k notional, 90-day horizon, before costs):
  OU theta=0.080/bar (half-life 8.7d, matching the cointegration screen), 
  mean P&L +$577/path, median +$658, 5th pct $0 (paths that never trigger an
  entry), 95th pct +$1,646, **prob(profit) 62.6%**.
- 13 new tests (OU parameter recovery, seeded reproducibility, path anchoring,
  terminal-distribution sanity, HTML artifact creation). **51/51 green.**
- Synthetic test artifacts (`*SYN*`) excluded from the repo via .gitignore.

## 2026-07-16 (final push) — Forward-test journal + cost-adjusted Monte Carlo (parallel subagents)

Gap analysis question from the user ("anything missing or should we start
testing?") answered: the biggest gap WAS the testing infrastructure. Two
subagents ran in parallel on disjoint files:

- **`reporting/journal.py` + `run_journal.py`**: the forward-test record.
  `append_signals` logs each day's signal (hedge ratio, z, recommendation)
  with a UTC `logged_at` timestamp, idempotent per (as_of, pair) so re-runs
  never duplicate. `grade_journal` later rebuilds the spread with the LOGGED
  hedge ratio (no hindsight refit) and grades whether |z| moved toward zero
  over the next 10 trading bars; entries with |z| < 1 are `no_signal` (no
  directional prediction to grade), open windows are `too_recent`. The
  journal CSV is deliberately git-tracked — an un-fudgeable paper-trail.
  `run_screen.py --journal` also appends. Day-1 entry logged (GDX/GLD,
  z=-0.59, no_signal — correctly below the prediction threshold).
- **Cost-aware Monte Carlo**: `simulate_strategy_pnl` now models entry+exit
  transaction costs and slippage (mirroring the backtester's bps convention)
  plus short-leg borrow cost (50bps annual on half the notional, pro-rated by
  holding period). GDX/GLD headline moved from mean +$577/62.6% prob-of-profit
  (gross) to **+$556/62.3% net of costs** — barely dented because the
  strategy trades infrequently, which is itself a useful robustness signal.
  run_montecarlo.py prints both; the histogram shows net.
- 13 new tests across the two workstreams; **64/64 green**.

## 2026-07-17 — Kalman innovations, websocket streaming, macro regime overlay (3 parallel subagents) + GitHub Pages

User asked: what model are we running, is it dynamic, would Kalman/TradingView
help, and "make sure real-time tracking happens" — then "make it as dynamic as
possible" and finally "get the interactive charts onto GitHub." Three
subagents ran in parallel on disjoint files; all verified and merged.

- **Kalman innovation signals** (`hedge_ratio_mode="kalman_innovation"`):
  trades the filter's standardized one-step-ahead surprises instead of a
  rolling z of the Kalman spread — the canonical fix for the documented
  under-trading. Key calibration finding: the filter's observation variance
  sets the innovation z-score's noise floor; at the class default (1e-3,
  ~3% daily noise) real ETF pairs (~1% noise) never reach |z|=2, so a new
  `kalman_innovation_obs_variance` config field (default 1e-4) calibrates
  it. On GDX/GLD all three |z|>2 innovation signals fell in periods the
  rolling re-cointegration gate had (correctly) marked untradeable — the
  risk gate and the signal model are genuinely independent checks.
- **Websocket real-time monitor**: `run_live_monitor.py --mode stream`
  (default) subscribes to Yahoo's websocket via yfinance 1.5's WebSocket
  client — push-based live ticks, no API key — recomputing spread/z at most
  every 2s, with automatic fallback to polling on connect failure and a
  market-closed quiet mode. Verified LIVE during market hours: ticks flowed,
  z updated in real time (GDX/GLD z=-1.35 at verification). Dashboard shows
  a STREAMING/POLLING badge and refreshes every 5s. TradingView was
  considered and rejected: no public data API — it's a charting front-end
  over the same exchange data; websocket streaming is the actual upgrade.
- **Macro regime overlay** (`screening/regime.py`): causal rolling
  percentile-rank stress mask from VIX + gold-vol (GVZ), macro panel also
  tracking 10y yields, the dollar, and oil; spread-vs-macro correlation
  diagnostics. Current reading at build time: CALM (VIX pct 0.45, GVZ 0.63),
  164/619 lookback days stressed. Informational in run_screen.py for now —
  wiring the mask into the backtester's tradeable gating is the next step.
- **105/105 tests green** across all three workstreams (25 monitor + 11
  regime + 6 innovation tests added).
- **GitHub Pages**: interactive Plotly dashboards published under `docs/`
  (fan chart, net-P&L distribution, spread/z with bands, equity curve,
  live-monitor snapshot) with a styled index.html linking them; README
  documents the one-time Settings->Pages step. Fresh regeneration at
  publish time: net-of-costs prob(profit) 63.7% on GDX/GLD.

## Open threads

- [ ] User flips on GitHub Pages (Settings -> Pages -> main /docs) to make
      the dashboards live at ryanakratzer-cpu.github.io/pairs-trading-engine
- [ ] Wire the regime stress mask into PairBacktester's tradeable gating
      (currently informational only)
- [ ] **Forward test running** — grade the journal after ~2 weeks of entries;
      the first gradable verdicts land ~10 trading days after the first
      |z| >= 1 entry
- [ ] Walk-forward validation of the trading rule parameters (entry/exit/stop
      were chosen with full-sample knowledge; textbook defaults, but unproven
      out-of-window)
- [ ] Kalman **innovation-based** signal generation (trade the filter's
      innovations rather than a rolling z of the Kalman spread) — the proper
      fix for the under-trading found above
- [ ] Hedge-ratio-weighted (rather than flat dollar-neutral) leg sizing
- [ ] Basket/multivariate cointegration (Johansen test) as a complement to
      pairwise Engle-Granger
- [ ] Event/regime filters (e.g. exclude entries around earnings, skip
      extreme-volatility regimes)
- [ ] Watch GDX/GLD divergence: trailing-regime stats are strong (p=0.0095,
      HL 8.7d) but the last week saw z > +3 short-spread stop-outs — either
      an opportunity setting up or early regime breakdown

## 2026-07-18 — Efficient frontier portfolio allocation

- **portfolio/optimizer.py**: each pair strategy = one asset. mu from the
  pair's OU fit (analytic reversion heuristic, (entry-exit)z * stationary_std
  per trade, ~4 half-lives per cycle) instead of noisy historical means;
  covariance = Ledoit-Wolf (2004) shrinkage toward scaled identity,
  implemented directly (no sklearn dependency). Long-only, fully invested,
  40% per-pair cap; SLSQP via scipy. Tangency (max Sharpe), min-variance,
  efficient frontier, Dirichlet random-portfolio cloud.
- **Universe widened** 73 -> 105 tickers, 13 -> 21 sectors, 197 -> 248 pairs
  (airlines, autos, discount retail, insurance, defense, homebuilders,
  exchanges, rates/credit ETFs + 10 cross-sector known pairs). Raw screen
  survivors went 29 -> 43; OOS survivors still 2 (SO/XLU, AEP/DUK decayed).
- **run_portfolio.py** live run: allocated across top-8 raw-pool pairs
  (labeled EXPLORATORY - <3 OOS survivors). Shrinkage intensity 0.37.
  Model: tangency Sharpe 16.79 vs 1/N 15.43. Realized (in-sample): tangency
  0.11, min-var 0.28, 1/N 0.03 - min-variance won on realized Sharpe and
  halved max drawdown (-2.0% vs -3.8%). Key honest finding: OU expected
  returns (+20-55%/yr) vastly exceed realized (~0%) because the heuristic
  assumes always-tradeable cycles while the re-cointegration gate blocks
  entries most of the time - model Sharpes are RANKING information only.
- **Visuals**: interactive_efficient_frontier_live.html (Sharpe-colored
  cloud + frontier + CAL + labeled tangency/min-var/1/N markers, percent
  ticks, major+minor gridlines) and interactive_allocation_comparison.html
  (realized equity overlay with per-scheme Sharpe in legend). Verified
  rendering in browser (all 6 traces, gridlines, colorbar).
- **122/122 tests green** (17 portfolio tests added).

### New open threads
- [ ] Scale OU mu by expected fraction-of-time-tradeable (gate-aware
      expected returns) so model and realized Sharpe live on one scale
- [ ] Walk-forward the allocator: fit weights on formation window, measure
      on held-out window (the in-sample comparison can't crown a winner)
- [ ] Wire optimized weights into PairBacktester (per-pair capital instead
      of flat capital_per_pair)

## 2026-07-19 — Walk-forward harness, regime/event gating, gate-aware mu

Committed+pushed 57cb7ab (portfolio optimizer, user-committed 07-18), then:

- **backtest/walkforward.py**: formation/holdout rolling splits (252d/63d,
  step 63d), three studies: pair_survival_study (screen decay rate),
  parameter_study (formation-picked bands vs textbook defaults on holdout),
  allocation_study (formation-fit weights vs 1/N on holdout). ADF power
  caveat documented: 63-bar holdout rates are conservative lower bounds.
- **Regime gate WIRED IN**: PairBacktester.run(..., entries_allowed=mask);
  run_screen.py now gates with VIX/GVZ stress mask AND screening/events.py
  FOMC/election blackout windows (24 FOMC decision days 2024-2026 + 2
  elections, day-before through day-after). Blocks new entries only.
- **Gate-aware OU mu**: ou_expected_annual_return(tradeable_fraction=...)
  scales cycles/yr by the realized fraction of entry-allowed days;
  run_portfolio.py passes each pair's prepared["tradeable"].mean().
- **141/141 tests green** (19 added).

### Walk-forward live results (5 windows, 105 tickers, 248 pairs)
- **SURVIVAL RATE 7%** at p<0.05 (11% at p<0.10, median holdout p=0.44):
  screened pairs overwhelmingly decay within the next quarter. Most
  persistent: ABT/MRK, AEP/DUK, ALL/TRV, D/XLU, DUK/SO (4/5 formation
  passes each, 25% holdout survival).
- **Textbook bands beat optimization**: mean holdout Sharpe 1.32 (defaults)
  vs 1.01 (formation-optimized) on GDX/GLD. Overfitting tax 0.17. Verdict:
  do NOT tune entry/exit/stop.
- **Min-variance wins again OOS**: mean holdout Sharpe +0.19 vs 1/N -0.15,
  tangency -0.24 (only 5 windows - directional, not decisive).
- Gated live screen: combined gate blocks 209/617 days; VIX pct rank 0.71
  (rising, threshold 0.80). OOS survivors down to 1 pair. Journal appended:
  GDX/GLD re-cointegrated on trailing window (p=0.034, HL 11d, z=-1.26,
  drifting toward long-entry).

### Strategic read
The 7% survival rate is the project's most important number: pair selection
decays faster than a quarterly rebalance can exploit. Cleverness downstream
of the screen (signals, allocation) cannot overcome it. Directions that
respect this: trade only multi-window-persistent pairs (ABT/MRK, AEP/DUK,
ALL/TRV, D/XLU cluster), shorten the holding horizon, or accept regime-level
rather than pair-level stability (min-variance across many weak pairs).

## 2026-07-20 — Focus book (persistent-pair portfolio mode)

Answered the user's structure question (many pairs vs two): the engine trades
a BOOK of dollar-neutral pair spreads, never a single pair. GDX/GLD was only
the deep-dive/diagnostics pair. The 7% OOS survival rate makes single-pair
concentration unsafe, so the evidence points to a diversified book of the most
PERSISTENT pairs. Built that as a first-class mode.

- **screening/focus_book.py**: fixed FOCUS_BOOK of 5 FocusPair records, one
  per sector, selected by walk-forward formation-passes (not day p-value):
  ABT/MRK (healthcare, 4/5), ALL/TRV (insurance, 4/5 + OOS-validated),
  DUK/SO (utilities, 4/5), COST/PEP (staples, 3/5), COP/SLB (energy, 3/5).
  Insurance cluster (ALL/TRV/AIG/PRU/MET) deduped to one slot. Each member
  carries its evidence string. Helpers focus_pairs()/focus_tickers().
- **run_focus_book.py**: fetch focus tickers -> macro+event entry gate ->
  gated portfolio backtest -> min-variance allocation across the book ->
  daily signal report -> journal append. Flags: --review (print evidence),
  --journal (append+grade).
- **148/148 tests green** (7 focus-book integrity tests: dedup, one-per-sector,
  no structural near-twins, helpers).

### Live focus-book run (2026-07-20)
- Gate blocks 208/616 days. Backtest (conservative, gated): 30 trades,
  total_return -0.02%, Sharpe -0.02, maxDD -0.56%, win_rate 50%.
- Min-variance weights: DUK/SO 40% (capped), ALL/TRV 22.6%, COST/PEP 15%,
  ABT/MRK 13.2%, COP/SLB 9.2%. Min-var vol 2.0% vs equal-weight 2.5%;
  realized Sharpe min-var -0.21 vs equal-weight -0.50 (min-var less bad,
  consistent with walk-forward).
- Today's signals: **ALL/TRV ENTER_LONG_SPREAD (z=-2.03, p=0.014)** — the
  strongest pair at an actionable entry. ABT/MRK HOLD z=+2.19, COP/SLB HOLD
  z=+2.47, DUK/SO HOLD z=-1.44, COST/PEP NO_POSITION (dropped out, p=0.056).
  5 rows appended to journal.

### Open threads
- [ ] Grade the focus-book journal after ~2 weeks; ALL/TRV entry is the first
      real forward-test with an actionable signal
- [ ] Refresh focus-book membership from a monthly walk-forward run (drift
      check); COST/PEP is the weakest member (p=0.047, 29d half-life)
- [ ] Wire min-variance focus weights into PairBacktester capital allocation
      (still flat capital_per_pair inside the backtest itself)

## 2026-07-20 (later) — Two parallel subagents: book refresh + backtest audit

Ran two subagents on disjoint file scopes (no commits by agents; orchestrator
verified + committed).

### Subagent A — monthly book-refresh tooling
- **screening/book_refresh.py**: rank_pairs() re-runs the persistence machinery
  (screen_universe FDR+OOS, pair_survival_study walk-forward), joins per pair,
  ranks by formation-passes desc / median holdout p asc / adf_p asc, excludes
  near-twins + sub-5d half-life, dedupes one-per-sector via SECTOR_ETFS.
  compare_to_current_book() flags each FOCUS_BOOK member KEEP/DROP? and lists
  challengers. PROPOSAL ONLY — never mutates focus_book.py.
- **run_book_refresh.py**: full-universe run -> dated markdown report to
  vault\01_RAW_CLIPS\quant_research\book_refresh_reports\book_refresh_YYYY-MM-DD.md
- **9 refresh tests** (no network). 
- Live run finding (BOOK IS DRIFTING): fresh proposed book = ALL/TRV (keep),
  D/XLU (NEW utilities, outranks our DUK/SO), ABT/MRK (keep), LUV/UAL (NEW
  airlines), COST/PEP (keep). Challengers: D/XLU (rank 2), LUV/UAL (rank 8).
  Under the strict "passes FDR+OOS screen" rule none currently "qualify" — a
  faithful reflection of the 7% survival context; report keeps passes_screen
  and in_top_N as separate columns so nuance is visible.

### Subagent B — backtest audit (checks, rechecks, improves)
- **1 REAL BUG FIXED** (backtest/simulator.py END_OF_SAMPLE): on a ragged
  price panel, an open position on a pair whose data ends before the global
  last date was silently DROPPED (entry cost charged, P&L never realized,
  no trade-log row) — broke the "always fully realized" contract. Fix:
  liquidate at that pair's OWN last bar. Aligned-panel behavior unchanged.
- Ruled out (verified, no defect): look-ahead/causality (future-price
  perturbation test = bit-identical equity), cost/slippage accounting,
  sizing/sign conventions, concurrency cap off-by-one, metrics math
  (hand-computed vs toy fixtures), walkforward split leakage.
- **5 regression/coverage tests added** (ragged liquidation, causality guard,
  hand-computed metrics, degenerate-curve safety, profit-factor edges).

### Combined: 162/162 tests green. Committed + pushed.

### Open threads
- [ ] Decide on the drifting book: D/XLU vs DUK/SO (utilities), add LUV/UAL
      (airlines)? Next monthly refresh report will re-confirm.
- [ ] Schedule run_book_refresh.py monthly (local, needs the vault filesystem).

## 2026-07-21 — Council of agents: book-drift decision → KEEP AS-IS (4-0)

Convened a 4-lens council (statistical, economic, portfolio-risk, skeptic) to
decide whether to act on the 2026-07-20 refresh drift (D/XLU outranking DUK/SO;
LUV/UAL challenger). UNANIMOUS: keep the book unchanged. No edit to focus_book.py.

Decisive facts (agreed across lenses):
- D/XLU does NOT cointegrate on the full window (ADF p=0.70, no computable
  half-life); DUK/SO does (p=0.006, ~18d half-life). D/XLU outranks it ONLY
  because the refresh sort key weights median_holdout_p above adf_p.
- D/XLU is a stock-vs-own-sector-ETF (Dominion is a constituent of XLU) — a
  mechanically-tight index-inclusion relationship, the same pathology the
  near-twin exclusion exists for. It slips through because NEAR_TWINS only
  enumerates ETF-vs-ETF duplicates + share classes, not stock-vs-own-index.
- LUV/UAL: ADF p=0.314 (not cointegrated), 0 holdout survivals, fails screen;
  airlines carry fat-tailed idiosyncratic gap risk the event-gate can't catch.
- Portfolio lens (empirical): book already near-orthogonal (avg pairwise
  strategy-return corr ~ -0.02, div ratio ~1.04) — neither change adds
  diversification; nothing to gain, only signal-validity to lose.
- Every current member is flagged DROP? / fails today's strict FDR+OOS screen,
  but all remain genuinely cointegrated (ADF p<=0.056) and market-neutral,
  while both challengers are neither. The drift is inside the noise band of 5
  windows, in the ~7% survival regime.

Two tool-improvement findings the council surfaced (NOT yet implemented):
- [ ] book_refresh.py ranking key can promote a non-cointegrated pair: sort
      weights median_holdout_p above adf_p, so a pair with no full-window
      cointegration / no half-life can outrank a solid one. Fix: gate on
      full-window cointegration (or make adf_p primary / require a computable
      half-life) before persistence tie-breaks.
- [ ] NEAR_TWINS filter misses stock-vs-own-sector-ETF (e.g. D/XLU). Add a
      rule: exclude a single stock paired with a sector ETF that contains it.
- [ ] Governance (skeptic's proposed rule): only drop a member after the SAME
      challenger out-ranks it for N (2-3) consecutive monthly refreshes AND the
      challenger passes the FDR+OOS screen at least once. Codify to prevent
      churning on one noisy run.

Reaffirmed: ALL/TRV remains the one genuinely differentiated member (rank 1,
sole OOS-validated, ~12d half-life); its ENTER_LONG_SPREAD journal entry is the
first real forward test — dropping members now would also reset that record.

## 2026-07-23 — Implemented the council's 3 book_refresh improvements

Acted on the 2026-07-21 council's tool-improvement findings (screening/book_refresh.py):
1. **Cointegration gate as PRIMARY sort key**: is_cointegrated_full (screen's
   is_cointegrated + computable half-life) now sorts first, so a non-cointegrated
   pair can never outrank a cointegrated one on formation-passes. Kills the
   D/XLU-over-DUK/SO artifact.
2. **Stock-vs-own-sector-ETF exclusion**: _is_stock_vs_own_sector_etf drops a
   single stock paired with a sector ETF that holds it (SECTOR_ETF_TICKERS +
   same SECTOR_ETFS group). D/XLU now excluded like a near-twin.
3. **Hysteresis governance** (governance_actions, GovernanceConfig.n_consecutive=2):
   a member is REPLACE only after the SAME challenger out-ranks it AND passes the
   FDR+OOS screen for N consecutive monthly refreshes; else WATCH/KEEP.
   run_book_refresh.py persists refresh_history.json (idempotent per date) and
   prints/report a governance action table.

- **170/170 tests green** (8 new book_refresh tests).
- **Live re-run confirms**: D/XLU GONE (excluded), DUK/SO restored as utilities
  leader (KEEP); LUV/UAL GONE (cointegration gate sank it). Governance verdict:
  NO REPLACE — keep book as-is. New drift noted but correctly WATCH-ed: AIG/PRU
  (5 formation passes) now out-ranks ALL/TRV in insurance, but fails the screen,
  so ALL/TRV = WATCH not REPLACE. Hysteresis working as designed.
- Committed + pushed.

Note: the book itself (focus_book.py) remains UNCHANGED — the council decided
keep-as-is, and the refresh tool now correctly re-confirms that rather than
flagging phantom drift each month.

## 2026-07-28/29 — Multi-day tracking, decision grading, dashboard track record

Four subagents (one died on a session limit; orchestrator finished its work).
211/211 tests green (was 170).

### New: persistent multi-day track record
- **reporting/daily_performance.py** + **run_daily_tracker.py**: append-only
  session log at outputs/daily_performance.csv, idempotent per (date, pair);
  grade-on-read of past decisions (journal.py pattern, never mutates the log);
  performance_summary() = cumulative P&L/return vs SPY, days-beating-market,
  active-vs-flat session counts, Sharpe. Handles the "today's bar not published
  yet" case by falling back to the last COMPLETED session and SAYING SO.
- Registered as a daily post-close scheduled task (see below).

### BUG FOUND + FIXED: the tracker's return convention was dishonest
First build set pair_return_pct = position-AGNOSTIC spread move, then
benchmarked it against SPY. Result on the real first run: ALL/TRV NO_POSITION,
pnl $0.00, but recorded +2.97% return and +2.74% excess — and the summary said
"cumulative return +0.712%, beat the market 1/1 days (100%)" while cumulative
P&L was -$62.39. i.e. it claimed a 100% win rate against the market on a losing
day. Fixed by splitting the columns:
  strategy_return_pct = position * spread move  (0 when flat; the ONLY column
                        benchmarked vs SPY; pair_pnl_usd derived from it)
  spread_move_pct     = position-agnostic diagnostic, clearly labelled
After fix, same session: P&L -$62.16, return -0.124%, beat market 0/1. Signs
agree. Regression tests pin flat-day=0%, sign agreement, and summary coherence.

### New: historical decision grading (answers "were our buy/sell calls right?")
- **reporting/decision_grader.py** + **run_grade_decisions.py**: grades every
  decision by OUTCOME (P&L net of costs) and THESIS (did the spread revert).
- Live focus-book result (900d, conservative, 124 decisions / 55 round trips):
  entry hit-rate 42% by outcome; 86% by thesis RAW but 41% once artifacts are
  discounted; 0/22 stop-losses justified; total +$118 on $100k over ~2.5y.
  ALL/TRV (+$285, 62%) carries the book; ABT/MRK (-$339) loses more than the
  book makes.
- Verdict: NOT trustworthy yet. 42% on 55 trades ~ coin flip (95% CI ~29-56%),
  and it is in-sample/circular (book selected on this same history).

### TWO REAL DEFECTS SURFACED (not yet fixed — awaiting user decision)
1. **Entry-before-stop ordering in signals/spread.py**: when flat, generate_signals
   checks entry with NO guard that |z| < stop_z, so it opens a position at e.g.
   z=-5.28 that is instantly stop-eligible. 24 of 63 entries fired beyond the
   stop band. These score as "converged" on a small bounce while reliably losing.
   This artifact is what inflated the thesis hit-rate from 41% to 86%.
   Proposed fix (a): refuse new entries when |z| >= stop_z.
2. **Stop-loss fires on noise**: 0 of 22 stops justified — in all 22 the spread
   snapped back within 10 bars. stop_z=3.0 (conservative) converts recoverable
   drawdowns into realized losses. Proposed fix (b): re-tune/widen stop_z.
   Recommendation: do (a) first, re-grade, THEN (b) — doing both at once
   confounds the evidence.

### Dashboard
- Two tabs: "Pair analysis" (now with Sharpe / annualized return / annualized
  vol / profit factor / max drawdown / half-life) and "Track record (multi-day)"
  reading the tracker CSV defensively (friendly empty state, never crashes).
  Cumulative-vs-SPY line chart, daily-excess bar chart, session table with
  "Our return" vs a separately-labelled "Spread move (diag.)" column.

## 2026-07-29 — FIRST LIVE REAL-TIME TEST (model vs market)

Goal: measure THE MODEL (not discretionary calls) against the market, intraday.

### Council (3 agents: statistical / risk / skeptic) — UNANIMOUS 3-0 NO NEW TRADE
Decisive reason was MECHANICAL, not judgment: **2026-07-29 is an FOMC decision
date** in screening/events.py; the entry gate blocks 07-28..07-30. The engine
refuses new entries by design. Verified independently.
Supporting findings (all verified):
- ABT/MRK beta = -0.580 => "short spread" is short ABT AND short MRK: net
  -$15,800 on $10k notional. NOT market-neutral; a directional sector bet.
  Beta unstable (-0.14 -> -0.69 in 6 months). COST/PEP beta FLIPPED SIGN
  (-0.74 -> +0.49) => cointegrating vector unidentified.
- On the trailing 252d window the engine actually fits, ABT/MRK is NOT
  cointegrated (rolling refits p=0.4984, p=0.253) despite full-history p=0.0101.
- z=+2.05 was 0.05 past the line after 1.89/1.87 — inside parameter noise.
- Statistical agent flagged (UNVERIFIED, for a future session): test_pair_
  cointegration uses plain adfuller() on ESTIMATED residuals with standard ADF
  critical values; Engle-Granger/MacKinnon values are required. If right, EVERY
  p-value in the project is biased toward "cointegrated". HIGH PRIORITY to check.

### My error, corrected in the open
First "pre-registered" snapshot was written from a script that BYPASSED the
event gate and stamped the prior session's prices. Did NOT delete: appended
5 reversing CLOSE rows at identical prices (exactly $0.00 P&L) with the reason
in the row. Erroneous rows remain visible in outputs/paper_trades.csv.

### BUG FOUND + FIXED (cb2c0b3)
reporting/daily_performance.py generated signals WITHOUT the event gate, so the
multi-day tracker would log entries on blackout dates the live engine refuses —
inflating the track record with trades that never happen. Now gated by default
(apply_event_gate=False for synthetic-date tests). Regression test pins that a
fresh |z|=2.5 gives an entry ungated and NO_POSITION gated. 212 tests green.

### Live result (intraday 13:25 ET, ~2.5h to close)
Model book: ABT/MRK SHORT_SPREAD (-1, legacy position opened pre-blackout);
ALL/TRV, DUK/SO, COST/PEP, COP/SLB all FLAT ($0 contribution).
  ABT +1.61%  MRK +0.05%  => spread WIDENED against the short
  Model P&L: -$163 (-1.63%) beta-weighted | -$78 (-0.78%) dollar-neutral
  SPY -0.52%  => LOSING, and BEHIND the market on both conventions.
  Cash (flat) would have beaten us: 0% vs SPY -0.52%.
The council's specific warning is playing out: ABT gapped +10.7% on 07-16 and
keeps drifting up; shorting the spread = fading a live re-rating.

### OPEN ISSUE raised by this test (priority)
P&L convention mismatch: backtest/simulator.py sizes legs DOLLAR-NEUTRAL and
ignores beta's sign, while reporting/daily_performance.py and paper_trades.py
mark beta-weighted log-spread. Harmless at beta>0, NEARLY OPPOSITE at beta<0
(ABT/MRK -1.63% vs -0.78%, a 2x gap). Must reconcile before any track-record
number involving a negative-beta pair is trusted.

Post-close gated number recorded by run_daily_tracker.py (scheduled 16:30).

## 2026-07-30 (evening) — Book reconstituted; close-to-close measurement

### Result of the day (2 open positions, close-to-close)
ABT/MRK short-spread +$249.11 (ABT -2.21%, MRK -0.44%); DUK/SO +$42.08.
BOOK +$291.19 = +1.46% on $20k. SPY +1.68%. Profitable, slightly behind a
strong tape. Forecast scorecard 4 HIT / 2 MISS.
CAVEAT: nearly all of it was the OVERNIGHT GAP (+$276 vs -$24 intraday), earned
from a directional healthcare short the new negative-beta gate now forbids. And
prediction #3 was backwards - I forecast MRK weakness; ABT collapsed instead.
Right P&L direction, wrong mechanism = luck, logged as a miss.

### FIX 1 - close-to-close measurement (the day's most important finding)
Tracker measured OPEN->CLOSE. Positions are held overnight (half-lives 6-12d),
so the overnight gap was invisible: 07-30 booked ABT/MRK at -$43 when the true
close-to-close was +$252. The gap WAS the day. Strategy return AND the SPY
benchmark now anchor on the prior close. Regression test: 5% overnight gap with
zero intraday move must not score ~0.

### FIX 2 - book reconstituted on proper Engle-Granger evidence
Re-screened all 248 pairs with statsmodels coint() on the trailing 252d window
(the window beta is actually fit on). Only 15 of 248 survived; only DUK/SO from
the old book. Admission now requires: EG p<0.05 on 252d; beta POSITIVE and in
[0.25,4.0] (near-zero beta = ~94% directional despite being nominally a pair);
half-life 5-30d; no near-twins; one per sector.
  OUT: ABT/MRK (beta -0.578, EG p=0.56), COST/PEP (beta -0.642, 0.31),
       COP/SLB (0.40), ALL/TRV (0.11)
  IN : DUK/SO 0.0027 b+0.893 hl8.0 | JNJ/MRK 0.0106 b+0.799 hl6.4 |
       AVGO/NVDA 0.0122 b+1.236 hl7.2 | GDX/GLD 0.0185 b+1.419 hl7.3 |
       ROST/TGT 0.0435 b+0.966 hl12.1 | GD/RTX 0.0474 b+0.448 hl8.0 (weakest)
  Excluded despite strong p-values: CL/XLP (0.0027) and NVDA/SMH (0.0150) are
  stock-vs-own-sector-ETF; AIG cluster collapsed to one slot (AIG/PRU 0.0002
  rejected anyway on beta +0.191 = too directional); GDX/IAU dropped as a
  duplicate of GDX/GLD.

### Track record archived, NOT backfilled
outputs/archive/daily_performance_openclose_pre_2026-07-31.csv holds 07-28..30.
Cannot append: different convention AND different book. Deliberately not
backfilled - the new book was selected using data through 07-29, so computing
"its" performance over those days would be look-ahead. Live record for the new
book starts fresh 2026-07-31.

### 228 tests green. Committed 5240c28.

### Setup for 2026-07-31 (gate OPEN - first tradeable day since 07-27)
  DUK/SO    z=+1.90  (0.10 from entry)  <- strongest pair, nearly firing
  AVGO/NVDA z=+1.75  (0.25 from entry)
  JNJ/MRK   z=+0.83 | GDX/GLD z=-0.80 | ROST/TGT z=+0.72 | GD/RTX z=-0.12

## 2026-07-31 (Fri) — First clean win; exit-attribution bug; prep for Mon 08-03

### Result: BOOK +$86.57 (+0.144%), SPY +0.72%
DUK/SO completed a FULL mean-reversion cycle: z +1.11 -> +0.02, hit the exit
band, closed +$86.57 (+0.87% on the position). All other pairs flat.
This is the first THESIS-CONSISTENT win: positive beta (+0.894, genuinely
market-neutral), strongest cointegration in the book (EG p=0.0027), and the
profit came from spread CONVERGENCE — not, as with Wed's ABT/MRK, from a
directional selloff on a negative-beta position.
Forecast scorecard 4 HIT / 2 MISS (missed: AVGO/NVDA never fired, z collapsed
2.09->0.89; and I predicted DUK/SO would NOT exit — it exited by converging).

### BUG FOUND + FIXED (c092e15): winning exits booked as ZERO
Tracker attributed P&L using the POST-signal position. On an EXIT day that is 0,
so the session booked $0.00 despite the position having been held all day and
the profit realized. Live: DUK/SO spread moved -0.87% in our favour (+$86.57),
tracker recorded 0.00. This zeroed out EVERY WINNING EXIT — precisely the
sessions where P&L is realized. P&L now accrues on `position_held` (prior bar's
position), added as a column so attribution is auditable. On an ENTRY bar
position_held=0, which is correct: we enter at the close, holding nothing during
that session. 3 tests corrected to key off position_held (not weakened).

### Refresh aligned with the book's rules (e90a6c9)
The monthly refresh (fires Sat 08-01 08:00) still ranked on the BIASED
ADF-on-residuals p-value with no hedge-ratio constraint — it would have proposed
pairs the 07-30 reconstitution rejected, contradicting the book monthly. Now
gates on the proper EG p-value when available and requires beta in [0.25, 4.0].
230 tests green.

### Running tally of live-only bugs (3 days, 4 bugs)
All four would have MISREPRESENTED the track record, and none was visible from
the test suite alone:
  1. open->close window hid overnight gaps (07-30)
  2. stale CSV cache meant the current session could never be recorded (07-29)
  3. ungated signals logged entries the real engine refuses (07-29)
  4. winning exits booked as $0.00 (07-31)

### State going into Mon 2026-08-03 (gate OPEN, no FOMC)
BOOK IS ENTIRELY FLAT — DUK/SO exited Friday, nothing else held.
  pair        z      to entry   beta     hl    EG p(252d)
  DUK/SO     +0.02    1.98     +0.894   9.4    0.0369
  JNJ/MRK    -0.81    1.19     +0.795   6.3    0.0112
  AVGO/NVDA  +0.89    1.11     +1.230   6.9    0.0124
  GDX/GLD    -0.51    1.49     +1.407   6.7    0.0077
  ROST/TGT   +0.94    1.06     +0.961  11.5    0.0395
  GD/RTX     +0.16    1.84     +0.449   7.9    0.0435
Nothing is within 1.0 of a trigger; all six still cointegrated on the fitted
window. Base case for Monday is a flat, quiet book earning exactly 0%.
Closest to firing: ROST/TGT (1.06) and AVGO/NVDA (1.11).
