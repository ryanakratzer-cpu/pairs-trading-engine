# HANDOFF — Pairs Trading Engine

**Written 2026-08-05. Read this before touching anything.**
Compiled from three independent read-only audits (architecture / dashboards+data / findings+history).

---

## 0. What this is, and the one thing you must internalise

A cointegration-based statistical-arbitrage research engine. Python, ~13,000 lines, 230 passing tests.
**It never places an order, never talks to a broker, never sizes a real position.** Every module docstring says so. Keep it that way.

**THE HONEST VERDICT: this strategy has no measurable gross alpha at daily frequency.**

Measured 2026-08-03 across all 248 then-candidate pairs, n = 1,681–3,460 round trips:
gross P&L per trade **−$7.54 to +$0.62** — indistinguishable from zero, *before* the ~20bp round-trip cost.
Net across the universe: **−$33k to −$95k**.

Everything that once looked like edge was **selection bias**: the 6 book pairs showed +$67/trade while the other 242 showed −$23/trade, and *all* of the book's outperformance sat inside the half of the sample the book was selected on.

**The project's real output is its measurement discipline.** Sixteen defects found and fixed, eight of them in the measurement layer, every one of which had misrepresented results. Four were found *only* by running live against numbers that could be checked independently — the 230-test suite was green throughout.

Do not present any performance number from this repo as evidence of edge. The README leads with this; keep it that way.

---

## 1. Locations

| What | Where |
|---|---|
| Repo | `C:\Users\ryana\OneDrive\Desktop\Ryan's Obsidian\01_RAW_CLIPS\quant_research\pairs-trading-engine` |
| Vault parent | `...\01_RAW_CLIPS\quant_research` |
| GitHub | `https://github.com/ryanakratzer-cpu/pairs-trading-engine` (branch `main`) |
| GitHub Pages | `https://ryanakratzer-cpu.github.io/pairs-trading-engine/` (source: `main` / `docs`) |
| Forecasts | `...\quant_research\forecasts\forecast_YYYY-MM-DD.md` |
| Book refresh reports | `...\quant_research\book_refresh_reports\` |
| Project diary | `...\quant_research\Pairs Trading Engine Implementation Log.md` ⚠️ **ends 2026-07-31, stale** |

**Interpreter: `py`, never `python`** (the Store alias is broken on this machine).
`py` → `C:\Users\ryana\AppData\Local\Python\pythoncore-3.14-64\python.exe` (3.14.5).
No packaging — **run everything from the repo root**; imports are bare top-level names.

```
py -m pip install -r requirements.txt
py -m pytest -q                  # expect 230 passed
py run_demo.py                   # offline; exit 0 = every stage verified
py run_focus_book.py --review    # offline; prints the book + its evidence
```

---

## 2. Sign and naming conventions — memorise these

- `spread = log(A) − β·log(B)` (no intercept).
- **`position = +1` = LONG spread** = long A, short β·B, entered when `z < −entry_z`.
  **`position = −1` = SHORT spread**, entered when `z > +entry_z`.
- Events: `ENTER_LONG_SPREAD`, `ENTER_SHORT_SPREAD`, `EXIT`, `STOP_LOSS`, `TIME_EXIT`, `HOLD`, `NO_POSITION`.
- **Every gate blocks NEW ENTRIES ONLY.** Re-cointegration, macro stress, event blackout, negative-beta, `max_entry_z` — an open position always still EXITs / STOP_LOSSes / TIME_EXITs on its own terms. Tests encode this; "fixing" a gate to force exits breaks causality and attribution tests.
- **Missing data defaults to allowed/calm everywhere.** Deliberate: a spotty macro feed must never silently halt the strategy.
- Sharpe convention project-wide: **rf = 0, 252 periods/year, std(ddof=1)**.

### The four columns people get wrong

| Column | Meaning |
|---|---|
| `position` | **Post-signal** state at the close — what we hold going into tomorrow. **0 on an EXIT day.** |
| `position_held` | The position **actually held across the session** (prior bar's state). **P&L ACCRUES ON THIS ONE.** 0 on an entry bar (we enter at the close, holding nothing during it). |
| `spread_move_pct` | **DIAGNOSTIC ONLY**, position-agnostic. Never a return, never benchmarked, never summed into a track record. |
| `strategy_return_pct` | **What we earned** = `position_held × spread_move_pct`. Flat = exactly 0.0. The only column benchmarked. |

Using `position` instead of `position_held` **zeroes out every winning exit** — that was a real live bug (§5, B15).

### The two p-values

`adf_pvalue` is a plain `adfuller()` on **estimated** OLS residuals → wrong critical values → **systematically 2–4× too optimistic**.
`eg_pvalue` (from `statsmodels.tsa.stattools.coint`) is the correct MacKinnon reference.
`is_cointegrated` and `screen_universe`'s `tradeable` still key off the **biased** value — deliberately, so the fix stayed purely additive. `eg_disagrees` is where the truth surfaces. `focus_book`'s admission rule and `book_refresh.build_ranking` both require `eg_pvalue`.

---

## 3. Architecture map

```
data/loader.py          fetch_price_history (yfinance, adj close only, sha1 CSV cache), align_and_clean
signals/spread.py       SignalConfig; StaticOLS + KalmanHedgeRatio; build_spread; rolling_zscore;
                        generate_signals  <- the state machine, the semantic core
screening/
  universe.py           SECTOR_ETFS (54 groups, 303 tickers), generate_candidate_pairs (919 pairs)
  cointegration.py      EngleGrangerResult, test_pair_cointegration, compute_half_life,
                        validate_out_of_sample, _benjamini_hochberg, screen_universe, is_market_neutral
  regime.py             VIX/GVZ macro stress mask (True = calm = entries allowed)
  events.py             FOMC/election blackout mask (~18 trading days/yr)
  focus_book.py         FOCUS_BOOK: the fixed 12-pair watchlist + admission rules
  book_refresh.py       monthly re-rank + hysteresis governance. PROPOSES, NEVER MUTATES.
backtest/
  simulator.py          PairBacktestConfig, PairBacktester  <- read _prepare_pair_series carefully
  metrics.py            compute_metrics (Sharpe rf=0/252, drawdown, win rate, profit factor)
  walkforward.py        pair_survival_study / parameter_study / allocation_study
portfolio/optimizer.py  Ledoit-Wolf cov, OU expected returns, tangency/min-var/frontier
montecarlo/simulator.py OU fit + 1000-path simulation through the REAL state machine
reporting/
  daily_report.py       "what would the engine say today" (no memory, no gating)
  journal.py            forward-test signal journal, 10-bar grading
  daily_performance.py  THE track record. record_session / grade_previous_sessions / performance_summary
  decision_grader.py    OUTCOME lens vs THESIS lens
  paper_trades.py       append-only discretionary log (model_agrees column)
visualization/          plots.py (matplotlib PNG), interactive.py (Plotly HTML)
```

### Runners

| Script | Purpose | Network |
|---|---|---|
| `run_demo.py` | **Deterministic verification.** Exit 0 = all stages verified. | No |
| `run_focus_book.py [--review] [--journal]` | The book as one gated portfolio | Yes (`--review` no) |
| `run_daily_tracker.py [--summary]` | **Daily after close.** Record → grade → summarise | Yes |
| `build_dashboards.py` | **Daily after close.** Rebuild + publish all dashboards | Yes |
| `run_screen.py [--journal]` | Full-universe live screen | Yes |
| `run_walkforward.py` | The three out-of-sample studies | Yes |
| `run_grade_decisions.py` | Historical decision hit-rates, both lenses | Yes |
| `run_book_refresh.py` | Monthly proposal (writes OUTSIDE the repo) | Yes |
| `run_montecarlo.py [A B]` | 1000-path OU + interactive dashboards | Yes |
| `run_pair_study.py [A B]` | Single-pair 3-mode comparison | Yes |
| `run_live_monitor.py [A B]` | Intraday websocket monitor | Yes |
| `dashboard.py` | `py -m streamlit run dashboard.py` | Yes |

### Config defaults worth knowing

`SignalConfig`: `zscore_window=30, entry_z=2.0, exit_z=0.5, stop_z=3.75, max_holding_bars=None, max_entry_z=None (→ uses stop_z, so the guard is ACTIVE by default), require_positive_hedge_ratio=True`.

`PairBacktestConfig`: `transaction_cost_bps=5, slippage_bps=5, capital_per_pair=10_000, max_concurrent_pairs=5, initial_capital=100_000, recheck_freq_days=60, recheck_window_days=252, hedge_ratio_mode="regime", kalman_innovation_obs_variance=1e-4`.
Presets: `conservative()` ($5k/3 pairs/stop 3.0) — **used by run_screen, run_focus_book, run_grade_decisions**; `moderate()`; `aggressive()`.

`WalkForwardConfig`: `formation_days=252, holdout_days=63, step_days=63`.
`GovernanceConfig`: `n_consecutive=2`.

---

## 4. The current book — 12 pairs (adopted 2026-08-05)

FNV/WPM `gold_royalties` (EG p=0.0007, β+0.764, hl 8.2d) · EPD/OKE `midstream` (0.0017, +0.857, 5.4) · AVGO/NVDA `semis` (0.0124, +1.213, 6.6) · JNJ/MRK `pharma` (0.0130, +0.792, 6.4) · AU/NEM `gold_miners` (0.0148, +1.222, 6.5) · ADBE/CRM `software` (0.0178, +1.079, 6.6) · CHD/CL `household_products` (0.0246, +0.834, 6.8) · SRE/XEL `utilities` (0.0354, +1.018, 7.3) · ROST/TGT `discount_retail` (0.0389, +0.948, 11.0) · APD/LIN `industrial_gases` (0.0440, +0.789, 8.1) · AMGN/VRTX `biotech` (0.0457, +1.215, 7.6) · FITB/RF `regional_banks` (0.0492, +1.281, 8.4 — weakest, at the margin)

**Admission rules (all four required):**
1. Proper **Engle-Granger** p < 0.05 on the **trailing 252 days** (the window β is actually fit on).
2. Hedge ratio **positive and in [0.70, 1.40]** — the engine z-scores on `log(A) − β·log(B)` but holds **equal dollars**, so signalled ≠ traded portfolio unless β ≈ 1.
3. Half-life in **[5, 30] days**.
4. No structural near-twins, no stock-vs-own-sector-ETF (CL/XLP, NVDA/SMH rejected), **one pair per sector**.

**Multiple-testing caveat, stated in the module itself:** 919 tests at p<0.05 throws ~46 false positives by chance; 34 survived. **These are best-evidenced candidates, NOT established discoveries.** The book exists to generate enough trades to be *measurable*, not to assert an edge.

Retired 2026-08-05 (kept as `_RETIRED_2026_08_05`): DUK/SO, GDX/GLD, GD/RTX + readmitted JNJ/MRK, AVGO/NVDA, ROST/TGT.

---

## 5. The sixteen bugs — the project's hardest-won lessons

Four early ones (B1–B4) were **test-harness** bugs where production code was correct. Do not "fix" working code.

| # | Bug | Lesson |
|---|---|---|
| B1 | pytest collected the production `test_pair_cointegration` as a test | any production symbol prefixed `test_` is a collection hazard |
| B2 | concurrency test double-counted a same-day exit→entry handoff | the simulator legitimately reuses a freed slot same-day |
| B3 | force-liquidation test assumed a fixed holding period | scan the trade log, don't assume |
| B4 | OOS fixture wasn't strong enough (p=0.073, intermittent) | fixture power is per-test, not shared |
| **B5**★ | Equity only moved when a trade CLOSED — open drawdown invisible | Sharpe 1.43→0.87, maxDD −0.17%→−0.77% after fixing |
| **B6**★ | Screen and report used different lookback windows | same run showed contradictory `is_cointegrated` two sections apart |
| **B7**★ | Ragged-panel END_OF_SAMPLE silently dropped positions | entry cost charged, P&L never realised |
| **B8**★ | Tracker benchmarked a **position-agnostic** spread move vs SPY | claimed "beat market 100% of days" on a −$62 day |
| **B9**★ | Tracker generated signals **without** the event gate | logged entries the live engine refuses |
| **B10**★ | Stale CSV cache ⇒ current session could never be recorded | cache key includes `end`; a pre-publication fetch poisons it |
| **B11**★ | **ADF-on-residuals: every p-value biased 2–4×** | COST/PEP 0.0393 → 0.1207, crosses 0.05 and flips |
| **B12**★ | Entry-before-stop: entries fired at \|z\| already past the stop | 24 of 63 entries; inflated thesis hit-rate 41%→86% |
| **B13**★ | Negative β ≠ market-neutral | ABT/MRK β −0.578 ⇒ net −$15,800 on a $10k notional |
| **B14**★ | Open→close measurement hid overnight gaps | ABT/MRK booked −$43 when true close-to-close was +$252 |
| **B15**★ | **Winning EXITs booked as $0.00** | DUK/SO earned +$86.57 and recorded 0.00 |
| B16 | Monthly refresh ranked on the biased p-value | would have contradicted the book every month |

**The pattern (the single most important lesson): every one of these was a quantity computed on one basis and aggregated on another, with no invariant checking that the pieces agree. Four were found only by running live. The test suite was green throughout.**

---

## 6. Settled findings — do NOT re-litigate

| Finding | Number |
|---|---|
| Walk-forward pair survival | **7%** at p<0.05 (11% at p<0.10), median holdout p 0.44 |
| Parameter tuning is negative-value | textbook bands holdout Sharpe **1.32** vs **1.01** optimized |
| Stop-loss justification | **0 of 22** — every stop realised a loss that would have recovered |
| Entry-band buckets | 1.0–1.5 loses **−$12.73/trade**; only 2.0–2.5 is positive (+$23.94, t=1.06, not significant) |
| Risk profiles | moderate/aggressive take near-identical signals at **2–4× the cost** |
| Capital utilisation | 6-pair book deployed **~1.3%** of capital, ~6 trades/yr |
| Statistical power | **~91 trades ≈ 15 years** to distinguish the most optimistic edge from zero |
| Screen is anti-predictive | EG p<0.05 pairs earned **less** than p≥0.05; the 5–30d half-life band was the **worst** region |
| Book orthogonality | avg pairwise strategy-return correlation ≈ **−0.02** |
| Sharpe at small n | SE ≈ √(252/N) → **±15.9 at N=1**. Refuse to print below N=252. |

Also ruled out by a dedicated audit — **verified, no defect**: look-ahead/causality (future-price perturbation → bit-identical equity), cost/slippage accounting, sizing/sign conventions, concurrency off-by-one, metrics math, walk-forward split leakage.

---

## 7. Decisions already made — do NOT reverse blindly

1. **The book is HARDCODED, not screened daily.** With 7% quarterly survival, a daily screen churns on noise. `book_refresh.py` proposes only.
2. **Min-variance was deliberately NOT wired into `PairBacktester`.** It wins in-sample and on 5 OOS windows, but that is "directional, not decisive." Known open thread since 07-18.
3. **Beta-weighted sizing is a CORRECTNESS fix, not a performance fix.** It buys ~2.5% vol reduction and makes market exposure **7–19× worse** for similar-β pairs (AVGO/NVDA, GD/RTX), because the cointegrating β is fit on price *levels* while market exposure lives in *returns*.
4. **The prior track record was ARCHIVED, not backfilled.** `outputs/archive/` — different convention *and* different book; backfilling would be look-ahead. Do not "helpfully" reconstruct it.
5. **moderate/aggressive profiles are deprecated** (still in code, deprecated by decision).
6. **"Days beating SPY" is being RETIRED.** A dollar-neutral book has no market exposure to be compensated for; on a flat day it measures SPY's sign. Benchmark to **zero**.
7. **Do NOT tune entry/exit/stop bands.** Settled 07-19.
8. **"More aggressive" = more shots on goal, not bigger size or looser bands.** Both of the latter measured strictly worse.
9. TradingView rejected (no public data API). Regime-OLS is the recommended mode; Kalman beta is a drift diagnostic only.

---

## 8. Daily operational routine

All from the repo root. **Order matters** — grade the forecast BEFORE rebuilding dashboards, or new HIT/MISS rows won't reach the predictions chart.

1. **Pre-open (~09:30 ET)** — write `..\forecasts\forecast_YYYY-MM-DD.md`: a state table plus numbered, confidence-tagged, falsifiable predictions. **Written before the outcome is known and never edited afterwards.**
   ```
   py run_focus_book.py --review
   ```
2. **Post-close (≥16:35 ET)** — `py run_daily_tracker.py`
3. **Grade the forecast** — append a `# RESULTS` section with a `| # | prediction | conf | HIT/MISS |` table. Predictions above are never edited.
4. **Rebuild dashboards** — `py build_dashboards.py`
5. **Publish** — `git add docs outputs/daily_performance.csv ../forecasts && git commit && git push`

Monthly: `py run_book_refresh.py` (proposal only).

---

## 9. Data artifacts and integrity rules

- **`outputs/daily_performance.csv`** — the track record. 20 columns (`PERFORMANCE_COLUMNS`). **Append-only**, idempotent on `(date, ticker_a, ticker_b)`, **grade-on-read** (never written back), **close-to-close**, event-gated by default, fetched with `use_cache=False`.
- **`outputs/archive/`** — pre-2026-07-31 sessions, old convention + old book. Cannot be appended. Deliberately not backfilled.
- **`outputs/paper_trades.csv`** — append-only discretionary log. Contains 5 erroneous OPENs plus 5 reversing CLOSEs at identical prices ($0.00) — **the errors were left visible on purpose**, with the reason in-row. That is the integrity model: correct by appending, never by deleting.
- **`outputs/signal_journal.csv`** — forward-test journal, frozen 07-21, superseded in practice.
- **`data_cache/`** — 73 CSVs, gitignored, safe to delete. ⚠️ A fetch of a window ending "today" before the daily bar publishes will **poison** the cache for that window.
- **`..\forecasts\*.md`** — pre-registered predictions. Integrity rule: the prediction block is never edited; only RESULTS is appended.

---

## 10. Open threads, by priority

**P0 — blocking honest measurement**
1. **Build the out-of-selection backtest harness.** *The* top recommendation. Select the book as of date T using only data < T, trade T→T+63, roll. Machinery exists in `walkforward.py`. Until this exists every experiment reproduces the selection-bias illusion.
2. **Metrics overhaul.** Retire days-beating-SPY (still computed in `performance_summary`); gate Sharpe at N≥252; gate `annualized_return` at N≥63.
3. **`trade_log` lacks gross/cost columns** — only net `pnl`. The gross-vs-net analysis at the heart of the 08-03 review cannot be reproduced from the artifact.
4. **`metrics.py` has no capital-utilisation metric** — the ~1.3% figure that drives the whole breadth thesis isn't produced by the codebase.

**P1 — operational**
5. **`PairsTradingBookRefresh` has NEVER run.** Result `267011` = `SCHED_S_TASK_HAS_NOT_RUN`; the 08-01 trigger elapsed and silently rescheduled to 08-29. Likely `DisallowStartIfOnBatteries=true` + `LogonType=InteractiveToken`. `refresh_history.json` is stuck on the retired book.
6. **`PairsTradingDailyTracker` works (Last Result 0) but fires late** — 08-04 landed at 20:04 instead of 16:30 (`StartWhenAvailable` catch-up after wake).
7. **Neither task logs anything** — both invoke `python.exe` directly instead of the `.cmd` wrappers that carry the redirect. Fix: point at the wrappers, disable the battery conditions, enable the TaskScheduler Operational log.

**P2 — model work**
8. Stop-loss re-tune (0/22 justified) — the recommended sequence was *entry guard first (done), re-grade, then widen `stop_z`*. The re-grade hasn't happened, and `.conservative()` still passes the indicted `stop_z=3.0`.
9. Reconcile the P&L convention mismatch (dollar-neutral simulator vs beta-weighted tracker) — mitigated by the β band, not fixed.
10. Wire min-variance weights into `PairBacktester` (carried since 07-18).
11. Is the 30-day z-window too slow for 6–12 day half-lives? **Measured answer: no — longer windows were monotonically better.** An earlier hypothesis that shorter would help was wrong.
12. Walk-forward the allocator; Johansen baskets; productionise Kalman innovation mode.

**P3 — hygiene**
13. CRLF churn / missing `.gitattributes` (`* text=auto eol=lf`).
14. `_watch_0803.sh` is committed and hardcodes the vault path + username — remove.
15. README Layout section still says "105-test suite / 73 tickers / 5 pairs" (actual: 230 / 303 / 12).
16. `Pairs Trading Engine (MOC).md` is **the most misleading file in the vault** — retired book, withdrawn numbers, dated 07-17.
17. Implementation Log ends 07-31 — missing the 08-03 reviews and 08-03/04/05 sessions.
18. `book_refresh.py` still uses β band [0.25, 4.0] while `focus_book.py` requires [0.70, 1.40] — the refresh can propose pairs the book rejects.

---

## 11. Known-stale artifacts (do not trust)

- `outputs/daily_performance.csv` contains only the **retired 6-pair book** (3 sessions: 07-31, 08-03, 08-04). The 12-pair book's first session records on the next tracker run.
- `docs/index.html` card #4 and `interactive_predictions_vs_market.html` describe the retired 6-pair book and **have no generator script** — they cannot be refreshed by any command in the repo.
- `dashboard.py`'s `TRACK_RECORD_COLUMNS` is missing `position_held` (predates the exit-attribution fix), so a winning EXIT day shows "Flat" next to a non-zero P&L.
- Every GDX/GLD dashboard and every `spread_zscore_*.png` is from a pair no longer in the book.
- Only 1 of 4 forecast files has a graded RESULTS table, so the predictions dashboard is built from one day's six predictions.
- `daily_performance.csv`'s `adf_pvalue` / `is_cointegrated` columns are still the **biased** ADF figure.

---

## 12. Twenty gotchas

1. `py`, not `python`. 2. Run from repo root. 3. **yfinance `end` is EXCLUSIVE** — add a day. 4. The CSV cache can freeze an incomplete panel; delete `data_cache/*.csv` before debugging stale data. 5. `position` vs `position_held`. 6. `spread_move_pct` is not a return. 7. Two p-values; `is_cointegrated` uses the biased one. 8. `_is_stock_vs_own_sector_etf` only applies in `book_refresh`/`focus_book`. 9. β band divergence between `book_refresh` and `focus_book`. 10. Gates never force exits. 11. Missing data = allowed/calm. 12. Dollar-neutral ≠ beta-weighted. 13. `kalman` mode under-trades by design; don't raise `kalman_innovation_obs_variance` to 1e-3. 14. `run_book_refresh.py` writes outside the repo to a hardcoded absolute path. 15. The book is 12 pairs now — check `len(FOCUS_BOOK)`, don't trust prose. 16. `.gitignore` excludes `outputs/interactive_*.html` but `docs/` copies ARE tracked. 17. `generate_daily_signal_report` has no memory across runs. 18. `holding_days` is calendar days; `max_holding_bars` is bars. 19. `compute_half_life` returns None on no reversion. 20. Path contains an apostrophe (`Ryan's Obsidian`) — always double-quote it.

---

## 13. If you do only one thing

Build the **out-of-selection harness** (§10.1). Every number this repo produces is currently in-sample and circular — the book is graded on the history it was selected from. Until that harness exists, every improvement you measure will be the selection-bias illusion regenerating itself, exactly as the +$118 did.

And keep the honesty. The most valuable thing here is not the engine — it is that the engine has repeatedly caught itself lying, and each time the correction was appended rather than the record quietly edited.
