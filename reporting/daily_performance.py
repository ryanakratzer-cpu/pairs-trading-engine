"""Persistent multi-day performance tracker for the pairs engine.

WHY this exists: `reporting/daily_report.py` answers "what would the engine say
today?" and `reporting/journal.py` forward-tests whether the *z-score thesis*
paid off. Neither accumulates a track record you can point at and say "over the
last N sessions this strategy made/lost X and beat/lagged the market on Y% of
days". This module does exactly that: one append-only CSV row per pair per
session, recording both the engine's DECISION and the realized OPEN->CLOSE
outcome of that session, plus a grade-on-read loop that judges yesterday's
decisions against what actually happened afterwards.

Design mirrors reporting/journal.py deliberately:
  * append is IDEMPOTENT per (date, ticker_a, ticker_b) so a daily post-close
    cron can re-run safely,
  * grading is GRADE-ON-READ - it never mutates the CSV, so the recorded
    history stays an un-fudgeable forward record,
  * the price fetcher is INJECTABLE so tests run with zero network.

RETURN CONVENTIONS (stated explicitly rather than left implicit):

  The headline performance number is POSITION-AWARE. An earlier version of this
  module reported the raw spread move as "the pair return" and benchmarked THAT
  against SPY, which meant a session where the engine held NOTHING could still
  claim it beat the market by whatever the spread happened to do. Cumulative
  return and cumulative P&L then told contradictory stories (e.g. "+0.71%, beat
  the market 100% of days" while P&L was -$62). A track record whose purpose is
  to be trusted over many sessions cannot do that, so the two concepts are now
  split and named so they can never be confused:

  * `spread_move_pct` - DIAGNOSTIC, position-agnostic log-spread change over
    the session, in percent:
        spread_t        = log(P_a) - hedge_ratio * log(P_b)
        spread_move_pct = 100 * (spread_close - spread_open)
    It answers "what did the spread do today?" and is the same number whether
    the engine was long, short, or flat. It is NEVER our return and is never
    benchmarked against SPY.

  * `strategy_return_pct` - WHAT WE ACTUALLY EARNED that session, in percent:
        strategy_return_pct = position * spread_move_pct
    Long spread (+1) earns the spread's rise, short spread (-1) earns its fall
    (matching signals.spread's convention), and a FLAT session (position 0)
    earns exactly 0.0 - not whatever the spread happened to do. This is the
    only number used for cumulative return, the SPY comparison and Sharpe.

  * `pair_pnl_usd` - the same result in dollars on `notional`, DERIVED from
    strategy_return_pct so the two can never disagree in sign or magnitude:
        pair_pnl_usd = strategy_return_pct / 100 * notional

  * `excess_return_pct` = strategy_return_pct - spy_return_pct, an honest "did
    we beat the market". On a flat day this is simply -spy_return_pct: holding
    nothing lags a rising market, which is the truth.

  * Leg returns (`ticker_a_return_pct`, `ticker_b_return_pct`,
    `spy_return_pct`) are simple open->close percent returns.

Like everything in this project this is signal research only. Nothing here
places an order, sizes a real position, or talks to a broker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data.loader import fetch_price_history
from screening.cointegration import test_pair_cointegration
from screening.events import event_exclusion_mask
from signals.spread import SignalConfig, build_spread, generate_signals, rolling_zscore

DEFAULT_PERFORMANCE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "daily_performance.csv"

BENCHMARK_TICKER = "SPY"
TRADING_DAYS_PER_YEAR = 252  # matches backtest/metrics.py

DISCLAIMER = "SIGNAL RESEARCH ONLY - no orders are placed and no position is held."

# Identity key: the SESSION date (not logged_at), so re-running the same
# post-close job - or backfilling - cannot create a second row per observation.
PERFORMANCE_KEY = ("date", "ticker_a", "ticker_b")

# The on-disk schema, in order. Written exactly once (header) and re-used on
# every append so a future code change cannot silently shift columns.
PERFORMANCE_COLUMNS = [
    "date",
    "ticker_a",
    "ticker_b",
    "hedge_ratio",
    "adf_pvalue",
    "is_cointegrated",
    "half_life_days",
    "open_z",
    "close_z",
    "signal",
    "position",
    "strategy_return_pct",
    "spread_move_pct",
    "pair_pnl_usd",
    "ticker_a_return_pct",
    "ticker_b_return_pct",
    "spy_return_pct",
    "excess_return_pct",
    "logged_at",
]

# Grading statuses - mirror reporting/journal.py.
STATUS_GRADED = "graded"
STATUS_TOO_RECENT = "too_recent"
STATUS_INSUFFICIENT_DATA = "insufficient_data"

# Decision families for grading. HOLD is ambiguous on its own (the state
# machine emits it both for "still in a position" and for "no z-score yet"),
# so the recorded `position` disambiguates it at grade time.
ENTRY_SIGNALS = frozenset({"ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD"})
CLOSING_SIGNALS = frozenset({"EXIT", "STOP_LOSS", "TIME_EXIT"})
FLAT_SIGNALS = frozenset({"NO_POSITION", "HOLD"})


# ------------------------------------------------------------------ fetching


def fetch_session_ohlc(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Session Open/Close bars for `tickers`, as a (field, ticker) MultiIndex frame.

    data.loader.fetch_price_history deliberately returns adjusted CLOSE only,
    which cannot express an intraday open->close session return - so this is a
    thin sibling that keeps the Open column. Injectable everywhere it is used
    so tests never touch the network.
    """
    import yfinance as yf  # local import: keeps the module importable offline

    tickers = sorted(set(tickers))
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise ValueError(f"yfinance returned no OHLC for tickers={tickers}, {start}..{end}")

    if not isinstance(raw.columns, pd.MultiIndex):
        # Single-ticker shape: promote to the same (field, ticker) layout.
        raw = pd.concat({tickers[0]: raw}, axis=1).swaplevel(axis=1)
    return raw.sort_index()


def _field(ohlc: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pull one OHLC field out as a plain ticker-columned frame."""
    if field not in ohlc.columns.get_level_values(0):
        raise ValueError(f"OHLC frame has no '{field}' field")
    return ohlc[field]


# ------------------------------------------------------------------ recording


def _date_key(value) -> str | None:
    """Normalize a session date to YYYY-MM-DD so idempotency keys survive the
    CSV round-trip (a Timestamp repr would never match a parsed string)."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def record_session(
    pairs: list[tuple[str, str]],
    as_of: str | None = None,
    notional: float = 10_000.0,
    path: Path | str = DEFAULT_PERFORMANCE_PATH,
    lookback_days: int = 400,
    signal_config: SignalConfig | None = None,
    benchmark: str = BENCHMARK_TICKER,
    price_fetcher: Callable[..., pd.DataFrame] = fetch_price_history,
    ohlc_fetcher: Callable[..., pd.DataFrame] = fetch_session_ohlc,
    verbose: bool = True,
    apply_event_gate: bool = True,
) -> int:
    """Record one session's decision + realized outcome for each pair.

    For every pair: fetch trailing daily closes, refit the Engle-Granger hedge
    ratio, rebuild the spread and rolling z, run the signal state machine, and
    take the decision on the session bar. Then measure that session's
    open->close outcome (see the module docstring for the return conventions)
    and append one row per pair.

    Idempotent per (date, ticker_a, ticker_b): re-running for a session already
    on disk writes nothing and returns 0.

    DATA CAVEAT - the current session's bar is frequently not published yet
    (yfinance publishes the daily bar after the close, and intraday-partial
    bars are unreliable). Rather than mislabel a stale bar with today's date,
    this falls back to the most recent COMPLETED session present in the data
    and says so loudly (printed when `verbose`, and the recorded `date` column
    is the true session date, never the requested one).

    Returns the number of new rows written.
    """
    path = Path(path)
    signal_config = signal_config or SignalConfig()
    if not pairs:
        return 0

    requested = pd.Timestamp(as_of).normalize() if as_of else pd.Timestamp.today().normalize()
    start = (requested - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    # yfinance's `end` is exclusive; +1 day so the requested session can appear.
    end = (requested + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    tickers = sorted({t for pair in pairs for t in pair} | {benchmark})

    # BYPASS THE CSV CACHE for the close panel. data/loader.py caches on
    # (tickers, start, end), and this window always extends to `requested` — so
    # a run that happens before the provider publishes that day's daily bar
    # writes an INCOMPLETE panel to cache, and every later run reuses it and can
    # never see the session. Observed live 2026-07-29: the 16:30 scheduled run
    # cached a panel ending 07-28, so re-runs at 18:09+ still reported "no pair
    # had enough data" even though the bar was available. The ohlc fetcher is
    # uncached and did see it, which is exactly how the split was diagnosed.
    # Injected test fetchers accept **kwargs, so this stays test-friendly.
    try:
        closes = price_fetcher(tickers, start=start, end=end, use_cache=False)
    except TypeError:
        # A custom fetcher that doesn't take use_cache: fall back rather than fail.
        closes = price_fetcher(tickers, start=start, end=end)
    ohlc = ohlc_fetcher(tickers, start=start, end=end)

    opens_all = _field(ohlc, "Open")
    closes_all = _field(ohlc, "Close")

    session_date, fallback = _resolve_session(requested, opens_all, closes_all, tickers)
    if session_date is None:
        if verbose:
            print(
                f"[daily_performance] no completed session with full OHLC on or before "
                f"{requested.date()} - nothing recorded."
            )
        return 0
    if fallback and verbose:
        print(
            f"[daily_performance] CAVEAT: requested session {requested.date()} has no "
            f"completed bar yet; recording the most recent COMPLETED session "
            f"{session_date.date()} instead."
        )
    elif verbose:
        print(f"[daily_performance] recording completed session {session_date.date()}")

    spy_open, spy_close = _leg_prices(opens_all, closes_all, benchmark, session_date)
    spy_return_pct = _pct_return(spy_open, spy_close)

    rows: list[dict] = []
    for ticker_a, ticker_b in pairs:
        row = _session_row(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            session_date=session_date,
            closes=closes,
            opens_all=opens_all,
            closes_all=closes_all,
            signal_config=signal_config,
            notional=notional,
            spy_return_pct=spy_return_pct,
            apply_event_gate=apply_event_gate,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        if verbose:
            # Say WHY, and do not let the caller print "already recorded": a
            # data gap and an idempotent no-op both return 0 but mean opposite
            # things, and conflating them hid a real bug for a full session.
            print(
                f"[daily_performance] DATA GAP - session {session_date.date()} resolved but no "
                f"pair could be priced from it (missing close/open bar or too little history). "
                f"NOTHING was recorded; this is NOT an idempotent no-op."
            )
        return 0

    return _append_rows(pd.DataFrame(rows), path)


def _resolve_session(
    requested: pd.Timestamp,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    tickers: list[str],
) -> tuple[pd.Timestamp | None, bool]:
    """Most recent session <= `requested` that has a complete Open AND Close for
    every ticker. Returns (session_date, used_fallback)."""
    have = [t for t in tickers if t in opens.columns and t in closes.columns]
    if not have:
        return None, False
    complete = opens[have].notna().all(axis=1) & closes[have].notna().all(axis=1)
    usable = complete[complete].index
    usable = usable[usable <= requested]
    if len(usable) == 0:
        return None, False
    session_date = pd.Timestamp(usable[-1])
    return session_date, session_date != requested


def _leg_prices(
    opens: pd.DataFrame, closes: pd.DataFrame, ticker: str, session_date: pd.Timestamp
) -> tuple[float, float]:
    return float(opens.loc[session_date, ticker]), float(closes.loc[session_date, ticker])


def _pct_return(open_price: float, close_price: float) -> float:
    return float(100.0 * (close_price / open_price - 1.0))


def _session_row(
    ticker_a: str,
    ticker_b: str,
    session_date: pd.Timestamp,
    closes: pd.DataFrame,
    opens_all: pd.DataFrame,
    closes_all: pd.DataFrame,
    signal_config: SignalConfig,
    notional: float,
    spy_return_pct: float,
    apply_event_gate: bool = True,
) -> dict | None:
    """Build one CSV row, or None when the pair lacks usable data."""
    for frame in (closes, opens_all, closes_all):
        if ticker_a not in frame.columns or ticker_b not in frame.columns:
            return None
    if session_date not in opens_all.index or session_date not in closes_all.index:
        return None

    history = closes.loc[:session_date, [ticker_a, ticker_b]].dropna()
    if len(history) < signal_config.zscore_window + 10:
        return None
    if pd.Timestamp(history.index[-1]) != session_date:
        # The close panel does not actually carry the session bar; refuse to
        # grade an older bar as if it were this session.
        return None

    eg = test_pair_cointegration(
        history[ticker_a], history[ticker_b], ticker_a=ticker_a, ticker_b=ticker_b
    )
    hedge_ratio = eg.hedge_ratio

    spread = build_spread(history[ticker_a], history[ticker_b], hedge_ratio)
    zscore = rolling_zscore(spread, signal_config.zscore_window)
    # GATED, not raw: the tracker must record what the REAL engine would do, and
    # the real engine refuses new entries inside an FOMC/election blackout
    # (screening/events.py), exactly as run_screen.py and run_focus_book.py do.
    # Recording ungated signals would log entries the live book would never take
    # and quietly inflate the track record with trades that never happen. The
    # mask gates NEW entries only; an already-open position still exits/stops.
    # apply_event_gate=False is for tests that isolate other behaviour on
    # synthetic date ranges; production always records the GATED decision.
    entries_allowed = event_exclusion_mask(zscore.index) if apply_event_gate else None
    signals = generate_signals(zscore, signal_config, tradeable=entries_allowed)
    latest = signals.iloc[-1]
    close_z = float(latest["zscore"]) if pd.notna(latest["zscore"]) else np.nan
    position = int(latest["position"])
    signal = str(latest["event"])

    # z as it stood at the OPEN: same trailing history, but the session bar's
    # closes swapped for its opens. Same rolling window, so it is directly
    # comparable to close_z.
    open_a, close_a = _leg_prices(opens_all, closes_all, ticker_a, session_date)
    open_b, close_b = _leg_prices(opens_all, closes_all, ticker_b, session_date)
    at_open = history.copy()
    at_open.iloc[-1] = [open_a, open_b]
    open_spread_series = build_spread(at_open[ticker_a], at_open[ticker_b], hedge_ratio)
    open_z_series = rolling_zscore(open_spread_series, signal_config.zscore_window)
    open_z = float(open_z_series.iloc[-1]) if pd.notna(open_z_series.iloc[-1]) else np.nan

    # Diagnostic: position-agnostic log-spread change over the session.
    spread_open = float(np.log(open_a) - hedge_ratio * np.log(open_b))
    spread_close = float(np.log(close_a) - hedge_ratio * np.log(close_b))
    spread_move_pct = float(100.0 * (spread_close - spread_open))
    # What WE earned: apply the position. Flat (0) earns exactly 0.0 - the
    # `or 0.0` normalizes a -0.0 product so a flat row never prints "-0.0".
    strategy_return_pct = float(position * spread_move_pct) or 0.0
    # Derived from strategy_return_pct by construction so the dollar figure and
    # the percentage figure can never disagree.
    pair_pnl_usd = float(strategy_return_pct / 100.0 * notional) or 0.0

    return {
        "date": _date_key(session_date),
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "hedge_ratio": hedge_ratio,
        "adf_pvalue": eg.adf_pvalue,
        "is_cointegrated": eg.is_cointegrated,
        "half_life_days": eg.half_life_days,
        "open_z": open_z,
        "close_z": close_z,
        "signal": signal,
        "position": position,
        "strategy_return_pct": strategy_return_pct,
        "spread_move_pct": spread_move_pct,
        "pair_pnl_usd": pair_pnl_usd,
        "ticker_a_return_pct": _pct_return(open_a, close_a),
        "ticker_b_return_pct": _pct_return(open_b, close_b),
        "spy_return_pct": spy_return_pct,
        "excess_return_pct": strategy_return_pct - spy_return_pct,
        "logged_at": None,  # stamped at write time in _append_rows
    }


def _append_rows(new_rows: pd.DataFrame, path: Path) -> int:
    """Idempotent append on (date, ticker_a, ticker_b). Returns rows written."""
    if new_rows.empty:
        return 0

    new_rows = new_rows.copy()
    new_rows["date"] = new_rows["date"].map(_date_key)
    new_rows = new_rows[new_rows["date"].notna()]
    if new_rows.empty:
        return 0

    existing_keys: set[tuple[str, str, str]] = set()
    if path.exists():
        # dtype=str so keys compare as the exact strings on disk rather than
        # whatever pandas would infer on read.
        existing = pd.read_csv(path, dtype=str)
        if not existing.empty:
            existing_keys = {
                (row.date, row.ticker_a, row.ticker_b) for row in existing.itertuples()
            }

    is_new = [
        (row.date, row.ticker_a, row.ticker_b) not in existing_keys
        for row in new_rows.itertuples()
    ]
    new_rows = new_rows[is_new]
    if new_rows.empty:
        return 0

    new_rows = new_rows.copy()
    new_rows["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows = new_rows.reindex(columns=PERFORMANCE_COLUMNS)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    new_rows.to_csv(path, mode="a", header=write_header, index=False)
    return len(new_rows)


# ------------------------------------------------------------------- grading


def grade_previous_sessions(
    path: Path | str = DEFAULT_PERFORMANCE_PATH,
    horizon_days: int = 5,
    zscore_window: int = 30,
    entry_z: float = 2.0,
    price_fetcher: Callable[..., pd.DataFrame] = fetch_price_history,
) -> tuple[pd.DataFrame, dict]:
    """Grade recorded sessions whose `horizon_days` outcome window has closed.

    GRADE-ON-READ: the CSV is never mutated. For each gradable row the pair's
    prices are re-fetched around the session date, the spread is rebuilt with
    the RECORDED hedge ratio (refitting would leak hindsight into the grade),
    and the rolling z is recomputed. The core statistic follows journal.py:

        spread_change_z = sign(close_z) * (z_end - z_start)

    negative = the spread converged toward its mean from the recorded side.

    What "correct" means depends on the decision that was made:
      * ENTER_LONG_SPREAD / ENTER_SHORT_SPREAD (and HOLD while in a position):
        the trade is a bet on convergence -> correct when spread_change_z < 0.
      * EXIT / STOP_LOSS / TIME_EXIT: closing is a bet that convergence is over
        -> correct when the spread did NOT keep converging, spread_change_z >= 0.
      * NO_POSITION (and HOLD while flat): standing aside is correct when no
        |z| >= entry_z opportunity appeared in the window - i.e. staying flat
        cost nothing.

    Statuses mirror journal.py: graded / too_recent / insufficient_data.
    `price_fetcher` is injectable so tests grade synthetic frames offline.

    Returns (graded DataFrame with status/correct/spread_change_z columns,
    summary dict with n_graded, hit_rate and a per-decision-type breakdown).
    """
    path = Path(path)
    empty_summary = {"n_graded": 0, "hit_rate": None, "by_decision": {}}
    if not path.exists():
        return pd.DataFrame(), empty_summary

    history = pd.read_csv(path)
    if history.empty:
        return history, empty_summary

    today = pd.Timestamp.today().normalize()
    # Calendar-day buffers around trading-bar counts; ~2.5 calendar days per
    # trading bar comfortably covers weekends and holidays.
    buffer_before = int(zscore_window * 2.5) + 10
    buffer_after = int(horizon_days * 2.5) + 10

    statuses: list[str] = []
    corrects: list[object] = []
    changes: list[float] = []
    for row in history.itertuples():
        status, correct, change = _grade_row(
            row,
            today=today,
            horizon_days=horizon_days,
            zscore_window=zscore_window,
            entry_z=entry_z,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            price_fetcher=price_fetcher,
        )
        statuses.append(status)
        corrects.append(correct)
        changes.append(change)

    result = history.copy()
    result["status"] = statuses
    result["correct"] = corrects
    result["spread_change_z"] = changes

    graded = result[result["status"] == STATUS_GRADED]
    n_graded = int(len(graded))
    hit_rate = float(graded["correct"].astype(bool).mean()) if n_graded else None

    by_decision: dict[str, dict] = {}
    for signal, group in graded.groupby("signal"):
        by_decision[str(signal)] = {
            "n": int(len(group)),
            "hit_rate": float(group["correct"].astype(bool).mean()),
        }

    return result, {"n_graded": n_graded, "hit_rate": hit_rate, "by_decision": by_decision}


def _decision_family(signal: str, position) -> str:
    """entry | close | flat — HOLD is disambiguated by the recorded position."""
    if signal in ENTRY_SIGNALS:
        return "entry"
    if signal in CLOSING_SIGNALS:
        return "close"
    if signal == "HOLD":
        try:
            in_position = int(position) != 0
        except (TypeError, ValueError):
            in_position = False
        # HOLD while in a position is still a live convergence bet; HOLD while
        # flat (emitted when z is NaN) is a stay-flat decision.
        return "entry" if in_position else "flat"
    if signal in FLAT_SIGNALS:
        return "flat"
    return "flat"


def _grade_row(
    row,
    today: pd.Timestamp,
    horizon_days: int,
    zscore_window: int,
    entry_z: float,
    buffer_before: int,
    buffer_after: int,
    price_fetcher: Callable[..., pd.DataFrame],
) -> tuple[str, object, float]:
    """Grade one recorded session. Returns (status, correct, spread_change_z)."""
    session_date = pd.Timestamp(row.date)
    if (today - session_date).days < horizon_days:
        # Cheap calendar guard before any fetch: horizon_days trading bars can
        # never fit inside fewer calendar days.
        return STATUS_TOO_RECENT, pd.NA, np.nan

    if pd.isna(row.hedge_ratio):
        return STATUS_INSUFFICIENT_DATA, pd.NA, np.nan

    start = (session_date - pd.Timedelta(days=buffer_before)).strftime("%Y-%m-%d")
    end = (session_date + pd.Timedelta(days=buffer_after)).strftime("%Y-%m-%d")
    try:
        prices = price_fetcher([row.ticker_a, row.ticker_b], start=start, end=end)
    except Exception:
        return STATUS_INSUFFICIENT_DATA, pd.NA, np.nan

    if row.ticker_a not in prices.columns or row.ticker_b not in prices.columns:
        return STATUS_INSUFFICIENT_DATA, pd.NA, np.nan
    pair = prices[[row.ticker_a, row.ticker_b]].dropna()
    if session_date not in pair.index:
        return STATUS_INSUFFICIENT_DATA, pd.NA, np.nan

    spread = build_spread(pair[row.ticker_a], pair[row.ticker_b], float(row.hedge_ratio))
    zscore = rolling_zscore(spread, zscore_window)

    start_pos = pair.index.get_loc(session_date)
    end_pos = start_pos + horizon_days
    if end_pos >= len(zscore):
        # The outcome window has not finished printing yet.
        return STATUS_TOO_RECENT, pd.NA, np.nan

    z_start = zscore.iloc[start_pos]
    z_end = zscore.iloc[end_pos]
    if pd.isna(z_start) or pd.isna(z_end):
        return STATUS_INSUFFICIENT_DATA, pd.NA, np.nan

    logged_z = row.close_z
    # sign(0) is 0 and would annihilate the statistic; default to +1 so the
    # change is still reported on a consistent orientation.
    direction = float(np.sign(float(logged_z))) if pd.notna(logged_z) else 1.0
    if direction == 0.0:
        direction = 1.0
    change = float(direction * (z_end - z_start))

    family = _decision_family(str(row.signal), row.position)
    if family == "entry":
        correct = bool(change < 0)
    elif family == "close":
        correct = bool(change >= 0)
    else:
        # Stay-flat: standing aside only costs something if a genuine entry
        # opportunity (|z| >= entry_z) printed inside the window.
        window = zscore.iloc[start_pos + 1 : end_pos + 1].dropna()
        missed = bool((window.abs() >= entry_z).any()) if len(window) else False
        correct = not missed

    return STATUS_GRADED, correct, change


# ------------------------------------------------------------------- summary


def performance_summary(path: Path | str = DEFAULT_PERFORMANCE_PATH) -> dict:
    """The accumulated multi-day track record.

    EVERY performance aggregate here is built from `strategy_return_pct` - the
    POSITION-AWARE return - never from `spread_move_pct`. That is what keeps
    cumulative return and cumulative P&L telling the same story: a session where
    the book held nothing contributes 0% and $0, instead of the old behaviour
    where a flat day could still be scored as "beat the market".

    Rows are aggregated to one observation per SESSION: the day's strategy
    return is the equal-weight mean of that day's recorded pairs (an
    equal-weight book of whatever was tracked), and the day's benchmark return
    is the SPY return for that session. Cumulative return compounds the daily
    equal-weight strategy returns; cumulative P&L sums the recorded dollars.

    `n_active_sessions` / `n_flat_sessions` split the record by whether ANY
    tracked pair actually held a position that day. A track record dominated by
    flat sessions has to say so plainly - otherwise a flat, do-nothing history
    reads like a working strategy.

    The Sharpe-style ratio uses backtest/metrics.py's convention:
    risk-free = 0, 252 periods per year, sample std (ddof=1), on daily STRATEGY
    returns.

    Key naming: `cumulative_strategy_return_pct` is the canonical name;
    `cumulative_pair_return_pct` is kept as an alias with the identical value so
    existing callers (run_daily_tracker.py) keep working. Same for the
    `strategy_return_pct` / `pair_return_pct` entries in best_day / worst_day.
    """
    path = Path(path)
    empty = {
        "n_sessions": 0,
        "n_rows": 0,
        "n_active_sessions": 0,
        "n_flat_sessions": 0,
        "first_date": None,
        "last_date": None,
        "cumulative_pnl_usd": 0.0,
        "cumulative_strategy_return_pct": 0.0,
        "cumulative_pair_return_pct": 0.0,
        "cumulative_spy_return_pct": 0.0,
        "days_beating_market": 0,
        "pct_days_beating_market": None,
        "avg_daily_excess_return_pct": None,
        "best_day": None,
        "worst_day": None,
        "sharpe_ratio": None,
    }
    if not path.exists():
        return empty

    history = pd.read_csv(path)
    if history.empty:
        return empty

    history["date"] = pd.to_datetime(history["date"])
    if "position" in history.columns:
        history["position"] = pd.to_numeric(history["position"], errors="coerce").fillna(0)
    else:  # defensive: a hand-edited CSV without the column reads as all-flat
        history["position"] = 0
    daily = history.groupby("date").agg(
        strategy_return_pct=("strategy_return_pct", "mean"),
        spy_return_pct=("spy_return_pct", "mean"),
        pnl_usd=("pair_pnl_usd", "sum"),
        # A session counts as ACTIVE when at least one tracked pair was in a
        # position; otherwise the book held nothing all day.
        n_active_rows=("position", lambda s: int((s != 0).sum())),
    )
    daily = daily.sort_index()
    daily["excess_return_pct"] = daily["strategy_return_pct"] - daily["spy_return_pct"]

    n_sessions = int(len(daily))
    n_active_sessions = int((daily["n_active_rows"] > 0).sum())
    n_flat_sessions = n_sessions - n_active_sessions

    strat_frac = daily["strategy_return_pct"] / 100.0
    spy_frac = daily["spy_return_pct"] / 100.0

    cumulative_strategy = float((1.0 + strat_frac).prod() - 1.0) * 100.0
    cumulative_spy = float((1.0 + spy_frac).prod() - 1.0) * 100.0

    beat = daily["excess_return_pct"] > 0
    days_beating = int(beat.sum())

    if n_sessions > 1 and float(strat_frac.std(ddof=1)) > 0:
        sharpe = float(
            strat_frac.mean() / strat_frac.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe = None

    best_idx = daily["strategy_return_pct"].idxmax()
    worst_idx = daily["strategy_return_pct"].idxmin()
    best_value = float(daily.loc[best_idx, "strategy_return_pct"])
    worst_value = float(daily.loc[worst_idx, "strategy_return_pct"])

    return {
        "n_sessions": n_sessions,
        "n_rows": int(len(history)),
        "n_active_sessions": n_active_sessions,
        "n_flat_sessions": n_flat_sessions,
        "first_date": daily.index[0].strftime("%Y-%m-%d"),
        "last_date": daily.index[-1].strftime("%Y-%m-%d"),
        "cumulative_pnl_usd": float(daily["pnl_usd"].sum()),
        "cumulative_strategy_return_pct": cumulative_strategy,
        "cumulative_pair_return_pct": cumulative_strategy,  # back-compat alias
        "cumulative_spy_return_pct": cumulative_spy,
        "days_beating_market": days_beating,
        "pct_days_beating_market": float(100.0 * days_beating / n_sessions),
        "avg_daily_excess_return_pct": float(daily["excess_return_pct"].mean()),
        "best_day": {
            "date": pd.Timestamp(best_idx).strftime("%Y-%m-%d"),
            "strategy_return_pct": best_value,
            "pair_return_pct": best_value,  # back-compat alias
        },
        "worst_day": {
            "date": pd.Timestamp(worst_idx).strftime("%Y-%m-%d"),
            "strategy_return_pct": worst_value,
            "pair_return_pct": worst_value,  # back-compat alias
        },
        "sharpe_ratio": sharpe,
    }
