"""Forward-test signal journal: log daily signals, grade them after the fact.

WHY this exists: a backtest can always be (accidentally) overfit to the data
it was built on. The journal records what the engine said *at the time* -
hedge ratio, z-score, recommendation - and only later grades each entry
against market data that did not exist when the entry was written. That makes
it an un-fudgeable forward-test record. Appending is idempotent per
(as_of, ticker_a, ticker_b) so a daily cron-style run can be re-executed
safely without duplicating rows.

Like everything in this project, this module is signal research only: it
never places an order and has no execution capability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data.loader import fetch_price_history
from signals.spread import build_spread, rolling_zscore

DEFAULT_JOURNAL_PATH = Path(__file__).resolve().parent.parent / "outputs" / "signal_journal.csv"

# Journal identity key. as_of (the bar date) rather than logged_at, so that
# re-running the report later the same day - or backfilling - cannot create
# a second row for the same observation.
JOURNAL_KEY = ("as_of", "ticker_a", "ticker_b")

# Grading statuses
STATUS_GRADED = "graded"
STATUS_NO_SIGNAL = "no_signal"
STATUS_TOO_RECENT = "too_recent"
STATUS_INSUFFICIENT_DATA = "insufficient_data"


def _as_of_key(value) -> str | None:
    """Normalize as_of to a plain YYYY-MM-DD string so idempotency keys survive
    the round-trip through CSV (Timestamp repr vs parsed string would never match)."""
    # Scalar-only na check: covers None, float nan, and NaT (pandas coerces a
    # None as_of to NaT once the column holds real Timestamps).
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def append_signals(
    report: pd.DataFrame,
    journal_path: Path | str = DEFAULT_JOURNAL_PATH,
) -> int:
    """Append rows from generate_daily_signal_report's output to the journal CSV.

    Idempotent per (as_of, ticker_a, ticker_b): rows whose key already exists
    in the journal are skipped, so re-running the same day is a no-op. Rows
    with no as_of (the NO_DATA placeholder rows) are skipped too - they carry
    no signal to forward-test and have no usable key. Returns the number of
    new rows written.
    """
    journal_path = Path(journal_path)
    if report is None or report.empty:
        return 0

    new_rows = report.copy()
    new_rows["as_of"] = new_rows["as_of"].map(_as_of_key)
    new_rows = new_rows[new_rows["as_of"].notna()]
    if new_rows.empty:
        return 0

    existing_keys: set[tuple[str, str, str]] = set()
    existing_columns: list[str] | None = None
    if journal_path.exists():
        # dtype=str so keys compare as the exact strings on disk, not as
        # whatever pandas would infer (dates, floats) on read.
        existing = pd.read_csv(journal_path, dtype=str)
        existing_columns = list(existing.columns)
        if not existing.empty:
            existing_keys = {
                (row.as_of, row.ticker_a, row.ticker_b) for row in existing.itertuples()
            }

    is_new = [
        (row.as_of, row.ticker_a, row.ticker_b) not in existing_keys
        for row in new_rows.itertuples()
    ]
    new_rows = new_rows[is_new]
    if new_rows.empty:
        return 0

    new_rows = new_rows.copy()
    new_rows["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if existing_columns is not None:
        # Align to the on-disk header so a future schema tweak in the report
        # cannot silently shift values into the wrong columns.
        new_rows = new_rows.reindex(columns=existing_columns)

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows.to_csv(journal_path, mode="a", header=not journal_path.exists(), index=False)
    return len(new_rows)


def grade_journal(
    journal_path: Path | str = DEFAULT_JOURNAL_PATH,
    horizon_days: int = 10,
    zscore_window: int = 30,
    signal_z_threshold: float = 1.0,
    price_fetcher: Callable[..., pd.DataFrame] = fetch_price_history,
) -> tuple[pd.DataFrame, dict]:
    """Grade journal entries that are at least `horizon_days` old.

    For each gradable row the pair's prices are re-fetched around as_of, the
    spread is rebuilt with the LOGGED hedge ratio (not a refit one - refitting
    would let hindsight leak into the grade), and the rolling z-score is
    recomputed. The grade asks: over the next `horizon_days` trading bars,
    did |z| move toward 0 from the side the signal was logged on?

        spread_change_z = sign(logged_z) * (z_end - z_start)

    Negative means the spread converged toward its mean from the logged side
    (the mean-reversion thesis paid off); positive means it diverged further.

    Rows are never dropped; each gets a status:
      - graded: converged/spread_change_z are filled in
      - no_signal: |logged z| < signal_z_threshold, so the entry carried no
        directional prediction and grading it would just add noise
      - too_recent: the outcome window has not finished yet
      - insufficient_data: prices could not support the computation

    `price_fetcher` is injectable so tests can grade against synthetic frames
    without any network access; the default is the cached yfinance loader.

    Returns (graded DataFrame, summary dict with n_graded and hit_rate).
    """
    journal_path = Path(journal_path)
    empty_summary = {"n_graded": 0, "hit_rate": None}
    if not journal_path.exists():
        return pd.DataFrame(), empty_summary

    journal = pd.read_csv(journal_path)
    if journal.empty:
        return journal, empty_summary

    today = pd.Timestamp.today().normalize()
    # Calendar-day buffers around the trading-bar counts: ~2.5 calendar days
    # per trading day comfortably covers weekends and holidays, and the exact
    # bar arithmetic below is what actually decides gradability.
    buffer_before = int(zscore_window * 2.5) + 10
    buffer_after = int(horizon_days * 2.5) + 10

    statuses: list[str] = []
    convergeds: list[object] = []
    changes: list[float] = []

    for row in journal.itertuples():
        status, converged, change = _grade_row(
            row,
            today=today,
            horizon_days=horizon_days,
            zscore_window=zscore_window,
            signal_z_threshold=signal_z_threshold,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            price_fetcher=price_fetcher,
        )
        statuses.append(status)
        convergeds.append(converged)
        changes.append(change)

    result = journal.copy()
    result["status"] = statuses
    result["converged"] = convergeds
    result["spread_change_z"] = changes

    graded_mask = result["status"] == STATUS_GRADED
    n_graded = int(graded_mask.sum())
    hit_rate = (
        float(result.loc[graded_mask, "converged"].astype(bool).mean()) if n_graded else None
    )
    return result, {"n_graded": n_graded, "hit_rate": hit_rate}


def _grade_row(
    row,
    today: pd.Timestamp,
    horizon_days: int,
    zscore_window: int,
    signal_z_threshold: float,
    buffer_before: int,
    buffer_after: int,
    price_fetcher: Callable[..., pd.DataFrame],
) -> tuple[str, object, float]:
    """Grade one journal row. Returns (status, converged, spread_change_z)."""
    z_logged = row.zscore
    if pd.isna(z_logged) or abs(float(z_logged)) < signal_z_threshold:
        # Near-zero z carries no directional prediction: "spread is roughly at
        # its mean" is not a forecast, so there is nothing to hit or miss.
        return STATUS_NO_SIGNAL, pd.NA, np.nan

    as_of = pd.Timestamp(row.as_of)
    if (today - as_of).days < horizon_days:
        # Cheap calendar check before any fetch: horizon_days trading bars can
        # never fit inside fewer calendar days.
        return STATUS_TOO_RECENT, pd.NA, np.nan

    start = (as_of - pd.Timedelta(days=buffer_before)).strftime("%Y-%m-%d")
    end = (as_of + pd.Timedelta(days=buffer_after)).strftime("%Y-%m-%d")
    try:
        prices = price_fetcher([row.ticker_a, row.ticker_b], start=start, end=end)
    except Exception:
        return STATUS_INSUFFICIENT_DATA, pd.NA, np.nan

    if row.ticker_a not in prices.columns or row.ticker_b not in prices.columns:
        return STATUS_INSUFFICIENT_DATA, pd.NA, np.nan
    pair = prices[[row.ticker_a, row.ticker_b]].dropna()
    if as_of not in pair.index:
        return STATUS_INSUFFICIENT_DATA, pd.NA, np.nan

    spread = build_spread(pair[row.ticker_a], pair[row.ticker_b], float(row.hedge_ratio))
    zscore = rolling_zscore(spread, zscore_window)

    start_pos = pair.index.get_loc(as_of)
    end_pos = start_pos + horizon_days
    if end_pos >= len(zscore):
        # Not enough bars after as_of yet: the outcome window is still open.
        return STATUS_TOO_RECENT, pd.NA, np.nan

    z_start = zscore.iloc[start_pos]
    z_end = zscore.iloc[end_pos]
    if pd.isna(z_start) or pd.isna(z_end):
        return STATUS_INSUFFICIENT_DATA, pd.NA, np.nan

    change = float(np.sign(float(z_logged)) * (z_end - z_start))
    return STATUS_GRADED, bool(change < 0), change
