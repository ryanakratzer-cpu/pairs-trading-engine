"""Journal append idempotency and grading math on fully synthetic data.

No network anywhere: grading tests inject a price_fetcher that serves a
hand-built price panel, so every assertion is deterministic and the suite
never touches yfinance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reporting.journal import (
    STATUS_GRADED,
    STATUS_NO_SIGNAL,
    STATUS_TOO_RECENT,
    append_signals,
    grade_journal,
)

# Grading geometry shared by the synthetic panel and the journal rows below.
# zscore_window deliberately exceeds horizon so the rolling mean cannot fully
# adapt to a diverging spread within the grading window (with window <=
# horizon, rolling z mechanically reverts and everything would look converged).
ZSCORE_WINDOW = 20
HORIZON = 5
AS_OF_POS = 50


def _report_row(
    ticker_a: str,
    ticker_b: str,
    as_of,
    zscore: float,
    hedge_ratio: float = 1.0,
    recommendation: str = "ENTER_SHORT_SPREAD",
) -> dict:
    """One row in the schema generate_daily_signal_report produces."""
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "as_of": as_of,
        "hedge_ratio": hedge_ratio,
        "is_cointegrated": True,
        "adf_pvalue": 0.01,
        "half_life_days": 12.0,
        "zscore": zscore,
        "recommendation": recommendation,
    }


@pytest.fixture
def journal_dates():
    return pd.bdate_range("2024-01-01", periods=80)


@pytest.fixture
def synthetic_report(journal_dates):
    as_of = journal_dates[AS_OF_POS]
    return pd.DataFrame(
        [
            _report_row("AAA", "BBB", as_of, zscore=2.4),
            _report_row("CCC", "DDD", as_of, zscore=2.4),
        ]
    )


@pytest.fixture
def synthetic_panel(journal_dates):
    """Price panel with a known post-signal outcome per pair.

    Spread design (hedge ratio 1.0, B pinned at 100 so spread == log A drift):
    - base: +/-0.01 oscillation, mean 0, so the rolling z-score has a stable
      nonzero std to divide by
    - AAA/BBB: spike to 0.03 at as_of, then straight back to the oscillation
      (the spread CONVERGES over the horizon)
    - CCC/DDD: spike at as_of then exponential blow-up (DIVERGES, decisively
      enough that the adapting rolling mean cannot mask it)
    """
    n = len(journal_dates)
    base = 0.01 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)

    spread_conv = base.copy()
    spread_conv[AS_OF_POS] = 0.03

    spread_div = base.copy()
    spread_div[AS_OF_POS : AS_OF_POS + HORIZON + 1] = 0.03 * (2.0 ** np.arange(HORIZON + 1))

    log_b = np.log(100.0)
    panel = pd.DataFrame(
        {
            "AAA": np.exp(log_b + spread_conv),
            "BBB": np.full(n, 100.0),
            "CCC": np.exp(log_b + spread_div),
            "DDD": np.full(n, 100.0),
        },
        index=journal_dates,
    )
    return panel


def _make_fetcher(panel: pd.DataFrame):
    """Injectable stand-in for data.loader.fetch_price_history."""
    calls: list[list[str]] = []

    def fetcher(tickers, start, end, **kwargs):
        calls.append(list(tickers))
        return panel.loc[pd.Timestamp(start) : pd.Timestamp(end), list(tickers)]

    return fetcher, calls


def _forbidden_fetcher(tickers, start, end, **kwargs):
    raise AssertionError(f"price_fetcher must not be called, got {tickers}")


# ---------------------------------------------------------------- append


def test_append_twice_is_idempotent(tmp_path, synthetic_report):
    path = tmp_path / "journal.csv"
    n_first = append_signals(synthetic_report, journal_path=path)
    n_second = append_signals(synthetic_report, journal_path=path)

    assert n_first == 2
    assert n_second == 0
    on_disk = pd.read_csv(path)
    assert len(on_disk) == 2
    assert "logged_at" in on_disk.columns
    assert on_disk["logged_at"].notna().all()


def test_append_only_writes_genuinely_new_rows(tmp_path, synthetic_report, journal_dates):
    path = tmp_path / "journal.csv"
    append_signals(synthetic_report, journal_path=path)

    extra = pd.concat(
        [synthetic_report, pd.DataFrame([_report_row("EEE", "FFF", journal_dates[AS_OF_POS], zscore=1.5)])],
        ignore_index=True,
    )
    n_new = append_signals(extra, journal_path=path)

    assert n_new == 1
    assert len(pd.read_csv(path)) == 3


def test_append_skips_no_data_rows(tmp_path, journal_dates):
    path = tmp_path / "journal.csv"
    report = pd.DataFrame(
        [
            _report_row("AAA", "BBB", journal_dates[AS_OF_POS], zscore=2.0),
            _report_row("ZZZ", "YYY", None, zscore=np.nan, recommendation="NO_DATA"),
        ]
    )
    n_written = append_signals(report, journal_path=path)

    assert n_written == 1
    on_disk = pd.read_csv(path)
    assert list(on_disk["ticker_a"]) == ["AAA"]


def test_append_empty_report_is_noop(tmp_path):
    path = tmp_path / "journal.csv"
    assert append_signals(pd.DataFrame(), journal_path=path) == 0
    assert not path.exists()


# ---------------------------------------------------------------- grading


def test_grading_converged_and_diverged(tmp_path, synthetic_report, synthetic_panel):
    path = tmp_path / "journal.csv"
    append_signals(synthetic_report, journal_path=path)
    fetcher, calls = _make_fetcher(synthetic_panel)

    graded, summary = grade_journal(
        journal_path=path,
        horizon_days=HORIZON,
        zscore_window=ZSCORE_WINDOW,
        price_fetcher=fetcher,
    )

    assert len(graded) == 2  # nothing dropped
    assert (graded["status"] == STATUS_GRADED).all()

    conv = graded[graded["ticker_a"] == "AAA"].iloc[0]
    div = graded[graded["ticker_a"] == "CCC"].iloc[0]
    assert bool(conv["converged"]) is True
    assert conv["spread_change_z"] < 0  # negative = converged from the logged side
    assert bool(div["converged"]) is False
    assert div["spread_change_z"] > 0

    assert summary == {"n_graded": 2, "hit_rate": 0.5}
    assert len(calls) == 2  # one fetch per gradable row, nothing extra


def test_no_signal_rows_are_not_graded_and_never_fetch(tmp_path, journal_dates):
    path = tmp_path / "journal.csv"
    report = pd.DataFrame([_report_row("AAA", "BBB", journal_dates[AS_OF_POS], zscore=0.3)])
    append_signals(report, journal_path=path)

    graded, summary = grade_journal(
        journal_path=path,
        horizon_days=HORIZON,
        zscore_window=ZSCORE_WINDOW,
        price_fetcher=_forbidden_fetcher,
    )

    assert len(graded) == 1
    assert graded.iloc[0]["status"] == STATUS_NO_SIGNAL
    assert pd.isna(graded.iloc[0]["converged"])
    assert summary == {"n_graded": 0, "hit_rate": None}


def test_too_recent_rows_stay_ungraded_and_never_fetch(tmp_path):
    path = tmp_path / "journal.csv"
    today = pd.Timestamp.today().normalize()
    report = pd.DataFrame([_report_row("AAA", "BBB", today, zscore=2.5)])
    append_signals(report, journal_path=path)

    graded, summary = grade_journal(
        journal_path=path,
        horizon_days=HORIZON,
        zscore_window=ZSCORE_WINDOW,
        price_fetcher=_forbidden_fetcher,
    )

    assert len(graded) == 1  # returned ungraded, not dropped
    assert graded.iloc[0]["status"] == STATUS_TOO_RECENT
    assert pd.isna(graded.iloc[0]["converged"])
    assert summary == {"n_graded": 0, "hit_rate": None}


def test_recent_row_with_partial_price_window_is_too_recent(
    tmp_path, journal_dates, synthetic_panel
):
    """A row old enough by calendar days but whose horizon bars have not all
    printed yet must also stay ungraded (the outcome window is still open)."""
    path = tmp_path / "journal.csv"
    late_as_of = journal_dates[-2]  # only 1 bar after it in the panel
    report = pd.DataFrame([_report_row("AAA", "BBB", late_as_of, zscore=2.0)])
    append_signals(report, journal_path=path)
    fetcher, _ = _make_fetcher(synthetic_panel)

    graded, summary = grade_journal(
        journal_path=path,
        horizon_days=HORIZON,
        zscore_window=ZSCORE_WINDOW,
        price_fetcher=fetcher,
    )

    assert graded.iloc[0]["status"] == STATUS_TOO_RECENT
    assert summary["n_graded"] == 0


def test_grading_missing_journal_returns_empty(tmp_path):
    graded, summary = grade_journal(
        journal_path=tmp_path / "does_not_exist.csv", price_fetcher=_forbidden_fetcher
    )
    assert graded.empty
    assert summary == {"n_graded": 0, "hit_rate": None}
