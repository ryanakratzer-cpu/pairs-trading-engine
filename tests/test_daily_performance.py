"""Multi-day performance tracker: append idempotency, schema, grading, summary math.

No network anywhere. Both fetchers (daily closes and session OHLC) are
injected with hand-built frames, so every assertion is deterministic and the
suite never touches yfinance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reporting.daily_performance import (
    PERFORMANCE_COLUMNS,
    STATUS_GRADED,
    STATUS_TOO_RECENT,
    grade_previous_sessions,
    performance_summary,
    record_session,
)
from signals.spread import SignalConfig

# Grading geometry, shared by the synthetic panel and the recorded rows.
# zscore_window deliberately exceeds horizon so the rolling mean cannot fully
# adapt to a diverging spread inside the grading window.
ZSCORE_WINDOW = 20
HORIZON = 5
AS_OF_POS = 50
NOTIONAL = 10_000.0


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def session_dates():
    return pd.bdate_range("2024-01-01", periods=80)


@pytest.fixture
def close_panel(session_dates):
    """Daily closes: AAA/BBB is a genuine cointegrated pair (OU spread on a
    random-walk B), SPY is an independent drifting benchmark."""
    n = len(session_dates)
    rng = np.random.default_rng(7)

    log_b = np.log(60.0) + np.cumsum(0.01 * rng.standard_normal(n))
    spread = np.empty(n)
    spread[0] = 0.0
    for t in range(1, n):
        spread[t] = spread[t - 1] + 0.15 * (0.0 - spread[t - 1]) + 0.02 * rng.standard_normal()
    log_a = 0.9 * log_b + spread

    log_spy = np.log(450.0) + np.cumsum(0.008 * rng.standard_normal(n))

    return pd.DataFrame(
        {"AAA": np.exp(log_a), "BBB": np.exp(log_b), "SPY": np.exp(log_spy)},
        index=session_dates,
    )


@pytest.fixture
def ohlc_panel(close_panel):
    """(field, ticker) MultiIndex OHLC frame. Opens are a fixed haircut off the
    close so every session has a known, nonzero open->close return per leg."""
    haircut = {"AAA": 0.990, "BBB": 0.995, "SPY": 0.998}
    opens = close_panel.apply(lambda col: col * haircut[col.name])
    return pd.concat({"Open": opens, "Close": close_panel}, axis=1)


def _close_fetcher(panel: pd.DataFrame):
    def fetcher(tickers, start, end, **kwargs):
        cols = [t for t in tickers if t in panel.columns]
        return panel.loc[pd.Timestamp(start) : pd.Timestamp(end), cols]

    return fetcher


def _ohlc_fetcher(ohlc: pd.DataFrame):
    def fetcher(tickers, start, end, **kwargs):
        return ohlc.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    return fetcher


def _forbidden_fetcher(tickers, start, end, **kwargs):
    raise AssertionError(f"price_fetcher must not be called, got {tickers}")


def _record(path, close_panel, ohlc_panel, as_of, pairs=None):
    return record_session(
        pairs or [("AAA", "BBB")],
        # These tests run on synthetic date ranges and isolate session-recording
        # mechanics; the FOMC/election gate is pinned separately below so the
        # mask can't decide the signal here. Production defaults to gated.
        apply_event_gate=False,
        as_of=as_of,
        notional=NOTIONAL,
        path=path,
        signal_config=SignalConfig(zscore_window=ZSCORE_WINDOW),
        price_fetcher=_close_fetcher(close_panel),
        ohlc_fetcher=_ohlc_fetcher(ohlc_panel),
        verbose=False,
    )


# ----------------------------------------------------------------- recording


def test_record_session_writes_exact_schema(tmp_path, close_panel, ohlc_panel, session_dates):
    path = tmp_path / "perf.csv"
    as_of = session_dates[60].strftime("%Y-%m-%d")

    assert _record(path, close_panel, ohlc_panel, as_of) == 1

    on_disk = pd.read_csv(path)
    assert list(on_disk.columns) == PERFORMANCE_COLUMNS
    row = on_disk.iloc[0]
    assert row["date"] == session_dates[60].strftime("%Y-%m-%d")
    assert row["ticker_a"] == "AAA" and row["ticker_b"] == "BBB"
    assert row["position"] in (-1, 0, 1)
    assert pd.notna(row["logged_at"])


def test_record_session_is_idempotent(tmp_path, close_panel, ohlc_panel, session_dates):
    path = tmp_path / "perf.csv"
    as_of = session_dates[60].strftime("%Y-%m-%d")

    first = _record(path, close_panel, ohlc_panel, as_of)
    second = _record(path, close_panel, ohlc_panel, as_of)

    assert first == 1
    assert second == 0
    assert len(pd.read_csv(path)) == 1


def test_record_session_return_conventions(tmp_path, close_panel, ohlc_panel, session_dates):
    """spread_move_pct is the position-agnostic log-spread change (diagnostic);
    strategy_return_pct applies the position, and pair_pnl_usd is derived from
    strategy_return_pct so the two can never disagree."""
    path = tmp_path / "perf.csv"
    session = session_dates[60]
    _record(path, close_panel, ohlc_panel, session.strftime("%Y-%m-%d"))
    row = pd.read_csv(path).iloc[0]

    # UPDATED 2026-07-30: the anchor is the PRIOR SESSION'S CLOSE, not this
    # session's open. The strategy holds positions overnight (6-12 day
    # half-lives), so an open->close window silently discards the overnight gap
    # — which on 2026-07-30 was the entire day's P&L (+$252 real vs -$43
    # booked). This assertion previously encoded the open->close convention and
    # is deliberately changed, not weakened: it still pins an exact value.
    h = float(row["hedge_ratio"])
    prior_session = close_panel.index[close_panel.index.get_loc(session) - 1]
    anchor_a = close_panel.loc[prior_session, "AAA"]
    anchor_b = close_panel.loc[prior_session, "BBB"]
    close_a = close_panel.loc[session, "AAA"]
    close_b = close_panel.loc[session, "BBB"]

    expected_move = 100.0 * (
        (np.log(close_a) - h * np.log(close_b)) - (np.log(anchor_a) - h * np.log(anchor_b))
    )
    assert row["spread_move_pct"] == pytest.approx(expected_move)
    assert row["strategy_return_pct"] == pytest.approx(
        int(row["position"]) * expected_move
    )
    assert row["pair_pnl_usd"] == pytest.approx(
        row["strategy_return_pct"] / 100.0 * NOTIONAL
    )

    # Per-leg returns stay open->close: they are intraday DIAGNOSTICS, not the
    # benchmarked performance number.
    assert row["ticker_a_return_pct"] == pytest.approx(100.0 * (1 / 0.990 - 1))
    assert row["ticker_b_return_pct"] == pytest.approx(100.0 * (1 / 0.995 - 1))
    # UPDATED 2026-07-30: the benchmark now uses the SAME close-to-close window
    # as strategy_return_pct. Comparing an overnight-inclusive strategy return
    # against an intraday-only SPY would make excess_return_pct meaningless.
    spy_prior = close_panel.loc[prior_session, "SPY"]
    spy_close = close_panel.loc[session, "SPY"]
    assert row["spy_return_pct"] == pytest.approx(100.0 * (spy_close / spy_prior - 1))
    # Excess is benchmarked against what WE earned, not against the spread move.
    assert row["excess_return_pct"] == pytest.approx(
        row["strategy_return_pct"] - row["spy_return_pct"]
    )


def test_recorded_flat_session_earns_nothing(tmp_path, close_panel, ohlc_panel, session_dates):
    """REGRESSION: the recorder must never book a return on a day we held nothing.

    Sweep every recordable session; on each flat one the strategy return, the
    P&L and the excess must all reflect "we were out of the market", even though
    the spread itself moved.
    """
    flat_seen = 0
    for i in range(45, 80):
        path = tmp_path / f"perf_{i}.csv"
        if _record(path, close_panel, ohlc_panel, session_dates[i].strftime("%Y-%m-%d")) != 1:
            continue
        row = pd.read_csv(path).iloc[0]
        if int(row["position"]) != 0:
            continue
        flat_seen += 1
        assert row["strategy_return_pct"] == 0.0
        assert row["pair_pnl_usd"] == 0.0
        assert row["excess_return_pct"] == pytest.approx(-row["spy_return_pct"])
        # ...and the diagnostic spread move is still recorded, unchanged.
        assert pd.notna(row["spread_move_pct"])

    assert flat_seen > 0, "fixture produced no flat sessions to regression-test"


def test_recorded_pnl_and_return_agree_in_sign(tmp_path, close_panel, ohlc_panel, session_dates):
    """pair_pnl_usd is derived from strategy_return_pct, so signs always match."""
    for i in range(45, 80):
        path = tmp_path / f"perf_{i}.csv"
        if _record(path, close_panel, ohlc_panel, session_dates[i].strftime("%Y-%m-%d")) != 1:
            continue
        row = pd.read_csv(path).iloc[0]
        assert np.sign(row["pair_pnl_usd"]) == np.sign(row["strategy_return_pct"])
        # A short position inverts the spread move; a long one passes it through.
        assert row["strategy_return_pct"] == pytest.approx(
            int(row["position"]) * row["spread_move_pct"]
        )


@pytest.fixture
def short_entry_panels(session_dates):
    """A panel engineered to force a SHORT-spread entry on session AS_OF_POS,
    with a POSITIVE intraday spread move.

    BBB's open EQUALS its close, so the session's log-spread move collapses to
    log(close_A) - log(open_A) whatever hedge ratio gets fitted; A's open is a
    haircut BELOW its close, so the spread rose. A +0.03 spike against a +/-0.01
    oscillation puts the rolling z near +3, which trips ENTER_SHORT_SPREAD.
    (BBB still drifts day-to-day: a perfectly constant leg makes the OLS design
    matrix rank-deficient.)
    """
    n = len(session_dates)
    t = np.arange(n)
    base = 0.01 * np.where(t % 2 == 0, 1.0, -1.0)
    base[AS_OF_POS] = 0.03
    log_b = np.log(100.0) + 0.004 * np.sin(t / 5.0)
    closes = pd.DataFrame(
        {
            "AAA": np.exp(log_b + base),
            "BBB": np.exp(log_b),
            "SPY": np.full(n, 450.0),
        },
        index=session_dates,
    )
    haircut = {"AAA": 0.98, "BBB": 1.0, "SPY": 1.0}
    opens = closes.apply(lambda col: col * haircut[col.name])
    return closes, pd.concat({"Open": opens, "Close": closes}, axis=1)


def test_short_spread_session_with_positive_spread_move_loses(tmp_path, short_entry_panels, session_dates):
    """REGRESSION: sign agreement. Short the spread and the spread RISES ->
    strategy_return_pct and pair_pnl_usd must both be NEGATIVE, even though the
    diagnostic spread_move_pct is positive."""
    closes, ohlc = short_entry_panels
    path = tmp_path / "perf.csv"

    assert _record(path, closes, ohlc, session_dates[AS_OF_POS].strftime("%Y-%m-%d")) == 1
    row = pd.read_csv(path).iloc[0]

    assert int(row["position"]) == -1
    assert row["spread_move_pct"] > 0  # the spread moved AGAINST the short
    assert row["strategy_return_pct"] < 0
    assert row["pair_pnl_usd"] < 0
    assert row["strategy_return_pct"] == pytest.approx(-row["spread_move_pct"])
    assert row["pair_pnl_usd"] == pytest.approx(row["strategy_return_pct"] / 100.0 * NOTIONAL)


def test_record_session_falls_back_to_last_completed_session(
    tmp_path, close_panel, ohlc_panel, session_dates, capsys
):
    """Today's bar is often unpublished. The recorded date must be the real
    completed session, never the (mislabeled) requested one."""
    path = tmp_path / "perf.csv"
    requested = session_dates[-1] + pd.Timedelta(days=3)

    n = record_session(
        [("AAA", "BBB")],
        apply_event_gate=False,
        as_of=requested.strftime("%Y-%m-%d"),
        notional=NOTIONAL,
        path=path,
        signal_config=SignalConfig(zscore_window=ZSCORE_WINDOW),
        price_fetcher=_close_fetcher(close_panel),
        ohlc_fetcher=_ohlc_fetcher(ohlc_panel),
        verbose=True,
    )
    printed = capsys.readouterr().out

    assert n == 1
    assert pd.read_csv(path).iloc[0]["date"] == session_dates[-1].strftime("%Y-%m-%d")
    assert "CAVEAT" in printed
    assert session_dates[-1].strftime("%Y-%m-%d") in printed


def test_record_session_multiple_pairs_one_row_each(tmp_path, close_panel, ohlc_panel, session_dates):
    path = tmp_path / "perf.csv"
    panel = close_panel.rename(columns={"AAA": "AAA"})
    panel = panel.assign(CCC=close_panel["AAA"] * 1.01, DDD=close_panel["BBB"] * 0.99)
    haircut = {"AAA": 0.990, "BBB": 0.995, "SPY": 0.998, "CCC": 0.993, "DDD": 0.997}
    ohlc = pd.concat(
        {"Open": panel.apply(lambda c: c * haircut[c.name]), "Close": panel}, axis=1
    )

    n = _record(
        path, panel, ohlc, session_dates[60].strftime("%Y-%m-%d"),
        pairs=[("AAA", "BBB"), ("CCC", "DDD")],
    )
    assert n == 2
    assert sorted(pd.read_csv(path)["ticker_a"]) == ["AAA", "CCC"]


# ------------------------------------------------------------------ grading


@pytest.fixture
def grading_panel(session_dates):
    """Post-decision outcomes engineered per pair (hedge ratio 1.0, B pinned at
    100 so the spread is just log A's drift):
      - base +/-0.01 oscillation, mean 0, giving the rolling z a stable std
      - AAA/BBB: single spike at AS_OF then straight back to the oscillation
        (spread CONVERGES over the horizon)
      - CCC/DDD: spike then exponential blow-up (DIVERGES decisively enough
        that the adapting rolling mean cannot mask it)
    """
    n = len(session_dates)
    base = 0.01 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)

    conv = base.copy()
    conv[AS_OF_POS] = 0.03

    div = base.copy()
    div[AS_OF_POS : AS_OF_POS + HORIZON + 1] = 0.03 * (2.0 ** np.arange(HORIZON + 1))

    log_b = np.log(100.0)
    return pd.DataFrame(
        {
            "AAA": np.exp(log_b + conv),
            "BBB": np.full(n, 100.0),
            "CCC": np.exp(log_b + div),
            "DDD": np.full(n, 100.0),
        },
        index=session_dates,
    )


def _grading_fetcher(panel: pd.DataFrame):
    calls: list[list[str]] = []

    def fetcher(tickers, start, end, **kwargs):
        calls.append(list(tickers))
        return panel.loc[pd.Timestamp(start) : pd.Timestamp(end), list(tickers)]

    return fetcher, calls


def _perf_row(ticker_a, ticker_b, date, close_z, signal, position, **overrides) -> dict:
    row = {
        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "hedge_ratio": 1.0,
        "adf_pvalue": 0.01,
        "is_cointegrated": True,
        "half_life_days": 12.0,
        "open_z": close_z,
        "close_z": close_z,
        "signal": signal,
        "position": position,
        "strategy_return_pct": 0.1,
        "spread_move_pct": 0.1,
        "pair_pnl_usd": 10.0,
        "ticker_a_return_pct": 0.2,
        "ticker_b_return_pct": 0.1,
        "spy_return_pct": 0.05,
        "excess_return_pct": 0.05,
        "logged_at": "2024-01-01T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def _write_perf_csv(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).reindex(columns=PERFORMANCE_COLUMNS).to_csv(path, index=False)


def test_grading_entry_converged_vs_diverged(tmp_path, session_dates, grading_panel):
    path = tmp_path / "perf.csv"
    as_of = session_dates[AS_OF_POS]
    _write_perf_csv(
        path,
        [
            _perf_row("AAA", "BBB", as_of, 2.4, "ENTER_SHORT_SPREAD", -1),
            _perf_row("CCC", "DDD", as_of, 2.4, "ENTER_SHORT_SPREAD", -1),
        ],
    )
    fetcher, calls = _grading_fetcher(grading_panel)

    graded, summary = grade_previous_sessions(
        path=path, horizon_days=HORIZON, zscore_window=ZSCORE_WINDOW, price_fetcher=fetcher
    )

    assert len(graded) == 2  # nothing dropped
    assert (graded["status"] == STATUS_GRADED).all()

    conv = graded[graded["ticker_a"] == "AAA"].iloc[0]
    div = graded[graded["ticker_a"] == "CCC"].iloc[0]
    assert bool(conv["correct"]) is True
    assert conv["spread_change_z"] < 0  # negative = converged from the logged side
    assert bool(div["correct"]) is False
    assert div["spread_change_z"] > 0

    assert summary["n_graded"] == 2
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["by_decision"]["ENTER_SHORT_SPREAD"] == {"n": 2, "hit_rate": 0.5}
    assert len(calls) == 2  # one fetch per gradable row, nothing extra


def test_grading_close_decisions_invert_the_test(tmp_path, session_dates, grading_panel):
    """Closing is a bet that convergence is OVER: the diverging pair vindicates
    the exit, the converging pair means the exit was premature."""
    path = tmp_path / "perf.csv"
    as_of = session_dates[AS_OF_POS]
    _write_perf_csv(
        path,
        [
            _perf_row("AAA", "BBB", as_of, 2.4, "EXIT", 0),
            _perf_row("CCC", "DDD", as_of, 2.4, "STOP_LOSS", 0),
        ],
    )
    fetcher, _ = _grading_fetcher(grading_panel)

    graded, summary = grade_previous_sessions(
        path=path, horizon_days=HORIZON, zscore_window=ZSCORE_WINDOW, price_fetcher=fetcher
    )

    premature = graded[graded["ticker_a"] == "AAA"].iloc[0]
    vindicated = graded[graded["ticker_a"] == "CCC"].iloc[0]
    assert bool(premature["correct"]) is False  # it kept converging without us
    assert bool(vindicated["correct"]) is True  # it kept diverging, good exit
    assert summary["n_graded"] == 2


def test_grading_stay_flat(tmp_path, session_dates, grading_panel):
    """Standing aside is correct only when no |z| >= entry_z chance appeared."""
    path = tmp_path / "perf.csv"
    as_of = session_dates[AS_OF_POS]
    _write_perf_csv(
        path,
        [
            _perf_row("AAA", "BBB", as_of, 0.4, "NO_POSITION", 0),
            _perf_row("CCC", "DDD", as_of, 0.4, "NO_POSITION", 0),
        ],
    )
    fetcher, _ = _grading_fetcher(grading_panel)

    graded, summary = grade_previous_sessions(
        path=path, horizon_days=HORIZON, zscore_window=ZSCORE_WINDOW, entry_z=2.0,
        price_fetcher=fetcher,
    )

    quiet = graded[graded["ticker_a"] == "AAA"].iloc[0]
    missed = graded[graded["ticker_a"] == "CCC"].iloc[0]
    assert bool(quiet["correct"]) is True  # nothing to miss, flat cost nothing
    assert bool(missed["correct"]) is False  # a 2-sigma entry printed; we missed it
    assert summary["n_graded"] == 2
    assert summary["by_decision"]["NO_POSITION"]["hit_rate"] == pytest.approx(0.5)


def test_fresh_row_is_too_recent_and_never_fetches(tmp_path):
    path = tmp_path / "perf.csv"
    today = pd.Timestamp.today().normalize()
    _write_perf_csv(path, [_perf_row("AAA", "BBB", today, 2.5, "ENTER_SHORT_SPREAD", -1)])

    graded, summary = grade_previous_sessions(
        path=path, horizon_days=HORIZON, zscore_window=ZSCORE_WINDOW,
        price_fetcher=_forbidden_fetcher,
    )

    assert len(graded) == 1  # returned ungraded, not dropped
    assert graded.iloc[0]["status"] == STATUS_TOO_RECENT
    assert pd.isna(graded.iloc[0]["correct"])
    assert summary == {"n_graded": 0, "hit_rate": None, "by_decision": {}}


def test_grading_missing_file_returns_empty(tmp_path):
    graded, summary = grade_previous_sessions(
        path=tmp_path / "nope.csv", price_fetcher=_forbidden_fetcher
    )
    assert graded.empty
    assert summary == {"n_graded": 0, "hit_rate": None, "by_decision": {}}


def test_grading_never_mutates_the_csv(tmp_path, session_dates, grading_panel):
    path = tmp_path / "perf.csv"
    as_of = session_dates[AS_OF_POS]
    _write_perf_csv(path, [_perf_row("AAA", "BBB", as_of, 2.4, "ENTER_SHORT_SPREAD", -1)])
    before = path.read_text()

    fetcher, _ = _grading_fetcher(grading_panel)
    grade_previous_sessions(
        path=path, horizon_days=HORIZON, zscore_window=ZSCORE_WINDOW, price_fetcher=fetcher
    )

    assert path.read_text() == before


# ------------------------------------------------------------------ summary


def test_performance_summary_math(tmp_path):
    """Hand-checked two-session track record.

    day 1: strategy +1.00%, spy +0.50%, pnl +$100  -> beats the market
    day 2: strategy -0.50%, spy +0.25%, pnl  -$50  -> lags the market
    """
    path = tmp_path / "perf.csv"
    _write_perf_csv(
        path,
        [
            _perf_row(
                "AAA", "BBB", "2024-03-01", 2.4, "ENTER_SHORT_SPREAD", -1,
                strategy_return_pct=1.0, spread_move_pct=-1.0, pair_pnl_usd=100.0,
                spy_return_pct=0.5, excess_return_pct=0.5,
            ),
            _perf_row(
                "AAA", "BBB", "2024-03-04", -0.5, "HOLD", -1,
                strategy_return_pct=-0.5, spread_move_pct=0.5, pair_pnl_usd=-50.0,
                spy_return_pct=0.25, excess_return_pct=-0.75,
            ),
        ],
    )

    s = performance_summary(path)

    assert s["n_sessions"] == 2
    assert s["n_rows"] == 2
    assert s["n_active_sessions"] == 2
    assert s["n_flat_sessions"] == 0
    assert s["first_date"] == "2024-03-01"
    assert s["last_date"] == "2024-03-04"
    assert s["cumulative_pnl_usd"] == pytest.approx(50.0)
    # compounded: 1.01 * 0.995 - 1 = 0.00495
    assert s["cumulative_strategy_return_pct"] == pytest.approx(0.495)
    assert s["cumulative_pair_return_pct"] == pytest.approx(0.495)  # back-compat alias
    # compounded: 1.005 * 1.0025 - 1 = 0.0075125
    assert s["cumulative_spy_return_pct"] == pytest.approx(0.75125)
    assert s["days_beating_market"] == 1
    assert s["pct_days_beating_market"] == pytest.approx(50.0)
    assert s["avg_daily_excess_return_pct"] == pytest.approx(-0.125)
    assert s["best_day"]["date"] == "2024-03-01"
    assert s["best_day"]["strategy_return_pct"] == pytest.approx(1.0)
    assert s["worst_day"]["date"] == "2024-03-04"
    assert s["worst_day"]["strategy_return_pct"] == pytest.approx(-0.5)
    # mean/std(ddof=1) * sqrt(252) on [0.01, -0.005]
    assert s["sharpe_ratio"] == pytest.approx(0.0025 / 0.010606602 * np.sqrt(252), rel=1e-6)
    # Cumulative return and cumulative P&L must tell the SAME story.
    assert np.sign(s["cumulative_strategy_return_pct"]) == np.sign(s["cumulative_pnl_usd"])


def test_summary_all_flat_sessions_are_flat_everywhere(tmp_path):
    """REGRESSION: the exact bug. Two flat sessions where the spread moved a lot
    must produce 0% cumulative return, $0 P&L, and ZERO days beating a market
    that rose - not "+X%, beat the market 100% of days"."""
    path = tmp_path / "perf.csv"
    _write_perf_csv(
        path,
        [
            _perf_row(
                "ALL", "TRV", "2024-03-01", -0.28, "NO_POSITION", 0,
                strategy_return_pct=0.0, spread_move_pct=2.967, pair_pnl_usd=0.0,
                spy_return_pct=0.226, excess_return_pct=-0.226,
            ),
            _perf_row(
                "ALL", "TRV", "2024-03-04", 0.4, "NO_POSITION", 0,
                strategy_return_pct=0.0, spread_move_pct=-1.5, pair_pnl_usd=0.0,
                spy_return_pct=0.1, excess_return_pct=-0.1,
            ),
        ],
    )

    s = performance_summary(path)

    assert s["n_sessions"] == 2
    assert s["n_flat_sessions"] == 2
    assert s["n_active_sessions"] == 0
    assert s["cumulative_strategy_return_pct"] == pytest.approx(0.0)
    assert s["cumulative_pnl_usd"] == pytest.approx(0.0)
    assert s["days_beating_market"] == 0
    assert s["pct_days_beating_market"] == pytest.approx(0.0)
    assert s["avg_daily_excess_return_pct"] == pytest.approx(-0.163)
    assert s["cumulative_spy_return_pct"] > 0  # the market rose; we did not


def test_summary_counts_active_and_flat_sessions(tmp_path):
    """A record dominated by flat days must say so: one active session, two flat."""
    path = tmp_path / "perf.csv"
    _write_perf_csv(
        path,
        [
            _perf_row("AAA", "BBB", "2024-03-01", 2.4, "ENTER_SHORT_SPREAD", -1,
                      strategy_return_pct=1.0, spread_move_pct=-1.0,
                      pair_pnl_usd=100.0, spy_return_pct=0.5),
            # Same session, a second pair that stayed flat -> session still ACTIVE.
            _perf_row("CCC", "DDD", "2024-03-01", 0.1, "NO_POSITION", 0,
                      strategy_return_pct=0.0, spread_move_pct=4.0,
                      pair_pnl_usd=0.0, spy_return_pct=0.5),
            _perf_row("AAA", "BBB", "2024-03-04", 0.2, "NO_POSITION", 0,
                      strategy_return_pct=0.0, spread_move_pct=3.0,
                      pair_pnl_usd=0.0, spy_return_pct=0.1),
            _perf_row("AAA", "BBB", "2024-03-05", 0.3, "HOLD", 0,
                      strategy_return_pct=0.0, spread_move_pct=-2.0,
                      pair_pnl_usd=0.0, spy_return_pct=-0.2),
        ],
    )

    s = performance_summary(path)

    assert s["n_sessions"] == 3
    assert s["n_rows"] == 4
    assert s["n_active_sessions"] == 1
    assert s["n_flat_sessions"] == 2
    # The active session's book return is the equal-weight mean of +1% and 0%.
    assert s["best_day"]["strategy_return_pct"] == pytest.approx(0.5)
    # 03-01: +0.5% vs SPY +0.5% -> tie, not a beat. 03-04: 0% vs +0.1% -> lag.
    # 03-05: 0% vs -0.2% -> sitting out beat a falling market. Exactly one.
    assert s["days_beating_market"] == 1


def test_summary_flat_day_beats_only_a_falling_market(tmp_path):
    """Holding nothing beats the market exactly when the market fell."""
    path = tmp_path / "perf.csv"
    _write_perf_csv(
        path,
        [
            _perf_row("AAA", "BBB", "2024-03-01", 0.2, "NO_POSITION", 0,
                      strategy_return_pct=0.0, spread_move_pct=5.0,
                      pair_pnl_usd=0.0, spy_return_pct=1.0),
            _perf_row("AAA", "BBB", "2024-03-04", 0.3, "NO_POSITION", 0,
                      strategy_return_pct=0.0, spread_move_pct=-5.0,
                      pair_pnl_usd=0.0, spy_return_pct=-1.0),
        ],
    )

    s = performance_summary(path)
    assert s["days_beating_market"] == 1
    assert s["cumulative_strategy_return_pct"] == pytest.approx(0.0)
    assert s["cumulative_pnl_usd"] == pytest.approx(0.0)


def test_performance_summary_averages_pairs_within_a_session(tmp_path):
    """Two pairs on one date = ONE session, equal-weighted; P&L still sums."""
    path = tmp_path / "perf.csv"
    _write_perf_csv(
        path,
        [
            _perf_row("AAA", "BBB", "2024-03-01", 2.4, "ENTER_SHORT_SPREAD", -1,
                      strategy_return_pct=1.0, spread_move_pct=-1.0,
                      pair_pnl_usd=100.0, spy_return_pct=0.5),
            _perf_row("CCC", "DDD", "2024-03-01", -2.4, "ENTER_LONG_SPREAD", 1,
                      strategy_return_pct=-3.0, spread_move_pct=-3.0,
                      pair_pnl_usd=-300.0, spy_return_pct=0.5),
        ],
    )

    s = performance_summary(path)
    assert s["n_sessions"] == 1
    assert s["n_rows"] == 2
    assert s["n_active_sessions"] == 1
    assert s["cumulative_pnl_usd"] == pytest.approx(-200.0)
    assert s["cumulative_strategy_return_pct"] == pytest.approx(-1.0)  # mean of +1 and -3
    assert s["days_beating_market"] == 0
    assert s["sharpe_ratio"] is None  # one session: no dispersion to measure


def test_performance_summary_missing_file(tmp_path):
    s = performance_summary(tmp_path / "nope.csv")
    assert s["n_sessions"] == 0
    assert s["n_active_sessions"] == 0
    assert s["n_flat_sessions"] == 0
    assert s["cumulative_pnl_usd"] == 0.0
    assert s["cumulative_strategy_return_pct"] == 0.0
    assert s["sharpe_ratio"] is None


def test_event_gate_blocks_new_entries_on_fomc_blackout():
    """REGRESSION (2026-07-29): the tracker generated signals WITHOUT the
    event-exclusion gate, so it would log entries on FOMC blackout dates that
    the real gated engine (run_screen.py / run_focus_book.py) refuses to take —
    silently inflating the multi-day track record with trades that never happen.

    Pins the gate directly: on a blackout date a fresh |z| past the entry band
    must NOT produce an entry, while the same z ungated must.
    """
    from screening.events import FOMC_DECISION_DATES, event_exclusion_mask
    from signals.spread import generate_signals

    blackout = pd.Timestamp(FOMC_DECISION_DATES[-1])
    index = pd.bdate_range(blackout - pd.Timedelta(days=10), blackout)
    # Flat, then a dislocation past the +2.0 entry band on the blackout bar.
    z = pd.Series(0.0, index=index)
    z.iloc[-1] = 2.5

    gate = event_exclusion_mask(index)
    assert not bool(gate.loc[blackout]), "the blackout date must be gated off"

    gated = generate_signals(z, SignalConfig(), tradeable=gate)
    ungated = generate_signals(z, SignalConfig())

    assert ungated["event"].iloc[-1] == "ENTER_SHORT_SPREAD"
    assert gated["event"].iloc[-1] == "NO_POSITION"
    assert int(gated["position"].iloc[-1]) == 0


def test_record_session_runs_the_gated_production_path(tmp_path, close_panel, ohlc_panel, session_dates):
    """REGRESSION: every other test passes apply_event_gate=False, so the GATED
    branch — the one production actually uses — was never executed, and a missing
    `event_exclusion_mask` import survived a green suite and only blew up live
    with a NameError. This test exercises record_session with the gate ON.
    """
    path = tmp_path / "perf.csv"
    n = record_session(
        [("AAA", "BBB")],
        apply_event_gate=True,  # the production default; must not raise
        as_of=session_dates[60].strftime("%Y-%m-%d"),
        notional=NOTIONAL,
        path=path,
        signal_config=SignalConfig(zscore_window=ZSCORE_WINDOW),
        price_fetcher=_close_fetcher(close_panel),
        ohlc_fetcher=_ohlc_fetcher(ohlc_panel),
        verbose=False,
    )
    assert n == 1
    row = pd.read_csv(path).iloc[0]
    assert list(pd.read_csv(path).columns) == PERFORMANCE_COLUMNS
    # Gated or not, a recorded position is still one of the three valid states.
    assert int(row["position"]) in (-1, 0, 1)


def test_overnight_gap_is_captured_not_discarded(tmp_path, close_panel, ohlc_panel, session_dates):
    """REGRESSION (2026-07-30): the tracker measured OPEN->CLOSE, so for a
    strategy that holds overnight the gap between yesterday's close and today's
    open was invisible. Live, that hid the entire day's P&L: the tracker booked
    ABT/MRK at -$43 (intraday) while close-to-close was +$252.

    Build a session whose overnight gap is large and whose intraday move is
    nearly nil; the recorded move must reflect the GAP, not ~0.
    """
    session = session_dates[60]
    prior = session_dates[59]

    closes = close_panel.copy()
    # Today's closes jump well away from yesterday's closes...
    closes.loc[session, "AAA"] = closes.loc[prior, "AAA"] * 1.05
    closes.loc[session, "BBB"] = closes.loc[prior, "BBB"]
    ohlc = pd.concat({"Open": closes * 1.0, "Close": closes}, axis=1)
    # ...and today's OPEN equals today's CLOSE, so the intraday move is exactly 0.
    ohlc.loc[session, ("Open", "AAA")] = closes.loc[session, "AAA"]
    ohlc.loc[session, ("Open", "BBB")] = closes.loc[session, "BBB"]

    path = tmp_path / "perf.csv"
    assert _record(path, closes, ohlc, session.strftime("%Y-%m-%d")) == 1
    row = pd.read_csv(path).iloc[0]

    # Under the OLD open->close convention this would have been ~0.0.
    assert abs(row["spread_move_pct"]) > 1.0, (
        "overnight gap was discarded: spread_move_pct is ~0 despite a 5% gap"
    )
