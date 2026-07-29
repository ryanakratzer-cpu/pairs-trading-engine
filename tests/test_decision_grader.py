"""Decision-grading logic on hand-built signal paths and synthetic panels.

No network anywhere: the two lenses and the exit verdicts are exercised
against signal frames constructed bar by bar, so every verdict has an
analytically known right answer, and the end-to-end path runs on the
conftest fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.simulator import PairBacktestConfig
from reporting.decision_grader import (
    ENTRY_EVENTS,
    EXIT_JUSTIFIED,
    EXIT_PREMATURE,
    EXIT_TOO_EARLY,
    OUTCOME_LOSS,
    OUTCOME_PROFIT,
    THESIS_CONVERGED,
    THESIS_DIVERGED,
    grade_decisions,
    grade_signal_frame,
    summarize_decisions,
)
from signals.spread import SignalConfig

EXIT_Z = SignalConfig().exit_z


def _signal_frame(zscores: list[float], events: list[str], start: str = "2024-01-01") -> pd.DataFrame:
    """Hand-built equivalent of generate_signals' output — z path plus the
    labeled transition on each bar. Position is not read by the grader."""
    assert len(zscores) == len(events)
    index = pd.bdate_range(start, periods=len(zscores))
    return pd.DataFrame({"zscore": zscores, "event": events}, index=index)


def _trade_row(ticker_a: str, ticker_b: str, entry_date, exit_date, pnl: float, reason: str) -> dict:
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "position": -1,
        "entry_date": pd.Timestamp(entry_date),
        "exit_date": pd.Timestamp(exit_date),
        "holding_days": (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days,
        "pnl": pnl,
        "exit_reason": reason,
    }


def _decision_row(**overrides) -> dict:
    """A schema-complete decision row; overrides set the fields under test."""
    row = {
        "date": pd.Timestamp("2024-01-01"),
        "pair": "A/B",
        "ticker_a": "A",
        "ticker_b": "B",
        "event": "ENTER_SHORT_SPREAD",
        "zscore": 2.5,
        "direction": "SHORT_SPREAD",
        "taken": True,
        "exit_date": pd.Timestamp("2024-01-10"),
        "exit_event": "EXIT",
        "z_exit": 0.2,
        "thesis_move": -2.3,
        "thesis_verdict": THESIS_CONVERGED,
        "pnl": 100.0,
        "outcome_verdict": OUTCOME_PROFIT,
        "holding_days": 9,
        "forward_z": np.nan,
        "forward_move": np.nan,
        "exit_verdict": "NOT_GRADED",
        "lenses_agree": True,
    }
    row.update(overrides)
    return row


class TestThesisLens:
    def test_converging_entry_grades_correct(self):
        """Short-spread entry at z=+2.5 that decays to the exit band: the
        mean-reversion thesis paid off, so sign(z_entry)*(z_exit-z_entry) < 0."""
        frame = _signal_frame(
            [2.5, 2.0, 1.4, 0.8, 0.3],
            ["ENTER_SHORT_SPREAD", "HOLD", "HOLD", "HOLD", "EXIT"],
        )
        graded = grade_signal_frame("A", "B", frame)
        entry = graded[graded["event"].isin(ENTRY_EVENTS)].iloc[0]

        assert entry["thesis_verdict"] == THESIS_CONVERGED
        assert entry["thesis_move"] == pytest.approx(0.3 - 2.5)
        assert entry["z_exit"] == pytest.approx(0.3)
        assert entry["exit_event"] == "EXIT"

    def test_diverging_entry_grades_incorrect(self):
        """Same entry side, but the spread blows out to the stop instead:
        the prediction was simply wrong, independent of any P&L."""
        frame = _signal_frame(
            [2.5, 2.8, 3.2, 3.6, 4.0],
            ["ENTER_SHORT_SPREAD", "HOLD", "HOLD", "HOLD", "STOP_LOSS"],
        )
        graded = grade_signal_frame("A", "B", frame)
        entry = graded[graded["event"].isin(ENTRY_EVENTS)].iloc[0]

        assert entry["thesis_verdict"] == THESIS_DIVERGED
        assert entry["thesis_move"] == pytest.approx(4.0 - 2.5)

    def test_long_spread_entry_uses_the_sign_convention(self):
        """A long-spread entry sits at NEGATIVE z; converging means z RISES
        toward zero, and the sign flip has to make that a CONVERGED grade."""
        frame = _signal_frame(
            [-2.4, -1.5, -0.4],
            ["ENTER_LONG_SPREAD", "HOLD", "EXIT"],
        )
        graded = grade_signal_frame("A", "B", frame)
        entry = graded[graded["event"].isin(ENTRY_EVENTS)].iloc[0]

        assert entry["direction"] == "LONG_SPREAD"
        assert entry["thesis_verdict"] == THESIS_CONVERGED
        # sign(-2.4) * (-0.4 - -2.4) = -1 * 2.0
        assert entry["thesis_move"] == pytest.approx(-2.0)


class TestExitGrading:
    def test_stop_loss_followed_by_more_divergence_is_justified(self):
        """The stop fires at z=3.8 and the spread keeps blowing out — bailing
        out saved money, so the stop was the right call."""
        zscores = [2.5, 3.0, 3.8] + [4.2, 4.6, 5.0, 5.4, 5.8]
        events = ["ENTER_SHORT_SPREAD", "HOLD", "STOP_LOSS"] + ["NO_POSITION"] * 5
        graded = grade_signal_frame("A", "B", _signal_frame(zscores, events), forward_bars=5)
        stop = graded[graded["event"] == "STOP_LOSS"].iloc[0]

        assert stop["exit_verdict"] == EXIT_JUSTIFIED
        assert stop["forward_move"] > 0

    def test_stop_loss_followed_by_snapback_was_too_early(self):
        zscores = [2.5, 3.0, 3.8] + [3.0, 2.0, 1.0, 0.4, 0.1]
        events = ["ENTER_SHORT_SPREAD", "HOLD", "STOP_LOSS"] + ["NO_POSITION"] * 5
        graded = grade_signal_frame("A", "B", _signal_frame(zscores, events), forward_bars=5)
        stop = graded[graded["event"] == "STOP_LOSS"].iloc[0]

        assert stop["exit_verdict"] == EXIT_TOO_EARLY
        assert stop["forward_move"] < 0

    def test_time_exit_that_converged_afterwards_was_premature(self):
        """Abandoning the thesis right before it worked is the failure mode a
        time-based exit has, and the grade has to name it."""
        zscores = [2.5, 2.4, 2.3] + [1.8, 1.2, 0.6, 0.2, 0.1]
        events = ["ENTER_SHORT_SPREAD", "HOLD", "TIME_EXIT"] + ["NO_POSITION"] * 5
        graded = grade_signal_frame("A", "B", _signal_frame(zscores, events), forward_bars=5)
        time_exit = graded[graded["event"] == "TIME_EXIT"].iloc[0]

        assert time_exit["exit_verdict"] == EXIT_PREMATURE

    def test_clean_exit_inside_the_band_is_a_consistency_check(self):
        frame = _signal_frame([2.5, 1.2, 0.3], ["ENTER_SHORT_SPREAD", "HOLD", "EXIT"])
        graded = grade_signal_frame("A", "B", frame)
        clean = graded[graded["event"] == "EXIT"].iloc[0]

        assert clean["exit_verdict"] == "CONVERGED"

    def test_exit_far_outside_the_band_is_flagged_not_converged(self):
        """A long spread that rockets to +5 trips the EXIT branch (z >= -exit_z)
        even though |z| is nowhere near the band. The verdict must say so."""
        frame = _signal_frame([-2.1, 1.0, 5.2], ["ENTER_LONG_SPREAD", "HOLD", "EXIT"])
        graded = grade_signal_frame("A", "B", frame)
        clean = graded[graded["event"] == "EXIT"].iloc[0]

        assert clean["exit_verdict"] == "NOT_CONVERGED"


class TestThesisLensBlindSpots:
    """The raw thesis rate flatters the engine in two structural ways. Both are
    flagged per-row so the headline number can be discounted honestly."""

    def test_overshoot_through_the_mean_is_flagged(self):
        """Entry at z=-2.3, exit at z=+5.2: the signed move is negative, so the
        arithmetic says CONVERGED — but the spread blew out the far side."""
        frame = _signal_frame([-2.3, 1.0, 5.2], ["ENTER_LONG_SPREAD", "HOLD", "EXIT"])
        graded = grade_signal_frame("A", "B", frame)
        entry = graded[graded["event"].isin(ENTRY_EVENTS)].iloc[0]

        assert entry["thesis_verdict"] == THESIS_CONVERGED
        assert bool(entry["overshoot_exit"])

    def test_ordinary_convergence_is_not_flagged_as_overshoot(self):
        frame = _signal_frame([-2.3, -1.0, -0.3], ["ENTER_LONG_SPREAD", "HOLD", "EXIT"])
        graded = grade_signal_frame("A", "B", frame)
        entry = graded[graded["event"].isin(ENTRY_EVENTS)].iloc[0]

        assert entry["thesis_verdict"] == THESIS_CONVERGED
        assert not entry["overshoot_exit"]
        assert not entry["entry_beyond_stop"]

    def test_entry_already_past_the_stop_band_is_flagged(self):
        """The state machine checks entry before stop, so a z of -5.2 opens a
        position that is instantly eligible to be stopped out."""
        config = SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=3.0)
        frame = _signal_frame([-5.2, -3.6], ["ENTER_LONG_SPREAD", "STOP_LOSS"])
        graded = grade_signal_frame("A", "B", frame, signal_config=config)
        entry = graded[graded["event"].isin(ENTRY_EVENTS)].iloc[0]

        assert bool(entry["entry_beyond_stop"])
        # Arithmetically it "converged" (-5.2 -> -3.6 moved toward zero)...
        assert entry["thesis_verdict"] == THESIS_CONVERGED
        # ...but the discounted rate must refuse to count it.
        assert summarize_decisions(graded)["entry_hit_rate_thesis"] == 1.0
        assert summarize_decisions(graded)["entry_hit_rate_thesis_strict"] == 0.0

    def test_discounted_rate_counts_only_clean_convergence(self):
        config = SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=3.0)
        clean = _signal_frame([-2.3, -1.0, -0.3], ["ENTER_LONG_SPREAD", "HOLD", "EXIT"],
                              start="2024-01-01")
        overshoot = _signal_frame([-2.3, 1.0, 5.2], ["ENTER_LONG_SPREAD", "HOLD", "EXIT"],
                                  start="2024-03-01")
        decisions = pd.concat(
            [
                grade_signal_frame("A", "B", clean, signal_config=config),
                grade_signal_frame("A", "B", overshoot, signal_config=config),
            ],
            ignore_index=True,
        )
        summary = summarize_decisions(decisions)

        assert summary["n_entries_graded_thesis"] == 2
        assert summary["entry_hit_rate_thesis"] == 1.0  # both "converged"
        assert summary["n_overshoot_exits"] == 1
        assert summary["entry_hit_rate_thesis_strict"] == pytest.approx(0.5)


class TestLensDisagreement:
    def test_converged_but_lost_money_is_flagged_as_disagreement(self):
        """The point of running both lenses: the spread reverted exactly as
        predicted, and the round trip still lost after costs."""
        frame = _signal_frame(
            [2.5, 1.1, 0.3],
            ["ENTER_SHORT_SPREAD", "HOLD", "EXIT"],
        )
        trade_log = pd.DataFrame(
            [_trade_row("A", "B", frame.index[0], frame.index[2], pnl=-18.0, reason="EXIT")]
        )
        graded = grade_signal_frame("A", "B", frame, trade_log=trade_log)
        entry = graded[graded["event"].isin(ENTRY_EVENTS)].iloc[0]

        assert entry["taken"]
        assert entry["thesis_verdict"] == THESIS_CONVERGED
        assert entry["outcome_verdict"] == OUTCOME_LOSS
        assert entry["lenses_agree"] is False

        summary = summarize_decisions(graded)
        assert summary["entry_hit_rate_thesis"] == 1.0
        assert summary["entry_hit_rate_outcome"] == 0.0
        assert summary["n_lens_disagreements"] == 1

    def test_agreement_when_both_lenses_say_right(self):
        frame = _signal_frame([2.5, 1.1, 0.3], ["ENTER_SHORT_SPREAD", "HOLD", "EXIT"])
        trade_log = pd.DataFrame(
            [_trade_row("A", "B", frame.index[0], frame.index[2], pnl=240.0, reason="EXIT")]
        )
        graded = grade_signal_frame("A", "B", frame, trade_log=trade_log)
        entry = graded[graded["event"].isin(ENTRY_EVENTS)].iloc[0]

        assert entry["outcome_verdict"] == OUTCOME_PROFIT
        assert entry["lenses_agree"] is True

    def test_unfilled_entry_has_no_outcome_but_still_has_a_thesis(self):
        """The portfolio caps concurrent positions, so a signalled entry is not
        always a filled trade. It still carries a prediction worth grading."""
        frame = _signal_frame([2.5, 1.1, 0.3], ["ENTER_SHORT_SPREAD", "HOLD", "EXIT"])
        graded = grade_signal_frame("A", "B", frame, trade_log=None)
        entry = graded[graded["event"].isin(ENTRY_EVENTS)].iloc[0]

        assert not entry["taken"]
        assert pd.isna(entry["pnl"])
        assert entry["outcome_verdict"] == "NOT_GRADED"
        assert entry["thesis_verdict"] == THESIS_CONVERGED

        summary = summarize_decisions(graded)
        assert summary["n_entries"] == 1
        assert summary["n_entries_taken"] == 0
        assert summary["n_entries_graded_outcome"] == 0
        assert summary["entry_hit_rate_outcome"] is None
        assert summary["entry_hit_rate_thesis"] == 1.0


class TestSummaryMath:
    def test_hit_rates_on_a_hand_built_decision_set(self):
        """Four entries: 3 converged / 1 diverged on the thesis lens, 2 profits
        / 2 losses on the outcome lens. Every rate is checkable by hand."""
        decisions = pd.DataFrame(
            [
                _decision_row(thesis_verdict=THESIS_CONVERGED, outcome_verdict=OUTCOME_PROFIT,
                              pnl=300.0, lenses_agree=True),
                _decision_row(thesis_verdict=THESIS_CONVERGED, outcome_verdict=OUTCOME_PROFIT,
                              pnl=100.0, lenses_agree=True),
                _decision_row(thesis_verdict=THESIS_CONVERGED, outcome_verdict=OUTCOME_LOSS,
                              pnl=-40.0, lenses_agree=False),
                _decision_row(thesis_verdict=THESIS_DIVERGED, outcome_verdict=OUTCOME_LOSS,
                              pnl=-360.0, lenses_agree=True),
            ]
        )
        summary = summarize_decisions(decisions)

        assert summary["n_entries"] == 4
        assert summary["n_entries_taken"] == 4
        assert summary["entry_hit_rate_thesis"] == pytest.approx(0.75)
        assert summary["entry_hit_rate_outcome"] == pytest.approx(0.5)
        assert summary["n_lens_comparable"] == 4
        assert summary["n_lens_disagreements"] == 1
        assert summary["lens_agreement_rate"] == pytest.approx(0.75)
        # (300 + 100 - 40) / 3 vs the single diverged entry
        assert summary["avg_pnl_correct_thesis"] == pytest.approx(120.0)
        assert summary["avg_pnl_incorrect_thesis"] == pytest.approx(-360.0)
        assert summary["avg_pnl_winner"] == pytest.approx(200.0)
        assert summary["avg_pnl_loser"] == pytest.approx(-200.0)
        assert summary["total_pnl"] == pytest.approx(0.0)

    def test_stop_loss_justification_rate(self):
        """Two stops justified, one too early -> 2/3."""
        decisions = pd.DataFrame(
            [
                _decision_row(event="STOP_LOSS", exit_verdict=EXIT_JUSTIFIED),
                _decision_row(event="STOP_LOSS", exit_verdict=EXIT_JUSTIFIED),
                _decision_row(event="STOP_LOSS", exit_verdict=EXIT_TOO_EARLY),
            ]
        )
        summary = summarize_decisions(decisions)

        assert summary["n_stop_losses"] == 3
        assert summary["stop_loss_justified_rate"] == pytest.approx(2 / 3)
        # Stop rows are exits, not entries — they must not pollute entry counts.
        assert summary["n_entries"] == 0
        assert summary["entry_hit_rate_thesis"] is None

    def test_empty_decision_set_returns_none_rates_not_a_crash(self):
        summary = summarize_decisions(pd.DataFrame())

        assert summary["n_decisions"] == 0
        assert summary["entry_hit_rate_outcome"] is None
        assert summary["entry_hit_rate_thesis"] is None
        assert summary["stop_loss_justified_rate"] is None


class TestEndToEnd:
    def test_grades_a_synthetic_cointegrated_panel_with_no_network(self, cointegrated_pair_prices):
        """Full path: backtest a known mean-reverting pair, then grade every
        decision it made. The pair is OU by construction, so the thesis lens
        should be right more often than not."""
        price_a, price_b, _hedge_ratio, _theta = cointegrated_pair_prices
        panel = pd.DataFrame({"A": price_a, "B": price_b})

        result = grade_decisions(panel, [("A", "B")], config=PairBacktestConfig.moderate())

        assert not result.decisions.empty, "an OU spread should cross the entry band at least once"
        entries = result.entries
        assert len(entries) > 0
        assert set(result.decisions["pair"]) == {"A/B"}
        assert result.summary["n_entries"] == len(entries)
        assert 0.0 <= result.summary["entry_hit_rate_thesis"] <= 1.0
        assert result.summary["entry_hit_rate_thesis"] >= 0.5
        assert not result.per_pair.empty

    def test_every_trade_log_entry_is_matched_to_a_graded_decision(self, sector_universe_fixture):
        """The outcome lens is only trustworthy if entries link to the right
        trades: each filled entry must carry that trade's P&L."""
        panel, (ticker_a, ticker_b) = sector_universe_fixture
        result = grade_decisions(panel, [(ticker_a, ticker_b)])

        taken = result.entries[result.entries["taken"]]
        assert len(taken) == len(result.trade_log)
        if not result.trade_log.empty:
            logged = sorted(result.trade_log["pnl"].round(6))
            graded = sorted(taken["pnl"].round(6))
            assert logged == graded

    def test_missing_pair_yields_an_empty_but_valid_result(self, cointegrated_pair_prices):
        price_a, price_b, _hedge_ratio, _theta = cointegrated_pair_prices
        panel = pd.DataFrame({"A": price_a, "B": price_b})

        result = grade_decisions(panel, [("NOPE", "ALSO_NOPE")])

        assert result.decisions.empty
        assert result.summary["n_decisions"] == 0
        assert result.per_pair.empty
