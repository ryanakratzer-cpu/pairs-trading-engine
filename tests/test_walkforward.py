"""Tests for backtest.walkforward (window generation + three studies),
screening.events (event-exclusion masks), the backtester's external entry
gate, and gate-aware OU expected returns."""

import numpy as np
import pandas as pd
import pytest

from backtest.simulator import PairBacktestConfig, PairBacktester
from backtest.walkforward import (
    WalkForwardConfig,
    allocation_study,
    default_parameter_grid,
    generate_windows,
    pair_survival_study,
    parameter_study,
    strategy_returns,
)
from montecarlo.simulator import OUFit
from portfolio.optimizer import ou_expected_annual_return
from screening.events import FOMC_DECISION_DATES, event_exclusion_mask
from signals.spread import SignalConfig

SMALL_WF = WalkForwardConfig(formation_days=150, holdout_days=50, step_days=50)


# ----------------------------------------------------------------- windows


def test_generate_windows_counts_and_disjointness(business_day_index):
    windows = generate_windows(business_day_index, SMALL_WF)
    # 500 bars, 200 per window, step 50 -> (500-200)/50 + 1 = 7 windows.
    assert len(windows) == 7
    for formation, holdout in windows:
        assert len(formation) == 150
        assert len(holdout) == 50
        # Holdout strictly after formation - no leakage.
        assert formation[-1] < holdout[0]


def test_generate_windows_too_short_index_gives_none():
    short = pd.bdate_range("2024-01-01", periods=100)
    assert generate_windows(short, SMALL_WF) == []


# ------------------------------------------------------------ pair survival


# Survival tests need POWER: ADF on a 50-bar holdout almost never rejects,
# and the fixture pair's true half-life (~5.8d) sits on the 5-day band edge,
# so the production band would drop it on estimation noise. Longer windows +
# a wider half-life band keep these tests about the study's logic, not about
# ADF small-sample power.
SURVIVAL_WF = WalkForwardConfig(
    formation_days=200, holdout_days=150, step_days=75,
    min_half_life_days=2.0, max_half_life_days=60.0,
)


def test_survival_study_synthetic_pair_survives(sector_universe_fixture):
    panel, injected = sector_universe_fixture
    result = pair_survival_study(panel, [injected], SURVIVAL_WF)
    assert not result.empty
    # A stationary-by-construction OU spread should survive most holdouts
    # once the holdout is long enough for ADF to have power.
    assert result["survived"].mean() >= 0.5
    assert set(result.columns) >= {
        "window_end", "formation_pvalue", "formation_half_life", "holdout_pvalue", "survived",
    }


def test_survival_study_ranks_real_pair_above_noise(sector_universe_fixture):
    """The study's documented use is COMPARATIVE: a true cointegrated pair
    must pass formation more often and show a stronger (lower) median holdout
    p-value than an independent-random-walk pair. (An absolute zero-survival
    claim for noise is too strong: two walks can wander together for 500 bars
    by chance — that's precisely the false-discovery problem the FDR
    correction in the main screen exists for.)"""
    panel, injected = sector_universe_fixture
    real = pair_survival_study(panel, [injected], SURVIVAL_WF)
    noise = pair_survival_study(panel, [("CCC", "DDD")], SURVIVAL_WF)
    assert len(real) >= len(noise)
    if not noise.empty:
        assert real["holdout_pvalue"].median() < noise["holdout_pvalue"].median()


def test_survival_study_missing_ticker_skipped(sector_universe_fixture):
    panel, _ = sector_universe_fixture
    assert pair_survival_study(panel, [("AAA", "ZZZ")], SURVIVAL_WF).empty


# ------------------------------------------------------------ parameter WF


def test_strategy_returns_zero_without_signals():
    dates = pd.bdate_range("2024-01-02", periods=100)
    flat = pd.Series(0.0, index=dates)
    returns = strategy_returns(flat, SignalConfig())
    assert (returns == 0).all()


def test_strategy_returns_costs_charged_on_turnover(known_half_life_ou_series):
    series, _ = known_half_life_ou_series
    gross = strategy_returns(series, SignalConfig(), cost_bps_per_side=0.0)
    net = strategy_returns(series, SignalConfig(), cost_bps_per_side=10.0)
    assert net.sum() < gross.sum()  # costs must bite once there are trades
    assert (gross - net >= -1e-15).all()


def test_parameter_study_shape_and_grid_membership(cointegrated_pair_prices):
    price_a, price_b, _, _ = cointegrated_pair_prices
    result = parameter_study(price_a, price_b, SMALL_WF)
    assert len(result) == 7
    grid_entries = {c.entry_z for c in default_parameter_grid()}
    assert set(result["chosen_entry_z"]).issubset(grid_entries)
    assert result["holdout_sharpe_chosen"].notna().all()
    assert result["holdout_sharpe_default"].notna().all()


# ------------------------------------------------------------ allocation WF


def test_allocation_study_schemes_and_determinism():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2023-01-02", periods=400)
    panel = pd.DataFrame(
        rng.normal(0.0002, 0.008, size=(400, 4)), index=dates, columns=list("ABCD")
    )
    result = allocation_study(panel, SMALL_WF)
    assert set(result["scheme"]) == {"equal_weight", "tangency", "min_variance"}
    # Every scheme measured on every window: count divisible by 3.
    assert len(result) % 3 == 0
    again = allocation_study(panel, SMALL_WF)
    pd.testing.assert_frame_equal(result, again)


def test_allocation_study_empty_on_short_panel():
    dates = pd.bdate_range("2024-01-01", periods=50)
    panel = pd.DataFrame(np.zeros((50, 2)), index=dates, columns=["A", "B"])
    assert allocation_study(panel, SMALL_WF).empty


# ------------------------------------------------------------- event masks


def test_event_mask_blocks_window_around_fomc():
    index = pd.bdate_range("2026-01-20", "2026-02-05")
    mask = event_exclusion_mask(index, days_before=1, days_after=1)
    # 2026-01-28 FOMC decision: 27th, 28th, 29th blocked.
    for day in ("2026-01-27", "2026-01-28", "2026-01-29"):
        assert not mask[pd.Timestamp(day)]
    assert mask[pd.Timestamp("2026-01-26")]
    assert mask[pd.Timestamp("2026-01-30")]


def test_event_mask_weekend_reachback():
    # 2024-11-05 election is a Tuesday; days_before=2 must block Monday AND
    # reach back across the weekend boundary correctly (calendar days).
    index = pd.bdate_range("2024-10-28", "2024-11-08")
    mask = event_exclusion_mask(index, event_dates=["2024-11-05"], days_before=2, days_after=1)
    assert not mask[pd.Timestamp("2024-11-04")]  # Monday
    assert not mask[pd.Timestamp("2024-11-06")]
    assert mask[pd.Timestamp("2024-11-01")]  # Friday, 4 calendar days out
    assert mask[pd.Timestamp("2024-11-07")]


def test_event_mask_defaults_cover_all_fomc_dates():
    index = pd.bdate_range("2024-01-01", "2026-12-31")
    mask = event_exclusion_mask(index)
    for raw in FOMC_DECISION_DATES:
        day = pd.Timestamp(raw)
        if day in mask.index:
            assert not mask[day], f"{raw} should be blocked"


def test_event_mask_empty_and_eventless_index():
    assert event_exclusion_mask(pd.DatetimeIndex([])).empty
    quiet = pd.bdate_range("2025-02-03", periods=5)  # no events that week
    assert event_exclusion_mask(quiet).all()


# ------------------------------------------------- backtester entry gating


def test_backtester_entries_allowed_false_blocks_all_trades(sector_universe_fixture):
    panel, injected = sector_universe_fixture
    config = PairBacktestConfig(recheck_window_days=100, recheck_freq_days=30)
    never = pd.Series(False, index=panel.index)
    gated = PairBacktester(config).run(panel, [injected], entries_allowed=never)
    assert gated["trade_log"].empty
    # Equity stays exactly at initial capital: no entries, no costs.
    assert (gated["equity_curve"] == config.initial_capital).all()


def test_backtester_entries_allowed_true_matches_ungated(sector_universe_fixture):
    panel, injected = sector_universe_fixture
    config = PairBacktestConfig(recheck_window_days=100, recheck_freq_days=30)
    always = pd.Series(True, index=panel.index)
    gated = PairBacktester(config).run(panel, [injected], entries_allowed=always)
    ungated = PairBacktester(config).run(panel, [injected])
    pd.testing.assert_series_equal(gated["equity_curve"], ungated["equity_curve"])


def test_backtester_prepared_exposes_tradeable_mask(sector_universe_fixture):
    panel, injected = sector_universe_fixture
    config = PairBacktestConfig(recheck_window_days=100, recheck_freq_days=30)
    result = PairBacktester(config).run(panel, [injected])
    tradeable = result["per_pair"][injected]["tradeable"]
    assert tradeable.dtype == bool
    assert 0.0 < tradeable.mean() <= 1.0  # warm-up alone guarantees some False


# --------------------------------------------------------- gate-aware OU mu


def test_ou_expected_return_scales_with_tradeable_fraction():
    fit = OUFit(theta=0.1, mu=0.0, sigma=0.02, half_life_days=6.9, n_obs=400)
    full = ou_expected_annual_return(fit, SignalConfig(), tradeable_fraction=1.0)
    half = ou_expected_annual_return(fit, SignalConfig(), tradeable_fraction=0.5)
    zero = ou_expected_annual_return(fit, SignalConfig(), tradeable_fraction=0.0)
    assert half == pytest.approx(full / 2)
    assert zero == 0.0


def test_ou_expected_return_rejects_invalid_fraction():
    fit = OUFit(theta=0.1, mu=0.0, sigma=0.02, half_life_days=6.9, n_obs=400)
    with pytest.raises(ValueError, match="tradeable_fraction"):
        ou_expected_annual_return(fit, SignalConfig(), tradeable_fraction=1.5)
