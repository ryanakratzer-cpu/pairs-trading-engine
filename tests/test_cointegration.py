import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from screening.cointegration import _benjamini_hochberg, compute_half_life, screen_universe, validate_out_of_sample
from screening.cointegration import test_pair_cointegration as engle_granger_test


def test_detects_cointegrated_pair(cointegrated_pair_prices):
    price_a, price_b, hedge_ratio_true, _theta = cointegrated_pair_prices
    result = engle_granger_test(price_a, price_b, ticker_a="A", ticker_b="B")

    assert result.is_cointegrated
    assert result.adf_pvalue < 0.05
    assert result.hedge_ratio == pytest.approx(hedge_ratio_true, abs=0.1)


def test_rejects_non_cointegrated_pair(non_cointegrated_pair_prices):
    price_a, price_b = non_cointegrated_pair_prices
    result = engle_granger_test(price_a, price_b, ticker_a="A", ticker_b="B")

    assert not result.is_cointegrated


def test_half_life_recovers_known_theta(known_half_life_ou_series):
    series, theta = known_half_life_ou_series
    half_life = compute_half_life(series)
    expected = np.log(2) / theta

    assert half_life == pytest.approx(expected, rel=0.4)


def test_cointegrated_pair_half_life_in_plausible_band(cointegrated_pair_prices):
    price_a, price_b, _hedge_ratio_true, theta = cointegrated_pair_prices
    result = engle_granger_test(price_a, price_b, ticker_a="A", ticker_b="B")
    expected_half_life = np.log(2) / theta

    assert result.half_life_days is not None
    assert result.half_life_days == pytest.approx(expected_half_life, rel=0.6)


def test_agrees_with_statsmodels_coint(cointegrated_pair_prices):
    price_a, price_b, _hedge_ratio_true, _theta = cointegrated_pair_prices
    log_a, log_b = np.log(price_a), np.log(price_b)
    _stat, pvalue, _crit = coint(log_a, log_b)
    result = engle_granger_test(price_a, price_b)

    assert (pvalue < 0.05) == result.is_cointegrated


def test_benjamini_hochberg_flags_only_pvalues_surviving_correction():
    # Standard textbook example: at fdr_level=0.05 over n=8 hypotheses, only
    # the two smallest p-values (rank 1: 0.001, rank 2: 0.01) survive — rank 3's
    # p=0.02 exceeds its threshold (3/8)*0.05=0.01875, and no larger rank passes.
    pvalues = pd.Series([0.001, 0.01, 0.02, 0.03, 0.04, 0.06, 0.5, 0.7])
    significant = _benjamini_hochberg(pvalues, fdr_level=0.05)

    assert significant.tolist() == [True, True, False, False, False, False, False, False]


def test_benjamini_hochberg_flags_nothing_when_all_pvalues_fail():
    pvalues = pd.Series([0.2, 0.3, 0.4, 0.5])
    significant = _benjamini_hochberg(pvalues, fdr_level=0.05)

    assert not significant.any()


def test_screen_universe_multiple_testing_correction_is_opt_in_and_never_loosens_tradeable(
    sector_universe_fixture,
):
    panel, (ticker_a, ticker_b) = sector_universe_fixture
    pairs = [(ticker_a, ticker_b), ("CCC", "DDD"), ("AAA", "CCC"), ("BBB", "DDD")]

    uncorrected = screen_universe(panel, pairs, min_half_life_days=1.0, max_half_life_days=60.0)
    corrected = screen_universe(
        panel,
        pairs,
        min_half_life_days=1.0,
        max_half_life_days=60.0,
        apply_multiple_testing_correction=True,
    )

    assert uncorrected["bh_significant"].isna().all()  # not computed unless opted in
    assert corrected["bh_significant"].dtype == bool
    # BH correction can only remove tradeable pairs relative to the uncorrected
    # screen, never add ones that weren't already tradeable pre-correction.
    tradeable_pairs_corrected = set(zip(corrected.loc[corrected["tradeable"], "ticker_a"], corrected.loc[corrected["tradeable"], "ticker_b"]))
    tradeable_pairs_uncorrected = set(zip(uncorrected.loc[uncorrected["tradeable"], "ticker_a"], uncorrected.loc[uncorrected["tradeable"], "ticker_b"]))
    assert tradeable_pairs_corrected <= tradeable_pairs_uncorrected


def test_out_of_sample_validates_a_persistent_relationship():
    # Stronger/longer than the shared cointegrated_pair_prices fixture: applying
    # the formation-fitted hedge ratio (not re-estimated) to a held-out window
    # is a stricter test, so this needs enough signal to stay significant there
    # too, not just over the full sample.
    dates = pd.bdate_range("2023-01-02", periods=600)
    n = len(dates)
    rng_b = np.random.default_rng(31)
    rng_ou = np.random.default_rng(32)

    hedge_ratio_true = 0.75
    theta = 0.2
    log_b = np.cumsum(0.01 * rng_b.standard_normal(n)) + np.log(80)
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = ou[t - 1] + theta * (0.0 - ou[t - 1]) + 0.015 * rng_ou.standard_normal()
    log_a = hedge_ratio_true * log_b + ou

    price_a = pd.Series(np.exp(log_a), index=dates, name="A")
    price_b = pd.Series(np.exp(log_b), index=dates, name="B")

    result = validate_out_of_sample(price_a, price_b, ticker_a="A", ticker_b="B")

    assert result.formation_is_cointegrated
    assert result.validation_is_stationary
    assert result.out_of_sample_validated
    assert result.formation_hedge_ratio == pytest.approx(hedge_ratio_true, abs=0.15)


def test_out_of_sample_rejects_a_relationship_that_breaks_down_after_formation():
    # Cointegrated for the formation window, then permanently drifts (no mean
    # reversion) for the entire validation window — a pair that looks like a
    # great in-sample find but would never have been tradeable going forward.
    dates = pd.bdate_range("2023-01-02", periods=500)
    n = len(dates)
    split_idx = int(n * 0.7)
    rng = np.random.default_rng(21)

    log_b = np.cumsum(0.01 * rng.standard_normal(n)) + np.log(60)
    ou = np.zeros(n)
    for t in range(1, split_idx):
        ou[t] = ou[t - 1] + 0.15 * (0.0 - ou[t - 1]) + 0.02 * rng.standard_normal()
    for t in range(split_idx, n):
        ou[t] = ou[t - 1] + 0.02 * rng.standard_normal() + 0.015  # permanent drift, no reversion

    log_a = 0.9 * log_b + ou
    price_a = pd.Series(np.exp(log_a), index=dates, name="A")
    price_b = pd.Series(np.exp(log_b), index=dates, name="B")

    result = validate_out_of_sample(price_a, price_b, ticker_a="A", ticker_b="B", formation_fraction=0.7)

    assert result.formation_is_cointegrated
    assert not result.validation_is_stationary
    assert not result.out_of_sample_validated


def test_out_of_sample_short_circuits_on_failed_formation(non_cointegrated_pair_prices):
    price_a, price_b = non_cointegrated_pair_prices
    result = validate_out_of_sample(price_a, price_b, ticker_a="A", ticker_b="B")

    assert not result.formation_is_cointegrated
    assert not result.out_of_sample_validated


def test_out_of_sample_raises_on_insufficient_data():
    dates = pd.bdate_range("2023-01-02", periods=40)
    price_a = pd.Series(np.linspace(100, 110, 40), index=dates)
    price_b = pd.Series(np.linspace(50, 55, 40), index=dates)

    with pytest.raises(ValueError):
        validate_out_of_sample(price_a, price_b, formation_fraction=0.7)


def test_screen_universe_out_of_sample_validation_is_opt_in_and_never_loosens_tradeable(
    sector_universe_fixture,
):
    panel, (ticker_a, ticker_b) = sector_universe_fixture
    pairs = [(ticker_a, ticker_b), ("CCC", "DDD"), ("AAA", "CCC"), ("BBB", "DDD")]

    without_oos = screen_universe(panel, pairs, min_half_life_days=1.0, max_half_life_days=60.0)
    with_oos = screen_universe(
        panel,
        pairs,
        min_half_life_days=1.0,
        max_half_life_days=60.0,
        require_out_of_sample_validation=True,
    )

    assert without_oos["oos_validated"].isna().all()  # not computed unless opted in
    assert with_oos["oos_validated"].isin([True, False]).all()
    tradeable_with = set(zip(with_oos.loc[with_oos["tradeable"], "ticker_a"], with_oos.loc[with_oos["tradeable"], "ticker_b"]))
    tradeable_without = set(zip(without_oos.loc[without_oos["tradeable"], "ticker_a"], without_oos.loc[without_oos["tradeable"], "ticker_b"]))
    assert tradeable_with <= tradeable_without
