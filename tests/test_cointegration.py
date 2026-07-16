import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from screening.cointegration import _benjamini_hochberg, compute_half_life, screen_universe
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
