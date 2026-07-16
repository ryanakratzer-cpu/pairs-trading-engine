import numpy as np
import pytest
from statsmodels.tsa.stattools import coint

from screening.cointegration import compute_half_life
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
