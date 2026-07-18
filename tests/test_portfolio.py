"""Tests for portfolio.optimizer: Ledoit-Wolf shrinkage, OU expected returns,
tangency / min-variance / frontier solutions and their invariants."""

import numpy as np
import pandas as pd
import pytest

from montecarlo.simulator import OUFit, fit_ou
from portfolio.optimizer import (
    DEFAULT_MAX_WEIGHT,
    PortfolioPoint,
    annualize_inputs,
    efficient_frontier,
    ledoit_wolf_cov,
    max_sharpe_weights,
    min_variance_weights,
    ou_expected_annual_return,
    random_portfolios,
)
from signals.spread import SignalConfig


@pytest.fixture
def three_asset_inputs():
    """Hand-built inputs with a known ordering: asset 0 high return / high vol,
    asset 1 moderate, asset 2 low return / low vol, mild correlations."""
    mu = np.array([0.12, 0.08, 0.03])
    vols = np.array([0.20, 0.12, 0.05])
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    cov = np.outer(vols, vols) * corr
    return mu, cov


@pytest.fixture
def random_returns_panel():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=300)
    data = rng.normal(0.0003, 0.01, size=(300, 4))
    return pd.DataFrame(data, index=dates, columns=["W", "X", "Y", "Z"])


# ---------------------------------------------------------------- Ledoit-Wolf


def test_ledoit_wolf_shape_symmetry_and_intensity(random_returns_panel):
    cov, intensity = ledoit_wolf_cov(random_returns_panel)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)
    assert 0.0 <= intensity <= 1.0
    # Shrunk covariance must be positive definite (that's the point of it).
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_ledoit_wolf_shrinks_toward_identity_scale(random_returns_panel):
    cov, intensity = ledoit_wolf_cov(random_returns_panel)
    x = random_returns_panel.to_numpy()
    x = x - x.mean(axis=0)
    sample = (x.T @ x) / len(x)
    # Off-diagonals move toward 0 (the identity target), never away.
    off = ~np.eye(4, dtype=bool)
    assert np.all(np.abs(cov[off]) <= np.abs(sample[off]) + 1e-15)
    # Total variance (trace) is preserved by the identity-scale target.
    assert np.trace(cov) == pytest.approx(np.trace(sample), rel=1e-9)


def test_ledoit_wolf_rejects_single_observation():
    single = pd.DataFrame([[0.01, 0.02]], columns=["A", "B"])
    with pytest.raises(ValueError, match="at least 2"):
        ledoit_wolf_cov(single)


def test_ledoit_wolf_handles_flat_zero_variance_column():
    dates = pd.bdate_range("2023-01-02", periods=100)
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(
        {"live": rng.normal(0, 0.01, 100), "flat": np.zeros(100)}, index=dates
    )
    cov, _ = ledoit_wolf_cov(panel)
    # Shrinkage lends the dead column some variance from the target - the
    # matrix stays invertible for the optimizer.
    assert np.all(np.linalg.eigvalsh(cov) > 0)


# ------------------------------------------------------- OU expected returns


def test_ou_expected_return_positive_for_reverting_fit(known_half_life_ou_series):
    series, _theta = known_half_life_ou_series
    result = ou_expected_annual_return(fit_ou(series), SignalConfig())
    assert result > 0


def test_ou_expected_return_zero_when_no_reversion():
    fit = OUFit(theta=-0.01, mu=0.0, sigma=0.02, half_life_days=None, n_obs=100)
    assert ou_expected_annual_return(fit, SignalConfig()) == 0.0


def test_ou_expected_return_decreases_with_half_life():
    """Slower reversion = fewer cycles per year = lower expected return,
    holding the spread's scale fixed."""
    fast = OUFit(theta=0.14, mu=0.0, sigma=0.02, half_life_days=5.0, n_obs=500)
    slow = OUFit(theta=0.028, mu=0.0, sigma=0.02, half_life_days=25.0, n_obs=500)
    config = SignalConfig()
    # Fix stationary_std by comparing at equal sigma via the property's inputs:
    # theta differs, so compare per-cycle-count effect using same-std fits.
    fast_ret = ou_expected_annual_return(fast, config)
    slow_ret = ou_expected_annual_return(slow, config)
    # fast has smaller stationary_std at equal sigma, so normalize per trade:
    # instead assert on trades/year scaling directly via ratio of half-lives.
    assert fast_ret != slow_ret  # sanity: they differ
    # With the same stationary_std, return scales as 1/half_life. Construct that:
    sigma_slow_matched = fast.sigma * np.sqrt(
        (1 - (1 - slow.theta) ** 2) / (1 - (1 - fast.theta) ** 2)
    )
    slow_matched = OUFit(theta=0.028, mu=0.0, sigma=sigma_slow_matched, half_life_days=25.0, n_obs=500)
    assert ou_expected_annual_return(slow_matched, config) < fast_ret


# ------------------------------------------------------------- optimization


def test_weights_are_valid_and_capped(three_asset_inputs):
    mu, cov = three_asset_inputs
    for weights in (max_sharpe_weights(mu, cov), min_variance_weights(cov)):
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= -1e-12)
        assert np.all(weights <= DEFAULT_MAX_WEIGHT + 1e-9)


def test_tangency_beats_alternatives_on_sharpe(three_asset_inputs):
    mu, cov = three_asset_inputs
    tangency = PortfolioPoint.from_weights(max_sharpe_weights(mu, cov), mu, cov)
    minvar = PortfolioPoint.from_weights(min_variance_weights(cov), mu, cov)
    equal = PortfolioPoint.from_weights(np.full(3, 1 / 3), mu, cov)
    assert tangency.sharpe >= minvar.sharpe - 1e-9
    assert tangency.sharpe >= equal.sharpe - 1e-9


def test_min_variance_has_lowest_volatility(three_asset_inputs):
    mu, cov = three_asset_inputs
    minvar = PortfolioPoint.from_weights(min_variance_weights(cov), mu, cov)
    for point in random_portfolios(mu, cov, n_portfolios=500, seed=11):
        assert minvar.volatility <= point.volatility + 1e-9


def test_frontier_is_monotone_and_contains_no_dominated_points(three_asset_inputs):
    mu, cov = three_asset_inputs
    frontier = efficient_frontier(mu, cov, n_points=25)
    assert len(frontier) >= 10
    rets = [p.expected_return for p in frontier]
    vols = [p.volatility for p in frontier]
    # Returns rise along the frontier; volatility never decreases with return
    # on the efficient branch.
    assert all(r2 >= r1 - 1e-9 for r1, r2 in zip(rets, rets[1:]))
    assert all(v2 >= v1 - 1e-6 for v1, v2 in zip(vols, vols[1:]))


def test_weight_cap_binds_when_one_asset_dominates():
    """One asset with a huge Sharpe would take 100% unconstrained - the cap
    must hold it at max_weight instead."""
    mu = np.array([0.50, 0.02, 0.02])
    cov = np.diag([0.01, 0.04, 0.04])
    weights = max_sharpe_weights(mu, cov, max_weight=0.40)
    assert weights[0] == pytest.approx(0.40, abs=1e-6)


def test_cap_below_one_over_n_is_lifted_to_feasibility():
    mu = np.array([0.05, 0.06])
    cov = np.diag([0.02, 0.03])
    # 0.3 cap with 2 assets is infeasible (sum <= 0.6 < 1); solver must lift it.
    weights = max_sharpe_weights(mu, cov, max_weight=0.30)
    assert weights.sum() == pytest.approx(1.0)


def test_two_asset_frontier_works():
    mu = np.array([0.04, 0.10])
    cov = np.array([[0.01, 0.002], [0.002, 0.05]])
    frontier = efficient_frontier(mu, cov, n_points=15, max_weight=1.0)
    assert len(frontier) >= 5


def test_random_portfolios_deterministic_with_seed(three_asset_inputs):
    mu, cov = three_asset_inputs
    a = random_portfolios(mu, cov, n_portfolios=200, seed=9)
    b = random_portfolios(mu, cov, n_portfolios=200, seed=9)
    assert len(a) == len(b)
    assert np.allclose(a[0].weights, b[0].weights)


def test_annualize_inputs_scales_by_252(random_returns_panel):
    mu_hist, cov, intensity = annualize_inputs(random_returns_panel)
    cov_daily, _ = ledoit_wolf_cov(random_returns_panel)
    assert np.allclose(cov, cov_daily * 252)
    assert np.allclose(mu_hist, random_returns_panel.mean().to_numpy() * 252)
    assert 0.0 <= intensity <= 1.0


def test_portfolio_point_sharpe_zero_when_no_volatility():
    point = PortfolioPoint.from_weights(
        np.array([1.0]), np.array([0.05]), np.array([[0.0]])
    )
    assert point.sharpe == 0.0
