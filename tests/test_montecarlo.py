"""Monte Carlo OU simulator and interactive-plot tests. Deterministic, no network."""

import numpy as np
import pandas as pd
import pytest

from montecarlo.simulator import fit_ou, simulate_spread_paths, simulate_strategy_pnl, summarize_pnl
from screening.cointegration import compute_half_life
from signals.spread import SignalConfig
from visualization.interactive import (
    plot_interactive_equity,
    plot_interactive_spread_zscore,
    plot_monte_carlo_paths,
    plot_pnl_distribution,
)
from visualization.plots import plot_monte_carlo_paths_png

MIN_HTML_BYTES = 10_000  # a real chart with data + layout is well above this; an empty shell isn't


@pytest.fixture
def ou_fit_and_series(known_half_life_ou_series):
    series, theta_true = known_half_life_ou_series
    return fit_ou(series), series, theta_true


def test_ou_fit_recovers_known_parameters(ou_fit_and_series):
    fit, series, theta_true = ou_fit_and_series
    # 500 obs of a theta=0.15 process: estimates are noisy but should land
    # near the generating values (mu=0, sigma=0.05 in the fixture).
    assert fit.theta == pytest.approx(theta_true, abs=0.06)
    assert fit.mu == pytest.approx(0.0, abs=0.03)
    assert fit.sigma == pytest.approx(0.05, rel=0.15)
    assert fit.n_obs == len(series) - 1


def test_ou_fit_half_life_matches_compute_half_life(ou_fit_and_series):
    # Same AR(1) regression as screening.cointegration.compute_half_life, so
    # the implied half-life must agree exactly - one estimator, not two.
    fit, series, _ = ou_fit_and_series
    assert fit.half_life_days == pytest.approx(compute_half_life(series), rel=1e-9)


def test_ou_fit_no_mean_reversion_flags_none():
    rng = np.random.default_rng(7)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(400)))
    fit = fit_ou(random_walk)
    if fit.theta <= 0:
        assert fit.half_life_days is None


def test_simulated_paths_shape_and_anchor(ou_fit_and_series):
    fit, series, _ = ou_fit_and_series
    paths = simulate_spread_paths(series, n_paths=50, horizon_days=30, seed=1, fit=fit)
    assert paths.shape == (50, 31)  # column 0 is the anchor at the current spread
    assert np.all(paths[:, 0] == series.iloc[-1])
    assert np.isfinite(paths).all()


def test_simulated_paths_seeded_reproducibility(ou_fit_and_series):
    fit, series, _ = ou_fit_and_series
    a = simulate_spread_paths(series, n_paths=20, horizon_days=25, seed=42, fit=fit)
    b = simulate_spread_paths(series, n_paths=20, horizon_days=25, seed=42, fit=fit)
    c = simulate_spread_paths(series, n_paths=20, horizon_days=25, seed=43, fit=fit)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_terminal_distribution_approaches_mu(ou_fit_and_series):
    # theta ~0.15 => half-life ~5 bars, so 250 bars is ~50 half-lives: the
    # terminal cross-section should be centered on mu regardless of the anchor.
    fit, series, _ = ou_fit_and_series
    paths = simulate_spread_paths(series, n_paths=1000, horizon_days=250, seed=3, fit=fit)
    terminal_mean = paths[:, -1].mean()
    assert terminal_mean == pytest.approx(fit.mu, abs=3 * fit.stationary_std / np.sqrt(1000) + 0.01)


def test_strategy_pnl_finite_with_summary_keys(ou_fit_and_series):
    fit, series, _ = ou_fit_and_series
    paths = simulate_spread_paths(series, n_paths=100, horizon_days=60, seed=11, fit=fit)
    config = SignalConfig(zscore_window=10, entry_z=2.0, exit_z=0.5, stop_z=3.0)
    result = simulate_strategy_pnl(paths, fit, config, notional=10_000.0)

    assert set(result) == {"pnl_per_path", "mean", "median", "p05", "p95", "prob_profit", "n_paths"}
    assert result["pnl_per_path"].shape == (100,)
    assert np.isfinite(result["pnl_per_path"]).all()
    assert result["n_paths"] == 100
    assert 0.0 <= result["prob_profit"] <= 1.0
    assert result["p05"] <= result["median"] <= result["p95"]


def test_summarize_pnl_stat_values():
    pnl = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    summary = summarize_pnl(pnl)
    assert summary["mean"] == pytest.approx(0.0)
    assert summary["median"] == pytest.approx(0.0)
    assert summary["prob_profit"] == pytest.approx(0.4)  # strictly positive only
    assert summary["n_paths"] == 5


def test_plot_monte_carlo_paths_html(ou_fit_and_series):
    fit, series, _ = ou_fit_and_series
    paths = simulate_spread_paths(series, n_paths=60, horizon_days=30, seed=5, fit=fit)
    config = SignalConfig(zscore_window=10, entry_z=2.0, exit_z=0.5, stop_z=3.0)
    out = plot_monte_carlo_paths(series, paths, "SYNA", "SYNB", signal_config=config, ou_fit=fit)
    assert out.exists()
    assert out.stat().st_size > MIN_HTML_BYTES


def test_plot_interactive_equity_html(toy_trade_log):
    dates = pd.bdate_range("2023-01-02", periods=250)
    rng = np.random.default_rng(9)
    equity = pd.Series(100_000 + np.cumsum(rng.normal(20, 200, len(dates))), index=dates)
    out = plot_interactive_equity(equity, toy_trade_log, label="SYN_TEST")
    assert out.exists()
    assert out.stat().st_size > MIN_HTML_BYTES


def test_plot_interactive_spread_zscore_html(known_half_life_ou_series):
    series, _ = known_half_life_ou_series
    zscore = (series - series.mean()) / series.std(ddof=1)
    config = SignalConfig(zscore_window=10, entry_z=2.0, exit_z=0.5, stop_z=3.0)
    out = plot_interactive_spread_zscore(series, zscore, "SYNA", "SYNB", config)
    assert out.exists()
    assert out.stat().st_size > MIN_HTML_BYTES


def test_plot_pnl_distribution_html():
    rng = np.random.default_rng(13)
    pnl = rng.normal(50, 200, 1000)
    out = plot_pnl_distribution(pnl, label="SYN_TEST")
    assert out.exists()
    assert out.stat().st_size > MIN_HTML_BYTES


def test_plot_monte_carlo_paths_png_subsamples(ou_fit_and_series):
    fit, series, _ = ou_fit_and_series
    paths = simulate_spread_paths(series, n_paths=300, horizon_days=20, seed=17, fit=fit)
    out = plot_monte_carlo_paths_png(series, paths, "SYNA", "SYNB", max_paths=50)
    assert out.exists()
    assert out.stat().st_size > 5_000
