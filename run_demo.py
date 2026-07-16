"""Deterministic, network-free demo of the pairs-trading engine's core pipeline:
cointegration screening, half-life estimation, spread/z-score signal generation,
and a portfolio backtest. Every stage asserts an expected outcome; exit 0 means
all stages verified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics import compute_metrics
from backtest.simulator import PairBacktestConfig, PairBacktester
from screening.cointegration import test_pair_cointegration
from signals.spread import SignalConfig, build_spread, generate_signals, rolling_zscore
from visualization.plots import plot_equity_curve, plot_spread_and_zscore

SEED = 42


def _make_ou_series(n, theta, mu, sigma, x0, rng):
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    return x


def _make_random_walk(n, sigma, x0, rng):
    steps = sigma * rng.standard_normal(n)
    steps[0] = 0.0
    return x0 + np.cumsum(steps)


def build_synthetic_universe(n_days: int = 500) -> tuple[pd.DataFrame, float, float]:
    """One synthetic cointegrated pair + one independent-random-walk control pair.

    Each series gets its own seeded RNG (rather than one shared RNG consumed
    sequentially) so hedge-ratio/half-life recovery quality is reproducible
    and insensitive to reordering the series below.
    """
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    hedge_ratio_true = 0.8
    theta = 0.1
    log_b = _make_random_walk(n_days, sigma=0.01, x0=np.log(100), rng=np.random.default_rng(SEED + 1))
    ou = _make_ou_series(n_days, theta=theta, mu=0.0, sigma=0.02, x0=0.0, rng=np.random.default_rng(SEED + 2))
    log_a = hedge_ratio_true * log_b + ou

    log_c = _make_random_walk(n_days, sigma=0.015, x0=np.log(50), rng=np.random.default_rng(SEED + 3))
    log_d = _make_random_walk(n_days, sigma=0.012, x0=np.log(80), rng=np.random.default_rng(SEED + 4))

    panel = pd.DataFrame(
        {
            "COKE_SIM": np.exp(log_a),
            "PEPS_SIM": np.exp(log_b),
            "NOISE_C": np.exp(log_c),
            "NOISE_D": np.exp(log_d),
        },
        index=dates,
    )
    return panel, hedge_ratio_true, theta


def main() -> None:
    print("=== Pairs Trading Engine - synthetic demo (deterministic, no network) ===\n")

    panel, hedge_ratio_true, theta = build_synthetic_universe()
    price_a, price_b = panel["COKE_SIM"], panel["PEPS_SIM"]
    price_c, price_d = panel["NOISE_C"], panel["NOISE_D"]

    print("[1/5] Engle-Granger cointegration test")
    coint_result = test_pair_cointegration(price_a, price_b, ticker_a="COKE_SIM", ticker_b="PEPS_SIM")
    control_result = test_pair_cointegration(price_c, price_d, ticker_a="NOISE_C", ticker_b="NOISE_D")
    print(
        f"  COKE_SIM/PEPS_SIM: adf_pvalue={coint_result.adf_pvalue:.4f}, "
        f"is_cointegrated={coint_result.is_cointegrated}, hedge_ratio={coint_result.hedge_ratio:.3f}"
    )
    print(
        f"  NOISE_C/NOISE_D:   adf_pvalue={control_result.adf_pvalue:.4f}, "
        f"is_cointegrated={control_result.is_cointegrated}"
    )
    assert coint_result.is_cointegrated, "expected synthetic pair to be classified cointegrated"
    assert not control_result.is_cointegrated, "expected independent random walks to NOT be cointegrated"
    assert abs(coint_result.hedge_ratio - hedge_ratio_true) < 0.1, "recovered hedge ratio too far from ground truth"

    print("\n[2/5] Half-life of mean reversion")
    expected_half_life = np.log(2) / theta
    print(f"  estimated={coint_result.half_life_days:.2f} days, expected~={expected_half_life:.2f} days")
    assert coint_result.half_life_days is not None
    assert 2.0 <= coint_result.half_life_days <= 20.0, "half-life outside plausible tradeable band"

    print("\n[3/5] Spread construction, z-score, and entry/exit/stop signals")
    signal_config = SignalConfig(zscore_window=30, entry_z=2.0, exit_z=0.5, stop_z=3.75)
    spread = build_spread(price_a, price_b, coint_result.hedge_ratio)
    zscore = rolling_zscore(spread, signal_config.zscore_window)
    signals = generate_signals(zscore, signal_config)
    n_entries = signals["event"].isin(["ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD"]).sum()
    print(f"  entries triggered: {n_entries}")
    assert n_entries > 0, "expected at least one entry signal over the synthetic sample"

    print("\n[4/5] Portfolio backtest")
    backtest_config = PairBacktestConfig(
        recheck_window_days=120,
        recheck_freq_days=60,
        signal_config=signal_config,
        max_concurrent_pairs=2,
    )
    result = PairBacktester(backtest_config).run(panel, [("COKE_SIM", "PEPS_SIM"), ("NOISE_C", "NOISE_D")])
    metrics = compute_metrics(result["equity_curve"], result["trade_log"])
    print(
        f"  trades={metrics['n_trades']}, total_return={metrics['total_return']:.2%}, "
        f"sharpe={metrics['sharpe_ratio']:.2f}, max_drawdown={metrics['max_drawdown']:.2%}"
    )
    assert metrics["n_trades"] > 0, "expected the backtest to execute at least one trade"

    print("\n[5/5] Saving diagnostic plots to outputs/")
    spread_plot = plot_spread_and_zscore(spread, zscore, "COKE_SIM", "PEPS_SIM", signal_config)
    equity_plot = plot_equity_curve(result["equity_curve"], label="demo_portfolio")
    print(f"  {spread_plot}")
    print(f"  {equity_plot}")

    print("\nAll stages verified. Exiting 0.")


if __name__ == "__main__":
    main()
