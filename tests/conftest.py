import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_ou_series(n: int, theta: float, mu: float, sigma: float, x0: float, seed: int) -> np.ndarray:
    """Discrete-time Ornstein-Uhlenbeck: x[t] = x[t-1] + theta*(mu - x[t-1]) + sigma*eps."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    return x


def _make_random_walk(n: int, sigma: float, x0: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = sigma * rng.standard_normal(n)
    steps[0] = 0.0
    return x0 + np.cumsum(steps)


@pytest.fixture
def business_day_index():
    return pd.bdate_range("2023-01-02", periods=500)


@pytest.fixture
def cointegrated_pair_prices(business_day_index):
    """B random-walks; A = hedge_ratio*B + OU-mean-reverting spread. Analytically
    known hedge ratio and half-life (ln(2)/theta) for ground-truth assertions."""
    n = len(business_day_index)
    hedge_ratio_true = 0.8
    theta = 0.1
    log_b = _make_random_walk(n, sigma=0.01, x0=np.log(100), seed=1)
    ou_spread = _make_ou_series(n, theta=theta, mu=0.0, sigma=0.02, x0=0.0, seed=2)
    log_a = hedge_ratio_true * log_b + ou_spread
    price_a = pd.Series(np.exp(log_a), index=business_day_index, name="A")
    price_b = pd.Series(np.exp(log_b), index=business_day_index, name="B")
    return price_a, price_b, hedge_ratio_true, theta


@pytest.fixture
def non_cointegrated_pair_prices(business_day_index):
    """Two independent random walks — false-positive control for the screener."""
    n = len(business_day_index)
    log_a = _make_random_walk(n, sigma=0.015, x0=np.log(50), seed=3)
    log_b = _make_random_walk(n, sigma=0.012, x0=np.log(80), seed=4)
    price_a = pd.Series(np.exp(log_a), index=business_day_index, name="A")
    price_b = pd.Series(np.exp(log_b), index=business_day_index, name="B")
    return price_a, price_b


@pytest.fixture
def negative_beta_pair_panel(business_day_index):
    """Panel whose two tickers cointegrate with a NEGATIVE hedge ratio.

    NEG is built against an INVERTED log_bbb, so OLS recovers beta ~= -0.9.
    spread = log(NEG) - beta*log(BBB) then has a same-signed hedge leg, i.e.
    trading it is net directional rather than market-neutral — exactly the
    ABT/MRK and COST/PEP situation the require_positive_hedge_ratio gate
    exists to block. The OU spread is strong enough that the pair still passes
    the re-cointegration checks, so the gate is the ONLY thing stopping it.
    """
    n = len(business_day_index)
    log_bbb = _make_random_walk(n, sigma=0.01, x0=np.log(60), seed=10)
    ou_spread = _make_ou_series(n, theta=0.12, mu=0.0, sigma=0.02, x0=0.0, seed=11)
    # +2*log(60) keeps prices positive despite the inverted loading.
    log_neg = -0.9 * log_bbb + ou_spread + 2 * np.log(60)

    panel = pd.DataFrame(
        {"NEG": np.exp(log_neg), "BBB": np.exp(log_bbb)},
        index=business_day_index,
    )
    return panel, ("NEG", "BBB")


@pytest.fixture
def eg_disagreement_pair_prices(business_day_index):
    """A weakly mean-reverting pair reproducing the live COST/PEP disagreement.

    The naive ADF-on-estimated-residuals p-value lands just under 0.05 while
    the proper Engle-Granger/MacKinnon p-value lands well above it — measured
    here at ~0.039 vs ~0.120, essentially the live COST/PEP numbers
    (0.0393 vs 0.1207). theta=0.03 makes reversion slow enough to sit in that
    borderline band.
    """
    n = len(business_day_index)
    log_b = _make_random_walk(n, sigma=0.01, x0=np.log(60), seed=20)
    ou_spread = _make_ou_series(n, theta=0.03, mu=0.0, sigma=0.02, x0=0.0, seed=120)
    log_a = 0.9 * log_b + ou_spread
    price_a = pd.Series(np.exp(log_a), index=business_day_index, name="A")
    price_b = pd.Series(np.exp(log_b), index=business_day_index, name="B")
    return price_a, price_b


@pytest.fixture
def known_half_life_ou_series(business_day_index):
    n = len(business_day_index)
    theta = 0.15
    series = _make_ou_series(n, theta=theta, mu=0.0, sigma=0.05, x0=0.0, seed=5)
    return pd.Series(series, index=business_day_index), theta


@pytest.fixture
def toy_equity_curve():
    dates = pd.bdate_range("2023-01-02", periods=6)
    values = [100_000, 101_000, 100_500, 102_000, 101_500, 103_000]
    return pd.Series(values, index=dates, dtype=float)


@pytest.fixture
def toy_trade_log():
    return pd.DataFrame(
        {
            "ticker_a": ["KO", "XOM"],
            "ticker_b": ["PEP", "CVX"],
            "position": [1, -1],
            "entry_date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "exit_date": pd.to_datetime(["2023-01-10", "2023-01-06"]),
            "holding_days": [7, 2],
            "pnl": [500.0, -200.0],
            "exit_reason": ["EXIT", "STOP_LOSS"],
        }
    )


@pytest.fixture
def sector_universe_fixture(business_day_index):
    """Small synthetic multi-ticker panel: AAA/BBB is an injected cointegrated
    pair, CCC/DDD are independent noise, for end-to-end screening/backtest tests."""
    n = len(business_day_index)
    log_bbb = _make_random_walk(n, sigma=0.01, x0=np.log(60), seed=10)
    ou_spread = _make_ou_series(n, theta=0.12, mu=0.0, sigma=0.02, x0=0.0, seed=11)
    log_aaa = 0.9 * log_bbb + ou_spread

    log_ccc = _make_random_walk(n, sigma=0.013, x0=np.log(40), seed=12)
    log_ddd = _make_random_walk(n, sigma=0.011, x0=np.log(90), seed=13)

    panel = pd.DataFrame(
        {
            "AAA": np.exp(log_aaa),
            "BBB": np.exp(log_bbb),
            "CCC": np.exp(log_ccc),
            "DDD": np.exp(log_ddd),
        },
        index=business_day_index,
    )
    return panel, ("AAA", "BBB")
