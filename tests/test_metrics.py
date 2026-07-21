import numpy as np
import pandas as pd
import pytest

from backtest.metrics import compute_metrics


def test_compute_metrics_matches_hand_computation(toy_equity_curve, toy_trade_log):
    metrics = compute_metrics(toy_equity_curve, toy_trade_log, periods_per_year=252)

    expected_total_return = toy_equity_curve.iloc[-1] / toy_equity_curve.iloc[0] - 1
    assert metrics["total_return"] == pytest.approx(expected_total_return)

    assert metrics["n_trades"] == 2
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["avg_win"] == pytest.approx(500.0)
    assert metrics["avg_loss"] == pytest.approx(-200.0)
    assert metrics["profit_factor"] == pytest.approx(500.0 / 200.0)
    assert metrics["avg_holding_days"] == pytest.approx((7 + 2) / 2)


def test_compute_metrics_return_and_risk_match_hand_computation(toy_equity_curve, toy_trade_log):
    """Nail down the return/risk block against fully hand-computed values on the
    toy curve — Sharpe, annualized vol, annualized return, and max drawdown were
    previously unasserted. rf=0, 252 periods/year, ddof=1 on the vol.
    """
    metrics = compute_metrics(toy_equity_curve, toy_trade_log, periods_per_year=252)

    returns = toy_equity_curve.pct_change().dropna()
    n = len(returns)  # 5 return observations from 6 equity points
    total_return = toy_equity_curve.iloc[-1] / toy_equity_curve.iloc[0] - 1

    expected_vol = returns.std(ddof=1) * np.sqrt(252)
    expected_sharpe = returns.mean() * 252 / expected_vol
    expected_ann_return = (1 + total_return) ** (252 / n) - 1
    # Equity dips 101,000 -> 100,500 then 102,000 -> 101,500; the deepest single
    # drop from a running peak is 100,500/101,000 - 1.
    expected_max_dd = 100_500 / 101_000 - 1

    assert metrics["annualized_vol"] == pytest.approx(expected_vol)
    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert metrics["annualized_return"] == pytest.approx(expected_ann_return)
    assert metrics["max_drawdown"] == pytest.approx(expected_max_dd)


def test_compute_metrics_single_point_and_flat_curve_are_degenerate_safe():
    """Zero-variance / single-observation guards: no div-by-zero, no NaNs."""
    single = compute_metrics(pd.Series([100_000.0], index=pd.bdate_range("2023-01-02", periods=1)),
                             pd.DataFrame())
    assert single["sharpe_ratio"] == 0.0
    assert single["annualized_vol"] == 0.0
    assert single["annualized_return"] == 0.0
    assert single["max_drawdown"] == 0.0

    flat = compute_metrics(pd.Series([100_000.0] * 5, index=pd.bdate_range("2023-01-02", periods=5)),
                           pd.DataFrame())
    assert flat["sharpe_ratio"] == 0.0  # zero variance must not divide by zero
    assert flat["annualized_vol"] == 0.0


def test_profit_factor_all_wins_is_inf_all_losses_is_zero():
    dates = pd.bdate_range("2023-01-02", periods=3)
    equity = pd.Series([100_000, 100_500, 101_000], index=dates, dtype=float)

    all_wins = pd.DataFrame({"pnl": [100.0, 200.0], "holding_days": [1, 2]})
    m_wins = compute_metrics(equity, all_wins)
    assert m_wins["win_rate"] == pytest.approx(1.0)
    assert m_wins["profit_factor"] == float("inf")

    all_losses = pd.DataFrame({"pnl": [-100.0, -200.0], "holding_days": [1, 2]})
    m_losses = compute_metrics(equity, all_losses)
    assert m_losses["win_rate"] == pytest.approx(0.0)
    assert m_losses["profit_factor"] == 0.0


def test_compute_metrics_handles_empty_trade_log(toy_equity_curve):
    metrics = compute_metrics(toy_equity_curve, pd.DataFrame())

    assert metrics["n_trades"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["profit_factor"] == 0.0


def test_max_drawdown_is_correctly_negative():
    dates = pd.bdate_range("2023-01-02", periods=4)
    equity = pd.Series([100_000, 120_000, 90_000, 110_000], index=dates, dtype=float)
    metrics = compute_metrics(equity, pd.DataFrame())

    expected_dd = 90_000 / 120_000 - 1
    assert metrics["max_drawdown"] == pytest.approx(expected_dd)
