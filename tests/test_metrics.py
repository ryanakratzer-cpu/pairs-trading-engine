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
