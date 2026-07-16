"""Performance metrics for backtested pair-trading equity curves and trade logs."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Headline performance metrics. Sharpe ratio convention: risk-free rate = 0,
    periods_per_year = 252 trading days, stated explicitly rather than left implicit.
    """
    returns = equity_curve.pct_change().dropna()
    n_periods = len(returns)

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0.0
    annualized_return = (
        (1 + total_return) ** (periods_per_year / n_periods) - 1 if n_periods > 0 else 0.0
    )
    annualized_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if n_periods > 1 else 0.0
    sharpe_ratio = (
        float(returns.mean() * periods_per_year) / annualized_vol if annualized_vol > 0 else 0.0
    )

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    if trade_log.empty:
        win_rate = avg_win = avg_loss = profit_factor = avg_holding_days = 0.0
        n_trades = 0
    else:
        pnl = trade_log["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        n_trades = len(trade_log)
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        avg_holding_days = float(trade_log["holding_days"].mean())

    return {
        "total_return": total_return,
        "annualized_return": float(annualized_return),
        "annualized_vol": annualized_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "n_trades": int(n_trades),
        "avg_holding_days": avg_holding_days,
    }
