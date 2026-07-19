"""Walk-forward validation: fit on a formation window, measure on a held-out
window that the fit never saw, roll forward, aggregate.

One framework reused three ways, matching the project's three open honesty
questions:

- `pair_survival_study` — do pairs that pass the cointegration screen in a
  formation window STAY cointegrated in the following holdout window? The
  aggregate survival rate is the out-of-sample decay rate of the screen
  itself: if it's low, screened pairs typically break before they can pay.
- `parameter_study` — if you pick the best entry/exit/stop bands on the
  formation window (the tempting thing), do they beat the textbook defaults
  on the holdout window? Reports both so overfitting shows up as a gap.
- `allocation_study` — fit tangency / min-variance weights on formation
  strategy returns, apply them (frozen) to holdout returns, compare against
  equal-weight. The in-sample comparison in run_portfolio.py cannot crown a
  winner; this can.

Everything downstream of a window boundary uses only formation-fitted
parameters (hedge ratios are never re-estimated inside a holdout), and
rolling statistics warm up on formation data so holdout bars are evaluated
causally from bar one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from portfolio.optimizer import annualize_inputs, max_sharpe_weights, min_variance_weights
from screening.cointegration import compute_half_life, test_pair_cointegration
from signals.spread import SignalConfig, generate_signals, rolling_zscore

TRADING_DAYS = 252
DEFAULT_COST_BPS_ROUND_TRIP_SIDE = 10.0  # 5 bps cost + 5 bps slippage, per side


@dataclass(frozen=True)
class WalkForwardConfig:
    formation_days: int = 252
    holdout_days: int = 63
    step_days: int = 63
    significance: float = 0.05
    min_half_life_days: float = 5.0
    max_half_life_days: float = 30.0
    use_log_prices: bool = True
    signal_config: SignalConfig = field(default_factory=SignalConfig)


def generate_windows(index: pd.Index, config: WalkForwardConfig) -> list[tuple[pd.Index, pd.Index]]:
    """Rolling (formation_index, holdout_index) splits. Windows advance by
    `step_days`; the last partial window is dropped rather than shortened so
    every holdout is measured on the same number of bars.
    """
    windows = []
    total = config.formation_days + config.holdout_days
    start = 0
    while start + total <= len(index):
        formation = index[start : start + config.formation_days]
        holdout = index[start + config.formation_days : start + total]
        windows.append((formation, holdout))
        start += config.step_days
    return windows


def _adf_pvalue(spread: pd.Series) -> float:
    return float(adfuller(spread.dropna().to_numpy(), autolag="AIC")[1])


def _sharpe(daily_returns: pd.Series) -> float:
    std = daily_returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(daily_returns.mean() / std * np.sqrt(TRADING_DAYS))


def strategy_returns(
    spread: pd.Series,
    config: SignalConfig,
    cost_bps_per_side: float = DEFAULT_COST_BPS_ROUND_TRIP_SIDE,
) -> pd.Series:
    """Daily strategy returns of trading `config`'s bands on a (log-price)
    spread: yesterday's position earns today's spread change, minus costs on
    each position change. Log-spread changes are already fractions of pair
    notional, so these are unit-capital returns — the walk-forward studies'
    common currency.
    """
    zscore = rolling_zscore(spread, config.zscore_window)
    positions = generate_signals(zscore, config)["position"]
    gross = positions.shift(1) * spread.diff()
    turnover = positions.diff().abs().fillna(0.0)
    return gross.fillna(0.0) - turnover * (cost_bps_per_side / 10_000)


# ------------------------------------------------------------ pair survival


def pair_survival_study(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    config: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    """For every window and every pair that passes the screen on formation
    data (cointegrated at `significance` + half-life inside the tradeable
    band), re-test the SAME formation-fitted spread on the holdout window.

    Returns one row per formation-passing (window, pair):
    window_end, ticker_a, ticker_b, formation_pvalue, formation_half_life,
    holdout_pvalue, survived. `survived.mean()` is the screen's out-of-sample
    survival rate — the number this project never had.

    Power caveat: the ADF test on a short holdout (63 bars) has LOW power, so
    a genuinely stationary spread will often fail to reject the unit root
    there — the survival rate is therefore a conservative LOWER BOUND on true
    persistence, and is most meaningful comparatively (pair vs pair, window
    vs window) rather than as an absolute probability.
    """
    config = config or WalkForwardConfig()
    rows = []
    for formation_idx, holdout_idx in generate_windows(prices.index, config):
        for ticker_a, ticker_b in pairs:
            if ticker_a not in prices.columns or ticker_b not in prices.columns:
                continue
            form_a = prices.loc[formation_idx, ticker_a].dropna()
            form_b = prices.loc[formation_idx, ticker_b].dropna()
            if len(form_a) < config.formation_days * 0.9:
                continue
            result = test_pair_cointegration(
                form_a, form_b, significance=config.significance, use_log_prices=config.use_log_prices
            )
            if not result.is_cointegrated:
                continue
            form_spread = (
                np.log(form_a) - result.hedge_ratio * np.log(form_b)
                if config.use_log_prices
                else form_a - result.hedge_ratio * form_b
            )
            half_life = compute_half_life(form_spread)
            if (
                half_life is None
                or not config.min_half_life_days <= half_life <= config.max_half_life_days
            ):
                continue

            hold_a = prices.loc[holdout_idx, ticker_a].dropna()
            hold_b = prices.loc[holdout_idx, ticker_b].dropna()
            hold_spread = (
                np.log(hold_a) - result.hedge_ratio * np.log(hold_b)
                if config.use_log_prices
                else hold_a - result.hedge_ratio * hold_b
            )
            holdout_pvalue = _adf_pvalue(hold_spread)
            rows.append(
                {
                    "window_end": holdout_idx[-1],
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "formation_pvalue": result.adf_pvalue,
                    "formation_half_life": half_life,
                    "holdout_pvalue": holdout_pvalue,
                    "survived": holdout_pvalue < config.significance,
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ parameter WF


def default_parameter_grid() -> list[SignalConfig]:
    """Modest grid around the textbook defaults — wide enough for overfitting
    to have room to show off, small enough to stay honest about search size
    (9 configs, not 900)."""
    grid = []
    for entry in (1.5, 2.0, 2.5):
        for exit_ in (0.25, 0.5, 1.0):
            grid.append(SignalConfig(entry_z=entry, exit_z=exit_, stop_z=entry + 1.75))
    return grid


def parameter_study(
    price_a: pd.Series,
    price_b: pd.Series,
    config: WalkForwardConfig | None = None,
    grid: list[SignalConfig] | None = None,
) -> pd.DataFrame:
    """Per window: pick the grid config with the best FORMATION Sharpe, then
    measure that choice AND the default config on the HOLDOUT window (same
    formation-fitted hedge ratio for both; the rolling z-score warms up on
    formation bars so holdout evaluation is causal from its first bar).

    Returns one row per window: chosen entry/exit/stop, formation Sharpe of
    the chosen config, holdout Sharpe of chosen vs default. If picking winners
    in-sample worked, holdout_sharpe_chosen would beat holdout_sharpe_default
    on average; the gap between formation and holdout Sharpe of the chosen
    config is the overfitting tax, stated per window.
    """
    config = config or WalkForwardConfig()
    grid = grid or default_parameter_grid()
    default_cfg = config.signal_config
    prices = pd.concat([price_a.rename("a"), price_b.rename("b")], axis=1).dropna()

    rows = []
    for formation_idx, holdout_idx in generate_windows(prices.index, config):
        form = prices.loc[formation_idx]
        result = test_pair_cointegration(
            form["a"], form["b"], significance=config.significance, use_log_prices=config.use_log_prices
        )
        # Spread over formation+holdout with the formation-fitted hedge ratio;
        # holdout returns are then sliced out after rolling stats warm up on
        # formation bars only.
        both = prices.loc[formation_idx.union(holdout_idx)]
        full_spread = (
            np.log(both["a"]) - result.hedge_ratio * np.log(both["b"])
            if config.use_log_prices
            else both["a"] - result.hedge_ratio * both["b"]
        )

        best_cfg, best_form_sharpe = None, -np.inf
        for candidate in grid:
            form_sharpe = _sharpe(strategy_returns(full_spread.loc[formation_idx], candidate))
            if form_sharpe > best_form_sharpe:
                best_cfg, best_form_sharpe = candidate, form_sharpe

        holdout_chosen = _sharpe(strategy_returns(full_spread, best_cfg).loc[holdout_idx])
        holdout_default = _sharpe(strategy_returns(full_spread, default_cfg).loc[holdout_idx])
        rows.append(
            {
                "window_end": holdout_idx[-1],
                "chosen_entry_z": best_cfg.entry_z,
                "chosen_exit_z": best_cfg.exit_z,
                "chosen_stop_z": best_cfg.stop_z,
                "formation_sharpe_chosen": best_form_sharpe,
                "holdout_sharpe_chosen": holdout_chosen,
                "holdout_sharpe_default": holdout_default,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ allocation WF


def allocation_study(
    strategy_returns_panel: pd.DataFrame,
    config: WalkForwardConfig | None = None,
    max_weight: float = 0.40,
) -> pd.DataFrame:
    """Per window: estimate mean/covariance on FORMATION strategy returns,
    solve tangency and min-variance weights there, then apply those frozen
    weights to the HOLDOUT returns. Equal-weight rides along as the 1/N
    benchmark (it needs no estimation, which is exactly its advantage).

    The panel should itself be causal (e.g. per-pair backtest returns) — this
    study adds the walk-forward split for the WEIGHTS, which is the part
    run_portfolio.py's in-sample comparison couldn't validate.

    Returns one row per (window, scheme) with the holdout Sharpe; aggregate
    with `.groupby("scheme")["holdout_sharpe"].mean()` and count wins vs 1/N.
    """
    config = config or WalkForwardConfig()
    n_assets = strategy_returns_panel.shape[1]
    rows = []
    for formation_idx, holdout_idx in generate_windows(strategy_returns_panel.index, config):
        formation = strategy_returns_panel.loc[formation_idx].dropna()
        holdout = strategy_returns_panel.loc[holdout_idx].dropna()
        if len(formation) < 30 or len(holdout) < 5:
            continue
        mu_hist, cov, _ = annualize_inputs(formation)
        schemes: dict[str, np.ndarray] = {"equal_weight": np.full(n_assets, 1.0 / n_assets)}
        try:
            schemes["tangency"] = max_sharpe_weights(mu_hist, cov, max_weight=max_weight)
            schemes["min_variance"] = min_variance_weights(cov, max_weight=max_weight)
        except RuntimeError:
            pass  # optimizer non-convergence: 1/N row still records the window
        for scheme, weights in schemes.items():
            holdout_returns = (holdout * weights).sum(axis=1)
            rows.append(
                {
                    "window_end": holdout_idx[-1],
                    "scheme": scheme,
                    "holdout_sharpe": _sharpe(holdout_returns),
                    "holdout_return_ann": float(holdout_returns.mean() * TRADING_DAYS),
                }
            )
    return pd.DataFrame(rows)
