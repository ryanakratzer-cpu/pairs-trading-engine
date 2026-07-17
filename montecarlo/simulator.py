"""Monte Carlo simulation of a pair's spread as an Ornstein-Uhlenbeck process.

The point of simulating rather than just backtesting: a single historical path
is one draw from the spread's distribution, so backtest P&L alone can't say
whether the strategy's edge is robust or lucky. Fitting an OU model to the
observed spread and pushing ~1000 synthetic paths through the SAME signal
state machine the backtester uses gives a forward-looking P&L distribution
(and a probability of profit) instead of a single number.

Model and fit share the discrete-time convention already used across the
project (tests/conftest.py's _make_ou_series, screening.cointegration's
compute_half_life):

    x[t] = x[t-1] + theta * (mu - x[t-1]) + sigma * eps[t],   eps ~ N(0, 1)

which is the AR(1) regression  delta_x[t] = a + b * x[t-1] + resid  with
theta = -b, mu = a / theta, and sigma = std(resid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from signals.spread import SignalConfig, generate_signals

DEFAULT_N_PATHS = 1000
DEFAULT_HORIZON_DAYS = 90


@dataclass(frozen=True)
class OUFit:
    """Per-bar OU parameters estimated from an observed spread series."""

    theta: float  # mean-reversion speed per bar; <= 0 means no reversion detected
    mu: float  # long-run mean the process reverts toward
    sigma: float  # one-bar diffusion (std of the AR(1) residual)
    half_life_days: float | None  # ln(2)/theta; None when theta <= 0
    n_obs: int

    @property
    def stationary_std(self) -> float | None:
        """Std of the process's stationary distribution, sigma / sqrt(1 - phi^2)
        with phi = 1 - theta. This is the natural scale for z-scoring a
        simulated path: unlike a rolling std it needs no warm-up window, so a
        short simulated horizon isn't wasted estimating what the fit already
        knows. None when the fit implies no stationary distribution.
        """
        phi = 1.0 - self.theta
        if abs(phi) >= 1.0:
            return None
        return float(self.sigma / np.sqrt(1.0 - phi**2))


def fit_ou(spread: pd.Series) -> OUFit:
    """Fit OU parameters via the same AR(1) regression compute_half_life uses.

    Regress delta_spread on lagged spread (with intercept). The slope gives
    theta, the intercept gives mu, and the residual std gives sigma - so the
    implied half-life here is numerically identical to compute_half_life's,
    keeping the simulator consistent with the screener's diagnostics rather
    than introducing a second, subtly different estimator.
    """
    spread = pd.Series(spread).astype(float).reset_index(drop=True)
    lagged = spread.shift(1)
    delta = spread.diff()

    valid = pd.concat([lagged, delta], axis=1).dropna()
    if len(valid) < 3:
        raise ValueError(f"Need at least 3 spread observations to fit OU, got {len(spread)}")
    lagged_valid = valid.iloc[:, 0].to_numpy()
    delta_valid = valid.iloc[:, 1].to_numpy()

    design = sm.add_constant(lagged_valid)
    model = sm.OLS(delta_valid, design).fit()
    intercept, slope = model.params

    theta = float(-slope)
    # When theta <= 0 the regression found no pull toward a mean, so a / theta
    # is meaningless; fall back to the sample mean so simulation still runs
    # (the paths will just wander) and flag it via half_life_days=None.
    mu = float(intercept / theta) if theta > 0 else float(spread.mean())
    sigma = float(np.std(model.resid, ddof=1))
    half_life = float(np.log(2) / theta) if theta > 0 else None

    return OUFit(theta=theta, mu=mu, sigma=sigma, half_life_days=half_life, n_obs=len(valid))


def simulate_spread_paths(
    spread: pd.Series,
    n_paths: int = DEFAULT_N_PATHS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    seed: int | None = None,
    fit: OUFit | None = None,
) -> np.ndarray:
    """Simulate OU spread trajectories forward from the spread's latest value.

    Returns an (n_paths, horizon_days + 1) array whose column 0 is the current
    spread value for every path - keeping the anchor point in the array means
    plots and P&L both see the fan open from where the spread actually is
    today, not one step into the future.

    Pass `fit` to reuse parameters already estimated (e.g. by the runner's
    console summary) instead of refitting; pass `seed` for reproducibility.
    """
    if n_paths < 1 or horizon_days < 1:
        raise ValueError("n_paths and horizon_days must both be >= 1")
    if fit is None:
        fit = fit_ou(spread)

    x0 = float(pd.Series(spread).iloc[-1])
    rng = np.random.default_rng(seed)

    paths = np.empty((n_paths, horizon_days + 1))
    paths[:, 0] = x0
    shocks = rng.standard_normal((n_paths, horizon_days))
    # Loop over time (typically ~90 steps), vectorized across paths - the
    # recursion can't be vectorized in t because each step feeds the next.
    for t in range(1, horizon_days + 1):
        prev = paths[:, t - 1]
        paths[:, t] = prev + fit.theta * (fit.mu - prev) + fit.sigma * shocks[:, t - 1]
    return paths


def summarize_pnl(pnl_per_path: np.ndarray) -> dict:
    """Distribution summary for a per-path P&L array. Keys are stable API:
    tests and the P&L histogram plot both key off them.
    """
    pnl = np.asarray(pnl_per_path, dtype=float)
    return {
        "mean": float(np.mean(pnl)),
        "median": float(np.median(pnl)),
        "p05": float(np.percentile(pnl, 5)),
        "p95": float(np.percentile(pnl, 95)),
        "prob_profit": float(np.mean(pnl > 0)),
        "n_paths": int(len(pnl)),
    }


def simulate_strategy_pnl(
    paths: np.ndarray,
    fit: OUFit,
    signal_config: SignalConfig,
    notional: float = 10_000.0,
) -> dict:
    """Run the project's real entry/exit/stop state machine on each simulated
    path and return the resulting per-path P&L distribution.

    Deliberate simplifications (this is a distribution estimate, not a full
    backtest):
    - Each path is z-scored against the OU fit's stationary distribution,
      (x - mu) / stationary_std, rather than a rolling window - a rolling
      z would burn `zscore_window` bars of an already-short horizon warming
      up, and the stationary z is the model-consistent choice for paths the
      same fit generated.
    - P&L of a held position is position * delta_spread per bar, scaled by
      `notional` (spread is in log-price units, so this approximates the
      dollar P&L of a notional-sized dollar-neutral pair position). No
      transaction costs, slippage, or per-leg share sizing.
    - A position still open at the horizon is implicitly marked at the final
      bar's value - the same END_OF_SAMPLE force-liquidation convention the
      backtester uses.

    Returns summarize_pnl()'s summary dict plus "pnl_per_path".
    """
    stationary_std = fit.stationary_std
    if stationary_std is None or stationary_std <= 0:
        raise ValueError(
            "OU fit implies no stationary distribution (theta <= 0 or phi >= 1); "
            "cannot z-score simulated paths for signal generation"
        )

    paths = np.asarray(paths, dtype=float)
    pnl_per_path = np.empty(paths.shape[0])
    for i, path in enumerate(paths):
        zscore = pd.Series((path - fit.mu) / stationary_std)
        signals = generate_signals(zscore, signal_config)
        positions = signals["position"].to_numpy()
        # Position established at bar t earns the spread change over t -> t+1.
        pnl_per_path[i] = notional * float(np.sum(positions[:-1] * np.diff(path)))

    return {"pnl_per_path": pnl_per_path, **summarize_pnl(pnl_per_path)}
