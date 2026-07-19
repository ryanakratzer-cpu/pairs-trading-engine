"""Mean-variance allocation across pair strategies: efficient frontier,
tangency (max-Sharpe) portfolio, minimum-variance portfolio.

Each *asset* here is one pair's mean-reversion strategy, not a raw ticker —
the optimizer decides how much capital each pair gets, replacing the flat
`capital_per_pair` split the backtester uses by default.

The two classically fragile Markowitz inputs are handled deliberately:

- Expected returns come from the OU fit each pair already has
  (`ou_expected_annual_return`), not from noisy historical strategy means.
  A fitted mean-reversion speed implies an analytic expected reversion P&L,
  which is a far lower-variance estimate than a sample mean of a strategy
  return series that trades a handful of times per year.
- The covariance matrix is shrunk toward a scaled identity via Ledoit-Wolf
  (2004), implemented directly (~20 lines) rather than pulling in sklearn.
  Pair-strategy return series are mostly zero (flat between trades), so the
  sample covariance is ill-conditioned; shrinkage keeps the optimizer from
  amplifying that estimation noise into corner solutions.

Constraints throughout: fully invested (weights sum to 1), long-only (you
can't short a strategy you simply wouldn't run), and a per-pair weight cap so
one seductive OU fit can't take the whole book. Sharpe convention matches the
rest of the project: risk-free rate 0, 252 trading days/year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from montecarlo.simulator import OUFit
from signals.spread import SignalConfig

TRADING_DAYS = 252
DEFAULT_MAX_WEIGHT = 0.40


@dataclass(frozen=True)
class PortfolioPoint:
    """One portfolio on the risk/return plane. Returns and vol are annualized."""

    expected_return: float
    volatility: float
    sharpe: float
    weights: np.ndarray

    @staticmethod
    def from_weights(
        weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free_rate: float = 0.0
    ) -> "PortfolioPoint":
        weights = np.asarray(weights, dtype=float)
        ret = float(weights @ mu)
        vol = float(np.sqrt(weights @ cov @ weights))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        return PortfolioPoint(expected_return=ret, volatility=vol, sharpe=sharpe, weights=weights)


def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled
    identity target. Returns (shrunk_covariance, shrinkage_intensity).

    Intensity 0 = pure sample covariance, 1 = pure identity target. The
    optimal intensity is estimated from the data itself: noisier sample
    covariances (short histories, many assets) shrink harder.
    """
    x = returns.to_numpy(dtype=float)
    t, n = x.shape
    if t < 2:
        raise ValueError(f"Need at least 2 return observations, got {t}")
    x = x - x.mean(axis=0)

    sample = (x.T @ x) / t
    mu_target = np.trace(sample) / n
    target = mu_target * np.eye(n)

    # d^2: distance between sample cov and target; b^2: estimation error of
    # the sample cov itself (capped at d^2 so intensity stays in [0, 1]).
    d2 = float(np.sum((sample - target) ** 2))
    b2_sum = 0.0
    for row in x:
        outer = np.outer(row, row)
        b2_sum += float(np.sum((outer - sample) ** 2))
    b2 = min(b2_sum / t**2, d2)

    intensity = 0.0 if d2 == 0 else b2 / d2
    shrunk = intensity * target + (1.0 - intensity) * sample
    return shrunk, intensity


def ou_expected_annual_return(
    fit: OUFit,
    signal_config: SignalConfig | None = None,
    notional: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    tradeable_fraction: float = 1.0,
) -> float:
    """Analytic-heuristic expected annual return (as a fraction of `notional`)
    of trading the entry/exit bands on an OU spread with these parameters.

    Decomposition (documented heuristic, not a theorem):
    - P&L per round trip: entering at |z| = entry_z and exiting at exit_z
      captures (entry_z - exit_z) * stationary_std in spread units, times
      `notional` (log-spread convention, same as the Monte Carlo simulator),
      minus round-trip costs on both ends.
    - Trades per year: one full divergence-reversion cycle is taken as
      ~4 half-lives (~2 out, ~2 back) — the same order of magnitude the
      half-life screening band (5-30d) implicitly assumes. Slower reversion
      therefore means both fewer trades AND (unchanged) per-trade capture,
      which is why short-half-life pairs dominate the tangency portfolio.

    A fit with no detected reversion (theta <= 0) returns 0.0 — the strategy
    has no modeled edge there, and the optimizer should see that, not a
    historical-mean artifact.

    `tradeable_fraction` is the gate-awareness fix: the raw heuristic assumes
    the pair is ALWAYS in a tradeable cycle, but in practice the rolling
    re-cointegration gate (and any macro/event gate) blocks entries much of
    the time — which is why raw OU expectations ran 20-55%/yr against ~0%
    realized. Pass the backtester's realized fraction of entry-allowed days
    (`prepared["tradeable"].mean()`) to scale the number of cycles the
    strategy can actually harvest, putting model and realized Sharpe on one
    scale.
    """
    signal_config = signal_config or SignalConfig()
    if fit.half_life_days is None or fit.stationary_std is None:
        return 0.0
    if not 0.0 <= tradeable_fraction <= 1.0:
        raise ValueError(f"tradeable_fraction must be in [0, 1], got {tradeable_fraction}")

    capture_per_trade = (signal_config.entry_z - signal_config.exit_z) * fit.stationary_std * notional
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    round_trip_cost = 2.0 * cost_rate * notional
    pnl_per_trade = capture_per_trade - round_trip_cost

    cycle_days = 4.0 * fit.half_life_days
    trades_per_year = tradeable_fraction * TRADING_DAYS / max(cycle_days, 1.0)

    return trades_per_year * pnl_per_trade / notional


def _effective_max_weight(n_assets: int, max_weight: float) -> float:
    """A cap below 1/n makes 'weights sum to 1' infeasible; lift it just enough."""
    return max(max_weight, 1.0 / n_assets + 1e-9)


def _solve(objective, n: int, max_weight: float, extra_constraints: list | None = None) -> np.ndarray:
    cap = _effective_max_weight(n, max_weight)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        constraints.extend(extra_constraints)
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed to converge: {result.message}")
    # Clean float dust so reported weights sum to exactly 1.
    weights = np.clip(result.x, 0.0, cap)
    return weights / weights.sum()


def min_variance_weights(cov: np.ndarray, max_weight: float = DEFAULT_MAX_WEIGHT) -> np.ndarray:
    n = cov.shape[0]
    return _solve(lambda w: w @ cov @ w, n, max_weight)


def max_sharpe_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.0,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> np.ndarray:
    """Tangency portfolio: the constrained-weight portfolio with the highest
    Sharpe ratio — where the capital allocation line touches the frontier.
    """
    n = cov.shape[0]

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(w @ cov @ w)
        if vol <= 0:
            return 0.0
        return -((w @ mu - risk_free_rate) / vol)

    return _solve(neg_sharpe, n, max_weight)


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 50,
    risk_free_rate: float = 0.0,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> list[PortfolioPoint]:
    """The upper (efficient) branch: minimum-variance portfolio at each target
    return between the global-min-variance return and the highest return
    reachable under the weight cap. Infeasible/non-converged targets are
    skipped rather than raising, so a coarse grid never kills the whole curve.
    """
    n = len(mu)
    cap = _effective_max_weight(n, max_weight)

    w_minvar = min_variance_weights(cov, max_weight)
    ret_low = float(w_minvar @ mu)
    # Highest expected return under the cap: greedily fill the best assets.
    order = np.argsort(mu)[::-1]
    w_max = np.zeros(n)
    remaining = 1.0
    for idx in order:
        take = min(cap, remaining)
        w_max[idx] = take
        remaining -= take
        if remaining <= 0:
            break
    ret_high = float(w_max @ mu)

    points: list[PortfolioPoint] = []
    for target in np.linspace(ret_low, ret_high, n_points):
        try:
            weights = _solve(
                lambda w: w @ cov @ w,
                n,
                max_weight,
                extra_constraints=[{"type": "eq", "fun": lambda w, t=target: w @ mu - t}],
            )
        except RuntimeError:
            continue
        points.append(PortfolioPoint.from_weights(weights, mu, cov, risk_free_rate))
    return points


def random_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    n_portfolios: int = 3000,
    risk_free_rate: float = 0.0,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    seed: int | None = 42,
) -> list[PortfolioPoint]:
    """Dirichlet-sampled long-only portfolios for the frontier chart's cloud.
    Samples breaching the weight cap are rejected (kept only if none qualify,
    so a tight cap can't return an empty cloud).
    """
    n = len(mu)
    cap = _effective_max_weight(n, max_weight)
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(n), size=n_portfolios)
    within_cap = samples[(samples <= cap + 1e-12).all(axis=1)]
    if len(within_cap) == 0:
        within_cap = samples
    return [PortfolioPoint.from_weights(w, mu, cov, risk_free_rate) for w in within_cap]


def annualize_inputs(daily_returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Shrunk, annualized covariance (and historical annualized mean, for
    reference) from a daily strategy-return panel. Returns (mu_hist, cov,
    shrinkage_intensity). The optimizer's default mu should come from
    `ou_expected_annual_return` — mu_hist is provided for comparison only.
    """
    cov_daily, intensity = ledoit_wolf_cov(daily_returns)
    mu_hist = daily_returns.mean().to_numpy() * TRADING_DAYS
    return mu_hist, cov_daily * TRADING_DAYS, intensity
