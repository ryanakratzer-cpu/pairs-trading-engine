"""Spread construction, rolling z-score, and the entry/exit/stop-loss signal state machine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class SignalConfig:
    zscore_window: int = 30
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.75
    # Time-based exit: if a position hasn't converged within this many bars,
    # the mean-reversion thesis has failed — exit rather than sit through more
    # divergence waiting for the price stop. Rule of thumb: 2-3x the pair's
    # half-life. None disables it.
    max_holding_bars: int | None = None

    def __post_init__(self) -> None:
        if not (0 < self.exit_z < self.entry_z < self.stop_z):
            raise ValueError("Require 0 < exit_z < entry_z < stop_z")
        if self.zscore_window < 2:
            raise ValueError("zscore_window must be >= 2")
        if self.max_holding_bars is not None and self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1 (or None to disable)")


class HedgeRatioModel(ABC):
    """Interface for estimating the hedge ratio used to construct a pair's spread.

    Only StaticOLSHedgeRatio is implemented for v1. A rolling-window or
    Kalman-filter variant that lets the hedge ratio vary over time is a
    documented future extension, not built here.
    """

    @abstractmethod
    def hedge_ratio(self, price_a: pd.Series, price_b: pd.Series, use_log_prices: bool = True) -> float:
        raise NotImplementedError


class StaticOLSHedgeRatio(HedgeRatioModel):
    """Fixed hedge ratio estimated once via OLS over the full input window."""

    def hedge_ratio(self, price_a: pd.Series, price_b: pd.Series, use_log_prices: bool = True) -> float:
        a = np.log(price_a) if use_log_prices else price_a
        b = np.log(price_b) if use_log_prices else price_b
        design = sm.add_constant(b.to_numpy())
        model = sm.OLS(a.to_numpy(), design).fit()
        return float(model.params[1])


class KalmanHedgeRatio(HedgeRatioModel):
    """Time-varying hedge ratio via a 2-state (intercept, beta) Kalman filter.

    Observation model:  a_t = alpha_t + beta_t * b_t + noise
    State model:        (alpha_t, beta_t) follow a random walk

    Strictly causal: the beta at time t is computed only from observations up
    to and including t, so it can feed a backtest or live monitor with no
    look-ahead. `delta` controls how fast the state is allowed to drift —
    larger values adapt faster but produce noisier betas; the 1e-5 default is
    the standard choice in the pairs-trading literature (e.g. Chan, Kinlay).
    """

    def __init__(self, delta: float = 1e-5, observation_variance: float = 1e-3):
        self.delta = delta
        self.observation_variance = observation_variance

    def hedge_ratio(self, price_a: pd.Series, price_b: pd.Series, use_log_prices: bool = True) -> float:
        """Latest (most recent) beta — the interface's single-float contract."""
        return float(self.hedge_ratio_series(price_a, price_b, use_log_prices).iloc[-1])

    def hedge_ratio_series(
        self, price_a: pd.Series, price_b: pd.Series, use_log_prices: bool = True
    ) -> pd.Series:
        """Full causal beta path, one value per bar, aligned to price_a's index."""
        a = (np.log(price_a) if use_log_prices else price_a).astype(float).to_numpy()
        b = (np.log(price_b) if use_log_prices else price_b).astype(float).to_numpy()
        n = len(a)

        state = np.zeros(2)  # (alpha, beta)
        state_cov = np.eye(2)  # diffuse prior: let early observations set the level
        transition_cov = (self.delta / (1 - self.delta)) * np.eye(2)

        betas = np.empty(n)
        for t in range(n):
            # Predict: random-walk state, covariance grows by transition noise
            state_cov = state_cov + transition_cov

            # Update with observation a_t = H @ state + noise, H = [1, b_t]
            obs_vector = np.array([1.0, b[t]])
            innovation = a[t] - obs_vector @ state
            innovation_var = obs_vector @ state_cov @ obs_vector + self.observation_variance
            kalman_gain = state_cov @ obs_vector / innovation_var
            state = state + kalman_gain * innovation
            state_cov = state_cov - np.outer(kalman_gain, obs_vector) @ state_cov

            betas[t] = state[1]

        return pd.Series(betas, index=price_a.index, name="kalman_beta")

    def spread_series(
        self, price_a: pd.Series, price_b: pd.Series, use_log_prices: bool = True
    ) -> pd.Series:
        """Spread built with the per-bar causal beta: a_t - beta_t * b_t."""
        a = np.log(price_a) if use_log_prices else price_a
        b = np.log(price_b) if use_log_prices else price_b
        betas = self.hedge_ratio_series(price_a, price_b, use_log_prices)
        return a - betas * b


def build_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
    use_log_prices: bool = True,
) -> pd.Series:
    a = np.log(price_a) if use_log_prices else price_a
    b = np.log(price_b) if use_log_prices else price_b
    return a - hedge_ratio * b


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Causal rolling z-score: each point uses only the trailing `window` observations."""
    rolling_mean = spread.rolling(window=window, min_periods=window).mean()
    rolling_std = spread.rolling(window=window, min_periods=window).std()
    return (spread - rolling_mean) / rolling_std


def generate_signals(
    zscore: pd.Series,
    config: SignalConfig,
    tradeable: pd.Series | None = None,
) -> pd.DataFrame:
    """Day-by-day entry/exit/stop-loss state machine over a z-score series.

    Path-dependent (today's position depends on yesterday's), so implemented
    as an explicit loop rather than vectorized. Position convention: +1 = long
    spread (long A, short hedge_ratio*B), entered when z < -entry_z; -1 = short
    spread, entered when z > entry_z. `tradeable` (aligned to zscore's index)
    gates new entries only — when False, no new position may open, but an
    already-open position still exits/stops normally on its own terms; this is
    how a failed rolling re-cointegration check disables a pair going forward.
    Returns a DataFrame with columns zscore, position ({-1, 0, 1}), and event
    (labeled state transition).
    """
    if tradeable is None:
        tradeable = pd.Series(True, index=zscore.index)

    positions: list[int] = []
    events: list[str] = []
    position = 0
    bars_held = 0

    for z, can_enter in zip(zscore, tradeable):
        if pd.isna(z):
            if position != 0:
                bars_held += 1
            event = "HOLD" if position != 0 else "NO_POSITION"
            positions.append(position)
            events.append(event)
            continue

        if position == 0:
            if can_enter and z > config.entry_z:
                position = -1
                bars_held = 0
                event = "ENTER_SHORT_SPREAD"
            elif can_enter and z < -config.entry_z:
                position = 1
                bars_held = 0
                event = "ENTER_LONG_SPREAD"
            else:
                event = "NO_POSITION"
        else:
            bars_held += 1
            time_exit_due = (
                config.max_holding_bars is not None and bars_held >= config.max_holding_bars
            )
            if position == 1:
                if z <= -config.stop_z:
                    position = 0
                    event = "STOP_LOSS"
                elif z >= -config.exit_z:
                    position = 0
                    event = "EXIT"
                elif time_exit_due:
                    position = 0
                    event = "TIME_EXIT"
                else:
                    event = "HOLD"
            else:  # position == -1
                if z >= config.stop_z:
                    position = 0
                    event = "STOP_LOSS"
                elif z <= config.exit_z:
                    position = 0
                    event = "EXIT"
                elif time_exit_due:
                    position = 0
                    event = "TIME_EXIT"
                else:
                    event = "HOLD"

        positions.append(position)
        events.append(event)

    return pd.DataFrame(
        {"zscore": zscore.to_numpy(), "position": positions, "event": events},
        index=zscore.index,
    )
