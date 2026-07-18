import numpy as np
import pandas as pd
import pytest

from signals.spread import KalmanHedgeRatio, SignalConfig, StaticOLSHedgeRatio, generate_signals, rolling_zscore


def test_rolling_zscore_is_causal():
    dates = pd.bdate_range("2023-01-02", periods=10)
    spread = pd.Series(range(10), index=dates, dtype=float)
    z = rolling_zscore(spread, window=3)

    assert z.iloc[:2].isna().all()

    truncated = spread.iloc[:3]
    z_truncated = rolling_zscore(truncated, window=3)
    assert z.iloc[2] == pytest.approx(z_truncated.iloc[2])


def test_signal_state_machine_entry_exit():
    config = SignalConfig(zscore_window=3, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    zscore = pd.Series([0.0, 2.5, 2.5, 1.0, 0.3, 0.0])
    result = generate_signals(zscore, config)

    assert result["event"].tolist() == [
        "NO_POSITION",
        "ENTER_SHORT_SPREAD",
        "HOLD",
        "HOLD",
        "EXIT",
        "NO_POSITION",
    ]
    assert result["position"].tolist() == [0, -1, -1, -1, 0, 0]


def test_signal_state_machine_stop_loss():
    config = SignalConfig(zscore_window=3, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    zscore = pd.Series([-2.5, -3.0, -3.6, -1.0])
    result = generate_signals(zscore, config)

    assert result["event"].tolist() == ["ENTER_LONG_SPREAD", "HOLD", "STOP_LOSS", "NO_POSITION"]
    assert result["position"].tolist() == [1, 1, 0, 0]


def test_tradeable_gate_blocks_new_entries():
    config = SignalConfig(zscore_window=3, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    zscore = pd.Series([2.5, 2.5])
    tradeable = pd.Series([False, False])
    result = generate_signals(zscore, config, tradeable=tradeable)

    assert result["position"].tolist() == [0, 0]
    assert result["event"].tolist() == ["NO_POSITION", "NO_POSITION"]


def test_tradeable_gate_does_not_block_exit():
    config = SignalConfig(zscore_window=3, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    zscore = pd.Series([2.5, 0.3])
    tradeable = pd.Series([True, False])
    result = generate_signals(zscore, config, tradeable=tradeable)

    assert result["event"].tolist() == ["ENTER_SHORT_SPREAD", "EXIT"]


def test_signal_config_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        SignalConfig(entry_z=1.0, exit_z=2.0, stop_z=3.0)


def test_time_exit_closes_position_that_never_converges():
    config = SignalConfig(zscore_window=3, entry_z=2.0, exit_z=0.5, stop_z=5.0, max_holding_bars=3)
    # Enters short at z=2.5, then z hovers between exit and stop forever —
    # without the time exit this would HOLD indefinitely.
    zscore = pd.Series([2.5, 2.0, 1.8, 1.9, 1.7, 1.6])
    result = generate_signals(zscore, config)

    assert result["event"].tolist() == [
        "ENTER_SHORT_SPREAD",
        "HOLD",
        "HOLD",
        "TIME_EXIT",
        "NO_POSITION",
        "NO_POSITION",
    ]
    assert result["position"].tolist() == [-1, -1, -1, 0, 0, 0]


def test_time_exit_does_not_fire_before_limit_or_when_disabled():
    zscore = pd.Series([2.5, 2.0, 1.8, 1.9, 1.7, 1.6])

    disabled = generate_signals(zscore, SignalConfig(zscore_window=3, stop_z=5.0, max_holding_bars=None))
    assert "TIME_EXIT" not in disabled["event"].tolist()

    loose = generate_signals(zscore, SignalConfig(zscore_window=3, stop_z=5.0, max_holding_bars=10))
    assert "TIME_EXIT" not in loose["event"].tolist()


def test_kalman_recovers_static_beta():
    # When the true relationship is constant, the Kalman beta should converge
    # to the same answer as static OLS.
    rng = np.random.default_rng(41)
    n = 400
    dates = pd.bdate_range("2023-01-02", periods=n)
    log_b = np.cumsum(0.01 * rng.standard_normal(n)) + np.log(70)
    log_a = 0.85 * log_b + 0.01 * rng.standard_normal(n)
    price_a = pd.Series(np.exp(log_a), index=dates)
    price_b = pd.Series(np.exp(log_b), index=dates)

    kalman_beta = KalmanHedgeRatio().hedge_ratio(price_a, price_b)
    ols_beta = StaticOLSHedgeRatio().hedge_ratio(price_a, price_b)

    assert kalman_beta == pytest.approx(0.85, abs=0.05)
    assert kalman_beta == pytest.approx(ols_beta, abs=0.05)


def test_kalman_tracks_a_drifting_beta_where_static_ols_cannot():
    # True beta shifts from 0.6 to 1.1 halfway through. The Kalman beta at the
    # end should be near 1.1; static OLS over the full window lands in between
    # and misrepresents both halves.
    rng = np.random.default_rng(42)
    n = 600
    dates = pd.bdate_range("2023-01-02", periods=n)
    log_b = np.cumsum(0.01 * rng.standard_normal(n)) + np.log(70)
    true_beta = np.where(np.arange(n) < n // 2, 0.6, 1.1)
    log_a = true_beta * log_b + 0.01 * rng.standard_normal(n)
    price_a = pd.Series(np.exp(log_a), index=dates)
    price_b = pd.Series(np.exp(log_b), index=dates)

    beta_series = KalmanHedgeRatio().hedge_ratio_series(price_a, price_b)
    ols_beta = StaticOLSHedgeRatio().hedge_ratio(price_a, price_b)

    # Kalman lands near each regime's true beta (some overshoot is inherent:
    # with a slowly-moving log price, the intercept and beta states are nearly
    # collinear, so the filter can't pin beta exactly). Full-window OLS doesn't
    # just land in between the two regimes — on this fixture it lands at ~-0.85,
    # wildly wrong for BOTH, because the level shift dominates the regression.
    assert beta_series.iloc[-1] == pytest.approx(1.1, abs=0.15)
    assert beta_series.iloc[n // 2 - 1] == pytest.approx(0.6, abs=0.1)
    assert abs(ols_beta - 1.1) > abs(beta_series.iloc[-1] - 1.1)  # OLS is worse at the end


def test_kalman_beta_is_causal():
    # The beta at bar t must not change when future bars are appended.
    rng = np.random.default_rng(43)
    n = 200
    dates = pd.bdate_range("2023-01-02", periods=n)
    log_b = np.cumsum(0.01 * rng.standard_normal(n)) + np.log(50)
    log_a = 0.9 * log_b + 0.01 * rng.standard_normal(n)
    price_a = pd.Series(np.exp(log_a), index=dates)
    price_b = pd.Series(np.exp(log_b), index=dates)

    kalman = KalmanHedgeRatio()
    full = kalman.hedge_ratio_series(price_a, price_b)
    truncated = kalman.hedge_ratio_series(price_a.iloc[:150], price_b.iloc[:150])

    assert full.iloc[149] == pytest.approx(truncated.iloc[149], abs=1e-12)


def test_innovation_series_is_causal(cointegrated_pair_prices):
    # The innovation at bar t is the surprise relative to the PRE-update state,
    # so appending future bars must leave every past value (innovation, its
    # predicted std, and the z-score) exactly unchanged.
    price_a, price_b, _hedge_ratio_true, _theta = cointegrated_pair_prices

    kalman = KalmanHedgeRatio()
    full = kalman.innovation_series(price_a, price_b)
    truncated = kalman.innovation_series(price_a.iloc[:300], price_b.iloc[:300])

    pd.testing.assert_frame_equal(full.iloc[:300], truncated)


def test_innovation_warmup_zscores_are_nan(cointegrated_pair_prices):
    # Under the diffuse prior the first predictions are dominated by state
    # uncertainty; those transient z-scores must be masked or they fire
    # bogus signals. Innovations themselves stay populated for diagnostics.
    price_a, price_b, _hedge_ratio_true, _theta = cointegrated_pair_prices
    kalman = KalmanHedgeRatio()

    default = kalman.innovation_series(price_a, price_b)
    assert default["zscore"].iloc[:30].isna().all()
    assert default["zscore"].iloc[30:].notna().all()
    assert default["innovation"].notna().all()
    assert default["innovation_std"].notna().all()

    custom = kalman.innovation_series(price_a, price_b, warmup_bars=10)
    assert custom["zscore"].iloc[:10].isna().all()
    assert custom["zscore"].iloc[10:].notna().all()


def test_innovation_zscore_trades_the_cointegrated_fixture(cointegrated_pair_prices):
    # THE regression test for the documented under-trading problem: a rolling
    # z-score of the Kalman spread barely trades because the adaptive state
    # absorbs divergences. The standardized innovation keeps them, so with
    # default SignalConfig thresholds it must produce real entries. The filter
    # R is set to the fixture's known observation variance (OU sigma=0.02,
    # so R=4e-4); the class default 1e-3 assumes noisier data than this
    # fixture has, which deflates every z below the 2.0 entry threshold.
    price_a, price_b, _hedge_ratio_true, _theta = cointegrated_pair_prices

    kalman = KalmanHedgeRatio(observation_variance=4e-4)
    zscore = kalman.innovation_series(price_a, price_b)["zscore"]
    signals = generate_signals(zscore, SignalConfig())

    entries = signals["event"].isin(["ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD"]).sum()
    assert entries >= 1
