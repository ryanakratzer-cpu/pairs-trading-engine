import pandas as pd
import pytest

from signals.spread import SignalConfig, generate_signals, rolling_zscore


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
