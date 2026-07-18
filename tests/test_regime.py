import numpy as np
import pandas as pd
import pytest

from screening.regime import (
    MACRO_TICKERS,
    RegimeConfig,
    align_mask_to,
    compute_stress_mask,
    macro_spread_diagnostics,
    stress_percentile_ranks,
)

# Small window so tests exercise warmup + spike behavior on short panels.
TEST_CONFIG = RegimeConfig(percentile_window=60)


def _make_macro_panel(
    n: int = 300,
    spike_start: int = 200,
    spike_len: int = 15,
    spike_column: str = "vix",
) -> pd.DataFrame:
    """Deterministic panel where every baseline series drifts slowly DOWN, so
    each new baseline value is the minimum of its trailing window (percentile
    rank near 0 -> unambiguously calm), and the spike segment rises strictly,
    so each spike value is the maximum of its window (rank 1.0 -> unambiguously
    stressed). This makes the expected mask exact rather than probabilistic."""
    idx = pd.bdate_range("2023-01-02", periods=n)
    t = np.arange(n, dtype=float)
    baseline_vix = 30.0 - 0.01 * t
    baseline_gvz = 25.0 - 0.01 * t

    panel = pd.DataFrame(
        {
            "vix": baseline_vix,
            "gold_vol": baseline_gvz,
            "ten_year_yield": 4.0 + 0.001 * t,
        },
        index=idx,
    )
    spike = 45.0 + 0.5 * np.arange(spike_len)
    panel.iloc[spike_start : spike_start + spike_len, panel.columns.get_loc(spike_column)] = spike
    return panel


def test_macro_tickers_is_subsettable_dict():
    assert isinstance(MACRO_TICKERS, dict)
    assert MACRO_TICKERS["vix"] == "^VIX"
    assert MACRO_TICKERS["gold_vol"] == "^GVZ"


def test_regime_config_validates_thresholds():
    with pytest.raises(ValueError):
        RegimeConfig(vix_stress_percentile=1.5)
    with pytest.raises(ValueError):
        RegimeConfig(percentile_window=1)


def test_mask_flags_vix_spike_exactly():
    spike_start, spike_len = 200, 15
    panel = _make_macro_panel(spike_start=spike_start, spike_len=spike_len)
    mask = compute_stress_mask(panel, TEST_CONFIG)

    spike_dates = panel.index[spike_start : spike_start + spike_len]
    calm_dates = panel.index.difference(spike_dates)

    assert not mask.loc[spike_dates].any(), "every spike date should be stressed"
    assert mask.loc[calm_dates].all(), "every non-spike date should be calm"


def test_gold_vol_spike_alone_triggers_stress():
    spike_start, spike_len = 200, 15
    panel = _make_macro_panel(spike_start=spike_start, spike_len=spike_len, spike_column="gold_vol")
    mask = compute_stress_mask(panel, TEST_CONFIG)

    spike_dates = panel.index[spike_start : spike_start + spike_len]
    assert not mask.loc[spike_dates].any()
    assert mask.loc[panel.index.difference(spike_dates)].all()


def test_spike_inside_warmup_window_stays_calm():
    # A spike before a full percentile window exists has no rank yet, so it
    # cannot be called stressed - warmup defaults to calm by design.
    panel = _make_macro_panel(spike_start=10, spike_len=15)
    mask = compute_stress_mask(panel, TEST_CONFIG)

    warmup_dates = panel.index[: TEST_CONFIG.percentile_window - 1]
    assert mask.loc[warmup_dates].all()


def test_mask_is_causal_under_truncation():
    panel = _make_macro_panel()
    mask_full = compute_stress_mask(panel, TEST_CONFIG)
    for cutoff in (100, 205, 250):
        mask_truncated = compute_stress_mask(panel.iloc[:cutoff], TEST_CONFIG)
        pd.testing.assert_series_equal(mask_full.iloc[:cutoff], mask_truncated)


def test_missing_macro_data_defaults_to_calm():
    panel = _make_macro_panel()
    # A stretch where both stress gauges go dark must not halt trading.
    dark = panel.index[150:170]
    panel.loc[dark, ["vix", "gold_vol"]] = np.nan
    mask = compute_stress_mask(panel, TEST_CONFIG)
    assert mask.loc[dark].all()

    # A panel with no stress gauges at all is entirely calm.
    no_gauges = panel[["ten_year_yield"]]
    assert compute_stress_mask(no_gauges, TEST_CONFIG).all()


def test_stress_percentile_ranks_only_includes_present_gauges():
    panel = _make_macro_panel()
    ranks = stress_percentile_ranks(panel[["vix", "ten_year_yield"]], TEST_CONFIG)
    assert list(ranks.columns) == ["vix"]
    # Warmup ranks are NaN; post-warmup ranks are within [0, 1].
    assert ranks["vix"].iloc[: TEST_CONFIG.percentile_window - 1].isna().all()
    post = ranks["vix"].dropna()
    assert ((post >= 0) & (post <= 1)).all()


def test_align_mask_to_ffills_onto_price_dates():
    mask_dates = pd.to_datetime(["2023-01-02", "2023-01-04", "2023-01-06"])
    mask = pd.Series([True, False, True], index=mask_dates)
    price_index = pd.date_range("2023-01-01", "2023-01-07", freq="D")

    aligned = align_mask_to(mask, price_index)

    assert aligned.dtype == bool
    assert list(aligned.index) == list(price_index)
    expected = [
        True,  # Jan 1: before first macro reading -> defaults to calm
        True,  # Jan 2: reading
        True,  # Jan 3: ffill of Jan 2
        False,  # Jan 4: reading
        False,  # Jan 5: ffill of Jan 4
        True,  # Jan 6: reading
        True,  # Jan 7: ffill of Jan 6
    ]
    assert aligned.tolist() == expected


def _make_diagnostics_panel(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Random-walk (in logs) macro series so returns have real variation."""
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(seed)
    log_vix = np.log(20.0) + np.cumsum(0.03 * rng.standard_normal(n))
    log_oil = np.log(70.0) + np.cumsum(0.02 * rng.standard_normal(n))
    return pd.DataFrame({"vix": np.exp(log_vix), "oil": np.exp(log_oil)}, index=idx)


def test_diagnostics_columns_and_coupling_detection():
    panel = _make_diagnostics_panel()
    # Spread built FROM the vix series: its daily changes are vix log returns,
    # so the spread-change vs vix-return correlation must be near 1, while the
    # independent oil series should show no strong coupling.
    spread = np.log(panel["vix"])

    diag = macro_spread_diagnostics(spread, panel, window=60)

    assert list(diag.columns) == ["macro_var", "full_sample_corr", "recent_corr", "n_obs"]
    assert set(diag["macro_var"]) == {"vix", "oil"}

    by_var = diag.set_index("macro_var")
    assert by_var.loc["vix", "full_sample_corr"] > 0.95
    assert by_var.loc["vix", "recent_corr"] > 0.95
    assert abs(by_var.loc["oil", "full_sample_corr"]) < 0.5
    assert by_var.loc["vix", "n_obs"] == len(panel) - 1  # first diff is NaN


def test_diagnostics_handles_thin_series_without_raising():
    panel = _make_diagnostics_panel(n=50)
    panel["thin"] = np.nan
    panel.iloc[-2:, panel.columns.get_loc("thin")] = [10.0, 11.0]
    spread = np.log(panel["vix"])

    diag = macro_spread_diagnostics(spread, panel, window=60)
    thin_row = diag.set_index("macro_var").loc["thin"]
    assert np.isnan(thin_row["full_sample_corr"])
