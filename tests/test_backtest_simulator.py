import numpy as np
import pandas as pd
import pytest

from backtest.simulator import (
    PairBacktestConfig,
    PairBacktester,
    _build_regime_hedge_ratios,
    _prepare_pair_series,
)
from signals.spread import SignalConfig


def _rw(n, sigma, x0, seed):
    rng = np.random.default_rng(seed)
    steps = sigma * rng.standard_normal(n)
    steps[0] = 0.0
    return x0 + np.cumsum(steps)


def _ou(n, theta, sigma, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (0.0 - x[t - 1]) + sigma * rng.standard_normal()
    return x


def test_risk_profile_presets_are_valid_and_ordered_by_exposure():
    conservative = PairBacktestConfig.conservative()
    moderate = PairBacktestConfig.moderate()
    aggressive = PairBacktestConfig.aggressive()

    for config in (conservative, moderate, aggressive):
        signal_config = config.signal_config
        assert 0 < signal_config.exit_z < signal_config.entry_z < signal_config.stop_z
        assert config.capital_per_pair > 0
        assert config.max_concurrent_pairs > 0

    # conservative should risk less per pair and in aggregate than aggressive
    assert conservative.capital_per_pair <= moderate.capital_per_pair <= aggressive.capital_per_pair
    assert conservative.max_concurrent_pairs <= moderate.max_concurrent_pairs <= aggressive.max_concurrent_pairs
    assert conservative.signal_config.stop_z <= moderate.signal_config.stop_z <= aggressive.signal_config.stop_z


def test_close_position_long_spread_pnl_and_costs():
    config = PairBacktestConfig(transaction_cost_bps=10.0, slippage_bps=0.0, capital_per_pair=10_000.0)
    backtester = PairBacktester(config)
    open_pos = backtester._open_position(
        ("A", "B"), position=1, date=pd.Timestamp("2023-01-02"), price_a=100.0, price_b=50.0
    )

    assert open_pos.shares_a == pytest.approx(5_000.0 / 100.0)
    assert open_pos.shares_b == pytest.approx(5_000.0 / 50.0)
    assert open_pos.entry_cost == pytest.approx((10.0 / 10_000) * 10_000.0)

    pnl = backtester._close_position(open_pos, exit_price_a=110.0, exit_price_b=45.0)

    gross = open_pos.shares_a * (110.0 - 100.0) + open_pos.shares_b * (50.0 - 45.0)
    notional_exit = open_pos.shares_a * 110.0 + open_pos.shares_b * 45.0
    exit_cost = (10.0 / 10_000) * notional_exit
    expected_pnl = gross - open_pos.entry_cost - exit_cost

    assert pnl == pytest.approx(expected_pnl)
    assert pnl > 0  # A rose, B fell — favorable for a long-spread position


def test_close_position_short_spread_pnl_zero_costs():
    config = PairBacktestConfig(transaction_cost_bps=0.0, slippage_bps=0.0, capital_per_pair=10_000.0)
    backtester = PairBacktester(config)
    open_pos = backtester._open_position(
        ("A", "B"), position=-1, date=pd.Timestamp("2023-01-02"), price_a=100.0, price_b=50.0
    )

    pnl = backtester._close_position(open_pos, exit_price_a=90.0, exit_price_b=55.0)

    gross = open_pos.shares_a * (100.0 - 90.0) + open_pos.shares_b * (55.0 - 50.0)
    assert pnl == pytest.approx(gross)
    assert pnl > 0  # A fell, B rose — favorable for a short-spread position


def test_mark_to_market_matches_close_minus_exit_cost():
    config = PairBacktestConfig(transaction_cost_bps=8.0, slippage_bps=2.0, capital_per_pair=10_000.0)
    backtester = PairBacktester(config)
    open_pos = backtester._open_position(
        ("A", "B"), position=1, date=pd.Timestamp("2023-01-02"), price_a=100.0, price_b=50.0
    )

    unrealized = backtester._mark_to_market(open_pos, price_a=105.0, price_b=48.0)
    closed_pnl = backtester._close_position(open_pos, exit_price_a=105.0, exit_price_b=48.0)

    notional_exit = open_pos.shares_a * 105.0 + open_pos.shares_b * 48.0
    exit_cost = ((config.transaction_cost_bps + config.slippage_bps) / 10_000) * notional_exit

    assert unrealized > 0  # A rose, B fell — favorable mark for a long-spread position
    assert closed_pnl == pytest.approx(unrealized - exit_cost)


def test_open_position_still_open_at_data_end_is_force_liquidated(cointegrated_pair_prices):
    price_a, price_b, _hedge_ratio_true, _theta = cointegrated_pair_prices

    # First pass over the full series to find a date where a position is open.
    config = PairBacktestConfig(
        recheck_window_days=100,
        recheck_freq_days=50,
        signal_config=SignalConfig(zscore_window=15, entry_z=1.0, exit_z=0.3, stop_z=4.0),
        max_concurrent_pairs=1,
    )
    panel = pd.concat([price_a.rename("A"), price_b.rename("B")], axis=1)
    full_result = PairBacktester(config).run(panel, [("A", "B")])
    assert not full_result["trade_log"].empty

    # Find a trade that was held for more than one bar, so there's an interior
    # date strictly between entry and exit to truncate the series at.
    trade_log = full_result["trade_log"]
    entry_idx = trade_log["entry_date"].map(panel.index.get_loc)
    exit_idx = trade_log["exit_date"].map(panel.index.get_loc)
    multi_bar_trades = trade_log[(exit_idx - entry_idx) >= 2]
    assert not multi_bar_trades.empty, "expected at least one trade held for 2+ bars"
    first_multi_bar_trade = multi_bar_trades.iloc[0]
    truncate_at_idx = panel.index.get_loc(first_multi_bar_trade["entry_date"]) + 1  # one bar into the trade

    # Truncate the data so the series ends while that trade is still open, then
    # confirm the truncated run force-closes it instead of leaving it dangling.
    truncated_panel = panel.iloc[: truncate_at_idx + 1]
    truncated_result = PairBacktester(config).run(truncated_panel, [("A", "B")])

    assert not truncated_result["trade_log"].empty
    last_trade = truncated_result["trade_log"].iloc[-1]
    assert last_trade["exit_date"] == truncated_panel.index[-1]
    assert last_trade["exit_reason"] == "END_OF_SAMPLE"

    final_equity = truncated_result["equity_curve"].iloc[-1]
    total_pnl = truncated_result["trade_log"]["pnl"].sum()
    assert final_equity == pytest.approx(config.initial_capital + total_pnl)


def test_position_open_at_a_pairs_own_data_end_is_liquidated_on_ragged_panel():
    """Regression: on a ragged panel one pair can stop trading before the
    global last date (differing listing/delisting windows survive the per-pair
    dropna). A position still open at THAT pair's own last bar must be
    force-liquidated there, not silently dropped. Previously the END_OF_SAMPLE
    block skipped any pair lacking the global last_date, discarding the open
    position's already-charged entry cost and its P&L entirely.
    """
    n = 400
    idx = pd.bdate_range("2022-01-03", periods=n)
    # LONG pair supplies the full-length index; SHORT pair is truncated early.
    long_b = _rw(n, 0.01, np.log(60), 1)
    long_a = 0.9 * long_b + _ou(n, 0.12, 0.02, 2)
    short_b = _rw(n, 0.01, np.log(50), 3)
    short_a = 0.9 * short_b + _ou(n, 0.12, 0.02, 4)
    panel = pd.DataFrame(
        {
            "LA": np.exp(long_a),
            "LB": np.exp(long_b),
            "SA": np.exp(short_a),
            "SB": np.exp(short_b),
        },
        index=idx,
    )
    config = PairBacktestConfig(
        recheck_window_days=100,
        recheck_freq_days=50,
        signal_config=SignalConfig(zscore_window=15, entry_z=1.0, exit_z=0.3, stop_z=4.0),
        max_concurrent_pairs=5,
    )

    # Find a bar in the tail where SA/SB genuinely holds an open position, and
    # truncate the short pair's data one bar later so the position is open at
    # its final available bar.
    short_signals = _prepare_pair_series(panel["SA"], panel["SB"], config)["signals"]
    open_bars = [i for i in range(260, 299) if short_signals.iloc[i]["position"] != 0]
    assert open_bars, "fixture must hold an open SA/SB position in the tail"
    trunc = open_bars[3]

    ragged = panel.copy()
    ragged.loc[idx[trunc + 1 :], ["SA", "SB"]] = np.nan
    assert ragged["SA"].last_valid_index() < idx[-1]  # short pair really ends early

    result = PairBacktester(config).run(ragged, [("LA", "LB"), ("SA", "SB")])
    trade_log = result["trade_log"]

    short_trades = trade_log[trade_log["ticker_a"] == "SA"]
    eos = short_trades[short_trades["exit_reason"] == "END_OF_SAMPLE"]
    assert len(eos) == 1, "the dangling short-pair position must be force-liquidated"
    assert eos.iloc[0]["exit_date"] == ragged["SA"].last_valid_index()

    # And with nothing dropped, total realized P&L reconciles with final equity.
    final_equity = result["equity_curve"].iloc[-1]
    assert final_equity == pytest.approx(config.initial_capital + trade_log["pnl"].sum())


def test_backtest_equity_is_causal_to_future_prices():
    """Regression / look-ahead guard: perturbing prices strictly AFTER a cutoff
    date must leave the equity curve and all trades closed on/before the cutoff
    bit-for-bit unchanged. Catches any signal, hedge-ratio recheck, rolling
    stat, or MTM that peeks at or beyond the bar it acts on.
    """
    n = 400
    idx = pd.bdate_range("2022-01-03", periods=n)
    log_b = _rw(n, 0.01, np.log(60), 10)
    log_a = 0.9 * log_b + _ou(n, 0.12, 0.02, 11)
    panel = pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=idx)

    config = PairBacktestConfig(
        recheck_window_days=100,
        recheck_freq_days=50,
        signal_config=SignalConfig(zscore_window=15, entry_z=1.0, exit_z=0.3, stop_z=4.0),
        max_concurrent_pairs=1,
    )
    base = PairBacktester(config).run(panel, [("A", "B")])

    cut = 250
    cut_date = idx[cut]
    perturbed_panel = panel.copy()
    rng = np.random.default_rng(999)
    shock = 1 + rng.normal(0.0, 0.05, size=(n - cut - 1, 2))
    perturbed_panel.iloc[cut + 1 :] = perturbed_panel.iloc[cut + 1 :].to_numpy() * shock
    perturbed = PairBacktester(config).run(perturbed_panel, [("A", "B")])

    base_eq = base["equity_curve"][base["equity_curve"].index <= cut_date]
    pert_eq = perturbed["equity_curve"][perturbed["equity_curve"].index <= cut_date]
    pd.testing.assert_series_equal(base_eq, pert_eq)

    cols = ["entry_date", "exit_date", "position", "pnl", "exit_reason"]
    base_closed = (
        base["trade_log"][base["trade_log"]["exit_date"] <= cut_date]
        .sort_values("entry_date")
        .reset_index(drop=True)
    )
    pert_closed = (
        perturbed["trade_log"][perturbed["trade_log"]["exit_date"] <= cut_date]
        .sort_values("entry_date")
        .reset_index(drop=True)
    )
    assert not base_closed.empty
    pd.testing.assert_frame_equal(base_closed[cols], pert_closed[cols])


def test_config_rejects_unknown_hedge_ratio_mode():
    with pytest.raises(ValueError):
        PairBacktestConfig(hedge_ratio_mode="ols")

    # All three supported modes must validate.
    for mode in ("regime", "kalman", "kalman_innovation"):
        assert PairBacktestConfig(hedge_ratio_mode=mode).hedge_ratio_mode == mode


def test_regime_hedge_ratio_disables_after_cointegration_breaks_down():
    dates = pd.bdate_range("2023-01-02", periods=400)
    n = len(dates)
    rng = np.random.default_rng(7)

    log_b = np.cumsum(0.01 * rng.standard_normal(n)) + np.log(60)
    ou = np.zeros(n)
    for t in range(1, 150):
        ou[t] = ou[t - 1] + 0.15 * (0.0 - ou[t - 1]) + 0.02 * rng.standard_normal()
    for t in range(150, n):
        ou[t] = ou[t - 1] + 0.02 * rng.standard_normal() + 0.01  # permanent drift, no reversion

    log_a = 0.9 * log_b + ou
    price_a = pd.Series(np.exp(log_a), index=dates)
    price_b = pd.Series(np.exp(log_b), index=dates)

    config = PairBacktestConfig(recheck_window_days=100, recheck_freq_days=50)
    _hedge_ratios, tradeable = _build_regime_hedge_ratios(price_a, price_b, config)

    assert tradeable.iloc[100:150].any()  # regime estimated from pre-break data
    assert not tradeable.iloc[-50:].any()  # regime estimated entirely from broken-down data


def test_max_concurrent_pairs_enforced(sector_universe_fixture):
    panel, (ticker_a, ticker_b) = sector_universe_fixture
    pairs = [(ticker_a, ticker_b), ("CCC", "DDD")]

    config = PairBacktestConfig(
        recheck_window_days=100,
        recheck_freq_days=50,
        signal_config=SignalConfig(zscore_window=15, entry_z=1.0, exit_z=0.3, stop_z=4.0),
        max_concurrent_pairs=1,
    )
    result = PairBacktester(config).run(panel, pairs)

    # A closed position frees its slot the same day a new one can open it, so
    # exit_date is exclusive here — otherwise a same-day handoff double-counts.
    open_count = pd.Series(0, index=result["equity_curve"].index)
    for _, trade in result["trade_log"].iterrows():
        span = (open_count.index >= trade["entry_date"]) & (open_count.index < trade["exit_date"])
        open_count[span] += 1

    assert open_count.max() <= config.max_concurrent_pairs


def test_end_to_end_backtest_smoke(sector_universe_fixture):
    panel, (ticker_a, ticker_b) = sector_universe_fixture
    pairs = [(ticker_a, ticker_b), ("CCC", "DDD")]

    config = PairBacktestConfig(
        recheck_window_days=100,
        recheck_freq_days=50,
        signal_config=SignalConfig(zscore_window=15, entry_z=1.0, exit_z=0.3, stop_z=4.0),
        max_concurrent_pairs=2,
    )
    result = PairBacktester(config).run(panel, pairs)

    assert len(result["equity_curve"]) == len(panel)
    assert not result["trade_log"].empty

    final_equity = result["equity_curve"].iloc[-1]
    total_pnl = result["trade_log"]["pnl"].sum()
    assert final_equity == pytest.approx(config.initial_capital + total_pnl)


def test_kalman_innovation_mode_backtests_end_to_end(sector_universe_fixture):
    # The innovation z-score must flow through the whole simulator: signal
    # generation, regime-gated entries, and P&L accounting. Same recheck and
    # threshold settings as the other end-to-end tests so trade frequency is
    # comparable across modes.
    panel, (ticker_a, ticker_b) = sector_universe_fixture
    pairs = [(ticker_a, ticker_b), ("CCC", "DDD")]

    config = PairBacktestConfig(
        recheck_window_days=100,
        recheck_freq_days=50,
        hedge_ratio_mode="kalman_innovation",
        signal_config=SignalConfig(zscore_window=15, entry_z=1.0, exit_z=0.3, stop_z=4.0),
        max_concurrent_pairs=2,
    )
    result = PairBacktester(config).run(panel, pairs)

    assert len(result["equity_curve"]) == len(panel)
    assert not result["trade_log"].empty  # at least one trade executed

    # The spread/zscore series exposed for plots must be the innovation series,
    # not a rolling z-score: after warmup they are fully populated (no rolling
    # window of NaNs) and the zscore is spread / predicted std at every bar.
    pair_data = result["per_pair"][(ticker_a, ticker_b)]
    assert pair_data["zscore"].iloc[:30].isna().all()
    assert pair_data["spread"].notna().all()

    final_equity = result["equity_curve"].iloc[-1]
    total_pnl = result["trade_log"]["pnl"].sum()
    assert final_equity == pytest.approx(config.initial_capital + total_pnl)
