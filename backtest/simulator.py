"""Portfolio-level backtest simulator for a set of candidate cointegrated pairs.

Equity curve reflects realized P&L at each trade's close (not daily
mark-to-market of open positions) — a documented simplification. Position
sizing is dollar-neutral per leg (capital_per_pair split evenly between the
two legs), not hedge-ratio-weighted dollar sizing — also documented in the
project README as a v1 simplification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from screening.cointegration import test_pair_cointegration
from signals.spread import SignalConfig, generate_signals, rolling_zscore

DEFAULT_RECHECK_FREQ_DAYS = 60
DEFAULT_RECHECK_WINDOW_DAYS = 252


@dataclass(frozen=True)
class PairBacktestConfig:
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    capital_per_pair: float = 10_000.0
    max_concurrent_pairs: int = 5
    initial_capital: float = 100_000.0
    recheck_freq_days: int = DEFAULT_RECHECK_FREQ_DAYS
    recheck_window_days: int = DEFAULT_RECHECK_WINDOW_DAYS
    significance: float = 0.05
    use_log_prices: bool = True
    signal_config: SignalConfig = field(default_factory=SignalConfig)


@dataclass
class _OpenPosition:
    ticker_a: str
    ticker_b: str
    position: int
    entry_date: pd.Timestamp
    entry_price_a: float
    entry_price_b: float
    shares_a: float
    shares_b: float
    entry_cost: float


def _build_regime_hedge_ratios(
    price_a: pd.Series,
    price_b: pd.Series,
    config: PairBacktestConfig,
) -> tuple[pd.Series, pd.Series]:
    """Piecewise-constant hedge ratio + tradeable flag, re-estimated every
    `recheck_freq_days` from the trailing `recheck_window_days` of data
    (causal — never uses data beyond the estimation date). A failed recheck
    sets tradeable=False until the next successful recheck; new entries are
    disabled during that stretch but open positions still exit/stop normally.
    """
    dates = price_a.index
    n = len(dates)
    hedge_ratios = pd.Series(np.nan, index=dates, dtype=float)
    tradeable = pd.Series(False, index=dates, dtype=bool)

    first_idx = config.recheck_window_days
    if first_idx >= n:
        return hedge_ratios, tradeable

    current_hedge_ratio = np.nan
    current_tradeable = False

    for i in range(n):
        if i >= first_idx and (i - first_idx) % config.recheck_freq_days == 0:
            window_a = price_a.iloc[i - config.recheck_window_days : i]
            window_b = price_b.iloc[i - config.recheck_window_days : i]
            result = test_pair_cointegration(
                window_a,
                window_b,
                significance=config.significance,
                use_log_prices=config.use_log_prices,
            )
            current_hedge_ratio = result.hedge_ratio
            current_tradeable = result.is_cointegrated

        hedge_ratios.iloc[i] = current_hedge_ratio
        tradeable.iloc[i] = current_tradeable

    return hedge_ratios, tradeable


def _prepare_pair_series(price_a: pd.Series, price_b: pd.Series, config: PairBacktestConfig) -> dict:
    hedge_ratios, tradeable = _build_regime_hedge_ratios(price_a, price_b, config)

    a = np.log(price_a) if config.use_log_prices else price_a
    b = np.log(price_b) if config.use_log_prices else price_b
    spread = a - hedge_ratios * b

    zscore = rolling_zscore(spread, config.signal_config.zscore_window)
    signals = generate_signals(zscore, config.signal_config, tradeable=tradeable)

    return {"hedge_ratios": hedge_ratios, "spread": spread, "zscore": zscore, "signals": signals}


class PairBacktester:
    """Simulates a portfolio of candidate pairs trading on their z-score signals,
    with transaction costs/slippage, dollar-neutral leg sizing, a max-concurrent-
    pairs cap (competing entries ranked by |z|-strength when slots are scarce),
    and periodic re-cointegration checks per pair.
    """

    def __init__(self, config: PairBacktestConfig | None = None):
        self.config = config or PairBacktestConfig()

    def run(self, price_panel: pd.DataFrame, pairs: list[tuple[str, str]]) -> dict:
        config = self.config
        prepared: dict[tuple[str, str], dict] = {}

        for ticker_a, ticker_b in pairs:
            if ticker_a not in price_panel.columns or ticker_b not in price_panel.columns:
                continue
            pair_prices = price_panel[[ticker_a, ticker_b]].dropna()
            if len(pair_prices) <= config.recheck_window_days:
                continue
            prepared[(ticker_a, ticker_b)] = {
                "prices": pair_prices,
                **_prepare_pair_series(pair_prices[ticker_a], pair_prices[ticker_b], config),
            }

        if not prepared:
            return {
                "equity_curve": pd.Series([config.initial_capital], index=[price_panel.index[-1]]),
                "trade_log": pd.DataFrame(),
                "per_pair": {},
            }

        all_dates = sorted(set().union(*(p["prices"].index for p in prepared.values())))

        open_positions: dict[tuple[str, str], _OpenPosition] = {}
        trade_log_rows = []
        realized_pnl = 0.0
        equity_curve = pd.Series(index=all_dates, dtype=float)

        for date in all_dates:
            for pair_key in list(open_positions.keys()):
                pair_data = prepared[pair_key]
                if date not in pair_data["signals"].index:
                    continue
                row = pair_data["signals"].loc[date]
                if row["event"] in ("EXIT", "STOP_LOSS"):
                    open_pos = open_positions.pop(pair_key)
                    price_a = pair_data["prices"].loc[date, pair_key[0]]
                    price_b = pair_data["prices"].loc[date, pair_key[1]]
                    pnl = self._close_position(open_pos, price_a, price_b)
                    realized_pnl += pnl
                    trade_log_rows.append(
                        {
                            "ticker_a": pair_key[0],
                            "ticker_b": pair_key[1],
                            "position": open_pos.position,
                            "entry_date": open_pos.entry_date,
                            "exit_date": date,
                            "holding_days": (date - open_pos.entry_date).days,
                            "pnl": pnl,
                            "exit_reason": row["event"],
                        }
                    )

            available_slots = config.max_concurrent_pairs - len(open_positions)
            if available_slots > 0:
                candidates = []
                for pair_key, pair_data in prepared.items():
                    if pair_key in open_positions:
                        continue
                    if date not in pair_data["signals"].index:
                        continue
                    row = pair_data["signals"].loc[date]
                    if row["event"] in ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD"):
                        candidates.append((pair_key, row, abs(row["zscore"])))

                candidates.sort(key=lambda c: c[2], reverse=True)
                for pair_key, row, _strength in candidates[:available_slots]:
                    pair_data = prepared[pair_key]
                    price_a = pair_data["prices"].loc[date, pair_key[0]]
                    price_b = pair_data["prices"].loc[date, pair_key[1]]
                    position = 1 if row["event"] == "ENTER_LONG_SPREAD" else -1
                    open_positions[pair_key] = self._open_position(pair_key, position, date, price_a, price_b)

            equity_curve.loc[date] = config.initial_capital + realized_pnl

        trade_log = pd.DataFrame(trade_log_rows)
        return {
            "equity_curve": equity_curve.ffill().fillna(config.initial_capital),
            "trade_log": trade_log,
            "per_pair": prepared,
        }

    def _open_position(self, pair_key, position, date, price_a, price_b) -> _OpenPosition:
        config = self.config
        notional_per_leg = config.capital_per_pair / 2
        shares_a = notional_per_leg / price_a
        shares_b = notional_per_leg / price_b
        cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000
        entry_cost = cost_rate * (notional_per_leg * 2)
        return _OpenPosition(
            ticker_a=pair_key[0],
            ticker_b=pair_key[1],
            position=position,
            entry_date=date,
            entry_price_a=price_a,
            entry_price_b=price_b,
            shares_a=shares_a,
            shares_b=shares_b,
            entry_cost=entry_cost,
        )

    def _close_position(self, open_pos: _OpenPosition, exit_price_a: float, exit_price_b: float) -> float:
        config = self.config
        if open_pos.position == 1:  # long A, short B
            gross_pnl = open_pos.shares_a * (exit_price_a - open_pos.entry_price_a) + open_pos.shares_b * (
                open_pos.entry_price_b - exit_price_b
            )
        else:  # short A, long B
            gross_pnl = open_pos.shares_a * (open_pos.entry_price_a - exit_price_a) + open_pos.shares_b * (
                exit_price_b - open_pos.entry_price_b
            )

        notional_exit = open_pos.shares_a * exit_price_a + open_pos.shares_b * exit_price_b
        cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000
        exit_cost = cost_rate * notional_exit

        return gross_pnl - open_pos.entry_cost - exit_cost
