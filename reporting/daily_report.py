"""Daily 'what would today's signal be' report.

Pure computation: pulls the latest bars, recomputes cointegration/hedge
ratio/z-score, and returns a labeled recommendation per pair. Never places
an order or calls a broker — this module has no execution capability at all.
Each report replays the full signal state machine from the start of the
lookback window, so it does not persist an actual held-position across
separate report runs; treat it as a research/monitoring tool, not a live
position tracker.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from data.loader import align_and_clean, fetch_price_history
from screening.cointegration import test_pair_cointegration
from signals.spread import SignalConfig, build_spread, generate_signals, rolling_zscore

DISCLAIMER = "SIGNAL ONLY - not an executed trade. No order has been placed."


def generate_daily_signal_report(
    pairs: list[tuple[str, str]],
    lookback_days: int = 400,
    signal_config: SignalConfig | None = None,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    signal_config = signal_config or SignalConfig()
    end = as_of_date or datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    tickers = sorted({t for pair in pairs for t in pair})
    raw_prices = fetch_price_history(tickers, start=start, end=end)
    prices, dropped = align_and_clean(raw_prices)
    if dropped:
        print(f"[daily_report] dropped thin-history tickers: {dropped}")

    rows = []
    for ticker_a, ticker_b in pairs:
        if ticker_a not in prices.columns or ticker_b not in prices.columns:
            rows.append(_no_data_row(ticker_a, ticker_b))
            continue

        pair_prices = prices[[ticker_a, ticker_b]].dropna()
        if len(pair_prices) < signal_config.zscore_window + 10:
            rows.append(_no_data_row(ticker_a, ticker_b))
            continue

        eg_result = test_pair_cointegration(
            pair_prices[ticker_a], pair_prices[ticker_b], ticker_a=ticker_a, ticker_b=ticker_b
        )
        spread = build_spread(pair_prices[ticker_a], pair_prices[ticker_b], eg_result.hedge_ratio)
        zscore = rolling_zscore(spread, signal_config.zscore_window)
        signals = generate_signals(zscore, signal_config)

        latest = signals.iloc[-1]
        rows.append(
            {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "as_of": pair_prices.index[-1],
                "hedge_ratio": eg_result.hedge_ratio,
                "is_cointegrated": eg_result.is_cointegrated,
                "adf_pvalue": eg_result.adf_pvalue,
                "half_life_days": eg_result.half_life_days,
                "zscore": latest["zscore"],
                "recommendation": latest["event"],
            }
        )

    return pd.DataFrame(rows)


def _no_data_row(ticker_a: str, ticker_b: str) -> dict:
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "as_of": None,
        "hedge_ratio": None,
        "is_cointegrated": None,
        "adf_pvalue": None,
        "half_life_days": None,
        "zscore": None,
        "recommendation": "NO_DATA",
    }


def print_report(report: pd.DataFrame) -> None:
    print(DISCLAIMER)
    print(report.to_string(index=False))
