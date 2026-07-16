"""Live screen: fetch real market data, screen for cointegration, backtest the
strongest candidates, and print today's signal report.

Network-dependent and non-deterministic (market data changes daily) — not part
of the reproducible test path; see run_demo.py for the deterministic,
no-network verification path. This script never places an order or calls a
broker: the final step is a computed signal report only.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from backtest.metrics import compute_metrics
from backtest.simulator import PairBacktestConfig, PairBacktester
from data.loader import align_and_clean, fetch_price_history
from reporting.daily_report import generate_daily_signal_report, print_report
from screening.cointegration import screen_universe
from screening.universe import default_universe, generate_candidate_pairs
from signals.spread import SignalConfig
from visualization.plots import plot_cointegration_heatmap, plot_equity_curve, plot_spread_and_zscore

LOOKBACK_DAYS = 900
TOP_N_TO_BACKTEST = 5


def main() -> None:
    print("=== Pairs Trading Engine - live screen (yfinance, non-deterministic) ===\n")

    end = datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    universe = default_universe()
    pairs = generate_candidate_pairs()

    print(f"[1/4] Fetching {len(universe)} tickers from {start} to {end} via yfinance")
    raw_prices = fetch_price_history(universe, start=start, end=end)
    prices, dropped = align_and_clean(raw_prices)
    if dropped:
        print(f"  dropped thin-history tickers: {dropped}")
    print(f"  {prices.shape[1]} tickers, {prices.shape[0]} trading days retained")

    print(f"\n[2/4] Screening {len(pairs)} candidate pairs for cointegration")
    results = screen_universe(prices, pairs)
    tradeable = results[results["tradeable"]]
    print(f"  {len(tradeable)}/{len(results)} pairs pass the cointegration + half-life filter")
    print(results.head(15).to_string(index=False))
    heatmap_path = plot_cointegration_heatmap(results.head(20), title="live_cointegration_pvalues")
    print(f"  saved {heatmap_path}")

    top_pairs = list(zip(tradeable["ticker_a"], tradeable["ticker_b"]))[:TOP_N_TO_BACKTEST]

    print(f"\n[3/4] Backtesting top {TOP_N_TO_BACKTEST} surviving pairs")
    if not top_pairs:
        print("  no pairs passed the screen - skipping backtest")
    else:
        backtest_config = PairBacktestConfig(max_concurrent_pairs=min(3, len(top_pairs)))
        result = PairBacktester(backtest_config).run(prices, top_pairs)
        metrics = compute_metrics(result["equity_curve"], result["trade_log"])
        print(
            f"  trades={metrics['n_trades']}, total_return={metrics['total_return']:.2%}, "
            f"sharpe={metrics['sharpe_ratio']:.2f}, max_drawdown={metrics['max_drawdown']:.2%}"
        )
        equity_plot = plot_equity_curve(result["equity_curve"], label="live_top_pairs")
        print(f"  saved {equity_plot}")
        for ticker_a, ticker_b in top_pairs:
            pair_data = result["per_pair"].get((ticker_a, ticker_b))
            if pair_data is None:
                continue
            spread_plot = plot_spread_and_zscore(
                pair_data["spread"], pair_data["zscore"], ticker_a, ticker_b, backtest_config.signal_config
            )
            print(f"  saved {spread_plot}")

    print("\n[4/4] Today's signal report")
    report_pairs = top_pairs or list(zip(tradeable["ticker_a"], tradeable["ticker_b"]))[:TOP_N_TO_BACKTEST]
    if report_pairs:
        report = generate_daily_signal_report(report_pairs, signal_config=SignalConfig())
        print_report(report)
    else:
        print("  no cointegrated pairs available for a signal report")


if __name__ == "__main__":
    main()
