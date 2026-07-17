"""Deep-dive study of a single pair: cointegration diagnostics, then a
regime-OLS vs Kalman-filter hedge-ratio backtest comparison with a config
tuned to the pair's measured half-life.

Usage:  py run_pair_study.py [TICKER_A TICKER_B]   (default: GDX GLD)

Network-dependent (yfinance). Signal/backtest research only — never places
an order.
"""

from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd

from backtest.metrics import compute_metrics
from backtest.simulator import PairBacktestConfig, PairBacktester
from data.loader import align_and_clean, fetch_price_history
from screening.cointegration import test_pair_cointegration, validate_out_of_sample
from signals.spread import SignalConfig
from visualization.plots import plot_equity_curve, plot_spread_and_zscore

LOOKBACK_DAYS = 900


def tuned_config(half_life_days: float, hedge_ratio_mode: str) -> PairBacktestConfig:
    """Config tuned to the pair's measured mean-reversion speed:
    - z-score window ~2x half-life (long enough to be stable, short enough to adapt)
    - time exit at ~2.5x half-life (if it hasn't converged by then, the thesis failed)
    - conservative sizing/stops per the project's conservative risk preset.
    """
    half_life = max(int(round(half_life_days)), 5)
    return PairBacktestConfig(
        capital_per_pair=5_000.0,
        max_concurrent_pairs=1,
        hedge_ratio_mode=hedge_ratio_mode,
        signal_config=SignalConfig(
            zscore_window=max(2 * half_life, 20),
            entry_z=2.0,
            exit_z=0.5,
            stop_z=3.0,
            max_holding_bars=int(2.5 * half_life),
        ),
    )


def main() -> None:
    ticker_a, ticker_b = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("GDX", "GLD")
    print(f"=== Pair study: {ticker_a} / {ticker_b} (signal research only, no orders) ===\n")

    end = datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    raw = fetch_price_history([ticker_a, ticker_b], start=start, end=end)
    prices, _ = align_and_clean(raw)
    pair_prices = prices[[ticker_a, ticker_b]].dropna()
    print(f"[1/3] {len(pair_prices)} trading days of data ({start} to {end})")

    coint = test_pair_cointegration(pair_prices[ticker_a], pair_prices[ticker_b], ticker_a, ticker_b)
    oos = validate_out_of_sample(pair_prices[ticker_a], pair_prices[ticker_b], ticker_a, ticker_b)
    print(
        f"  full-window: p={coint.adf_pvalue:.4f}, hedge_ratio={coint.hedge_ratio:.3f}, "
        f"half_life={coint.half_life_days if coint.half_life_days is None else round(coint.half_life_days, 1)}d"
    )
    print(
        f"  out-of-sample: formation p={oos.formation_pvalue:.4f}, "
        f"validation p={oos.validation_pvalue:.4f}, validated={oos.out_of_sample_validated}"
    )

    half_life = coint.half_life_days or 25.0

    print(f"\n[2/3] Backtest comparison (config tuned to half-life ~{half_life:.0f}d)")
    comparison_rows = []
    for mode in ("regime", "kalman"):
        config = tuned_config(half_life, mode)
        result = PairBacktester(config).run(pair_prices, [(ticker_a, ticker_b)])
        metrics = compute_metrics(result["equity_curve"], result["trade_log"])
        comparison_rows.append({"hedge_mode": mode, **{k: round(v, 4) for k, v in metrics.items()}})

        label = f"{ticker_a}_{ticker_b}_{mode}"
        plot_equity_curve(result["equity_curve"], label=label)
        pair_data = result["per_pair"].get((ticker_a, ticker_b))
        if pair_data is not None and mode == "kalman":
            plot_spread_and_zscore(
                pair_data["spread"].dropna(), pair_data["zscore"], ticker_a, ticker_b, config.signal_config
            )
        if not result["trade_log"].empty:
            print(f"\n  --- {mode} trade log ---")
            print(result["trade_log"].to_string(index=False))

    print("\n  --- metrics comparison ---")
    print(pd.DataFrame(comparison_rows).to_string(index=False))

    print("\n[3/3] Plots saved to outputs/. Done.")


if __name__ == "__main__":
    main()
