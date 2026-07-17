"""Monte Carlo study of a single pair: fit an OU process to the current
trailing-window spread, simulate ~1000 forward paths, push each through the
signal state machine for a P&L distribution, and render the interactive
Plotly dashboard (fan chart, equity/drawdown, spread/z-score, P&L histogram)
plus a static PNG fallback for vault embeds.

Usage:  py run_montecarlo.py [TICKER_A TICKER_B] [--seed N]   (default: GDX GLD, seed 42)

Network-dependent (yfinance, cached). Signal/research only - no order is
ever placed by this tool. The default seed makes runs reproducible; pass
--seed to draw a different simulation.
"""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pandas as pd

from backtest.simulator import PairBacktestConfig, PairBacktester
from data.loader import align_and_clean, fetch_price_history
from montecarlo.simulator import fit_ou, simulate_spread_paths, simulate_strategy_pnl
from screening.cointegration import test_pair_cointegration
from signals.spread import SignalConfig
from visualization.interactive import (
    plot_interactive_equity,
    plot_interactive_spread_zscore,
    plot_monte_carlo_paths,
    plot_pnl_distribution,
)
from visualization.plots import plot_monte_carlo_paths_png

LOOKBACK_DAYS = 900
REGIME_WINDOW_DAYS = 252  # match PairBacktestConfig.recheck_window_days / run_live_monitor
N_PATHS = 1000
HORIZON_DAYS = 90
DEFAULT_SEED = 42
NOTIONAL = 10_000.0
DISCLAIMER = "SIGNAL ONLY - research simulation. No order is ever placed by this tool."


def build_trailing_spread(pair_prices: pd.DataFrame, ticker_a: str, ticker_b: str):
    """Spread on the trailing REGIME_WINDOW_DAYS bars - the same trailing-regime
    convention the backtester trades on and the live monitor z-scores against,
    so the OU fit describes the spread the strategy would actually see today.
    """
    window = pair_prices.iloc[-REGIME_WINDOW_DAYS:]
    coint = test_pair_cointegration(window[ticker_a], window[ticker_b], ticker_a, ticker_b)
    spread = (
        np.log(window[ticker_a]) - coint.hedge_ratio * np.log(window[ticker_b]) - coint.intercept
    )
    return spread, coint


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ticker_a, ticker_b = (args[0], args[1]) if len(args) >= 2 else ("GDX", "GLD")
    seed = DEFAULT_SEED
    for i, arg in enumerate(sys.argv):
        if arg == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])

    print(f"=== Monte Carlo pair study: {ticker_a} / {ticker_b} (seed={seed}) ===")
    print(DISCLAIMER)

    end = datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    raw = fetch_price_history([ticker_a, ticker_b], start=start, end=end)
    prices, _ = align_and_clean(raw)
    pair_prices = prices[[ticker_a, ticker_b]].dropna()
    print(f"\n[1/4] {len(pair_prices)} trading days of data ({start} to {end})")

    spread, coint = build_trailing_spread(pair_prices, ticker_a, ticker_b)
    ou = fit_ou(spread)
    half_life_text = f"{ou.half_life_days:.1f}" if ou.half_life_days is not None else "n/a"
    print(f"  trailing-{REGIME_WINDOW_DAYS}d hedge_ratio={coint.hedge_ratio:.4f}, adf_p={coint.adf_pvalue:.4f}")
    print(
        f"  OU fit: theta={ou.theta:.4f}/bar, mu={ou.mu:.5f}, sigma={ou.sigma:.5f}/bar, "
        f"implied half-life={half_life_text} days ({ou.n_obs} obs)"
    )

    half_life = ou.half_life_days or 25.0
    signal_config = SignalConfig(
        zscore_window=max(2 * int(round(half_life)), 20),
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        max_holding_bars=int(2.5 * half_life),
    )

    print(f"\n[2/4] Simulating {N_PATHS} OU paths, {HORIZON_DAYS} business days ahead...")
    paths = simulate_spread_paths(spread, n_paths=N_PATHS, horizon_days=HORIZON_DAYS, seed=seed, fit=ou)
    # Same paths pushed through the state machine twice: once with all costs
    # zeroed (gross) and once with the defaults that mirror the backtester's
    # cost convention (net). Gross stays printed for comparability with old
    # runs, but net is the honest headline - it is what the histogram shows.
    gross = simulate_strategy_pnl(
        paths, ou, signal_config, notional=NOTIONAL,
        transaction_cost_bps=0.0, slippage_bps=0.0, short_borrow_bps_annual=0.0,
    )
    net = simulate_strategy_pnl(paths, ou, signal_config, notional=NOTIONAL)
    for name, result in (("gross of costs", gross), ("net of costs", net)):
        print(f"  P&L per path, {name} (notional ${NOTIONAL:,.0f}):")
        print(f"    mean   = {result['mean']:+,.2f}")
        print(f"    median = {result['median']:+,.2f}")
        print(f"    5th pct  = {result['p05']:+,.2f}")
        print(f"    95th pct = {result['p95']:+,.2f}")
        print(f"    prob(profit) = {result['prob_profit']:.1%}")

    print("\n[3/4] Backtesting pair for the equity/spread dashboards...")
    backtest_config = PairBacktestConfig(
        capital_per_pair=NOTIONAL, max_concurrent_pairs=1, signal_config=signal_config,
    )
    backtest = PairBacktester(backtest_config).run(pair_prices, [(ticker_a, ticker_b)])
    pair_data = backtest["per_pair"].get((ticker_a, ticker_b))

    print("\n[4/4] Writing interactive HTML + static PNG to outputs/ ...")
    saved = [
        plot_monte_carlo_paths(
            spread, paths, ticker_a, ticker_b, signal_config=signal_config, ou_fit=ou,
        ),
        plot_pnl_distribution(
            net["pnl_per_path"], summary=net, label=f"{ticker_a}_{ticker_b}",
            title_note="net of costs",
        ),
        plot_interactive_equity(
            backtest["equity_curve"], backtest["trade_log"], label=f"{ticker_a}_{ticker_b}",
        ),
        plot_monte_carlo_paths_png(spread, paths, ticker_a, ticker_b),
    ]
    if pair_data is not None:
        saved.insert(
            3,
            plot_interactive_spread_zscore(
                pair_data["spread"].dropna(), pair_data["zscore"], ticker_a, ticker_b, signal_config,
            ),
        )
    for path in saved:
        print(f"  saved {path}")

    print("\nDone. Open the interactive_*.html files in a browser.")


if __name__ == "__main__":
    main()
