"""Efficient-frontier allocation across screened pairs: screen the (widened)
universe, treat each surviving pair's mean-reversion strategy as an asset,
build OU-derived expected returns + a Ledoit-Wolf-shrunk covariance, compute
the efficient frontier / tangency (max-Sharpe) / minimum-variance portfolios,
and compare them against equal-weight (1/N) on realized backtest returns.

Network-dependent and non-deterministic (live market data), like run_screen.py.
Research only — no orders, no broker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime

from backtest.simulator import PairBacktestConfig, PairBacktester
from data.loader import align_and_clean, fetch_price_history
from montecarlo.simulator import fit_ou
from portfolio.optimizer import (
    PortfolioPoint,
    annualize_inputs,
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    ou_expected_annual_return,
    random_portfolios,
)
from screening.cointegration import screen_universe
from screening.universe import default_universe, generate_candidate_pairs
from visualization.interactive import plot_allocation_comparison, plot_efficient_frontier

LOOKBACK_DAYS = 900
MAX_PAIRS = 8          # frontier assets: enough to diversify, few enough to read
MAX_WEIGHT = 0.40      # per-pair cap so one seductive OU fit can't take the book
RISK_FREE_RATE = 0.0   # project-wide Sharpe convention
TRADING_DAYS = 252


def _realized_stats(daily_returns: pd.Series) -> dict:
    ann_ret = float(daily_returns.mean() * TRADING_DAYS)
    ann_vol = float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    equity = (1.0 + daily_returns).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
        "max_drawdown": max_dd,
    }


def main() -> None:
    print("=== Pairs Trading Engine - efficient frontier allocation (live data) ===\n")

    end = datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    universe = default_universe()
    pairs = generate_candidate_pairs()

    print(f"[1/5] Fetching {len(universe)} tickers from {start} to {end}")
    prices, dropped = align_and_clean(fetch_price_history(universe, start=start, end=end))
    if dropped:
        print(f"  dropped thin-history tickers: {dropped}")
    print(f"  {prices.shape[1]} tickers, {prices.shape[0]} trading days retained")

    print(f"\n[2/5] Screening {len(pairs)} candidate pairs")
    results = screen_universe(
        prices, pairs,
        apply_multiple_testing_correction=True,
        require_out_of_sample_validation=True,
    )
    passes = results[results["is_cointegrated"] & results["passes_half_life_filter"]]
    oos = results[results["oos_validated"] == True]  # noqa: E712 (nullable bool column)
    print(
        f"  {len(passes)}/{len(results)} pass raw screen; "
        f"{int(results['bh_significant'].sum())} survive FDR; {len(oos)} survive out-of-sample"
    )
    if len(oos) >= 3:
        selected, note = oos, "out-of-sample validated"
    else:
        # The frontier needs a menu of assets; with <3 validated pairs fall
        # back to the raw pool, clearly labeled exploratory.
        selected, note = passes, "raw p<0.05 pool - EXPLORATORY, not OOS-validated"
    selected = selected.head(MAX_PAIRS)
    pair_keys = list(zip(selected["ticker_a"], selected["ticker_b"]))
    print(f"  allocating across {len(pair_keys)} pairs ({note}): "
          + ", ".join(f"{a}/{b}" for a, b in pair_keys))
    if len(pair_keys) < 2:
        print("  <2 pairs available - a frontier needs at least two assets. Exiting.")
        return

    print("\n[3/5] Building per-pair strategy returns (single-pair backtests) and OU expected returns")
    config = PairBacktestConfig()
    returns_by_pair: dict[str, pd.Series] = {}
    mu_ou: dict[str, float] = {}
    for (ticker_a, ticker_b), hedge_ratio in zip(pair_keys, selected["hedge_ratio"]):
        name = f"{ticker_a}/{ticker_b}"
        result = PairBacktester(config).run(prices, [(ticker_a, ticker_b)])
        daily_pnl = result["equity_curve"].diff().dropna()
        returns_by_pair[name] = daily_pnl / config.capital_per_pair
        spread = np.log(prices[ticker_a]) - hedge_ratio * np.log(prices[ticker_b])
        # Gate-aware mu: scale the OU cycle count by the fraction of days the
        # pair's re-cointegration gate actually allowed entries, so the model
        # expectation lives on the same scale as realized returns.
        prepared = result["per_pair"].get((ticker_a, ticker_b))
        tradeable_frac = float(prepared["tradeable"].mean()) if prepared is not None else 1.0
        mu_ou[name] = ou_expected_annual_return(
            fit_ou(spread.dropna()),
            config.signal_config,
            notional=config.capital_per_pair,
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
            tradeable_fraction=tradeable_frac,
        )
        n_trades = len(result["trade_log"])
        print(
            f"  {name}: OU expected return {mu_ou[name]:+.1%}/yr "
            f"(entry gate open {tradeable_frac:.0%} of days), {n_trades} backtest trades"
        )

    returns = pd.DataFrame(returns_by_pair).dropna()
    flat = [c for c in returns.columns if returns[c].std() == 0]
    if flat:
        print(f"  NOTE: zero-variance return series (never traded in-window): {flat}")
    asset_names = list(returns.columns)
    mu = np.array([mu_ou[name] for name in asset_names])

    mu_hist, cov, intensity = annualize_inputs(returns)
    print(f"  Ledoit-Wolf shrinkage intensity: {intensity:.2f} "
          f"(0 = sample cov, 1 = identity target)")
    print("  expected returns - OU model vs historical mean (OU drives the optimizer):")
    for name, m_ou, m_h in zip(asset_names, mu, mu_hist):
        print(f"    {name:12s}  OU {m_ou:+7.1%}   hist {m_h:+7.1%}")

    print(f"\n[4/5] Optimizing (long-only, {MAX_WEIGHT:.0%} per-pair cap, rf={RISK_FREE_RATE:.0%})")
    w_tan = max_sharpe_weights(mu, cov, RISK_FREE_RATE, MAX_WEIGHT)
    w_min = min_variance_weights(cov, MAX_WEIGHT)
    w_eq = np.full(len(mu), 1.0 / len(mu))
    named = {
        "tangency (max Sharpe)": PortfolioPoint.from_weights(w_tan, mu, cov, RISK_FREE_RATE),
        "minimum variance": PortfolioPoint.from_weights(w_min, mu, cov, RISK_FREE_RATE),
        "equal weight (1/N)": PortfolioPoint.from_weights(w_eq, mu, cov, RISK_FREE_RATE),
    }
    for label, point in named.items():
        weights_str = ", ".join(
            f"{n} {w:.0%}" for n, w in zip(asset_names, point.weights) if w >= 0.005
        )
        print(f"  {label:24s} ret {point.expected_return:+6.1%}  vol {point.volatility:6.1%}  "
              f"Sharpe {point.sharpe:5.2f}  [{weights_str}]")

    frontier = efficient_frontier(mu, cov, n_points=50, risk_free_rate=RISK_FREE_RATE, max_weight=MAX_WEIGHT)
    cloud = random_portfolios(mu, cov, n_portfolios=3000, risk_free_rate=RISK_FREE_RATE, max_weight=MAX_WEIGHT)
    frontier_path = plot_efficient_frontier(
        frontier, cloud, named, asset_names, RISK_FREE_RATE, label="live",
    )
    print(f"  saved {frontier_path}")

    print("\n[5/5] Realized comparison - the three weightings applied to actual backtest returns")
    realized_sharpes: dict[str, float] = {}
    equity_curves: dict[str, pd.Series] = {}
    for label, point in named.items():
        port_returns = (returns * point.weights).sum(axis=1)
        stats = _realized_stats(port_returns)
        realized_sharpes[label] = stats["sharpe"]
        equity_curves[label] = 100_000.0 * (1.0 + port_returns).cumprod()
        print(f"  {label:24s} realized: ret {stats['ann_return']:+6.1%}/yr  "
              f"vol {stats['ann_vol']:6.1%}  Sharpe {stats['sharpe']:5.2f}  "
              f"maxDD {stats['max_drawdown']:6.2%}")
    comparison_path = plot_allocation_comparison(equity_curves, realized_sharpes)
    print(f"  saved {comparison_path}")

    print(
        "\nNOTE: the OU expected returns and the covariance are estimated on the SAME "
        "window the realized comparison runs on - this is an in-sample sanity check of "
        "the allocator's machinery, not out-of-sample evidence that optimization beats "
        "1/N (see DeMiguel et al. 2009 for why it often doesn't)."
    )


if __name__ == "__main__":
    main()
