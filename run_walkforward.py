"""Walk-forward validation runner: the out-of-sample answers to the three
questions the in-sample pipeline can't honestly answer —

1. What fraction of screened pairs STAY cointegrated out-of-sample?
   (the screen's decay rate)
2. Do formation-optimized entry/exit/stop bands beat the textbook defaults
   on held-out data? (the overfitting tax)
3. Do optimized portfolio weights beat equal-weight out-of-sample?
   (the DeMiguel question, asked of our own book)

Network-dependent (live market data). Research only — no orders, no broker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime

from backtest.simulator import PairBacktestConfig, PairBacktester
from backtest.walkforward import (
    WalkForwardConfig,
    allocation_study,
    pair_survival_study,
    parameter_study,
)
from data.loader import align_and_clean, fetch_price_history
from screening.cointegration import screen_universe
from screening.universe import default_universe, generate_candidate_pairs

LOOKBACK_DAYS = 900
MAX_ALLOC_PAIRS = 8
STUDY_PAIR = ("GDX", "GLD")  # the project's focus pair, kept under scrutiny


def main() -> None:
    print("=== Pairs Trading Engine - walk-forward validation (live data) ===\n")

    end = datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    universe = default_universe()
    pairs = generate_candidate_pairs()
    config = WalkForwardConfig()

    print(f"[1/4] Fetching {len(universe)} tickers from {start} to {end}")
    prices, dropped = align_and_clean(fetch_price_history(universe, start=start, end=end))
    if dropped:
        print(f"  dropped thin-history tickers: {dropped}")
    n_windows = max(0, (len(prices) - config.formation_days - config.holdout_days) // config.step_days + 1)
    print(
        f"  {prices.shape[1]} tickers, {prices.shape[0]} trading days -> ~{n_windows} "
        f"walk-forward windows ({config.formation_days}d formation / {config.holdout_days}d holdout)"
    )

    print(f"\n[2/4] Pair survival: screen on formation, re-test on holdout ({len(pairs)} candidates)")
    survival = pair_survival_study(prices, pairs, config)
    if survival.empty:
        print("  no pair passed the formation screen in any window")
    else:
        rate = survival["survived"].mean()
        rate10 = (survival["holdout_pvalue"] < 0.10).mean()
        print(
            f"  {len(survival)} formation-passing (window, pair) observations; "
            f"OUT-OF-SAMPLE SURVIVAL RATE: {rate:.0%} at p<0.05 ({rate10:.0%} at p<0.10; "
            f"median holdout p={survival['holdout_pvalue'].median():.2f}). "
            f"ADF has low power on a {config.holdout_days}-bar holdout - treat these as "
            f"conservative lower bounds, most meaningful comparatively."
        )
        by_window = survival.groupby("window_end")["survived"].agg(["count", "mean"])
        print("  per window (count passing formation, fraction surviving holdout):")
        print(by_window.to_string())
        repeat = (
            survival.groupby(["ticker_a", "ticker_b"])["survived"]
            .agg(["count", "mean"])
            .sort_values(["count", "mean"], ascending=False)
        )
        print("  most persistent pairs (windows passed, survival rate):")
        print(repeat.head(10).to_string())

    print(f"\n[3/4] Parameter walk-forward on {STUDY_PAIR[0]}/{STUDY_PAIR[1]} "
          f"(choose bands on formation, measure on holdout)")
    params = parameter_study(prices[STUDY_PAIR[0]], prices[STUDY_PAIR[1]], config)
    if params.empty:
        print("  not enough data for a single window")
    else:
        print(params.to_string(index=False))
        mean_chosen = params["holdout_sharpe_chosen"].mean()
        mean_default = params["holdout_sharpe_default"].mean()
        tax = params["formation_sharpe_chosen"].mean() - mean_chosen
        print(
            f"  mean holdout Sharpe - formation-optimized: {mean_chosen:.2f} "
            f"vs textbook defaults: {mean_default:.2f}; overfitting tax "
            f"(formation minus holdout Sharpe of chosen bands): {tax:.2f}"
        )

    print(f"\n[4/4] Allocation walk-forward (weights fit on formation, applied frozen to holdout)")
    ranked = screen_universe(prices, pairs)
    tradeable = ranked[ranked["is_cointegrated"] & ranked["passes_half_life_filter"]].head(MAX_ALLOC_PAIRS)
    alloc_pairs = list(zip(tradeable["ticker_a"], tradeable["ticker_b"]))
    if len(alloc_pairs) < 2:
        print("  <2 screened pairs - skipping allocation study")
    else:
        print(f"  building causal strategy returns for {len(alloc_pairs)} pairs: "
              + ", ".join(f"{a}/{b}" for a, b in alloc_pairs))
        bt_config = PairBacktestConfig()
        panel = {}
        for ticker_a, ticker_b in alloc_pairs:
            result = PairBacktester(bt_config).run(prices, [(ticker_a, ticker_b)])
            panel[f"{ticker_a}/{ticker_b}"] = (
                result["equity_curve"].diff().dropna() / bt_config.capital_per_pair
            )
        returns_panel = pd.DataFrame(panel).fillna(0.0)
        alloc = allocation_study(returns_panel, config)
        if alloc.empty:
            print("  not enough overlapping return history for a window")
        else:
            agg = alloc.groupby("scheme")["holdout_sharpe"].agg(["count", "mean"])
            print(agg.to_string())
            wide = alloc.pivot(index="window_end", columns="scheme", values="holdout_sharpe")
            if {"tangency", "equal_weight"}.issubset(wide.columns):
                wins_tan = int((wide["tangency"] > wide["equal_weight"]).sum())
                print(f"  tangency beats 1/N in {wins_tan}/{len(wide)} holdout windows")
            if {"min_variance", "equal_weight"}.issubset(wide.columns):
                wins_mv = int((wide["min_variance"] > wide["equal_weight"]).sum())
                print(f"  min-variance beats 1/N in {wins_mv}/{len(wide)} holdout windows")

    print(
        "\nNOTE: with ~2.5 years of data these are few-window estimates - directionally "
        "informative, not statistically decisive. The survival rate is the number to "
        "watch: it bounds how much any signal/allocation cleverness can matter."
    )


if __name__ == "__main__":
    main()
