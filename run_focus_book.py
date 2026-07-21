"""Focus-book runner: track the fixed, evidence-selected portfolio of the most
persistent cointegrated pairs (see screening/focus_book.py) as one book —
min-variance-weighted, macro/event-gated, with a daily signal report that
feeds the forward-test journal.

This is the "manage a diversified book of the best pairs" mode the walk-forward
evidence pointed to, as opposed to run_screen.py (rescreen the whole universe
daily) or run_journal.py GDX GLD (watch one pair). Signal research only — it
never places an order or calls a broker.

Usage:
  py run_focus_book.py            # backtest + allocate + today's signal report
  py run_focus_book.py --journal  # also append today's signals to the journal
  py run_focus_book.py --review   # print the evidence/rationale for each member
"""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pandas as pd

from backtest.metrics import compute_metrics
from backtest.simulator import PairBacktestConfig, PairBacktester
from data.loader import align_and_clean, fetch_price_history
from portfolio.optimizer import PortfolioPoint, annualize_inputs, min_variance_weights
from reporting.daily_report import generate_daily_signal_report, print_report
from reporting.journal import DEFAULT_JOURNAL_PATH, append_signals, grade_journal
from screening.events import event_exclusion_mask
from screening.focus_book import FOCUS_BOOK, focus_pairs, focus_tickers
from screening.regime import RegimeConfig, compute_stress_mask, fetch_macro_panel
from signals.spread import SignalConfig
from visualization.interactive import plot_allocation_comparison, plot_interactive_equity

LOOKBACK_DAYS = 900
MAX_WEIGHT = 0.40         # per-pair cap even in the focus book
RISK_PROFILE = "conservative"
TRADING_DAYS = 252


def print_review() -> None:
    print("=== Focus book — membership & evidence ===\n")
    print("A fixed, persistence-selected book (one pair per sector). Revisit when a")
    print("fresh walk-forward run changes the ranking. Research watchlist, not orders.\n")
    for i, pair in enumerate(FOCUS_BOOK, 1):
        print(f"  {i}. {pair.label:12s} [{pair.sector}]")
        print(f"       {pair.rationale}")
    print()


def _build_entry_gate(prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Macro stress mask AND event-exclusion windows, reindexed to the price
    dates. Degrades to event-only gating if the macro fetch fails, matching
    run_screen.py's fail-safe (missing macro defaults to calm)."""
    entries_allowed: pd.Series | None = None
    try:
        macro_panel = fetch_macro_panel(start, end)
        mask = compute_stress_mask(macro_panel, RegimeConfig())
        entries_allowed = mask.reindex(prices.index).fillna(True)
        n_stressed = int((~entries_allowed).sum())
        print(f"  [macro] stress mask blocks {n_stressed}/{len(entries_allowed)} days")
    except Exception as exc:
        print(f"  [macro] regime mask unavailable ({exc}); event-only gating")

    event_mask = event_exclusion_mask(prices.index)
    entries_allowed = event_mask if entries_allowed is None else (entries_allowed & event_mask)
    n_blocked = int((~entries_allowed).sum())
    print(f"  [gate] combined entry gate blocks {n_blocked}/{len(entries_allowed)} days")
    return entries_allowed


def main(journal: bool = False) -> None:
    print("=== Pairs Trading Engine — focus book (persistent-pair portfolio) ===\n")
    pairs = focus_pairs()
    labels = ", ".join(p.label for p in FOCUS_BOOK)
    print(f"Book ({len(pairs)} pairs, one per sector): {labels}\n")

    end = datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    print(f"[1/4] Fetching {len(focus_tickers())} tickers from {start} to {end}")
    prices, dropped = align_and_clean(fetch_price_history(focus_tickers(), start=start, end=end))
    if dropped:
        print(f"  dropped thin-history tickers: {dropped}")
    print(f"  {prices.shape[1]} tickers, {prices.shape[0]} trading days retained")
    entries_allowed = _build_entry_gate(prices, start, end)

    config = getattr(PairBacktestConfig, RISK_PROFILE)()

    print(f"\n[2/4] Backtesting the book as one portfolio ({RISK_PROFILE} profile, gated)")
    result = PairBacktester(config).run(prices, pairs, entries_allowed=entries_allowed)
    metrics = compute_metrics(result["equity_curve"], result["trade_log"])
    print(
        f"  trades={metrics['n_trades']}, total_return={metrics['total_return']:.2%}, "
        f"sharpe={metrics['sharpe_ratio']:.2f}, max_drawdown={metrics['max_drawdown']:.2%}, "
        f"win_rate={metrics['win_rate']:.0%}"
    )
    equity_plot = plot_interactive_equity(result["equity_curve"], result["trade_log"], label="focus_book")
    print(f"  saved {equity_plot}")

    print("\n[3/4] Min-variance allocation across the book (walk-forward's OOS winner)")
    per_pair = _per_pair_returns(prices, pairs, config)
    returns = pd.DataFrame(per_pair).dropna()
    live_pairs = [c for c in returns.columns if returns[c].std() > 0]
    if len(live_pairs) < 2:
        print(f"  only {len(live_pairs)} pair(s) traded in-window — allocation needs >=2; "
              f"skipping weights, book still tracked equal-notional")
    else:
        returns = returns[live_pairs]
        _mu_hist, cov, intensity = annualize_inputs(returns)
        weights = min_variance_weights(cov, MAX_WEIGHT)
        w_eq = np.full(len(live_pairs), 1.0 / len(live_pairs))
        mv_point = PortfolioPoint.from_weights(weights, _mu_hist, cov)
        eq_point = PortfolioPoint.from_weights(w_eq, _mu_hist, cov)
        print(f"  Ledoit-Wolf shrinkage intensity {intensity:.2f}; per-pair min-variance weights:")
        for name, w in zip(live_pairs, weights):
            print(f"    {name:12s} {w:6.1%}")
        print(f"  min-variance: vol {mv_point.volatility:.1%}  vs  equal-weight vol {eq_point.volatility:.1%}")

        equity_curves, realized_sharpes = {}, {}
        for label, wv in (("min_variance", weights), ("equal_weight", w_eq)):
            port_returns = (returns * wv).sum(axis=1)
            ann_ret = float(port_returns.mean() * TRADING_DAYS)
            ann_vol = float(port_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
            realized_sharpes[label] = ann_ret / ann_vol if ann_vol > 0 else 0.0
            equity_curves[label] = config.initial_capital * (1.0 + port_returns).cumprod()
        alloc_plot = plot_allocation_comparison(equity_curves, realized_sharpes)
        print(f"  realized Sharpe — min-variance {realized_sharpes['min_variance']:.2f}, "
              f"equal-weight {realized_sharpes['equal_weight']:.2f}")
        print(f"  saved {alloc_plot}")

    print("\n[4/4] Today's signal report for the book")
    report = generate_daily_signal_report(pairs, lookback_days=LOOKBACK_DAYS, signal_config=SignalConfig())
    print_report(report)
    if journal:
        n_new = append_signals(report)
        print(f"  [journal] {n_new} new row(s) appended to {DEFAULT_JOURNAL_PATH}")
        graded, summary = grade_journal()
        hit = "n/a" if summary["hit_rate"] is None else f"{summary['hit_rate']:.0%}"
        print(f"  [journal] graded={summary['n_graded']}, hit_rate={hit}")


def _per_pair_returns(prices: pd.DataFrame, pairs, config: PairBacktestConfig) -> dict:
    """Per-pair daily strategy returns (unit capital) from single-pair backtests
    — the panel the min-variance optimizer allocates over."""
    out = {}
    for ticker_a, ticker_b in pairs:
        single = PairBacktester(config).run(prices, [(ticker_a, ticker_b)])
        daily = single["equity_curve"].diff().dropna() / config.capital_per_pair
        out[f"{ticker_a}/{ticker_b}"] = daily
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--review" in args:
        print_review()
    else:
        main(journal="--journal" in args)
