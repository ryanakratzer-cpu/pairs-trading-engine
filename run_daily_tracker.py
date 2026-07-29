"""Persistent multi-day performance tracker runner.

Usage:
    py run_daily_tracker.py                 # focus book, record + grade + summary
    py run_daily_tracker.py GDX GLD         # explicit pair(s), flat ticker list
    py run_daily_tracker.py --summary       # print the track record only, record nothing

Run it once per day AFTER the close. Each run appends one row per pair for the
session (idempotent - re-running the same day writes nothing), grades the
decisions whose outcome window has now closed, and prints the accumulated
track record so you can see whether the engine is consistent over time.

SIGNAL RESEARCH ONLY: this script never places an order, never sizes a real
position, and never talks to a broker. The dollar figures are a paper track
record on a notional stake.

Network-dependent (yfinance, with local CSV caching).
"""

from __future__ import annotations

import sys

from reporting.daily_performance import (
    DEFAULT_PERFORMANCE_PATH,
    DISCLAIMER,
    STATUS_GRADED,
    grade_previous_sessions,
    performance_summary,
    record_session,
)
from screening.focus_book import focus_pairs

NOTIONAL = 10_000.0


def parse_pairs(argv: list[str]) -> list[tuple[str, str]]:
    """CLI tickers come flat (A B C D); pair them up. Default: the focus book."""
    tickers = [a for a in argv if not a.startswith("--")]
    if not tickers:
        return focus_pairs()
    if len(tickers) % 2 != 0:
        raise SystemExit(
            "Tickers must come in pairs, e.g.: py run_daily_tracker.py GDX GLD XOM CVX"
        )
    return [(tickers[i].upper(), tickers[i + 1].upper()) for i in range(0, len(tickers), 2)]


def _fmt(value, spec: str = ".2f", suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:{spec}}{suffix}"


def print_summary(summary: dict) -> None:
    if summary["n_sessions"] == 0:
        print("  no sessions recorded yet - run once after a close to start the record")
        return
    print(
        f"  sessions recorded : {summary['n_sessions']} "
        f"({summary['first_date']} .. {summary['last_date']}, "
        f"{summary['n_rows']} pair-rows)"
    )
    print(f"  cumulative P&L    : ${summary['cumulative_pnl_usd']:,.2f} on ${NOTIONAL:,.0f} notional/pair")
    # "(what we earned)" not "(pair spread)": this is the POSITION-AWARE return —
    # flat sessions contribute exactly 0%, so it agrees in sign with cumulative P&L.
    print(f"  cumulative return : {_fmt(summary['cumulative_pair_return_pct'], '+.3f', '%')} (what we earned)")
    print(f"  cumulative SPY    : {_fmt(summary['cumulative_spy_return_pct'], '+.3f', '%')}")
    print(
        f"  beat the market   : {summary['days_beating_market']}/{summary['n_sessions']} days "
        f"({_fmt(summary['pct_days_beating_market'], '.1f', '%')})"
    )
    print(f"  avg daily excess  : {_fmt(summary['avg_daily_excess_return_pct'], '+.4f', '%')}")
    best, worst = summary["best_day"], summary["worst_day"]
    if best:
        print(f"  best day          : {best['date']}  {best['pair_return_pct']:+.3f}%")
    if worst:
        print(f"  worst day         : {worst['date']}  {worst['pair_return_pct']:+.3f}%")
    print(f"  Sharpe (252, rf=0): {_fmt(summary['sharpe_ratio'], '.2f')}")


def print_grades(graded, summary: dict) -> None:
    if graded.empty:
        print("  nothing recorded yet, nothing to grade")
        return
    status_counts = graded["status"].value_counts().to_dict()
    print(f"  {len(graded)} recorded row(s), statuses: {status_counts}")

    done = graded[graded["status"] == STATUS_GRADED]
    if not done.empty:
        for row in done.tail(10).itertuples():
            verdict = "CORRECT" if bool(row.correct) else "wrong  "
            print(
                f"    {row.date}  {row.ticker_a}/{row.ticker_b:<5} "
                f"{row.signal:<19} -> {verdict}  "
                f"(spread_change_z={row.spread_change_z:+.3f})"
            )
    hit_rate = summary["hit_rate"]
    print(
        f"  verdict: n_graded={summary['n_graded']}, "
        f"hit_rate={'n/a' if hit_rate is None else f'{hit_rate:.0%}'}"
    )
    for decision, stats in sorted(summary["by_decision"].items()):
        print(f"    {decision:<19} n={stats['n']:<3} hit_rate={stats['hit_rate']:.0%}")


def main() -> None:
    argv = sys.argv[1:]
    summary_only = "--summary" in argv
    pairs = parse_pairs(argv)

    print("=== Multi-day performance tracker (SIGNAL RESEARCH ONLY - no orders ever) ===")
    print(DISCLAIMER)
    print(f"track record: {DEFAULT_PERFORMANCE_PATH}\n")

    if summary_only:
        print("[1/1] Cumulative track record (--summary: nothing recorded this run)")
        print_summary(performance_summary())
        return

    pair_labels = ", ".join(f"{a}/{b}" for a, b in pairs)
    print(f"[1/3] Recording today's session for: {pair_labels}")
    n_new = record_session(pairs, notional=NOTIONAL)
    if n_new:
        print(f"  {n_new} new row(s) written to {DEFAULT_PERFORMANCE_PATH}")
    else:
        print("  0 new rows - this session is already recorded (idempotent re-run)")

    print("\n[2/3] Grading decisions whose outcome window has closed")
    graded, grade_summary = grade_previous_sessions()
    print_grades(graded, grade_summary)

    print("\n[3/3] Cumulative track record")
    print_summary(performance_summary())


if __name__ == "__main__":
    main()
