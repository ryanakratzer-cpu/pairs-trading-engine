"""Forward-test journal runner: log today's signals, grade what's old enough.

Usage:  py run_journal.py [TICKER_A TICKER_B [TICKER_C TICKER_D ...]]
Default focus pair: GDX GLD.

Network-dependent (yfinance, with local CSV caching). Signal research only -
this script never places an order. The journal exists precisely to
forward-test the engine's calls on paper, out of sample, before trusting any
backtest number.
"""

from __future__ import annotations

import sys

from reporting.daily_report import generate_daily_signal_report, print_report
from reporting.journal import DEFAULT_JOURNAL_PATH, append_signals, grade_journal


def parse_pairs(argv: list[str]) -> list[tuple[str, str]]:
    """CLI tickers come flat (A B C D); pair them up. Default focus pair GDX GLD."""
    if not argv:
        return [("GDX", "GLD")]
    if len(argv) % 2 != 0:
        raise SystemExit("Tickers must come in pairs, e.g.: py run_journal.py GDX GLD XOM CVX")
    return [(argv[i].upper(), argv[i + 1].upper()) for i in range(0, len(argv), 2)]


def main() -> None:
    pairs = parse_pairs(sys.argv[1:])
    print("=== Forward-test signal journal (SIGNAL ONLY - no orders ever) ===\n")

    pair_labels = ", ".join(f"{a}/{b}" for a, b in pairs)
    print(f"[1/3] Generating today's signal report for: {pair_labels}")
    report = generate_daily_signal_report(pairs)
    print_report(report)

    print("\n[2/3] Appending to journal")
    n_new = append_signals(report)
    print(f"  {n_new} new row(s) written to {DEFAULT_JOURNAL_PATH}")

    print("\n[3/3] Grading journal entries whose outcome window has closed")
    graded, summary = grade_journal()
    if graded.empty:
        print("  journal is empty, nothing to grade")
    else:
        status_counts = graded["status"].value_counts().to_dict()
        print(f"  {len(graded)} journal row(s), statuses: {status_counts}")
    hit_rate = summary["hit_rate"]
    hit_rate_str = "n/a" if hit_rate is None else f"{hit_rate:.0%}"
    print(f"  summary: n_graded={summary['n_graded']}, hit_rate={hit_rate_str}")


if __name__ == "__main__":
    main()
