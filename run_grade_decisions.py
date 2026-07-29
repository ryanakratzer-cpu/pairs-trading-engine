"""Grade the engine's historical trading decisions: on the days it said BUY or
SELL, was it right?

This answers a question the equity curve does not. Sharpe and total return grade
the *portfolio*; this grades the *decisions* — every day |z| crossed the entry
band and the state machine took a position, plus every exit and stop-loss.

Two lenses, reported side by side because they disagree:
  OUTCOME — did the round-trip trade make money net of costs?
  THESIS  — did the spread actually mean-revert after entry?
            (sign(z_entry) * (z_exit - z_entry) < 0, same convention as the
            forward-test journal)

Network-dependent and non-deterministic. Signal research only: this script
never places an order or calls a broker.

Usage:
  py run_grade_decisions.py                  # the focus book (default)
  py run_grade_decisions.py GDX GLD          # one explicit pair
  py run_grade_decisions.py GDX GLD ABT MRK  # several pairs (as ticker pairs)
  py run_grade_decisions.py --lookback 1200  # longer history
  py run_grade_decisions.py --csv            # also write outputs/decision_grades.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

from backtest.simulator import PairBacktestConfig
from reporting.decision_grader import (
    DEFAULT_FORWARD_BARS,
    DEFAULT_LOOKBACK_DAYS,
    format_report,
    grade_focus_book,
)
from screening.focus_book import FOCUS_BOOK, focus_pairs

RISK_PROFILE = "conservative"  # matches run_focus_book.py so the grades describe the book as run
OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "decision_grades.csv"


def _parse_args(argv: list[str]) -> tuple[list[tuple[str, str]] | None, int, bool]:
    """-> (pairs or None for the focus book, lookback_days, write_csv)."""
    lookback = DEFAULT_LOOKBACK_DAYS
    write_csv = False
    tickers: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--lookback":
            if i + 1 >= len(argv):
                raise SystemExit("--lookback needs a number of days")
            lookback = int(argv[i + 1])
            i += 2
            continue
        if arg == "--csv":
            write_csv = True
            i += 1
            continue
        if arg.startswith("-"):
            raise SystemExit(f"unknown flag {arg!r} (see the module docstring for usage)")
        tickers.append(arg.upper())
        i += 1

    if not tickers:
        return None, lookback, write_csv
    if len(tickers) % 2 != 0:
        raise SystemExit(
            f"tickers must come in pairs; got {len(tickers)}: {' '.join(tickers)}"
        )
    pairs = [(tickers[i], tickers[i + 1]) for i in range(0, len(tickers), 2)]
    return pairs, lookback, write_csv


def main(argv: list[str] | None = None) -> None:
    pairs, lookback, write_csv = _parse_args(list(argv if argv is not None else sys.argv[1:]))

    print("=== Pairs Trading Engine — decision grader ===\n")
    if pairs is None:
        labels = ", ".join(p.label for p in FOCUS_BOOK)
        print(f"Grading the focus book ({len(focus_pairs())} pairs): {labels}")
    else:
        print(f"Grading {len(pairs)} explicit pair(s): "
              f"{', '.join(f'{a}/{b}' for a, b in pairs)}")
    print(f"Lookback {lookback} days, {RISK_PROFILE} risk profile, "
          f"{DEFAULT_FORWARD_BARS}-bar forward window for exit grading.\n")

    config = getattr(PairBacktestConfig, RISK_PROFILE)()
    print("[1/2] Fetching prices and replaying the engine's decisions...")
    result = grade_focus_book(pairs=pairs, lookback_days=lookback, config=config)
    print(f"  {len(result.decisions)} decision event(s) extracted, "
          f"{len(result.trade_log)} round-trip trade(s) in the log\n")

    print("[2/2] Grades\n")
    print(format_report(result))

    if not result.decisions.empty:
        print("\nFULL DECISION LOG")
        columns = ["date", "pair", "event", "zscore", "taken", "exit_date", "exit_event",
                   "z_exit", "thesis_move", "thesis_verdict", "pnl", "outcome_verdict",
                   "forward_move", "exit_verdict"]
        print(result.decisions[columns].to_string(index=False))

    if write_csv and not result.decisions.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        result.decisions.to_csv(OUTPUT_PATH, index=False)
        print(f"\n  saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
