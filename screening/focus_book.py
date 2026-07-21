"""The focus book: a fixed, evidence-selected portfolio of the most
*persistent* cointegrated pairs, one per sector.

Why a hardcoded book rather than "whatever the screen flags today": the
2026-07-19 walk-forward run showed a 7% out-of-sample survival rate — pairs
that pass a single day's screen mostly decay within the quarter, so chasing
the day's top p-values is churn. The pairs below were instead selected for
persistence across the walk-forward windows (how many formation windows each
survived) combined with a passing full-window screen, then deduplicated to
one pair per sector so the book is genuinely diversified rather than five
costumes on one insurance-sector bet.

Selection (see Pairs Trading Engine Implementation Log, 2026-07-19/20):
ranked by walk-forward formation-passes, excluding structural near-twins
(share classes / duplicate-index ETFs) and half-life < 5d. The insurance
cluster (ALL/TRV/AIG/PRU/MET all co-moved) contributes exactly one slot,
ALL/TRV — the only candidate that was also out-of-sample validated on the
2026-07-19 screen.

This is a research watchlist, not a trade list. Nothing here places an order.
Revisit the membership when a fresh walk-forward run materially changes the
persistence ranking; `run_focus_book.py --review` reprints the current
evidence for each member so drift is visible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FocusPair:
    ticker_a: str
    ticker_b: str
    sector: str
    rationale: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.ticker_a, self.ticker_b)

    @property
    def label(self) -> str:
        return f"{self.ticker_a}/{self.ticker_b}"


# Ordered by walk-forward persistence (formation-passes) at selection time.
FOCUS_BOOK: list[FocusPair] = [
    FocusPair(
        "ABT", "MRK", "healthcare",
        "large-cap healthcare (devices/diagnostics vs pharma); 4/5 formation "
        "passes, full-window p=0.006, half-life ~17d",
    ),
    FocusPair(
        "ALL", "TRV", "insurance",
        "P&C insurers; the insurance cluster's single slot — 4/5 formation "
        "passes AND out-of-sample validated 2026-07-19, half-life ~12d "
        "(shortest of the book), full-window p=0.009",
    ),
    FocusPair(
        "DUK", "SO", "utilities",
        "regulated southeastern utilities; 4/5 formation passes, "
        "full-window p=0.008, half-life ~18d",
    ),
    FocusPair(
        "COST", "PEP", "consumer_staples",
        "consumer staples (warehouse retail vs beverages/snacks); 3/5 "
        "formation passes, full-window p=0.047, half-life ~29d",
    ),
    FocusPair(
        "COP", "SLB", "energy",
        "energy (E&P vs oilfield services); 3/5 formation passes, "
        "full-window p=0.007, half-life ~17d",
    ),
]


def focus_pairs() -> list[tuple[str, str]]:
    """The book as plain (ticker_a, ticker_b) tuples for the backtester/report."""
    return [p.key for p in FOCUS_BOOK]


def focus_tickers() -> list[str]:
    """All distinct tickers in the book (deduplicated, sorted)."""
    return sorted({t for p in FOCUS_BOOK for t in p.key})
