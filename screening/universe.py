"""Default candidate ticker universe and pair generation for cointegration screening."""

from __future__ import annotations

from itertools import combinations

SECTOR_ETFS: dict[str, list[str]] = {
    "energy": ["XLE", "XOM", "CVX", "COP", "SLB"],
    "banks_financials": ["XLF", "JPM", "BAC", "WFC", "C", "GS", "MS"],
    "consumer_staples": ["XLP", "KO", "PEP", "PG", "CL", "WMT", "COST"],
    "metals_commodities": ["GLD", "SLV", "GDX"],
    "industrials": ["XLI", "UPS", "FDX", "HD", "LOW"],
    "payments": ["MA", "V"],
}

KNOWN_PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"),
    ("XOM", "CVX"),
    ("GLD", "SLV"),
    ("JPM", "BAC"),
    ("WFC", "C"),
    ("GS", "MS"),
    ("MA", "V"),
    ("HD", "LOW"),
    ("UPS", "FDX"),
]


def default_universe() -> list[str]:
    """All tickers across the default sector groups, deduplicated."""
    tickers: set[str] = set()
    for group in SECTOR_ETFS.values():
        tickers.update(group)
    return sorted(tickers)


def generate_candidate_pairs(
    universe: dict[str, list[str]] | None = None,
    group_by_sector: bool = True,
    known_pairs: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Generate candidate (ticker_a, ticker_b) pairs to screen for cointegration.

    When group_by_sector is True, pairs are only generated within each sector
    group (keeps the pair count well under O(n^2) for the full universe).
    Hand-picked known_pairs are always included regardless of grouping.
    """
    universe = SECTOR_ETFS if universe is None else universe
    known_pairs = KNOWN_PAIRS if known_pairs is None else known_pairs

    pairs: set[tuple[str, str]] = set()

    if group_by_sector:
        for group in universe.values():
            for a, b in combinations(sorted(group), 2):
                pairs.add((a, b))
    else:
        all_tickers = sorted({t for group in universe.values() for t in group})
        for a, b in combinations(all_tickers, 2):
            pairs.add((a, b))

    for a, b in known_pairs:
        pairs.add(tuple(sorted((a, b))))

    return sorted(pairs)
