"""Default candidate ticker universe and pair generation for cointegration screening."""

from __future__ import annotations

from itertools import combinations

SECTOR_ETFS: dict[str, list[str]] = {
    "energy": ["XLE", "XOM", "CVX", "COP", "SLB"],
    "refiners": ["MPC", "VLO", "PSX"],
    "banks_financials": ["XLF", "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW"],
    "consumer_staples": ["XLP", "KO", "PEP", "PG", "CL", "WMT", "COST", "MDLZ"],
    "metals_commodities": ["GLD", "SLV", "GDX", "IAU"],
    "industrials": ["XLI", "UPS", "FDX", "HD", "LOW", "CAT", "DE"],
    "payments": ["MA", "V", "AXP"],
    "tech_megacap": ["XLK", "AAPL", "MSFT", "GOOGL", "GOOG", "META", "ORCL"],
    "semis": ["SMH", "NVDA", "AMD", "AVGO", "TXN", "QCOM", "AMAT", "LRCX"],
    "healthcare": ["XLV", "JNJ", "MRK", "PFE", "ABT", "UNH", "TMO", "DHR"],
    "utilities": ["XLU", "NEE", "DUK", "SO", "D", "AEP"],
    "telecom": ["VZ", "T", "TMUS"],
    "broad_index": ["SPY", "IVV", "VOO"],
    # 2026-07 widening: the original 13 sectors produced only 2 out-of-sample
    # survivors on a live screen — too few assets for portfolio allocation to
    # matter. These groups add lower-correlation candidates (different
    # industries, plus rates/credit ETFs whose spreads are driven by curve
    # shape rather than equity beta).
    "airlines": ["DAL", "UAL", "LUV", "AAL"],
    "autos": ["F", "GM"],
    "discount_retail": ["DG", "DLTR", "TJX", "ROST", "TGT"],
    "insurance": ["MET", "PRU", "AIG", "ALL", "TRV"],
    "defense": ["LMT", "NOC", "RTX", "GD"],
    "homebuilders": ["DHI", "LEN", "PHM", "TOL"],
    "exchanges": ["CME", "ICE", "NDAQ", "CBOE"],
    "rates_credit": ["TLT", "IEF", "LQD", "HYG"],
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
    # Structural near-twins: same underlying economics by construction, so the
    # strongest cointegration candidates in the universe — but their spreads
    # are also the narrowest, so the half-life filter and transaction-cost
    # assumptions decide whether they're actually tradeable, not the ADF test.
    ("GOOG", "GOOGL"),  # Alphabet share classes
    ("GLD", "IAU"),     # two gold-bullion ETFs
    ("SPY", "IVV"),     # two S&P 500 ETFs
    ("SPY", "VOO"),
    ("IVV", "VOO"),
    ("AMAT", "LRCX"),   # semicap equipment duo
    ("CAT", "DE"),      # heavy machinery duo
    # Cross-sector economic-driver pairs (2026-07 widening): candidates whose
    # shared driver is a macro factor rather than sector membership, so their
    # spreads diversify a book of same-sector pairs.
    ("DAL", "UAL"),     # legacy carriers
    ("F", "GM"),        # Detroit duo
    ("DG", "DLTR"),     # dollar stores
    ("TJX", "ROST"),    # off-price retail
    ("MET", "PRU"),     # life insurers
    ("DHI", "LEN"),     # homebuilders
    ("RTX", "NOC"),     # defense primes
    ("CME", "ICE"),     # derivatives exchanges
    ("TLT", "IEF"),     # long vs intermediate Treasuries (curve trade)
    ("LQD", "HYG"),     # IG vs HY credit
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
