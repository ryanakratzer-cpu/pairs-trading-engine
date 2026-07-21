"""Tests for screening.focus_book: the fixed persistent-pair book's integrity
(structure, deduplication, no structural near-twins) and its helper accessors."""

from screening.focus_book import FOCUS_BOOK, FocusPair, focus_pairs, focus_tickers

# Structural near-twins the book must never contain: share classes and
# duplicate-index/duplicate-commodity ETFs whose spreads are trivially tight.
FORBIDDEN_TWINS = {
    frozenset({"GOOG", "GOOGL"}),
    frozenset({"SPY", "IVV"}),
    frozenset({"SPY", "VOO"}),
    frozenset({"IVV", "VOO"}),
    frozenset({"GLD", "IAU"}),
}


def test_book_is_nonempty_and_well_formed():
    assert len(FOCUS_BOOK) >= 3
    for pair in FOCUS_BOOK:
        assert isinstance(pair, FocusPair)
        assert pair.ticker_a and pair.ticker_b
        assert pair.ticker_a != pair.ticker_b
        assert pair.sector
        assert pair.rationale  # every member carries its evidence


def test_no_duplicate_pairs():
    keys = [frozenset(p.key) for p in FOCUS_BOOK]
    assert len(keys) == len(set(keys))


def test_one_pair_per_sector():
    """The whole point of the book: diversified, not five costumes on one
    sector bet. Each sector appears at most once."""
    sectors = [p.sector for p in FOCUS_BOOK]
    assert len(sectors) == len(set(sectors))


def test_no_structural_near_twins():
    for pair in FOCUS_BOOK:
        assert frozenset(pair.key) not in FORBIDDEN_TWINS, f"{pair.label} is a near-twin"


def test_focus_pairs_are_tuples():
    pairs = focus_pairs()
    assert len(pairs) == len(FOCUS_BOOK)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


def test_focus_tickers_deduplicated_and_sorted():
    tickers = focus_tickers()
    assert tickers == sorted(set(tickers))
    # Every ticker in the book appears in the flat list.
    for pair in FOCUS_BOOK:
        assert pair.ticker_a in tickers
        assert pair.ticker_b in tickers


def test_label_and_key_helpers():
    pair = FOCUS_BOOK[0]
    assert pair.label == f"{pair.ticker_a}/{pair.ticker_b}"
    assert pair.key == (pair.ticker_a, pair.ticker_b)
