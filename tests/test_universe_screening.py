from screening.cointegration import screen_universe
from screening.universe import SECTOR_ETFS, default_universe, generate_candidate_pairs


def test_generate_candidate_pairs_sector_grouped_and_known_pairs_included():
    pairs = generate_candidate_pairs(group_by_sector=True)

    assert ("XLE", "XLF") not in pairs  # cross-sector, not a known pair
    assert ("KO", "PEP") in pairs
    assert len(pairs) == len(set(pairs))
    assert all(a < b for a, b in pairs)


def test_default_universe_covers_all_sector_tickers():
    universe = default_universe()
    all_tickers = {t for group in SECTOR_ETFS.values() for t in group}

    assert set(universe) == all_tickers


def test_screen_universe_surfaces_injected_cointegrated_pair(sector_universe_fixture):
    panel, (ticker_a, ticker_b) = sector_universe_fixture
    pairs = [(ticker_a, ticker_b), ("CCC", "DDD"), ("AAA", "CCC"), ("BBB", "DDD")]

    results = screen_universe(panel, pairs, min_half_life_days=1.0, max_half_life_days=60.0)

    assert not results.empty
    top = results.iloc[0]
    assert {top["ticker_a"], top["ticker_b"]} == {ticker_a, ticker_b}
    assert bool(top["tradeable"])
