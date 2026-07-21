"""Tests for screening.book_refresh: the persistence ranking, one-per-sector
deduplication, near-twin exclusion, and current-book comparison logic.

The ranking/dedup/compare logic is exercised on small hand-built screen and
survival frames (no network, no heavy statsmodels calls), so the ordering and
flag rules are pinned deterministically. One end-to-end check runs the real
machinery on the synthetic sector_universe_fixture to confirm rank_pairs wires
screen_universe + pair_survival_study together and surfaces the injected pair.
"""

import numpy as np
import pandas as pd

from screening.book_refresh import (
    BookComparison,
    RefreshResult,
    build_ranking,
    compare_to_current_book,
    dedupe_one_per_sector,
    propose_book,
    rank_pairs,
)
from screening.focus_book import FocusPair

# ------------------------------------------------------------ hand-built frames


def _screen_row(ticker_a, ticker_b, adf_pvalue, half_life, tradeable=True):
    """One screen_universe-shaped row."""
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "adf_pvalue": adf_pvalue,
        "half_life_days": half_life,
        "bh_significant": tradeable,
        "oos_validated": tradeable,
        "tradeable": tradeable,
    }


def _survival_rows(ticker_a, ticker_b, holdout_pvalues, survived):
    """One pair_survival_study-shaped (window, pair) row per holdout p-value."""
    return [
        {
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "holdout_pvalue": p,
            "survived": s,
        }
        for p, s in zip(holdout_pvalues, survived)
    ]


def _make_frames():
    """Build a small screen + survival pair using real universe tickers so the
    sector map resolves. Persistence (formation-passes) is engineered so the
    intended ordering is unambiguous:
      COP/SLB (energy)      : 4 passes  -> should rank #1
      DUK/SO  (utilities)   : 3 passes  -> #2
      ABT/MRK (healthcare)  : 2 passes  -> #3
      JNJ/PFE (healthcare)  : 2 passes but weaker holdout p -> #4 (same sector as ABT/MRK)
    """
    screen = pd.DataFrame(
        [
            _screen_row("COP", "SLB", adf_pvalue=0.007, half_life=17.0),
            _screen_row("DUK", "SO", adf_pvalue=0.008, half_life=18.0),
            _screen_row("ABT", "MRK", adf_pvalue=0.006, half_life=17.0),
            _screen_row("JNJ", "PFE", adf_pvalue=0.010, half_life=20.0),
        ]
    )
    survival = pd.DataFrame(
        _survival_rows("COP", "SLB", [0.01, 0.02, 0.03, 0.04], [True, True, True, False])
        + _survival_rows("DUK", "SO", [0.02, 0.03, 0.05], [True, True, False])
        + _survival_rows("ABT", "MRK", [0.01, 0.02], [True, True])
        + _survival_rows("JNJ", "PFE", [0.04, 0.05], [False, False])
    )
    return screen, survival


# ------------------------------------------------------------ ranking order


def test_ranking_respects_persistence_key():
    screen, survival = _make_frames()
    ranked = build_ranking(screen, survival)

    # formation-passes descending is the primary key.
    assert list(ranked["label"]) == ["COP/SLB", "DUK/SO", "ABT/MRK", "JNJ/PFE"]
    assert list(ranked["rank"]) == [1, 2, 3, 4]
    assert list(ranked["n_formation_passes"]) == [4, 3, 2, 2]
    # tie on formation-passes (ABT/MRK vs JNJ/PFE, both 2) breaks on the lower
    # median holdout p-value.
    assert ranked.loc[ranked["label"] == "ABT/MRK", "median_holdout_pvalue"].iloc[0] < (
        ranked.loc[ranked["label"] == "JNJ/PFE", "median_holdout_pvalue"].iloc[0]
    )


def test_zero_formation_passes_sink_below_survivors():
    screen = pd.DataFrame(
        [
            _screen_row("COP", "SLB", adf_pvalue=0.001, half_life=17.0),  # great adf, no persistence
            _screen_row("DUK", "SO", adf_pvalue=0.040, half_life=18.0),   # weaker adf, has persistence
        ]
    )
    survival = pd.DataFrame(_survival_rows("DUK", "SO", [0.02, 0.03], [True, True]))
    ranked = build_ranking(screen, survival)
    # Persistence beats a strong single-window p-value: DUK/SO ranks first.
    assert list(ranked["label"]) == ["DUK/SO", "COP/SLB"]
    assert int(ranked.loc[ranked["label"] == "COP/SLB", "n_formation_passes"].iloc[0]) == 0


# ------------------------------------------------------------ exclusions


def test_near_twins_excluded():
    screen = pd.DataFrame(
        [
            _screen_row("GOOG", "GOOGL", adf_pvalue=0.0001, half_life=10.0),
            _screen_row("SPY", "IVV", adf_pvalue=0.0001, half_life=10.0),
            _screen_row("GLD", "IAU", adf_pvalue=0.0001, half_life=10.0),
            _screen_row("COP", "SLB", adf_pvalue=0.02, half_life=17.0),
        ]
    )
    ranked = build_ranking(screen, pd.DataFrame(columns=["ticker_a", "ticker_b", "holdout_pvalue", "survived"]))
    labels = set(ranked["label"])
    assert "GOOG/GOOGL" not in labels
    assert "SPY/IVV" not in labels
    assert "GLD/IAU" not in labels
    assert "COP/SLB" in labels  # a normal pair survives


def test_short_half_life_excluded():
    screen = pd.DataFrame(
        [
            _screen_row("COP", "SLB", adf_pvalue=0.01, half_life=3.0),   # < 5d -> excluded
            _screen_row("DUK", "SO", adf_pvalue=0.01, half_life=18.0),   # kept
        ]
    )
    ranked = build_ranking(screen, pd.DataFrame(columns=["ticker_a", "ticker_b", "holdout_pvalue", "survived"]))
    assert list(ranked["label"]) == ["DUK/SO"]


# ------------------------------------------------------------ dedup / propose


def test_one_per_sector_dedup():
    screen, survival = _make_frames()
    ranked = build_ranking(screen, survival)
    deduped = dedupe_one_per_sector(ranked)
    # ABT/MRK and JNJ/PFE are both healthcare; only the higher-ranked ABT/MRK survives.
    labels = list(deduped["label"])
    assert "ABT/MRK" in labels
    assert "JNJ/PFE" not in labels
    # Every sector bucket appears at most once.
    assert deduped["sector_key"].is_unique


def test_propose_book_caps_at_top_n():
    screen, survival = _make_frames()
    ranked = build_ranking(screen, survival)
    proposed = propose_book(ranked, top_n=2)
    assert len(proposed) == 2
    assert list(proposed["label"]) == ["COP/SLB", "DUK/SO"]


# ------------------------------------------------------------ compare


def test_compare_flags_dropped_member_and_challenger():
    screen, survival = _make_frames()
    ranked = build_ranking(screen, survival)
    proposed = propose_book(ranked, top_n=2)  # COP/SLB, DUK/SO
    result = RefreshResult(ranked=ranked, proposed=proposed, top_n=2)

    # Current book: ABT/MRK is a sitting member that is NOT in the top-2 proposed
    # -> should be flagged as not qualifying (drop). COP/SLB is a member that
    # stays. DUK/SO is a proposed pair NOT in the book -> a challenger.
    current_book = [
        FocusPair("COP", "SLB", "energy", "sitting member, still strong"),
        FocusPair("ABT", "MRK", "healthcare", "sitting member, slipping"),
    ]
    comparison = compare_to_current_book(result, current_book=current_book)
    assert isinstance(comparison, BookComparison)

    status = comparison.current_status.set_index("label")
    assert status.loc["COP/SLB", "qualifies"]          # kept
    assert not status.loc["ABT/MRK", "qualifies"]      # dropped (not in top-2)
    assert status.loc["ABT/MRK", "in_ranking"]         # present, just outranked

    challenger_labels = set(comparison.challengers["label"])
    assert "DUK/SO" in challenger_labels               # outranks ABT/MRK, not a member
    assert "COP/SLB" not in challenger_labels          # already a member, not a challenger


def test_compare_flags_member_absent_from_ranking():
    """A member excluded from the ranking entirely (e.g. near-twin / no data)
    is reported as not in the ranking and does not qualify."""
    screen, survival = _make_frames()
    ranked = build_ranking(screen, survival)
    proposed = propose_book(ranked, top_n=5)
    result = RefreshResult(ranked=ranked, proposed=proposed, top_n=5)

    current_book = [FocusPair("AAPL", "MSFT", "tech_megacap", "not in the fresh ranking at all")]
    comparison = compare_to_current_book(result, current_book=current_book)
    row = comparison.current_status.iloc[0]
    assert not row["in_ranking"]
    assert not row["qualifies"]
    assert pd.isna(row["current_rank"])


# ------------------------------------------------------------ end-to-end (real machinery, no network)


def test_rank_pairs_end_to_end_on_fixture(sector_universe_fixture):
    """rank_pairs wires screen_universe + pair_survival_study together on the
    synthetic panel and returns a well-formed ranking. (The fixture's injected
    AAA/BBB pair happens to have a full-window half-life just under 5 days, so
    the <5d exclusion legitimately drops it — which itself confirms the filter
    runs end to end.)"""
    panel, _injected = sector_universe_fixture
    pairs = [("AAA", "BBB"), ("CCC", "DDD")]
    result = rank_pairs(panel, pairs, top_n=5)

    assert isinstance(result, RefreshResult)
    # Ranks are contiguous and monotonic over the surviving candidates.
    assert result.ranked["rank"].is_monotonic_increasing
    assert list(result.ranked["rank"]) == list(range(1, len(result.ranked) + 1))
    # Every surviving pair clears the half-life floor the filter enforces.
    assert (result.ranked["half_life_days"].dropna() >= 5.0).all()
    # Proposed book is deduped one-per-sector and never exceeds the requested N.
    assert len(result.proposed) <= 5
    assert result.proposed["sector_key"].is_unique
