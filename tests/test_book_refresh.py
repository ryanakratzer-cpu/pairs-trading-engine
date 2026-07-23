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
    GovernanceConfig,
    RefreshResult,
    build_history_record,
    build_ranking,
    compare_to_current_book,
    dedupe_one_per_sector,
    governance_actions,
    propose_book,
    rank_pairs,
)
from screening.focus_book import FocusPair

EMPTY_SURVIVAL = pd.DataFrame(columns=["ticker_a", "ticker_b", "holdout_pvalue", "survived"])

# ------------------------------------------------------------ hand-built frames


def _screen_row(ticker_a, ticker_b, adf_pvalue, half_life, tradeable=True, is_cointegrated=None):
    """One screen_universe-shaped row. `is_cointegrated` defaults to the ADF
    verdict at p<0.05 (what the real screen would produce), but can be forced —
    e.g. a pair with a strong-looking p-value that still isn't cointegrated."""
    if is_cointegrated is None:
        is_cointegrated = adf_pvalue < 0.05
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "adf_pvalue": adf_pvalue,
        "half_life_days": half_life,
        "is_cointegrated": is_cointegrated,
        "bh_significant": tradeable,
        "oos_validated": tradeable,
        "tradeable": tradeable,
    }


def _mk_result(proposed_rows):
    """A RefreshResult with a minimal proposed frame (ranked == proposed) for
    governance tests. Each row needs ticker_a/ticker_b/label/sector_key/passes_screen."""
    proposed = pd.DataFrame(proposed_rows)
    return RefreshResult(ranked=proposed, proposed=proposed, top_n=len(proposed_rows))


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
    ranked = build_ranking(screen, EMPTY_SURVIVAL)
    assert list(ranked["label"]) == ["DUK/SO"]


def test_stock_vs_own_sector_etf_excluded():
    """D/XLU (Dominion is a constituent of the utilities SPDR) is a disguised
    near-twin and must be excluded, while the genuine two-name DUK/SO stays.
    This is the exact case the 2026-07-21 council flagged."""
    screen = pd.DataFrame(
        [
            _screen_row("D", "XLU", adf_pvalue=0.70, half_life=None),  # stock vs own sector ETF
            _screen_row("DUK", "SO", adf_pvalue=0.006, half_life=18.0),
        ]
    )
    ranked = build_ranking(screen, EMPTY_SURVIVAL)
    labels = set(ranked["label"])
    assert "D/XLU" not in labels
    assert "DUK/SO" in labels


def test_cointegration_gate_is_primary_sort_key():
    """A pair that does NOT cointegrate on the full window cannot outrank one
    that does, even with more formation passes — the artifact the council found
    (D/XLU floating above DUK/SO on formation-count despite ADF p=0.70)."""
    screen = pd.DataFrame(
        [
            # Strong persistence but not cointegrated full-window, no half-life.
            _screen_row("COP", "SLB", adf_pvalue=0.70, half_life=None, is_cointegrated=False),
            # Fewer formation passes but genuinely cointegrated.
            _screen_row("DUK", "SO", adf_pvalue=0.006, half_life=18.0),
        ]
    )
    survival = pd.DataFrame(
        _survival_rows("COP", "SLB", [0.10, 0.10, 0.10, 0.10], [True, True, True, True])  # 4 passes
        + _survival_rows("DUK", "SO", [0.20, 0.20], [True, True])                          # 2 passes
    )
    ranked = build_ranking(screen, survival)
    assert list(ranked["label"]) == ["DUK/SO", "COP/SLB"]
    assert bool(ranked.loc[ranked["label"] == "DUK/SO", "is_cointegrated_full"].iloc[0]) is True
    assert bool(ranked.loc[ranked["label"] == "COP/SLB", "is_cointegrated_full"].iloc[0]) is False


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


# ------------------------------------------------------------ governance (hysteresis)


def _proposed_row(ticker_a, ticker_b, label, sector_key, passes_screen, rank=1):
    return {
        "ticker_a": ticker_a, "ticker_b": ticker_b, "label": label,
        "sector_key": sector_key, "passes_screen": passes_screen, "rank": rank,
    }


def test_governance_keep_when_member_still_leads_sector():
    # DUK/SO is itself the proposed utilities leader -> nobody out-ranked it.
    result = _mk_result([_proposed_row("DUK", "SO", "DUK/SO", "utilities", passes_screen=False)])
    book = [FocusPair("DUK", "SO", "utilities", "x")]
    acts = governance_actions(result, history=[], current_book=book).set_index("label")
    assert acts.loc["DUK/SO", "recommended_action"] == "KEEP"
    assert acts.loc["DUK/SO", "challenger"] is None


def test_governance_watch_when_challenger_fails_screen():
    # D/XLU out-ranks DUK/SO but does NOT pass the screen -> WATCH, never REPLACE,
    # no matter the history. This is exactly the live 2026-07-21 situation.
    result = _mk_result([_proposed_row("D", "XLU", "D/XLU", "utilities", passes_screen=False)])
    book = [FocusPair("DUK", "SO", "utilities", "x")]
    history = [
        {"date": "2026-06-20", "members": {"DUK/SO": {"challenger": "D/XLU", "challenger_passes_screen": False}}},
        {"date": "2026-05-20", "members": {"DUK/SO": {"challenger": "D/XLU", "challenger_passes_screen": False}}},
    ]
    acts = governance_actions(result, history=history, current_book=book).set_index("label")
    assert acts.loc["DUK/SO", "recommended_action"] == "WATCH"
    assert acts.loc["DUK/SO", "challenger"] == "D/XLU"
    assert int(acts.loc["DUK/SO", "consecutive_count"]) == 0


def test_governance_replace_after_n_consecutive_screen_passing():
    # Same screen-passing challenger this run + one prior run = 2 consecutive
    # -> REPLACE at the default n_consecutive=2.
    result = _mk_result([_proposed_row("AEP", "NEE", "AEP/NEE", "utilities", passes_screen=True)])
    book = [FocusPair("DUK", "SO", "utilities", "x")]
    history = [
        {"date": "2026-06-20", "members": {"DUK/SO": {"challenger": "AEP/NEE", "challenger_passes_screen": True}}},
    ]
    acts = governance_actions(result, history=history, current_book=book).set_index("label")
    assert acts.loc["DUK/SO", "recommended_action"] == "REPLACE"
    assert int(acts.loc["DUK/SO", "consecutive_count"]) == 2


def test_governance_streak_resets_on_different_challenger():
    # This run's challenger passes, but the prior run's challenger was different
    # -> streak is 1, below n_consecutive=2 -> WATCH.
    result = _mk_result([_proposed_row("AEP", "NEE", "AEP/NEE", "utilities", passes_screen=True)])
    book = [FocusPair("DUK", "SO", "utilities", "x")]
    history = [
        {"date": "2026-06-20", "members": {"DUK/SO": {"challenger": "D/XLU", "challenger_passes_screen": True}}},
    ]
    acts = governance_actions(result, history=history, current_book=book).set_index("label")
    assert acts.loc["DUK/SO", "recommended_action"] == "WATCH"
    assert int(acts.loc["DUK/SO", "consecutive_count"]) == 1


def test_governance_respects_custom_n_consecutive():
    result = _mk_result([_proposed_row("AEP", "NEE", "AEP/NEE", "utilities", passes_screen=True)])
    book = [FocusPair("DUK", "SO", "utilities", "x")]
    # No history, single run: streak 1. n_consecutive=1 -> REPLACE; default 2 -> WATCH.
    strict = governance_actions(result, history=[], current_book=book, config=GovernanceConfig(1))
    assert strict.set_index("label").loc["DUK/SO", "recommended_action"] == "REPLACE"
    lenient = governance_actions(result, history=[], current_book=book)
    assert lenient.set_index("label").loc["DUK/SO", "recommended_action"] == "WATCH"


def test_build_history_record_captures_challenger_and_screen_flag():
    result = _mk_result([_proposed_row("D", "XLU", "D/XLU", "utilities", passes_screen=False)])
    book = [FocusPair("DUK", "SO", "utilities", "x")]
    record = build_history_record(result, "2026-07-21", current_book=book)
    assert record["date"] == "2026-07-21"
    assert record["members"]["DUK/SO"]["challenger"] == "D/XLU"
    assert record["members"]["DUK/SO"]["challenger_passes_screen"] is False
