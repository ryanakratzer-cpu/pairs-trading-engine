"""Monthly focus-book refresh: rank candidate pairs by PERSISTENCE and PROPOSE
book changes, so membership stays an evidence decision rather than drift.

WHY this exists separately from run_screen.py and focus_book.py: the 2026-07-19
walk-forward run showed a ~7% out-of-sample survival rate — pairs that pass a
single day's screen mostly decay within the quarter, so chasing the day's top
p-values is churn. The focus book (screening/focus_book.py) is therefore chosen
for *persistence* across walk-forward windows, deduplicated to one pair per
sector. But a book selected once and never revisited silently goes stale as
relationships decay and new ones form: the whole justification for a fixed book
is fresh persistence evidence, and that evidence has a shelf life.

This module re-runs the SAME persistence machinery the book was built from
(screen_universe with FDR + out-of-sample validation, plus pair_survival_study
walk-forward) on fresh data, ranks every candidate pair by how many formation
windows it survived, dedupes to one pair per sector, and reports how the CURRENT
book stacks up against the fresh ranking — which members still qualify, which
would be dropped, and which challengers now outrank a sitting member.

It PROPOSES, never mutates. focus_book.py is a human/evidence decision and this
code never edits it, never places an order, and never calls a broker. Research
only: the output is an auditable monthly proposal for a human to act on.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.walkforward import WalkForwardConfig, pair_survival_study
from screening.cointegration import screen_universe
from screening.focus_book import FOCUS_BOOK, FocusPair
from screening.universe import SECTOR_ETFS

# Structural near-twins: same underlying economics by construction (share
# classes / duplicate-index / duplicate-commodity ETFs), so their spreads are
# trivially tight and the ADF test always loves them — but they diversify
# nothing and their edge is eaten by costs. Excluded from any proposed book,
# mirroring screening/focus_book.py's selection rule.
NEAR_TWINS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"GOOG", "GOOGL"}),
        frozenset({"SPY", "IVV"}),
        frozenset({"SPY", "VOO"}),
        frozenset({"IVV", "VOO"}),
        frozenset({"GLD", "IAU"}),
    }
)

# Sector/index ETF tickers in the universe. A single stock paired with an ETF
# that HOLDS it (e.g. D/XLU — Dominion is a constituent of the utilities SPDR)
# is a stock-vs-own-index basis relationship: its cointegration is partly a
# mechanical accounting identity (the ETF return already contains the stock),
# it diversifies nothing, and it "breaks" on index rebalancing rather than
# fundamentals. It is the same pathology NEAR_TWINS exists for — the council of
# 2026-07-21 flagged that the frozenset above only catches ETF-vs-ETF and
# share-class twins, missing stock-vs-own-sector-ETF — so it is excluded by the
# same rule. See _is_stock_vs_own_sector_etf.
SECTOR_ETF_TICKERS: frozenset[str] = frozenset(
    {
        "XLE", "XLF", "XLP", "XLI", "XLK", "XLV", "XLU", "SMH",  # sector SPDRs / semis
        "SPY", "IVV", "VOO",                                      # broad-market index
        "GLD", "SLV", "GDX", "IAU",                               # commodity/metals
        "TLT", "IEF", "LQD", "HYG",                               # rates/credit
    }
)

MIN_HALF_LIFE_DAYS = 5.0
DEFAULT_TOP_N = 5
# A pair must clear ADF at this level on the FULL window to count as genuinely
# cointegrated for ranking. Matches the project-wide default significance.
COINT_SIGNIFICANCE = 0.05


@dataclass(frozen=True)
class RefreshResult:
    """The fresh persistence evidence, as two views of the same run.

    `ranked` is every surviving candidate pair scored and ordered by the
    persistence key (all sectors, before deduplication) with a 1-based `rank`.
    `proposed` is the deduplicated one-pair-per-sector top-N — the book the
    fresh evidence would build from scratch today.
    """

    ranked: pd.DataFrame
    proposed: pd.DataFrame
    top_n: int


@dataclass(frozen=True)
class BookComparison:
    """Current book measured against the fresh ranking.

    `current_status` is one row per sitting FOCUS_BOOK member with its fresh
    rank, evidence, and qualify/drop flags. `challengers` is the proposed-book
    pairs that are NOT current members — the pairs that would newly enter.
    """

    current_status: pd.DataFrame
    challengers: pd.DataFrame


def _canonical(ticker_a: str, ticker_b: str) -> tuple[str, str]:
    """Order-independent key for a pair, so (A, B) and (B, A) match."""
    return tuple(sorted((ticker_a, ticker_b)))


def _ticker_sector_map(universe: dict[str, list[str]] | None = None) -> dict[str, str]:
    """Invert SECTOR_ETFS to ticker -> sector. First sector wins on the rare
    ticker that appears in two groups; unmapped tickers resolve to 'unknown'
    at lookup time."""
    universe = SECTOR_ETFS if universe is None else universe
    mapping: dict[str, str] = {}
    for sector, tickers in universe.items():
        for ticker in tickers:
            mapping.setdefault(ticker, sector)
    return mapping


def _sector_key(ticker_a: str, ticker_b: str, mapping: dict[str, str]) -> str:
    """Sector bucket a pair dedupes into. Same-sector legs key on that sector;
    cross-sector legs key on the stable sorted combination of both sectors, so
    a genuinely cross-sector relationship isn't collapsed against unrelated
    same-sector pairs."""
    sector_a = mapping.get(ticker_a, "unknown")
    sector_b = mapping.get(ticker_b, "unknown")
    if sector_a == sector_b:
        return sector_a
    return "|".join(sorted((sector_a, sector_b)))


def _is_near_twin(ticker_a: str, ticker_b: str) -> bool:
    return frozenset({ticker_a, ticker_b}) in NEAR_TWINS


def _is_stock_vs_own_sector_etf(
    ticker_a: str, ticker_b: str, universe: dict[str, list[str]] | None = None
) -> bool:
    """True when one leg is a sector/index ETF and the other leg sits in that
    ETF's own SECTOR_ETFS group — i.e. the ETF holds the stock (D/XLU). This is
    a disguised near-twin: a stock-vs-own-index basis position whose spread is
    mechanically tight rather than a genuine cross-firm relationship, so it is
    excluded from any proposed book on the same structural grounds as the
    explicit NEAR_TWINS."""
    universe = SECTOR_ETFS if universe is None else universe
    for etf, other in ((ticker_a, ticker_b), (ticker_b, ticker_a)):
        if etf not in SECTOR_ETF_TICKERS:
            continue
        for members in universe.values():
            if etf in members and other in members:
                return True
    return False


def _aggregate_survival(survival: pd.DataFrame) -> pd.DataFrame:
    """Collapse pair_survival_study's (window, pair) rows to one row per pair:
    formation-passes (window count), holdout survivals, and median holdout
    p-value — the persistence signals the ranking is built on."""
    columns = ["ticker_a", "ticker_b", "n_formation_passes", "n_holdout_survivals", "median_holdout_pvalue"]
    if survival.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        survival.groupby(["ticker_a", "ticker_b"])
        .agg(
            n_formation_passes=("survived", "size"),
            n_holdout_survivals=("survived", "sum"),
            median_holdout_pvalue=("holdout_pvalue", "median"),
        )
        .reset_index()
    )
    grouped["n_holdout_survivals"] = grouped["n_holdout_survivals"].astype(int)
    return grouped


def build_ranking(
    screen: pd.DataFrame,
    survival: pd.DataFrame,
    universe: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Join the full-window screen with the walk-forward survival study and rank
    every candidate pair by persistence.

    Per pair the join carries: full-window adf_pvalue, half_life_days,
    bh_significant, oos_validated (from `screen`) and n_formation_passes,
    n_holdout_survivals, median_holdout_pvalue (from `survival`). `passes_screen`
    mirrors the screen's `tradeable` flag (cointegrated + half-life band + FDR +
    out-of-sample validated).

    Ranking key (cointegration GATE first, then persistence, then evidence):
      0. is_cointegrated_full descending — a pair that does not cointegrate on
         the full window (and thus has no established mean-reversion to trade)
         can NEVER outrank one that does, regardless of formation-pass count.
         This closes the artifact the 2026-07-21 council found: D/XLU (ADF
         p=0.70, no half-life) was floating above DUK/SO (p=0.006) purely on
         formation-passes + median-holdout-p, despite not cointegrating at all.
      1. n_formation_passes   descending — survived the most walk-forward windows
      2. median_holdout_pvalue ascending — held up hardest out of sample
      3. adf_pvalue           ascending — strongest full-window cointegration

    Structural near-twins, stock-vs-own-sector-ETF pairs, and pairs with
    half-life < 5 days are excluded before ranking. `rank` is 1-based over the
    surviving candidates; `is_sector_leader` marks the top-ranked pair in each
    sector bucket (the dedup survivor).
    """
    if screen.empty:
        return pd.DataFrame(
            columns=[
                "ticker_a", "ticker_b", "label", "sector_a", "sector_b", "sector_key",
                "adf_pvalue", "half_life_days", "bh_significant", "oos_validated",
                "passes_screen", "is_cointegrated_full", "n_formation_passes",
                "n_holdout_survivals", "median_holdout_pvalue", "rank", "is_sector_leader",
            ]
        )

    mapping = _ticker_sector_map(universe)
    agg = _aggregate_survival(survival)

    df = screen.merge(agg, on=["ticker_a", "ticker_b"], how="left")
    df["n_formation_passes"] = df["n_formation_passes"].fillna(0).astype(int)
    df["n_holdout_survivals"] = df["n_holdout_survivals"].fillna(0).astype(int)
    # median_holdout_pvalue stays NaN for pairs that never passed a formation
    # window — na_position='last' keeps them below any pair with evidence.

    # passes_screen mirrors the screen's own tradeable verdict when present.
    df["passes_screen"] = df["tradeable"].astype(bool) if "tradeable" in df.columns else False

    # Full-window cointegration gate: use the screen's own is_cointegrated flag
    # when present, else fall back to the ADF p-value; require a computable
    # half-life too (no half-life => no established mean-reversion to trade).
    if "is_cointegrated" in df.columns:
        coint = df["is_cointegrated"].astype(bool)
    else:
        coint = df["adf_pvalue"] < COINT_SIGNIFICANCE
    df["is_cointegrated_full"] = coint & df["half_life_days"].notna()

    # Exclusions: structural near-twins, stock-vs-own-sector-ETF basis pairs,
    # and sub-5-day half-lives. A NaN half-life (pair not cointegrated on the
    # full window) is NOT excluded here — it simply carries no half-life
    # evidence and, now, sinks below every cointegrated pair on the ranking key.
    df = df[~df.apply(lambda r: _is_near_twin(r["ticker_a"], r["ticker_b"]), axis=1)]
    df = df[~df.apply(lambda r: _is_stock_vs_own_sector_etf(r["ticker_a"], r["ticker_b"], universe), axis=1)]
    df = df[~(df["half_life_days"] < MIN_HALF_LIFE_DAYS)]

    df["label"] = df["ticker_a"] + "/" + df["ticker_b"]
    df["sector_a"] = df["ticker_a"].map(lambda t: mapping.get(t, "unknown"))
    df["sector_b"] = df["ticker_b"].map(lambda t: mapping.get(t, "unknown"))
    df["sector_key"] = df.apply(lambda r: _sector_key(r["ticker_a"], r["ticker_b"], mapping), axis=1)

    df = df.sort_values(
        by=["is_cointegrated_full", "n_formation_passes", "median_holdout_pvalue", "adf_pvalue"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    df["rank"] = df.index + 1

    # A pair leads its sector iff it is the highest-ranked (first) in its bucket.
    df["is_sector_leader"] = ~df["sector_key"].duplicated(keep="first")

    columns = [
        "ticker_a", "ticker_b", "label", "sector_a", "sector_b", "sector_key",
        "adf_pvalue", "half_life_days", "bh_significant", "oos_validated",
        "passes_screen", "is_cointegrated_full", "n_formation_passes",
        "n_holdout_survivals", "median_holdout_pvalue", "rank", "is_sector_leader",
    ]
    return df[columns]


def dedupe_one_per_sector(ranked: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-ranked pair per sector bucket. `ranked` must already be
    sorted by rank (build_ranking guarantees this), so 'first per sector_key'
    is the persistence winner for that sector."""
    if ranked.empty:
        return ranked.copy()
    return ranked.drop_duplicates(subset="sector_key", keep="first").reset_index(drop=True)


def propose_book(ranked: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """The one-pair-per-sector top-N the fresh evidence would build today."""
    return dedupe_one_per_sector(ranked).head(top_n).reset_index(drop=True)


def rank_pairs(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    top_n: int = DEFAULT_TOP_N,
    wf_config: WalkForwardConfig | None = None,
    universe: dict[str, list[str]] | None = None,
) -> RefreshResult:
    """Full refresh ranking: run the persistence machinery on `prices`/`pairs`
    and return the ranked candidates plus the proposed one-per-sector top-N.

    Uses the exact same knobs the book was selected under: FDR correction and
    out-of-sample validation on the full-window screen, and the default
    WalkForwardConfig for the survival study.
    """
    wf_config = wf_config or WalkForwardConfig()
    screen = screen_universe(
        prices,
        pairs,
        apply_multiple_testing_correction=True,
        require_out_of_sample_validation=True,
    )
    survival = pair_survival_study(prices, pairs, config=wf_config)
    ranked = build_ranking(screen, survival, universe=universe)
    proposed = propose_book(ranked, top_n=top_n)
    return RefreshResult(ranked=ranked, proposed=proposed, top_n=top_n)


def compare_to_current_book(
    result: RefreshResult,
    current_book: list[FocusPair] | None = None,
) -> BookComparison:
    """Measure the current FOCUS_BOOK against the fresh ranking.

    For each sitting member, report its fresh rank, whether it still passes the
    screen, whether it still lands in the proposed one-per-sector top-N, and a
    combined `qualifies` flag (both must hold). A member absent from the ranking
    (excluded as a near-twin, sub-5-day half-life, or too little data) is
    reported as dropped. `challengers` lists the proposed-book pairs that are
    NOT current members — the pairs that now outrank a sitting member and would
    newly enter the book.
    """
    current_book = FOCUS_BOOK if current_book is None else current_book
    ranked = result.ranked
    proposed = result.proposed

    proposed_keys = {
        _canonical(r["ticker_a"], r["ticker_b"]) for _, r in proposed.iterrows()
    }
    current_keys = {_canonical(p.ticker_a, p.ticker_b) for p in current_book}

    lookup = {
        _canonical(r["ticker_a"], r["ticker_b"]): r for _, r in ranked.iterrows()
    }

    status_rows = []
    for member in current_book:
        key = _canonical(member.ticker_a, member.ticker_b)
        row = lookup.get(key)
        if row is None:
            status_rows.append(
                {
                    "label": member.label,
                    "sector": member.sector,
                    "in_ranking": False,
                    "current_rank": pd.NA,
                    "passes_screen": False,
                    "in_proposed": False,
                    "qualifies": False,
                    "n_formation_passes": 0,
                    "n_holdout_survivals": 0,
                    "median_holdout_pvalue": pd.NA,
                    "adf_pvalue": pd.NA,
                    "half_life_days": pd.NA,
                }
            )
            continue
        in_proposed = key in proposed_keys
        passes_screen = bool(row["passes_screen"])
        status_rows.append(
            {
                "label": member.label,
                "sector": member.sector,
                "in_ranking": True,
                "current_rank": int(row["rank"]),
                "passes_screen": passes_screen,
                "in_proposed": in_proposed,
                "qualifies": in_proposed and passes_screen,
                "n_formation_passes": int(row["n_formation_passes"]),
                "n_holdout_survivals": int(row["n_holdout_survivals"]),
                "median_holdout_pvalue": row["median_holdout_pvalue"],
                "adf_pvalue": row["adf_pvalue"],
                "half_life_days": row["half_life_days"],
            }
        )

    current_status = pd.DataFrame(status_rows)

    challenger_mask = proposed.apply(
        lambda r: _canonical(r["ticker_a"], r["ticker_b"]) not in current_keys, axis=1
    ) if not proposed.empty else pd.Series(dtype=bool)
    challenger_cols = [
        "label", "sector_key", "rank", "passes_screen", "n_formation_passes",
        "n_holdout_survivals", "median_holdout_pvalue", "adf_pvalue", "half_life_days",
    ]
    if proposed.empty:
        challengers = pd.DataFrame(columns=challenger_cols)
    else:
        challengers = proposed[challenger_mask][challenger_cols].reset_index(drop=True)

    return BookComparison(current_status=current_status, challengers=challengers)


@dataclass(frozen=True)
class GovernanceConfig:
    """Hysteresis rule for acting on drift (2026-07-21 council recommendation).

    A sitting member is only recommended for REPLACE once the SAME challenger
    has out-ranked it (taken its sector slot in the proposed book) for
    `n_consecutive` monthly refreshes in a row AND that challenger passes the
    strict FDR+OOS screen. This prevents churning the book on a single noisy
    refresh — the whole reason a fixed book exists — while still surfacing
    genuine, sustained, screen-passing drift.
    """

    n_consecutive: int = 2


def _member_challenger(member: FocusPair, result: RefreshResult) -> tuple[str | None, bool]:
    """The proposed sector-leader that displaced `member` this run, plus whether
    that challenger passes the screen. Returns (None, False) when the member
    still leads its own sector bucket in the proposed book (nobody out-ranked
    it), or when it fell out of the top-N with no same-sector challenger."""
    member_key = _canonical(member.ticker_a, member.ticker_b)
    proposed = result.proposed
    if proposed.empty:
        return None, False
    for _, row in proposed.iterrows():
        if _canonical(row["ticker_a"], row["ticker_b"]) == member_key:
            return None, False  # member still holds its slot — not out-ranked
    for _, row in proposed.iterrows():
        if member.sector in str(row["sector_key"]).split("|"):
            return str(row["label"]), bool(row["passes_screen"])
    return None, False


def governance_actions(
    result: RefreshResult,
    history: list[dict] | None = None,
    current_book: list[FocusPair] | None = None,
    config: GovernanceConfig | None = None,
) -> pd.DataFrame:
    """Apply the hysteresis rule to each sitting member given prior refresh runs.

    `history` is the list of prior `build_history_record` dicts (oldest first;
    it must NOT already contain the current run). Returns one row per member:
    recommended_action in {KEEP, WATCH, REPLACE}, the current challenger (if
    any), whether it passes the screen, and the consecutive-run count backing a
    REPLACE. A member still leading its sector is KEEP; an out-ranked member is
    WATCH until the same screen-passing challenger has beaten it for
    `n_consecutive` runs, at which point it becomes REPLACE.
    """
    history = history or []
    current_book = FOCUS_BOOK if current_book is None else current_book
    config = config or GovernanceConfig()

    rows = []
    for member in current_book:
        challenger, passes = _member_challenger(member, result)
        if challenger is None:
            rows.append(
                {
                    "label": member.label, "sector": member.sector,
                    "recommended_action": "KEEP", "challenger": None,
                    "challenger_passes_screen": False, "consecutive_count": 0,
                }
            )
            continue

        # This run contributes to the streak only if the challenger passes the
        # screen; then count back over prior runs while the SAME challenger kept
        # winning AND passing. A gap (different challenger, or one that failed
        # the screen) resets the streak.
        streak = 1 if passes else 0
        if passes:
            for record in reversed(history):
                prior = record.get("members", {}).get(member.label)
                if prior and prior.get("challenger") == challenger and prior.get(
                    "challenger_passes_screen"
                ):
                    streak += 1
                else:
                    break
        action = "REPLACE" if streak >= config.n_consecutive else "WATCH"
        rows.append(
            {
                "label": member.label, "sector": member.sector,
                "recommended_action": action, "challenger": challenger,
                "challenger_passes_screen": passes, "consecutive_count": streak,
            }
        )
    return pd.DataFrame(rows)


def build_history_record(
    result: RefreshResult,
    date_str: str,
    current_book: list[FocusPair] | None = None,
) -> dict:
    """The persistable record of THIS run for the hysteresis history file: per
    member, which challenger (if any) out-ranked it and whether that challenger
    passed the screen. Appended to the history JSON so the next monthly run can
    measure consecutive-run streaks."""
    current_book = FOCUS_BOOK if current_book is None else current_book
    members = {}
    for member in current_book:
        challenger, passes = _member_challenger(member, result)
        members[member.label] = {
            "challenger": challenger,
            "challenger_passes_screen": bool(passes),
        }
    return {"date": date_str, "members": members}
