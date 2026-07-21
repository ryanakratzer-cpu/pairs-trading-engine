"""Monthly focus-book refresh runner: re-evaluate the fixed focus book
(screening/focus_book.py) against fresh walk-forward persistence evidence and
write a dated, auditable proposal to the vault.

This is the "is the book still the right book?" job, meant to run monthly. It
fetches the full universe exactly like run_screen.py (yfinance, 900-day
lookback), re-runs the SAME persistence machinery the book was selected under
(FDR + out-of-sample screen, plus the walk-forward survival study), ranks every
candidate pair, dedupes to one pair per sector, and reports how the current book
compares: which members still qualify, which would be dropped, and which
challenger pairs now outrank a sitting member.

It PROPOSES, never mutates. focus_book.py is a human/evidence decision — this
runner never edits it, never places an order, and never calls a broker. The
dated markdown report it leaves behind is the auditable monthly trail.

Network-dependent and non-deterministic (market data changes); not part of the
reproducible test path. See tests/test_book_refresh.py for the deterministic,
no-network verification of the ranking/dedup/compare logic.

Usage:
  py run_book_refresh.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from data.loader import align_and_clean, fetch_price_history
from screening.book_refresh import compare_to_current_book, rank_pairs
from screening.focus_book import FOCUS_BOOK
from screening.universe import default_universe, generate_candidate_pairs

LOOKBACK_DAYS = 900
TOP_N = 5
REPORT_DIR = Path(
    r"C:\Users\ryana\OneDrive\Desktop\Ryan's Obsidian\01_RAW_CLIPS\quant_research\book_refresh_reports"
)


def _fmt(value, spec: str = "") -> str:
    """Format a possibly-missing numeric cell for a table."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return "-"
    try:
        return format(value, spec) if spec else str(value)
    except (ValueError, TypeError):
        return str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Minimal GitHub-flavoured markdown table (no tabulate dependency)."""
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _current_status_rows(status: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, r in status.iterrows():
        verdict = "KEEP" if r["qualifies"] else "DROP?"
        rows.append(
            [
                r["label"],
                str(r["sector"]),
                _fmt(r["current_rank"]),
                _fmt(r["n_formation_passes"]),
                _fmt(r["n_holdout_survivals"]),
                _fmt(r["median_holdout_pvalue"], ".3f"),
                _fmt(r["adf_pvalue"], ".3f"),
                _fmt(r["half_life_days"], ".1f"),
                "yes" if r["passes_screen"] else "no",
                "yes" if r["in_proposed"] else "no",
                verdict,
            ]
        )
    return rows


def _proposed_rows(proposed: pd.DataFrame, current_labels: set[str]) -> list[list[str]]:
    rows = []
    for _, r in proposed.iterrows():
        rows.append(
            [
                _fmt(r["rank"]),
                r["label"],
                str(r["sector_key"]),
                _fmt(r["n_formation_passes"]),
                _fmt(r["n_holdout_survivals"]),
                _fmt(r["median_holdout_pvalue"], ".3f"),
                _fmt(r["adf_pvalue"], ".3f"),
                _fmt(r["half_life_days"], ".1f"),
                "yes" if r["passes_screen"] else "no",
                "current" if r["label"] in current_labels else "NEW",
            ]
        )
    return rows


def _build_markdown(date_str: str, result, comparison) -> str:
    status = comparison.current_status
    proposed = result.proposed
    challengers = comparison.challengers
    current_labels = set(status["label"])

    n_drop = int((~status["qualifies"]).sum())
    n_new = len(challengers)

    lines = [
        f"# Focus-book refresh — {date_str}",
        "",
        "**Proposal only.** This report re-evaluates the fixed focus book "
        "(`screening/focus_book.py`) against fresh walk-forward persistence "
        "evidence. It does not modify the book, place orders, or call a broker. "
        "Membership remains a human/evidence decision — use this as the "
        "auditable monthly trail.",
        "",
        f"- Lookback: {LOOKBACK_DAYS} days, top-N book size: {TOP_N}",
        f"- Current members flagged DROP?: **{n_drop}/{len(status)}**",
        f"- Challengers that would newly enter: **{n_new}**",
        "",
        "Ranking key: formation-passes (desc), then median holdout p-value "
        "(asc), then full-window ADF p-value (asc). Structural near-twins and "
        "sub-5-day half-lives are excluded; the proposed book is deduplicated "
        "to one pair per sector.",
        "",
        "## Current book — status vs fresh evidence",
        "",
        _md_table(
            ["pair", "sector", "rank", "form_passes", "holdout_surv",
             "med_holdout_p", "adf_p", "half_life_d", "passes_screen", "in_top_N", "verdict"],
            _current_status_rows(status),
        ),
        "",
        "`verdict = KEEP` when the member still lands in the proposed one-per-"
        "sector top-N AND still passes the screen; otherwise `DROP?` flags it "
        "for human review.",
        "",
        "## Proposed book — what the fresh evidence would build today",
        "",
        _md_table(
            ["rank", "pair", "sector_key", "form_passes", "holdout_surv",
             "med_holdout_p", "adf_p", "half_life_d", "passes_screen", "status"],
            _proposed_rows(proposed, current_labels),
        ),
        "",
    ]

    if n_new:
        lines += [
            "## Challengers — proposed pairs not currently in the book",
            "",
            _md_table(
                ["pair", "sector_key", "rank", "passes_screen", "form_passes",
                 "holdout_surv", "med_holdout_p", "adf_p", "half_life_d"],
                [
                    [
                        r["label"], str(r["sector_key"]), _fmt(r["rank"]),
                        "yes" if r["passes_screen"] else "no",
                        _fmt(r["n_formation_passes"]), _fmt(r["n_holdout_survivals"]),
                        _fmt(r["median_holdout_pvalue"], ".3f"),
                        _fmt(r["adf_pvalue"], ".3f"), _fmt(r["half_life_days"], ".1f"),
                    ]
                    for _, r in challengers.iterrows()
                ],
            ),
            "",
        ]
    else:
        lines += ["## Challengers", "", "None — every proposed member is already in the book.", ""]

    return "\n".join(lines)


def _print_summary(result, comparison) -> None:
    status = comparison.current_status
    print("\n--- Current book: status vs fresh evidence ---")
    for _, r in status.iterrows():
        verdict = "KEEP " if r["qualifies"] else "DROP?"
        rank = _fmt(r["current_rank"])
        print(
            f"  [{verdict}] {r['label']:12s} rank={rank:>4}  "
            f"form_passes={_fmt(r['n_formation_passes'])}  "
            f"holdout_surv={_fmt(r['n_holdout_survivals'])}  "
            f"med_holdout_p={_fmt(r['median_holdout_pvalue'], '.3f')}  "
            f"adf_p={_fmt(r['adf_pvalue'], '.3f')}  "
            f"passes_screen={'yes' if r['passes_screen'] else 'no'}"
        )

    print(f"\n--- Proposed book (top-{result.top_n}, one per sector) ---")
    current_labels = set(status["label"])
    for _, r in result.proposed.iterrows():
        tag = "current" if r["label"] in current_labels else "NEW"
        print(
            f"  #{int(r['rank']):<3} {r['label']:12s} [{r['sector_key']}]  "
            f"form_passes={_fmt(r['n_formation_passes'])}  "
            f"med_holdout_p={_fmt(r['median_holdout_pvalue'], '.3f')}  ({tag})"
        )

    print("\n--- Challengers (proposed pairs not in the current book) ---")
    if comparison.challengers.empty:
        print("  none — every proposed member is already in the book")
    else:
        for _, r in comparison.challengers.iterrows():
            print(
                f"  {r['label']:12s} rank={int(r['rank'])}  [{r['sector_key']}]  "
                f"form_passes={_fmt(r['n_formation_passes'])}  "
                f"passes_screen={'yes' if r['passes_screen'] else 'no'}"
            )


def main() -> None:
    print("=== Pairs Trading Engine — focus-book refresh (persistence re-evaluation) ===\n")
    print("PROPOSAL ONLY — this never edits focus_book.py, places an order, or calls a broker.\n")

    labels = ", ".join(p.label for p in FOCUS_BOOK)
    print(f"Current book ({len(FOCUS_BOOK)} pairs): {labels}\n")

    end = datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    universe = default_universe()
    pairs = generate_candidate_pairs()

    print(f"[1/3] Fetching {len(universe)} tickers from {start} to {end} via yfinance")
    prices, dropped = align_and_clean(fetch_price_history(universe, start=start, end=end))
    if dropped:
        print(f"  dropped thin-history tickers: {dropped}")
    print(f"  {prices.shape[1]} tickers, {prices.shape[0]} trading days retained")

    print(f"\n[2/3] Ranking {len(pairs)} candidate pairs by walk-forward persistence")
    result = rank_pairs(prices, pairs, top_n=TOP_N)
    comparison = compare_to_current_book(result)
    print(f"  {len(result.ranked)} pairs survived exclusions; proposed book has {len(result.proposed)} pairs")
    _print_summary(result, comparison)

    print("\n[3/3] Writing dated proposal report to the vault")
    date_str = datetime.today().strftime("%Y-%m-%d")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"book_refresh_{date_str}.md"
    report_path.write_text(_build_markdown(date_str, result, comparison), encoding="utf-8")
    print(f"  wrote {report_path}")


if __name__ == "__main__":
    main()
