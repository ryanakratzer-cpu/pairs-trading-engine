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

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.loader import align_and_clean, fetch_price_history
from screening.book_refresh import (
    build_history_record,
    compare_to_current_book,
    governance_actions,
    rank_pairs,
)
from screening.focus_book import FOCUS_BOOK
from screening.universe import default_universe, generate_candidate_pairs

LOOKBACK_DAYS = 900
TOP_N = 5
REPORT_DIR = Path(
    r"C:\Users\ryana\OneDrive\Desktop\Ryan's Obsidian\01_RAW_CLIPS\quant_research\book_refresh_reports"
)
# Hysteresis history: the record of prior refreshes the governance rule reads to
# decide whether drift is sustained (same screen-passing challenger for N runs)
# rather than a one-month blip.
HISTORY_PATH = REPORT_DIR / "refresh_history.json"


def _load_history(today: str) -> list[dict]:
    """Prior refresh records, oldest first, EXCLUDING any record already stamped
    with today's date so a same-day re-run doesn't double-count the current run
    into its own streak."""
    if not HISTORY_PATH.exists():
        return []
    try:
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [r for r in history if r.get("date") != today]


def _save_history(history: list[dict], record: dict) -> None:
    """Append this run's record (replacing any existing same-date record) and
    persist, so the sequence stays one-per-date and idempotent per day."""
    kept = [r for r in history if r.get("date") != record["date"]]
    kept.append(record)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(kept, indent=2), encoding="utf-8")


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


def _governance_rows(governance: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, r in governance.iterrows():
        rows.append(
            [
                r["label"],
                str(r["sector"]),
                r["recommended_action"],
                _fmt(r["challenger"]) if r["challenger"] else "-",
                "yes" if r["challenger_passes_screen"] else "no",
                _fmt(r["consecutive_count"]),
            ]
        )
    return rows


def _build_markdown(date_str: str, result, comparison, governance) -> str:
    status = comparison.current_status
    proposed = result.proposed
    challengers = comparison.challengers
    current_labels = set(status["label"])

    n_drop = int((~status["qualifies"]).sum())
    n_new = len(challengers)
    n_replace = int((governance["recommended_action"] == "REPLACE").sum())

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
        f"- Current members flagged DROP? (this run alone): **{n_drop}/{len(status)}**",
        f"- Challengers that would newly enter: **{n_new}**",
        f"- **Governance recommendation: {n_replace}/{len(governance)} members REPLACE** "
        "(sustained, screen-passing drift across consecutive refreshes)",
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
        "for human review. (This column reacts to a SINGLE run — see the "
        "governance recommendation below for the hysteresis-filtered action.)",
        "",
        "## Governance recommendation (hysteresis — the action to actually take)",
        "",
        "A member is only recommended REPLACE once the SAME challenger has "
        "out-ranked it AND passed the screen for enough consecutive monthly "
        "refreshes; otherwise WATCH (drift noted, not yet acted on) or KEEP. "
        "This is what prevents churning the book on one noisy run.",
        "",
        _md_table(
            ["pair", "sector", "action", "challenger", "challenger_passes_screen", "consecutive_runs"],
            _governance_rows(governance),
        ),
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


def _print_summary(result, comparison, governance) -> None:
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

    print("\n--- Governance recommendation (hysteresis-filtered ACTION) ---")
    for _, r in governance.iterrows():
        ch = f" vs {r['challenger']} (passes_screen={'yes' if r['challenger_passes_screen'] else 'no'}, "
        ch += f"{int(r['consecutive_count'])} consecutive)" if r["challenger"] else ""
        detail = ch if r["challenger"] else "  (still leads its sector — nobody out-ranked it)"
        print(f"  [{r['recommended_action']:7s}] {r['label']:12s}{detail}")
    n_replace = int((governance["recommended_action"] == "REPLACE").sum())
    if n_replace == 0:
        print("  => No REPLACE recommended: keep the book as-is this cycle.")


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

    date_str = datetime.today().strftime("%Y-%m-%d")
    print(f"\n[2/3] Ranking {len(pairs)} candidate pairs by walk-forward persistence")
    result = rank_pairs(prices, pairs, top_n=TOP_N)
    comparison = compare_to_current_book(result)
    history = _load_history(date_str)
    governance = governance_actions(result, history=history)
    print(f"  {len(result.ranked)} pairs survived exclusions; proposed book has {len(result.proposed)} pairs")
    print(f"  loaded {len(history)} prior refresh record(s) for the governance streak")
    _print_summary(result, comparison, governance)

    print("\n[3/3] Writing dated proposal report + updating history")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"book_refresh_{date_str}.md"
    report_path.write_text(_build_markdown(date_str, result, comparison, governance), encoding="utf-8")
    print(f"  wrote {report_path}")
    _save_history(history, build_history_record(result, date_str))
    print(f"  updated {HISTORY_PATH}")


if __name__ == "__main__":
    main()
