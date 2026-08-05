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

RECONSTITUTED 2026-07-30 on much stricter evidence. The prior book was selected
on walk-forward persistence using p-values we later proved were biased: the
screen ran plain ADF on ESTIMATED residuals with standard critical values,
overstating significance 2-4x. Re-screened all 248 candidate pairs with the
PROPER Engle-Granger test on the trailing 252 days — the same window the engine
actually fits beta on — only 15 pairs survived, and only ONE member of the old
book (DUK/SO) was among them. ABT/MRK, COST/PEP, COP/SLB and ALL/TRV all failed
(EG p 0.11 to 0.56, and the first two additionally had NEGATIVE betas, meaning
they were never market-neutral at all).

Current admission rules, all four required:
  1. proper Engle-Granger p < 0.05 on the trailing 252d window (not full history,
     not the biased ADF-on-residuals p-value);
  2. hedge ratio POSITIVE and in [0.25, 4.0] — positive so the position is
     genuinely long-one/short-other, and bounded because a beta near zero
     (e.g. AIG/TRV at +0.063) leaves the trade ~94% net directional despite
     nominally being a pair;
  3. half-life inside the 5-30 day tradeable band;
  4. no structural near-twins and one pair per sector. This excludes
     stock-vs-own-sector-ETF constructions that the raw screen loved —
     CL/XLP (EG p=0.0027) and NVDA/SMH (0.0150) both ranked highly but CL is a
     constituent of XLP and NVDA of SMH, so their spreads are mechanically
     tight. It also collapses the AIG insurance cluster (AIG/PRU, AIG/ALL,
     AIG/MET, AIG/TRV) and the NVDA semis cluster to one slot each, and drops
     GDX/IAU as a duplicate of GDX/GLD (IAU and GLD are both gold bullion).

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


# ---------------------------------------------------------------------------
# 2026-08-05 BREADTH REBUILD — 12 pairs, up from 6.
#
# WHY: the binding constraint was never signal quality, it was shots on goal.
# The 6-pair book deployed ~1.3% of capital and traded ~6x/year, which needs
# ~91 trades (~15 years) to distinguish its best-case edge from zero. Trade
# count and capital utilisation scale with the number of QUALIFYING pairs, so
# the universe was widened 105 -> 303 tickers (248 -> 919 candidate pairs) and
# re-screened under IDENTICAL admission rules. Qualifiers went 15 -> 34.
#
# The beta band is also TIGHTENED here to [0.70, 1.40] (was [0.25, 4.0]).
# Reason: the engine z-scores on log(A) - beta*log(B) but holds equal dollars,
# so the traded portfolio only matches the signalled portfolio when beta is
# near 1. At beta 0.45 (the retired GD/RTX) those are different positions.
# Tightening drops 34 -> 15 candidates, from which this is one pair per sector.
#
# NOTE ON MULTIPLE TESTING, stated plainly: 919 tests at p<0.05 would throw off
# roughly 46 false positives by chance alone. The count that survived every
# filter is 34. These pairs are therefore NOT established discoveries — they are
# the best-evidenced candidates available, and the book's purpose is to generate
# enough trades to be measurable, not to assert an edge exists.
# ---------------------------------------------------------------------------
FOCUS_BOOK: list[FocusPair] = [
    FocusPair("FNV", "WPM", "gold_royalties",
        "gold royalty/streaming duo — identical business model, same metal, no "
        "mine-operating risk. Best evidence in the whole 919-pair screen. "
        "EG p=0.0007, beta +0.764, half-life 8.2d."),
    FocusPair("EPD", "OKE", "midstream",
        "midstream energy infrastructure — fee-based, volume-driven, same "
        "basins. EG p=0.0017, beta +0.857, half-life 5.4d."),
    FocusPair("AVGO", "NVDA", "semis",
        "semiconductor duo; carried over from the previous book. "
        "EG p=0.0124, beta +1.213, half-life 6.6d."),
    FocusPair("JNJ", "MRK", "pharma",
        "large-cap pharma; carried over from the previous book. "
        "EG p=0.0130, beta +0.792, half-life 6.4d."),
    FocusPair("AU", "NEM", "gold_miners",
        "gold producers — distinct from the royalty slot above (operating "
        "leverage vs none). EG p=0.0148, beta +1.222, half-life 6.5d."),
    FocusPair("ADBE", "CRM", "software",
        "enterprise SaaS majors — same seat-based subscription economics. "
        "EG p=0.0178, beta +1.079, half-life 6.6d."),
    FocusPair("CHD", "CL", "household_products",
        "household/personal care staples. EG p=0.0246, beta +0.834, hl 6.8d."),
    FocusPair("SRE", "XEL", "utilities",
        "regulated utilities; REPLACES DUK/SO, which decayed from EG p=0.0027 "
        "to 0.0510 in four sessions and failed its own admission test. "
        "EG p=0.0354, beta +1.018, half-life 7.3d."),
    FocusPair("ROST", "TGT", "discount_retail",
        "off-price vs big-box retail; carried over. EG p=0.0389, beta +0.948, "
        "half-life 11.0d."),
    FocusPair("APD", "LIN", "industrial_gases",
        "industrial gases — a near-duopoly with near-identical cost structures. "
        "EG p=0.0440, beta +0.789, half-life 8.1d."),
    FocusPair("AMGN", "VRTX", "biotech",
        "large-cap biotech. EG p=0.0457, beta +1.215, half-life 7.6d."),
    FocusPair("FITB", "RF", "regional_banks",
        "regional banks — same rate/deposit-beta exposure. Weakest member, "
        "admitted at the margin. EG p=0.0492, beta +1.281, half-life 8.4d."),
]

# Retired 2026-08-05 (kept for the audit trail):
_RETIRED_2026_08_05: list[FocusPair] = [
    FocusPair(
        "DUK", "SO", "utilities",
        "regulated southeastern utilities; the ONLY survivor of the previous "
        "book. EG p=0.0027 on 252d, beta +0.893 (well balanced), half-life 8.0d. "
        "Strongest genuine cointegration in the book.",
    ),
    FocusPair(
        "JNJ", "MRK", "healthcare",
        "large-cap pharma pair; REPLACES ABT/MRK, which had a NEGATIVE beta "
        "(-0.578, never market-neutral) and EG p=0.56 on 252d. "
        "EG p=0.0106, beta +0.799, half-life 6.4d.",
    ),
    FocusPair(
        "AVGO", "NVDA", "semis",
        "semiconductor duo; the NVDA cluster's single slot (NVDA/SMH excluded "
        "as stock-vs-own-sector-ETF, NVDA/TXN and NVDA/QCOM as lower-ranked "
        "same-cluster). EG p=0.0122, beta +1.236, half-life 7.2d.",
    ),
    FocusPair(
        "GDX", "GLD", "metals_commodities",
        "gold miners vs bullion — the project's original deep-dive pair, now "
        "qualifying on strict evidence for the first time. EG p=0.0185, "
        "beta +1.419, half-life 7.3d. GDX/IAU dropped as a duplicate.",
    ),
    FocusPair(
        "ROST", "TGT", "discount_retail",
        "off-price vs big-box retail. EG p=0.0435, beta +0.966 (near dollar-"
        "neutral), half-life 12.1d — the longest holding period in the book.",
    ),
    FocusPair(
        "GD", "RTX", "defense",
        "defense primes. EG p=0.0474 — the weakest member, admitted at the "
        "margin; first candidate to drop at the next refresh. beta +0.448, "
        "half-life 8.0d.",
    ),
]


def focus_pairs() -> list[tuple[str, str]]:
    """The book as plain (ticker_a, ticker_b) tuples for the backtester/report."""
    return [p.key for p in FOCUS_BOOK]


def focus_tickers() -> list[str]:
    """All distinct tickers in the book (deduplicated, sorted)."""
    return sorted({t for p in FOCUS_BOOK for t in p.key})
