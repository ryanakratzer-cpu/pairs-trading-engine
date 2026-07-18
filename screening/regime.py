"""Macro regime overlay: flag stressed market regimes so mean-reversion entries
can be suspended when they are most likely to fail.

Cointegration relationships are fitted on mostly-calm history, but they break
precisely in stress: correlations converge, spreads blow through their fitted
bands, and half-life estimates stop meaning anything. This module produces a
per-date boolean "entries allowed" mask (compatible with
signals.spread.generate_signals' `tradeable` parameter) from rolling percentile
ranks of volatility gauges, plus a purely diagnostic report of which macro
variables currently co-move with a pair's spread.

Standalone by design: nothing here gates the screen or the backtester yet.
Integration into the simulator is a separate step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from data.loader import fetch_price_history

# Dict rather than a list so callers can subset ({"vix": MACRO_TICKERS["vix"]})
# or extend with their own symbols without touching this module.
MACRO_TICKERS: dict[str, str] = {
    "vix": "^VIX",
    "gold_vol": "^GVZ",
    "ten_year_yield": "^TNX",
    "dollar": "UUP",
    "oil": "USO",
}


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds for calling a regime stressed.

    percentile_window: trailing days over which each gauge's percentile rank is
    computed. 252 (one trading year) so "stressed" means high relative to the
    recent past, not to some fixed absolute level that would go stale as the
    vol base level drifts across years.

    Stress thresholds are percentile ranks in [0, 1]. gold_vol's is higher than
    vix's because GVZ is the secondary confirmation gauge: it should only veto
    entries on its own when gold vol is unusually extreme, not merely elevated.
    """

    percentile_window: int = 252
    vix_stress_percentile: float = 0.80
    gold_vol_stress_percentile: float = 0.85

    def __post_init__(self) -> None:
        if self.percentile_window < 2:
            raise ValueError("percentile_window must be >= 2")
        for name in ("vix_stress_percentile", "gold_vol_stress_percentile"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")


def fetch_macro_panel(
    start: str,
    end: str,
    tickers: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch daily macro series and return them under their friendly names.

    Goes through data.loader.fetch_price_history so downloads share the same
    local CSV cache as equity data. Macro indices are spotty (CBOE index
    calendars differ from equity calendars, some series start late), so each
    column keeps its own coverage: rows are only dropped when EVERY column is
    missing. Deliberately does not use data.loader.align_and_clean, whose
    joint-alignment dropna would discard perfectly good VIX readings just
    because a thinner series has a gap that day. Consumers handle NaN
    per-column.
    """
    if tickers is None:
        tickers = MACRO_TICKERS
    symbols = list(tickers.values())
    raw = fetch_price_history(symbols, start=start, end=end)

    symbol_to_name = {symbol: name for name, symbol in tickers.items()}
    panel = raw.rename(columns=symbol_to_name)
    ordered = [name for name in tickers if name in panel.columns]
    return panel[ordered].dropna(how="all")


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Causal percentile rank of each value within its trailing `window`
    observations (inclusive of the current one). Uses only past data, so
    truncating the series never changes earlier ranks - no look-ahead.
    Computed on the NaN-dropped series so a data gap widens the effective
    calendar span rather than poisoning every window it touches; ranks are NaN
    until a full window of history exists (warmup)."""
    clean = series.dropna()
    return clean.rolling(window, min_periods=window).rank(pct=True)


def stress_percentile_ranks(
    macro_panel: pd.DataFrame,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    """Rolling percentile ranks for whichever stress gauges (vix, gold_vol) are
    present in the panel, reindexed to the panel's dates. Exposed separately
    from compute_stress_mask so reports can print the actual percentile numbers
    behind a CALM/STRESSED call instead of an unexplained boolean."""
    config = config or RegimeConfig()
    ranks: dict[str, pd.Series] = {}
    for name in ("vix", "gold_vol"):
        if name in macro_panel.columns:
            ranks[name] = _rolling_percentile_rank(
                macro_panel[name], config.percentile_window
            ).reindex(macro_panel.index)
    return pd.DataFrame(ranks, index=macro_panel.index)


def compute_stress_mask(
    macro_panel: pd.DataFrame,
    config: RegimeConfig | None = None,
) -> pd.Series:
    """Per-date entries-allowed mask: True = calm (entries allowed), False =
    stressed. Feed the aligned result to generate_signals(tradeable=...),
    which gates NEW entries only - open positions still exit on their own terms.

    A date is stressed when vix's causal rolling percentile rank exceeds
    config.vix_stress_percentile OR gold_vol's exceeds
    config.gold_vol_stress_percentile, each rank computed on its own series
    over its own trailing window (no cross-series alignment, no look-ahead).

    Dates with no usable stress reading (missing data, warmup before a full
    percentile window, or a panel lacking the gauge columns entirely) default
    to CALM. This is deliberate: the overlay is a veto layer on top of an
    engine that already works without it, so the absence of evidence of stress
    must not silently halt trading - a spotty macro feed failing to print for
    a week should not shut the strategy down.
    """
    config = config or RegimeConfig()
    thresholds = {
        "vix": config.vix_stress_percentile,
        "gold_vol": config.gold_vol_stress_percentile,
    }
    ranks = stress_percentile_ranks(macro_panel, config)

    stressed = pd.Series(False, index=macro_panel.index)
    for name in ranks.columns:
        # NaN rank compares False against the threshold, i.e. defaults to calm.
        stressed |= ranks[name] > thresholds[name]
    return ~stressed


def align_mask_to(mask: pd.Series, price_index: pd.Index) -> pd.Series:
    """Project a macro-date mask onto a price date index.

    Forward-fill because the latest known regime reading stays in force until
    a newer macro observation arrives (macro and equity calendars do not match
    day-for-day). Price dates before the first macro reading default to calm
    (True), consistent with compute_stress_mask's missing-data policy: no
    stress reading is not evidence of stress.
    """
    aligned = mask.reindex(price_index, method="ffill")
    return aligned.fillna(True).astype(bool)


def macro_spread_diagnostics(
    spread: pd.Series,
    macro_panel: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Correlation of daily spread CHANGES against each macro variable's daily
    returns, full-sample and over the latest `window` observations.

    Purely diagnostic and NOT causal: a high correlation says the spread is
    currently coupled to that macro variable (so its "idiosyncratic
    mean-reversion" may really be a macro bet in disguise), not that the
    variable drives the spread. Differences/returns rather than levels because
    two trending level series correlate spuriously; changes are what a
    day-to-day hedged position is actually exposed to.

    Returns a DataFrame with columns: macro_var, full_sample_corr, recent_corr,
    n_obs. Correlations are NaN when fewer than 3 overlapping observations
    exist rather than raising, so one thin macro series cannot break the report.
    """
    spread_changes = spread.diff()
    rows = []
    for name in macro_panel.columns:
        macro_returns = macro_panel[name].dropna().pct_change()
        joined = pd.concat([spread_changes, macro_returns], axis=1, join="inner").dropna()
        joined.columns = ["spread_change", "macro_return"]

        full_corr = (
            joined["spread_change"].corr(joined["macro_return"]) if len(joined) >= 3 else float("nan")
        )
        recent = joined.tail(window)
        recent_corr = (
            recent["spread_change"].corr(recent["macro_return"]) if len(recent) >= 3 else float("nan")
        )
        rows.append(
            {
                "macro_var": name,
                "full_sample_corr": full_corr,
                "recent_corr": recent_corr,
                "n_obs": len(joined),
            }
        )
    return pd.DataFrame(rows, columns=["macro_var", "full_sample_corr", "recent_corr", "n_obs"])
