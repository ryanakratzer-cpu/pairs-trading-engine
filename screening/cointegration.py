"""Engle-Granger two-step cointegration testing and half-life estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

DEFAULT_SIGNIFICANCE = 0.05
DEFAULT_MIN_HALF_LIFE_DAYS = 5.0
DEFAULT_MAX_HALF_LIFE_DAYS = 30.0
MIN_OBSERVATIONS = 30


@dataclass(frozen=True)
class EngleGrangerResult:
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    intercept: float
    adf_stat: float
    adf_pvalue: float
    is_cointegrated: bool
    half_life_days: float | None


def _to_series(prices: pd.Series, use_log_prices: bool) -> pd.Series:
    return np.log(prices) if use_log_prices else prices


def test_pair_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
    ticker_a: str = "A",
    ticker_b: str = "B",
    significance: float = DEFAULT_SIGNIFICANCE,
    use_log_prices: bool = True,
) -> EngleGrangerResult:
    """Engle-Granger two-step cointegration test.

    Step 1: OLS-regress price_a on price_b to obtain the hedge ratio.
    Step 2: ADF-test the OLS residual (the spread) for stationarity.
    The hedge ratio returned is the same one the ADF test validated, rather
    than statsmodels' black-box coint() which recomputes internally.
    """
    a = _to_series(price_a, use_log_prices).astype(float).to_numpy()
    b = _to_series(price_b, use_log_prices).astype(float).to_numpy()

    design = sm.add_constant(b)
    model = sm.OLS(a, design).fit()
    intercept, hedge_ratio = model.params
    spread = model.resid

    adf_stat, adf_pvalue, *_ = adfuller(spread, autolag="AIC")
    is_cointegrated = adf_pvalue < significance

    half_life = compute_half_life(pd.Series(spread)) if is_cointegrated else None

    return EngleGrangerResult(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        hedge_ratio=float(hedge_ratio),
        intercept=float(intercept),
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_pvalue),
        is_cointegrated=is_cointegrated,
        half_life_days=half_life,
    )


def compute_half_life(spread: pd.Series) -> float | None:
    """Estimate mean-reversion half-life by fitting an AR(1)/OU model to the spread.

    Regresses delta_spread_t on spread_{t-1}; half_life = ln(2) / -slope.
    Returns None if the fitted slope implies no mean reversion (slope >= 0).
    """
    spread = pd.Series(spread).reset_index(drop=True)
    lagged = spread.shift(1)
    delta = spread.diff()

    valid = pd.concat([lagged, delta], axis=1).dropna()
    if len(valid) < 2:
        return None
    lagged_valid, delta_valid = valid.iloc[:, 0].to_numpy(), valid.iloc[:, 1].to_numpy()

    design = sm.add_constant(lagged_valid)
    model = sm.OLS(delta_valid, design).fit()
    slope = model.params[1]

    if slope >= 0:
        return None

    return float(np.log(2) / -slope)


def _benjamini_hochberg(pvalues: pd.Series, fdr_level: float) -> pd.Series:
    """Benjamini-Hochberg FDR procedure: find the largest rank k (p-values
    sorted ascending) with p_(k) <= (k/n)*fdr_level, then flag ranks 1..k as
    significant. Returns a boolean Series aligned to `pvalues`'s index.
    """
    n = len(pvalues)
    order = pvalues.sort_values().index
    ranks = np.arange(1, n + 1)
    thresholds = (ranks / n) * fdr_level
    sorted_pvalues = pvalues.loc[order].to_numpy()
    passes = sorted_pvalues <= thresholds

    significant = pd.Series(False, index=pvalues.index)
    if passes.any():
        cutoff_rank = int(np.max(np.where(passes)[0])) + 1
        significant.loc[order[:cutoff_rank]] = True
    return significant


def screen_universe(
    price_panel: pd.DataFrame,
    pairs: list[tuple[str, str]],
    significance: float = DEFAULT_SIGNIFICANCE,
    use_log_prices: bool = True,
    min_half_life_days: float = DEFAULT_MIN_HALF_LIFE_DAYS,
    max_half_life_days: float = DEFAULT_MAX_HALF_LIFE_DAYS,
    apply_multiple_testing_correction: bool = False,
    fdr_level: float = DEFAULT_SIGNIFICANCE,
) -> pd.DataFrame:
    """Test every candidate pair for cointegration and flag those in a tradeable half-life band.

    Returns a DataFrame ranked by ADF p-value (strongest evidence first). Pairs
    that fail cointegration or the half-life filter are included and flagged
    (is_cointegrated / passes_half_life_filter / tradeable columns) rather than
    silently dropped, so the screen is auditable end to end.

    Screening many pairs at a fixed significance level guarantees some false
    positives by chance (e.g. ~5 expected out of 100 pairs at p<0.05). Set
    apply_multiple_testing_correction=True to additionally require each pair's
    ADF p-value survive a Benjamini-Hochberg false-discovery-rate correction
    across all pairs tested (adds a `bh_significant` column; `tradeable` then
    requires both is_cointegrated, the half-life filter, AND bh_significant).
    Off by default to preserve v1 behavior.
    """
    rows = []
    for ticker_a, ticker_b in pairs:
        if ticker_a not in price_panel.columns or ticker_b not in price_panel.columns:
            continue
        pair_prices = price_panel[[ticker_a, ticker_b]].dropna()
        if len(pair_prices) < MIN_OBSERVATIONS:
            continue

        result = test_pair_cointegration(
            pair_prices[ticker_a],
            pair_prices[ticker_b],
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            significance=significance,
            use_log_prices=use_log_prices,
        )

        passes_half_life = (
            result.half_life_days is not None
            and min_half_life_days <= result.half_life_days <= max_half_life_days
        )

        rows.append(
            {
                "ticker_a": result.ticker_a,
                "ticker_b": result.ticker_b,
                "hedge_ratio": result.hedge_ratio,
                "adf_stat": result.adf_stat,
                "adf_pvalue": result.adf_pvalue,
                "is_cointegrated": result.is_cointegrated,
                "half_life_days": result.half_life_days,
                "passes_half_life_filter": passes_half_life,
                "tradeable": result.is_cointegrated and passes_half_life,
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    if apply_multiple_testing_correction:
        results["bh_significant"] = _benjamini_hochberg(results["adf_pvalue"], fdr_level)
        results["tradeable"] = results["tradeable"] & results["bh_significant"]
    else:
        results["bh_significant"] = None

    return results.sort_values("adf_pvalue", ascending=True).reset_index(drop=True)
