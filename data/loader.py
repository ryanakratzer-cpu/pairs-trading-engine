"""Historical adjusted-close price ingestion via yfinance, with local CSV caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
DEFAULT_MIN_HISTORY_FRACTION = 0.9
DEFAULT_MAX_FFILL_DAYS = 5


def _cache_key(tickers: list[str], start: str, end: str, interval: str) -> str:
    raw = f"{','.join(sorted(tickers))}|{start}|{end}|{interval}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def fetch_price_history(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted-close prices for `tickers` between `start` and `end`.

    Returns a wide DataFrame indexed by date, one column per ticker.
    threads=False avoids a sqlite cache lock intermittently hit when
    yfinance downloads multiple tickers concurrently.
    """
    tickers = sorted(set(tickers))
    key = _cache_key(tickers, start, end, interval)
    cache_path = CACHE_DIR / f"{key}.csv"

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached_tickers = [t for t in tickers if t in cached.columns]
        if cached_tickers == tickers:
            return cached[tickers]

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned no data for tickers={tickers}, start={start}, end={end}")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.sort_index()

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        prices.to_csv(cache_path)

    return prices


def align_and_clean(
    prices: pd.DataFrame,
    min_history_fraction: float = DEFAULT_MIN_HISTORY_FRACTION,
    max_ffill_days: int = DEFAULT_MAX_FFILL_DAYS,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop thin-history tickers, forward-fill small gaps, drop remaining NaN rows.

    Returns (cleaned_prices, dropped_tickers) so callers can log what was excluded.
    """
    n_rows = len(prices)
    coverage = prices.notna().sum() / n_rows
    keep = coverage[coverage >= min_history_fraction].index.tolist()
    dropped = sorted(set(prices.columns) - set(keep))

    cleaned = prices[keep].ffill(limit=max_ffill_days)
    cleaned = cleaned.dropna(how="any")

    return cleaned, dropped
