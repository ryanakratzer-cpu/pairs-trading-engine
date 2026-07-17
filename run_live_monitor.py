"""Live intraday pair monitor: polls prices, updates the spread z-score and
signal state in near-real-time, and renders an auto-refreshing HTML dashboard.

Usage:
    py run_live_monitor.py [TICKER_A TICKER_B] [--poll SECONDS] [--max-polls N]
    (defaults: GDX GLD, 30-second polls, unlimited)

Open outputs/live_monitor.html in a browser — it refreshes itself every 15s.

Honest data caveats: Yahoo intraday quotes can lag real-time by up to ~15
minutes depending on the exchange, and polling much faster than ~15-30s risks
rate-limiting. This is a monitoring/research tool: SIGNAL ONLY — it never
places an order and has no broker connectivity.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from data.loader import align_and_clean, fetch_price_history
from screening.cointegration import test_pair_cointegration
from signals.spread import KalmanHedgeRatio, SignalConfig

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
DAILY_LOOKBACK_DAYS = 900
DEFAULT_POLL_SECONDS = 30
STALE_AFTER_MINUTES = 20  # intraday bar older than this => treat market as closed/stale
DISCLAIMER = "SIGNAL ONLY - research monitor. No order is ever placed by this tool."


REGIME_WINDOW_DAYS = 252  # match PairBacktestConfig.recheck_window_days


def build_daily_context(ticker_a: str, ticker_b: str, signal_config: SignalConfig) -> dict:
    """Fit hedge ratio / z-score baseline from daily history once at startup.

    The hedge ratio is fit on the trailing REGIME_WINDOW_DAYS bars — the same
    trailing-regime convention the backtester trades on — so the monitor's live
    z-score is the same quantity the backtest validated, not a full-history
    OLS fit that the backtest never used.
    """
    end = datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    prices, _ = align_and_clean(fetch_price_history([ticker_a, ticker_b], start=start, end=end))
    pair = prices[[ticker_a, ticker_b]].dropna().iloc[-REGIME_WINDOW_DAYS:]

    coint = test_pair_cointegration(pair[ticker_a], pair[ticker_b], ticker_a, ticker_b)
    daily_spread = np.log(pair[ticker_a]) - coint.hedge_ratio * np.log(pair[ticker_b]) - coint.intercept
    window = daily_spread.iloc[-signal_config.zscore_window :]
    kalman_beta = KalmanHedgeRatio().hedge_ratio(pair[ticker_a], pair[ticker_b])

    return {
        "hedge_ratio": coint.hedge_ratio,
        "intercept": coint.intercept,
        "adf_pvalue": coint.adf_pvalue,
        "half_life_days": coint.half_life_days,
        "spread_mean": float(window.mean()),
        "spread_std": float(window.std(ddof=1)),
        "kalman_beta": kalman_beta,
        "daily_spread_tail": daily_spread.iloc[-90:],
        "last_daily_close": pair.iloc[-1].to_dict(),
    }


def fetch_latest_intraday(ticker_a: str, ticker_b: str) -> dict | None:
    """Most recent 1-minute bar for both tickers; None if unavailable."""
    raw = yf.download(
        [ticker_a, ticker_b], period="1d", interval="1m",
        auto_adjust=True, progress=False, threads=False,
    )
    if raw.empty:
        return None
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    closes = closes.dropna()
    if closes.empty or ticker_a not in closes.columns or ticker_b not in closes.columns:
        return None
    last = closes.iloc[-1]
    bar_time = closes.index[-1]
    if bar_time.tzinfo is None:
        bar_time = bar_time.tz_localize("UTC")
    age_minutes = (datetime.now(timezone.utc) - bar_time).total_seconds() / 60
    return {
        "price_a": float(last[ticker_a]),
        "price_b": float(last[ticker_b]),
        "bar_time": bar_time,
        "age_minutes": age_minutes,
        "is_stale": age_minutes > STALE_AFTER_MINUTES,
    }


def classify(z: float, config: SignalConfig) -> str:
    if abs(z) >= config.stop_z:
        return "BEYOND STOP BAND - do not enter; existing shorts of this spread would be stopped"
    if z >= config.entry_z:
        return "SHORT-SPREAD ENTRY ZONE (spread rich: short A / long B)"
    if z <= -config.entry_z:
        return "LONG-SPREAD ENTRY ZONE (spread cheap: long A / short B)"
    if abs(z) <= config.exit_z:
        return "MEAN ZONE - open positions would exit here"
    return "NEUTRAL - inside bands, no action"


def render_html(ticker_a: str, ticker_b: str, context: dict, history: pd.DataFrame, config: SignalConfig) -> None:
    """Self-contained dashboard, refreshes itself every 15s via meta tag."""
    latest = history.iloc[-1]
    z = latest["zscore"]
    status_color = "#c0392b" if abs(z) >= config.entry_z else ("#27ae60" if abs(z) <= config.exit_z else "#f39c12")
    rows = "".join(
        f"<tr><td>{idx:%H:%M:%S}</td><td>{r['price_a']:.2f}</td><td>{r['price_b']:.2f}</td>"
        f"<td>{r['spread']:.5f}</td><td>{r['zscore']:+.2f}</td></tr>"
        for idx, r in history.tail(30).iloc[::-1].iterrows()
    )
    points = " ".join(
        f"{i},{60 - min(max(r['zscore'], -4), 4) * 14:.1f}" for i, (_, r) in enumerate(history.tail(120).iterrows())
    )
    half_life = context["half_life_days"]
    half_life_text = f"{half_life:.1f}d" if half_life else "n/a"
    stale_note = (
        "<p style='color:#c0392b'><b>Market closed / data stale</b> - showing last available bar.</p>"
        if latest["is_stale"] else ""
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<title>{ticker_a}/{ticker_b} live pair monitor</title>
<style>
 body {{ font-family: Segoe UI, sans-serif; margin: 2em; background:#1e1e2e; color:#eee; }}
 .big {{ font-size: 3em; font-weight: bold; color: {status_color}; }}
 table {{ border-collapse: collapse; margin-top: 1em; }}
 td, th {{ padding: 4px 12px; border-bottom: 1px solid #444; text-align: right; }}
 .meta {{ color: #aaa; font-size: 0.9em; }}
 svg {{ background:#26263a; border-radius:8px; margin-top:1em; }}
</style></head><body>
<h1>{ticker_a} / {ticker_b} — live spread monitor</h1>
<p style="color:#e67e22"><b>{DISCLAIMER}</b></p>
{stale_note}
<div class="big">z = {z:+.2f}</div>
<p><b>{classify(z, config)}</b></p>
<p class="meta">as of {latest.name:%Y-%m-%d %H:%M:%S %Z} (bar age {latest['age_minutes']:.1f} min)
 &nbsp;|&nbsp; {ticker_a}={latest['price_a']:.2f} &nbsp; {ticker_b}={latest['price_b']:.2f}</p>
<p class="meta">hedge ratio (OLS)={context['hedge_ratio']:.4f} &nbsp;|&nbsp; Kalman beta={context['kalman_beta']:.4f}
 &nbsp;|&nbsp; ADF p={context['adf_pvalue']:.4f} &nbsp;|&nbsp; half-life={half_life_text}
 &nbsp;|&nbsp; bands: entry ±{config.entry_z}, exit ±{config.exit_z}, stop ±{config.stop_z}</p>
<svg width="740" height="120" viewBox="0 0 740 120">
 <line x1="0" y1="60" x2="740" y2="60" stroke="#555"/>
 <line x1="0" y1="{60 - config.entry_z * 14}" x2="740" y2="{60 - config.entry_z * 14}" stroke="#c0392b" stroke-dasharray="4"/>
 <line x1="0" y1="{60 + config.entry_z * 14}" x2="740" y2="{60 + config.entry_z * 14}" stroke="#c0392b" stroke-dasharray="4"/>
 <polyline points="{points}" fill="none" stroke="#3498db" stroke-width="2" transform="scale(6,1)"/>
</svg>
<table><tr><th>time</th><th>{ticker_a}</th><th>{ticker_b}</th><th>spread</th><th>z</th></tr>{rows}</table>
</body></html>"""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "live_monitor.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ticker_a, ticker_b = (args[0], args[1]) if len(args) >= 2 else ("GDX", "GLD")
    poll_seconds = DEFAULT_POLL_SECONDS
    max_polls = None
    for i, arg in enumerate(sys.argv):
        if arg == "--poll" and i + 1 < len(sys.argv):
            poll_seconds = max(int(sys.argv[i + 1]), 10)  # floor to avoid hammering the source
        if arg == "--max-polls" and i + 1 < len(sys.argv):
            max_polls = int(sys.argv[i + 1])

    config = SignalConfig(zscore_window=50, entry_z=2.0, exit_z=0.5, stop_z=3.0, max_holding_bars=62)

    print(f"=== Live pair monitor: {ticker_a}/{ticker_b}, polling every {poll_seconds}s ===")
    print(DISCLAIMER)
    print("Building daily context (hedge ratio, z-score baseline)...")
    context = build_daily_context(ticker_a, ticker_b, config)
    half_life = context["half_life_days"]
    print(
        f"  hedge_ratio={context['hedge_ratio']:.4f}, kalman_beta={context['kalman_beta']:.4f}, "
        f"adf_p={context['adf_pvalue']:.4f}, half_life={half_life if half_life is None else round(half_life, 1)}d"
    )

    history_rows, history_index = [], []
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        try:
            tick = fetch_latest_intraday(ticker_a, ticker_b)
        except Exception as exc:  # network hiccups shouldn't kill a long-running monitor
            print(f"  [{datetime.now():%H:%M:%S}] fetch error ({exc}); retrying next poll")
            time.sleep(poll_seconds)
            continue

        if tick is None:
            print(f"  [{datetime.now():%H:%M:%S}] no intraday data returned; retrying next poll")
            time.sleep(poll_seconds)
            continue

        spread = (
            np.log(tick["price_a"]) - context["hedge_ratio"] * np.log(tick["price_b"]) - context["intercept"]
        )
        z = (spread - context["spread_mean"]) / context["spread_std"]
        history_rows.append(
            {
                "price_a": tick["price_a"], "price_b": tick["price_b"],
                "spread": spread, "zscore": z,
                "age_minutes": tick["age_minutes"], "is_stale": tick["is_stale"],
            }
        )
        history_index.append(tick["bar_time"])
        history = pd.DataFrame(history_rows, index=pd.DatetimeIndex(history_index))
        render_html(ticker_a, ticker_b, context, history, config)

        stale_flag = " [STALE/CLOSED]" if tick["is_stale"] else ""
        print(
            f"  [{datetime.now():%H:%M:%S}] {ticker_a}={tick['price_a']:.2f} {ticker_b}={tick['price_b']:.2f} "
            f"z={z:+.2f}{stale_flag} -> {classify(z, config)}"
        )

        if max_polls is not None and polls >= max_polls:
            break
        # Poll gently when the market is closed - nothing changes anyway.
        time.sleep(poll_seconds * (4 if tick["is_stale"] else 1))

    print(f"\nDashboard: {OUTPUTS_DIR / 'live_monitor.html'} (auto-refreshes every 15s while this runs)")


if __name__ == "__main__":
    main()
