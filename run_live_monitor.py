"""Live intraday pair monitor: streams real-time prices over Yahoo's websocket
(with automatic fallback to polling), updates the spread z-score and signal
state in near-real-time, and renders an auto-refreshing HTML dashboard.

Usage:
    py run_live_monitor.py [TICKER_A TICKER_B] [--mode stream|poll]
                           [--min-update-interval SECONDS]
                           [--poll SECONDS] [--max-polls N]
    (defaults: GDX GLD, stream mode, 2s min recompute interval, 30s polls)

Modes:
    stream (default) - subscribes to both tickers on the yfinance websocket
        (yf.WebSocket). Every tick updates that ticker's latest price; once
        both tickers have a price, the spread/z-score is recomputed at most
        every --min-update-interval seconds. If the websocket fails to connect
        within ~15s, or errors/disconnects later, the monitor logs it and
        falls back to polling automatically. After 60s with no ticks (market
        closed), it renders the dashboard from the last daily close and keeps
        listening (no polling hammering).
    poll - the original loop: fetch the latest 1-minute bar every --poll
        seconds.

--max-polls semantics: in poll mode, N polls as before. In stream mode, the
run ends after N dashboard updates OR after N*60 seconds of wall time,
whichever comes first (so short test runs terminate even off-hours).

Open outputs/live_monitor.html in a browser - it refreshes itself every 5s in
stream mode, 15s in poll mode.

Honest data caveats: Yahoo intraday quotes can lag real-time by up to ~15
minutes depending on the exchange, and polling much faster than ~15-30s risks
rate-limiting. This is a monitoring/research tool: SIGNAL ONLY - it never
places an order and has no broker connectivity.
"""

from __future__ import annotations

import sys
import threading
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
DEFAULT_MIN_UPDATE_INTERVAL = 2.0
STALE_AFTER_MINUTES = 20  # intraday bar older than this => treat market as closed/stale
STREAM_CONNECT_TIMEOUT = 15.0  # seconds to wait for the websocket before falling back
STREAM_QUIET_SECONDS = 60.0  # no ticks for this long => market-closed note
STREAM_MAX_SECONDS_PER_UPDATE = 60.0  # stream --max-polls N also caps wall time at N * this
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


def compute_spread_point(price_a: float, price_b: float, context: dict) -> tuple[float, float]:
    """Pure math: (spread, zscore) for one pair of prices given the daily context.

    spread = ln(a) - hedge_ratio * ln(b) - intercept
    z      = (spread - spread_mean) / spread_std
    """
    spread = float(
        np.log(price_a) - context["hedge_ratio"] * np.log(price_b) - context["intercept"]
    )
    z = (spread - context["spread_mean"]) / context["spread_std"]
    return spread, z


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


def decide_stream_action(
    connected: bool,
    thread_alive: bool,
    seconds_since_last_event: float,
    connect_timeout: float = STREAM_CONNECT_TIMEOUT,
    quiet_seconds: float = STREAM_QUIET_SECONDS,
) -> str:
    """Pure fallback/quiet decision for the stream loop.

    Returns one of:
      'wait'     - not connected yet, still within the connect timeout
      'fallback' - websocket failed (never connected in time, or died) => poll
      'quiet'    - connected but no ticks for `quiet_seconds` (market closed)
      'ok'       - connected and ticking normally
    """
    if not connected:
        if not thread_alive or seconds_since_last_event >= connect_timeout:
            return "fallback"
        return "wait"
    if not thread_alive:
        return "fallback"
    if seconds_since_last_event >= quiet_seconds:
        return "quiet"
    return "ok"


def render_html(
    ticker_a: str,
    ticker_b: str,
    context: dict,
    history: pd.DataFrame,
    config: SignalConfig,
    mode: str = "poll",
) -> None:
    """Self-contained dashboard; refreshes itself via meta tag (5s stream, 15s poll)."""
    refresh_seconds = 5 if mode == "stream" else 15
    mode_badge = "STREAMING (Yahoo websocket)" if mode == "stream" else "POLLING"
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
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="{refresh_seconds}">
<title>{ticker_a}/{ticker_b} live pair monitor</title>
<style>
 body {{ font-family: Segoe UI, sans-serif; margin: 2em; background:#1e1e2e; color:#eee; }}
 .big {{ font-size: 3em; font-weight: bold; color: {status_color}; }}
 table {{ border-collapse: collapse; margin-top: 1em; }}
 td, th {{ padding: 4px 12px; border-bottom: 1px solid #444; text-align: right; }}
 .meta {{ color: #aaa; font-size: 0.9em; }}
 .mode {{ color: #3498db; font-weight: bold; font-size: 0.9em; letter-spacing: 1px; }}
 svg {{ background:#26263a; border-radius:8px; margin-top:1em; }}
</style></head><body>
<h1>{ticker_a} / {ticker_b} — live spread monitor</h1>
<p style="color:#e67e22"><b>{DISCLAIMER}</b></p>
<p class="mode">MODE: {mode_badge} &nbsp;|&nbsp; page refreshes every {refresh_seconds}s</p>
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


class _StreamState:
    """Shared state between the websocket listener thread and the main loop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prices: dict[str, float] = {}
        self.last_tick_monotonic: float | None = None
        self.connected = threading.Event()
        self.dead = threading.Event()
        self.error: str | None = None


def _stream_worker(tickers: list[str], state: _StreamState) -> None:
    """Background thread: subscribe and pump websocket ticks into `state`."""
    wanted = set(tickers)

    def handler(message: dict) -> None:
        symbol = message.get("id")
        price = message.get("price")
        if symbol in wanted and price is not None:
            with state.lock:
                state.prices[symbol] = float(price)
                state.last_tick_monotonic = time.monotonic()

    try:
        ws = yf.WebSocket(verbose=False)
        ws.subscribe(list(tickers))  # connects; raises if the socket can't open
        state.connected.set()
        ws.listen(handler)  # blocks; returns on error/disconnect
    except Exception as exc:  # noqa: BLE001 - any websocket failure => fallback signal
        state.error = str(exc)
    finally:
        state.dead.set()


def run_stream_loop(
    ticker_a: str,
    ticker_b: str,
    context: dict,
    config: SignalConfig,
    min_update_interval: float,
    max_polls: int | None,
) -> bool:
    """Websocket streaming loop. Returns True if it ran to completion,
    False if the websocket failed and the caller should fall back to polling."""
    state = _StreamState()
    thread = threading.Thread(target=_stream_worker, args=([ticker_a, ticker_b], state), daemon=True)
    thread.start()

    if not state.connected.wait(timeout=STREAM_CONNECT_TIMEOUT):
        detail = state.error or "timed out"
        print(f"  websocket did not connect within {STREAM_CONNECT_TIMEOUT:.0f}s ({detail})")
        return False
    print(f"  websocket connected; subscribed to {ticker_a}, {ticker_b}")

    history_rows: list[dict] = []
    history_index: list[pd.Timestamp] = []
    updates = 0
    start = time.monotonic()
    last_recompute = 0.0
    last_processed_tick = 0.0
    quiet_noted = False

    while True:
        if max_polls is not None and (
            updates >= max_polls or (time.monotonic() - start) >= max_polls * STREAM_MAX_SECONDS_PER_UPDATE
        ):
            break

        now = time.monotonic()
        with state.lock:
            prices = dict(state.prices)
            last_tick = state.last_tick_monotonic
        seconds_since_event = now - (last_tick if last_tick is not None else start)

        action = decide_stream_action(True, not state.dead.is_set(), seconds_since_event)
        if action == "fallback":
            detail = state.error or "connection closed"
            print(f"  [{datetime.now():%H:%M:%S}] websocket disconnected ({detail})")
            return False

        have_both = ticker_a in prices and ticker_b in prices
        fresh_tick = last_tick is not None and last_tick > last_processed_tick

        if have_both and fresh_tick and (now - last_recompute) >= min_update_interval:
            spread, z = compute_spread_point(prices[ticker_a], prices[ticker_b], context)
            history_rows.append(
                {
                    "price_a": prices[ticker_a], "price_b": prices[ticker_b],
                    "spread": spread, "zscore": z,
                    "age_minutes": seconds_since_event / 60, "is_stale": False,
                }
            )
            history_index.append(pd.Timestamp(datetime.now(timezone.utc)))
            history = pd.DataFrame(history_rows, index=pd.DatetimeIndex(history_index))
            render_html(ticker_a, ticker_b, context, history, config, mode="stream")
            print(
                f"  [{datetime.now():%H:%M:%S}] {ticker_a}={prices[ticker_a]:.2f} "
                f"{ticker_b}={prices[ticker_b]:.2f} z={z:+.2f} [STREAM] -> {classify(z, config)}"
            )
            last_recompute = now
            last_processed_tick = last_tick
            updates += 1
            quiet_noted = False
        elif action == "quiet" and not quiet_noted:
            print(
                f"  [{datetime.now():%H:%M:%S}] no ticks for {STREAM_QUIET_SECONDS:.0f}s - "
                "market likely closed; showing last daily close, still listening"
            )
            close = context["last_daily_close"]
            spread, z = compute_spread_point(close[ticker_a], close[ticker_b], context)
            history_rows.append(
                {
                    "price_a": float(close[ticker_a]), "price_b": float(close[ticker_b]),
                    "spread": spread, "zscore": z,
                    "age_minutes": seconds_since_event / 60, "is_stale": True,
                }
            )
            history_index.append(pd.Timestamp(datetime.now(timezone.utc)))
            history = pd.DataFrame(history_rows, index=pd.DatetimeIndex(history_index))
            render_html(ticker_a, ticker_b, context, history, config, mode="stream")
            print(f"  [{datetime.now():%H:%M:%S}] z={z:+.2f} [STALE/CLOSED] -> {classify(z, config)}")
            updates += 1
            quiet_noted = True

        time.sleep(0.25)

    return True


def run_poll_loop(
    ticker_a: str,
    ticker_b: str,
    context: dict,
    config: SignalConfig,
    poll_seconds: int,
    max_polls: int | None,
) -> None:
    """Original polling loop: latest 1-minute bar every `poll_seconds`."""
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

        spread, z = compute_spread_point(tick["price_a"], tick["price_b"], context)
        history_rows.append(
            {
                "price_a": tick["price_a"], "price_b": tick["price_b"],
                "spread": spread, "zscore": z,
                "age_minutes": tick["age_minutes"], "is_stale": tick["is_stale"],
            }
        )
        history_index.append(tick["bar_time"])
        history = pd.DataFrame(history_rows, index=pd.DatetimeIndex(history_index))
        render_html(ticker_a, ticker_b, context, history, config, mode="poll")

        stale_flag = " [STALE/CLOSED]" if tick["is_stale"] else ""
        print(
            f"  [{datetime.now():%H:%M:%S}] {ticker_a}={tick['price_a']:.2f} {ticker_b}={tick['price_b']:.2f} "
            f"z={z:+.2f}{stale_flag} -> {classify(z, config)}"
        )

        if max_polls is not None and polls >= max_polls:
            break
        # Poll gently when the market is closed - nothing changes anyway.
        time.sleep(poll_seconds * (4 if tick["is_stale"] else 1))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ticker_a, ticker_b = (args[0], args[1]) if len(args) >= 2 else ("GDX", "GLD")
    poll_seconds = DEFAULT_POLL_SECONDS
    min_update_interval = DEFAULT_MIN_UPDATE_INTERVAL
    mode = "stream"
    max_polls = None
    for i, arg in enumerate(sys.argv):
        if arg == "--poll" and i + 1 < len(sys.argv):
            poll_seconds = max(int(sys.argv[i + 1]), 10)  # floor to avoid hammering the source
        if arg == "--max-polls" and i + 1 < len(sys.argv):
            max_polls = int(sys.argv[i + 1])
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1].lower()
        if arg == "--min-update-interval" and i + 1 < len(sys.argv):
            min_update_interval = max(float(sys.argv[i + 1]), 0.5)  # floor to avoid render spam
    if mode not in ("stream", "poll"):
        print(f"Unknown --mode '{mode}'; expected stream or poll")
        sys.exit(2)

    config = SignalConfig(zscore_window=50, entry_z=2.0, exit_z=0.5, stop_z=3.0, max_holding_bars=62)

    mode_desc = (
        f"streaming (websocket, recompute every >={min_update_interval:g}s)"
        if mode == "stream" else f"polling every {poll_seconds}s"
    )
    print(f"=== Live pair monitor: {ticker_a}/{ticker_b}, {mode_desc} ===")
    print(DISCLAIMER)
    print("Building daily context (hedge ratio, z-score baseline)...")
    context = build_daily_context(ticker_a, ticker_b, config)
    half_life = context["half_life_days"]
    print(
        f"  hedge_ratio={context['hedge_ratio']:.4f}, kalman_beta={context['kalman_beta']:.4f}, "
        f"adf_p={context['adf_pvalue']:.4f}, half_life={half_life if half_life is None else round(half_life, 1)}d"
    )

    if mode == "stream":
        completed = run_stream_loop(ticker_a, ticker_b, context, config, min_update_interval, max_polls)
        if not completed:
            print("  falling back to polling mode")
            mode = "poll"  # so the dashboard note below reports the refresh actually in effect
            run_poll_loop(ticker_a, ticker_b, context, config, poll_seconds, max_polls)
    else:
        run_poll_loop(ticker_a, ticker_b, context, config, poll_seconds, max_polls)

    refresh = 5 if mode == "stream" else 15
    print(f"\nDashboard: {OUTPUTS_DIR / 'live_monitor.html'} (auto-refreshes every {refresh}s while this runs)")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
