"""Rebuild ALL dashboards from current data — run daily after the close.

One command refreshes every interactive view and republishes them to docs/ for
GitHub Pages, so the published dashboards never drift from the live record:

  1. trade_activity   — WHEN we bought/sold: z-score paths for every book pair
                        with entry/exit/stop markers, plus a position timeline
                        showing exactly what was held on which day.
  2. track_record     — cumulative P&L and per-session bars from the tracker's
                        append-only log.
  3. predictions      — pre-registered forecasts vs graded outcomes, with a
                        calibration panel.
  4. index.html       — the Pages landing page, restamped with today's date.

Research only — this visualises computed signals, never executed orders.

    py build_dashboards.py
"""

from __future__ import annotations

import shutil
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.loader import align_and_clean, fetch_price_history
from screening.cointegration import test_pair_cointegration
from screening.events import event_exclusion_mask
from screening.focus_book import FOCUS_BOOK, focus_pairs, focus_tickers
from signals.spread import SignalConfig, build_spread, generate_signals, rolling_zscore

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"
TRACK = OUT / "daily_performance.csv"
# Inside the repo (moved 2026-08-05) so the pre-registered prediction record
# travels with the code — a clone of this repo is the complete project.
FORECAST_DIR = ROOT / "forecasts"

BG = "#10151f"; GRID = "#2a3346"; INK = "#d6deeb"; MUTED = "#7b879c"
BLUE = "#5aa2f0"; GREEN = "#4fbf7e"; RED = "#e05c5c"; AMBER = "#f5b942"

ENTRY_EVENTS = ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD")
EXIT_EVENTS = ("EXIT", "STOP_LOSS", "TIME_EXIT")
LOOKBACK_DAYS = 400


def _theme(fig: go.Figure, title: str, height: int) -> None:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        font=dict(family="Segoe UI, sans-serif", size=12, color=INK),
        title=dict(text=title, x=0.01, xanchor="left"),
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=70, r=40, t=110, b=60),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)


def load_signals() -> dict:
    """Per-pair z-score, signals and hedge ratio over the recent window."""
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
    prices, _ = align_and_clean(fetch_price_history(focus_tickers(), start=start, end=end))
    cfg = SignalConfig()
    out = {}
    for a, b in focus_pairs():
        if a not in prices.columns or b not in prices.columns:
            continue
        r = test_pair_cointegration(prices[a].iloc[-252:], prices[b].iloc[-252:])
        z = rolling_zscore(build_spread(prices[a], prices[b], r.hedge_ratio), cfg.zscore_window).dropna()
        sig = generate_signals(z, cfg, tradeable=event_exclusion_mask(z.index))
        tail = sig.iloc[-LOOKBACK_DAYS:]
        out[f"{a}/{b}"] = {"signals": tail, "hedge_ratio": r.hedge_ratio, "cfg": cfg}
    return out


def build_trade_activity(data: dict) -> Path:
    """WHEN we traded: z paths with buy/sell markers + a position timeline."""
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.62, 0.38], shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Z-score vs signal bands — every entry and exit marked",
                        "Position timeline — what was held, and when"),
    )
    for name, d in data.items():
        s = d["signals"]
        fig.add_trace(go.Scatter(
            x=list(s.index), y=s["zscore"], mode="lines", name=name, line=dict(width=1.6),
            hovertemplate=f"{name} z=%{{y:+.2f}}<br>%{{x|%b %d}}<extra></extra>"), row=1, col=1)
        for events, sym, col, lbl in ((ENTRY_EVENTS, "triangle-up", GREEN, "BUY"),
                                      (EXIT_EVENTS, "triangle-down", RED, "SELL")):
            e = s[s["event"].isin(events)]
            if len(e):
                fig.add_trace(go.Scatter(
                    x=list(e.index), y=e["zscore"], mode="markers",
                    marker=dict(symbol=sym, size=13, color=col, line=dict(width=1, color=BG)),
                    showlegend=False,
                    hovertemplate=f"<b>{lbl}</b> {name}<br>%{{x|%b %d}} z=%{{y:+.2f}}<extra></extra>",
                ), row=1, col=1)
    cfg = next(iter(data.values()))["cfg"] if data else SignalConfig()
    for y, c in ((cfg.entry_z, RED), (-cfg.entry_z, RED), (cfg.exit_z, GREEN), (-cfg.exit_z, GREEN)):
        fig.add_hline(y=y, line=dict(color=c, width=1, dash="dash"), row=1, col=1)

    # Position timeline: one lane per pair, green long / red short.
    for i, (name, d) in enumerate(data.items()):
        pos = d["signals"]["position"]
        for val, col, lbl in ((1, GREEN, "LONG spread"), (-1, RED, "SHORT spread")):
            m = pos == val
            if m.any():
                fig.add_trace(go.Scatter(
                    x=list(pos.index[m]), y=[i] * int(m.sum()), mode="markers",
                    marker=dict(symbol="square", size=7, color=col), showlegend=False,
                    hovertemplate=f"{name} — {lbl}<br>%{{x|%b %d}}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(data))),
                     ticktext=list(data.keys()), row=2, col=1)
    fig.update_yaxes(title_text="z-score", row=1, col=1)
    _theme(fig, f"<b>Trade activity — when we bought and sold</b><br>"
                f"<span style='font-size:12px;color:{MUTED}'>Green ▲ = open spread position, "
                f"red ▼ = close. Updated {datetime.today():%Y-%m-%d}. Research only.</span>", 900)
    p = OUT / "interactive_trade_activity.html"
    fig.write_html(p, include_plotlyjs="cdn")
    return p


def build_track_record() -> Path | None:
    if not TRACK.exists() or TRACK.stat().st_size == 0:
        return None
    df = pd.read_csv(TRACK)
    if df.empty or "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    daily = df.groupby("date").agg(pnl=("pair_pnl_usd", "sum"), spy=("spy_return_pct", "first")).reset_index()
    daily["cum_pnl"] = daily["pnl"].cumsum()

    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True,
                        vertical_spacing=0.10,
                        subplot_titles=("Cumulative P&L ($)", "P&L per session ($)"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["cum_pnl"], mode="lines+markers",
                             line=dict(color=BLUE, width=2.5), name="cumulative P&L",
                             hovertemplate="%{x|%b %d}: $%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=daily["date"], y=daily["pnl"], name="session P&L",
                         marker_color=[GREEN if v >= 0 else RED for v in daily["pnl"]],
                         hovertemplate="%{x|%b %d}: $%{y:,.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=2, col=1)
    _theme(fig, f"<b>Track record — live sessions</b><br>"
                f"<span style='font-size:12px;color:{MUTED}'>{len(daily)} session(s). "
                f"Benchmark is ZERO, not SPY: a dollar-neutral book has no market exposure to "
                f"be compensated for. Updated {datetime.today():%Y-%m-%d}.</span>", 700)
    p = OUT / "interactive_track_record.html"
    fig.write_html(p, include_plotlyjs="cdn")
    return p


def build_predictions() -> Path | None:
    """Parse the pre-registered forecast files' RESULTS tables."""
    rows = []
    for f in sorted(FORECAST_DIR.glob("forecast_*.md")):
        day = f.stem.replace("forecast_", "")
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|") or "HIT" not in line and "MISS" not in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 4:
                continue
            conf = next((c for c in cells if c.endswith("%") and c[:-1].strip().isdigit()), None)
            if conf is None:
                continue
            rows.append({"day": day, "prediction": cells[1][:58],
                         "confidence": int(conf[:-1]) / 100.0,
                         "hit": "HIT" in line and "MISS" not in line.split("HIT")[0]})
    if not rows:
        return None
    P = pd.DataFrame(rows)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=(f"Pre-registered predictions "
                                        f"({int(P.hit.sum())} hits / {len(P)})",
                                        "Calibration: stated confidence vs hit rate"))
    for hit, col, nm in ((True, GREEN, "HIT"), (False, RED, "MISS")):
        s = P[P.hit == hit]
        fig.add_trace(go.Bar(x=s["prediction"], y=s["confidence"], name=nm, marker_color=col,
                             hovertemplate="%{x}<br>confidence %{y:.0%}<extra></extra>"), row=1, col=1)
    bins = [(0.45, 0.60, "45-60%"), (0.60, 0.80, "60-80%"), (0.80, 1.01, "80-100%")]
    bx, by, bn = [], [], []
    for lo, hi, lab in bins:
        s = P[(P.confidence >= lo) & (P.confidence < hi)]
        if len(s):
            bx.append(lab); by.append(s.hit.mean()); bn.append(len(s))
    fig.add_trace(go.Bar(x=bx, y=by, marker_color=BLUE, showlegend=False,
                         text=[f"{v:.0%} (n={n})" for v, n in zip(by, bn)],
                         textposition="outside", textfont=dict(color=INK),
                         hovertemplate="%{x}: %{y:.0%}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1); fig.update_xaxes(tickangle=-30, row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15], row=1, col=2)
    _theme(fig, f"<b>Prediction accuracy</b><br><span style='font-size:12px;color:{MUTED}'>"
                f"Registered before each open, graded mechanically from prices. "
                f"Updated {datetime.today():%Y-%m-%d}.</span>", 620)
    p = OUT / "interactive_predictions.html"
    fig.write_html(p, include_plotlyjs="cdn")
    return p


def main() -> None:
    print(f"=== Rebuilding dashboards {datetime.today():%Y-%m-%d %H:%M} ===")
    OUT.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    built = []

    data = load_signals()
    print(f"  loaded signals for {len(data)} pairs")
    built.append(build_trade_activity(data))

    for fn, label in ((build_track_record, "track record"), (build_predictions, "predictions")):
        p = fn()
        if p:
            built.append(p)
        else:
            print(f"  skipped {label} (no data yet)")

    for p in built:
        shutil.copy(p, DOCS / p.name)
        print(f"  built + published {p.name}")

    # Restamp the Pages index so the published date never lies about freshness.
    idx = DOCS / "index.html"
    if idx.exists():
        html = idx.read_text(encoding="utf-8")
        stamp = f"<!--updated-->Dashboards last rebuilt {datetime.today():%Y-%m-%d %H:%M}."
        import re
        html = re.sub(r"<!--updated-->[^<]*", stamp, html) if "<!--updated-->" in html else html.replace(
            '<div class="note">', f'<div class="note">{stamp} ')
        idx.write_text(html, encoding="utf-8")
        print("  restamped docs/index.html")
    print(f"=== {len(built)} dashboards refreshed ===")


if __name__ == "__main__":
    main()
