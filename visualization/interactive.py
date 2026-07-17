"""Interactive Plotly charts: Monte Carlo fan, equity/drawdown, spread/z-score, P&L histogram.

Each function writes a self-contained HTML file to outputs/ and returns its
Path. plotly.js is loaded from the CDN (include_plotlyjs="cdn") so each file
stays ~10-100 KB instead of ~4 MB with the library embedded - the same
convention the sibling convertible-bond-engine uses for its interactive HTML.

Styling is centralized in _apply_theme() so all four charts read as one
system: dark surface, recessive gridlines, restrained single-hue data color
with reserved status colors (entry/exit/stop) that are always labeled, and a
"research only" watermark on every chart to match the project's strict
no-execution stance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from montecarlo.simulator import OUFit, summarize_pnl
from signals.spread import SignalConfig

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

WATERMARK = "pairs-trading-engine — research only, not investment advice"

# Dark-surface palette: one data hue (blue) carries the series; amber/green/red
# are reserved status colors for entry/exit/stop and never reused for data.
PAPER_BG = "#10151f"
PLOT_BG = "#151c29"
GRID_COLOR = "#26314a"
INK = "#d6deeb"
INK_MUTED = "#7b879c"
DATA_BLUE = "#5aa2f0"
PATH_RGBA = "rgba(90, 162, 240, 0.06)"  # ~1000 overlaid paths need near-transparent lines
BAND_RGBA = "rgba(90, 162, 240, 0.16)"
MEDIAN_AMBER = "#f5b942"
HISTORY_INK = "#e8edf5"
ENTRY_ORANGE = "#e8843c"
EXIT_GREEN = "#4fbf7e"
STOP_RED = "#e05c5c"
DRAWDOWN_RED = "rgba(224, 92, 92, 0.55)"


def _apply_theme(fig: go.Figure, title: str, height: int = 640) -> None:
    """One shared look for every interactive chart, applied after traces exist
    so axis styling hits all subplots.
    """
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=19, color=INK)),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color=INK),
        height=height,
        margin=dict(l=70, r=40, t=85, b=60),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12),
        ),
        hoverlabel=dict(bgcolor="#1d2636", bordercolor=GRID_COLOR, font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    fig.add_annotation(
        text=WATERMARK, xref="paper", yref="paper", x=0.99, y=1.05,
        xanchor="right", yanchor="bottom", showarrow=False,
        font=dict(size=10, color=INK_MUTED),
    )


def _write(fig: go.Figure, filename: str) -> Path:
    out_path = OUTPUTS_DIR / filename
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def _future_index(historical_index: pd.Index, n_steps: int) -> list:
    """Continue the historical index forward for n_steps points, with the first
    point coinciding with the last historical date so the fan visually opens
    from the end of the observed series rather than one bar after it.
    """
    if isinstance(historical_index, pd.DatetimeIndex):
        # Short date strings, not Timestamps: the fan chart repeats this axis
        # once per path, and "YYYY-MM-DD" serializes at half the bytes of a
        # full ISO timestamp - that alone roughly halves the HTML size.
        return list(pd.bdate_range(start=historical_index[-1], periods=n_steps).strftime("%Y-%m-%d"))
    start = int(historical_index[-1]) if len(historical_index) else 0
    return list(range(start, start + n_steps))


def plot_monte_carlo_paths(
    historical_spread: pd.Series,
    paths: np.ndarray,
    ticker_a: str,
    ticker_b: str,
    signal_config: SignalConfig | None = None,
    ou_fit: OUFit | None = None,
    history_tail_days: int = 90,
) -> Path:
    """Fan chart: every simulated path at near-zero opacity so the ink density
    itself shows the probability mass, with the median path, a 5th-95th
    percentile band, the recent historical spread leading into the fan, and
    (when a fit + config are supplied) the signal bands converted from z-score
    units into spread units so they sit on the same axis as the paths.

    All ~1000 paths go into ONE scatter trace with None separators between
    paths - Plotly renders one trace of 90k points far faster than 1000
    separate traces, which is what keeps the HTML responsive.
    """
    paths = np.asarray(paths, dtype=float)
    n_paths, n_steps = paths.shape
    tail = historical_spread.iloc[-history_tail_days:]
    future_x = _future_index(tail.index, n_steps)

    # Round the fan's y-values to 6 decimals before serialization: the spread
    # is ~1e-2 scale so this is far below visual resolution, but it halves the
    # JSON payload versus full float64 repr across ~90k points.
    xs: list = []
    ys: list = []
    for path in np.round(paths, 6):
        xs.extend(future_x)
        xs.append(None)
        ys.extend(path.tolist())
        ys.append(None)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=PATH_RGBA, width=1),
            name=f"{n_paths} simulated paths",
            hovertemplate="%{y:.5f}<extra>simulated path</extra>",
        )
    )

    p05 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    median = np.median(paths, axis=0)
    fig.add_trace(
        go.Scatter(
            x=future_x, y=p95, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_x, y=p05, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=BAND_RGBA,
            name="5th-95th percentile",
            hovertemplate="%{y:.5f}<extra>5th-95th pct band</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_x, y=median, mode="lines",
            line=dict(color=MEDIAN_AMBER, width=3),
            name="median path",
            hovertemplate="%{y:.5f}<extra>median path</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(tail.index), y=tail.to_numpy(), mode="lines",
            line=dict(color=HISTORY_INK, width=2),
            name=f"historical spread (last {len(tail)}d)",
            hovertemplate="%{y:.5f}<extra>historical spread</extra>",
        )
    )

    # Signal levels live in z-score units; the fit's mu / stationary_std maps
    # them onto the spread axis so entries/exits are readable off this chart.
    if signal_config is not None and ou_fit is not None and ou_fit.stationary_std:
        sd = ou_fit.stationary_std
        fig.add_hline(y=ou_fit.mu, line=dict(color=INK_MUTED, width=1, dash="dot"),
                      annotation_text="OU mean", annotation_font_color=INK_MUTED)
        for z, color, label in (
            (signal_config.entry_z, ENTRY_ORANGE, "entry"),
            (signal_config.exit_z, EXIT_GREEN, "exit"),
            (signal_config.stop_z, STOP_RED, "stop"),
        ):
            fig.add_hline(
                y=ou_fit.mu + z * sd,
                line=dict(color=color, width=1, dash="dash"),
                annotation_text=f"{label} (+{z}z)", annotation_font_color=color,
            )
            fig.add_hline(
                y=ou_fit.mu - z * sd,
                line=dict(color=color, width=1, dash="dash"),
            )

    _apply_theme(
        fig,
        f"{ticker_a} / {ticker_b} spread — {n_paths} Monte Carlo OU paths, {n_steps - 1} business days ahead",
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Spread (log-price units)")
    return _write(fig, f"interactive_montecarlo_{ticker_a}_{ticker_b}.html")


def plot_interactive_equity(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame | None = None,
    label: str = "portfolio",
) -> Path:
    """Equity curve with its drawdown underneath on a shared x-axis, plus
    entry/exit markers when a trade log is available - drawdown gets its own
    panel (not a second y-axis) so both series keep an honest scale.
    """
    drawdown = equity_curve / equity_curve.cummax() - 1.0

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Scatter(
            x=list(equity_curve.index), y=equity_curve.to_numpy(), mode="lines",
            line=dict(color=DATA_BLUE, width=2), name="equity",
            hovertemplate="$%{y:,.0f}<extra>equity</extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(drawdown.index), y=drawdown.to_numpy(), mode="lines",
            line=dict(color=STOP_RED, width=1.5), fill="tozeroy", fillcolor=DRAWDOWN_RED,
            name="drawdown",
            hovertemplate="%{y:.2%}<extra>drawdown</extra>",
        ),
        row=2, col=1,
    )

    if trade_log is not None and not trade_log.empty:
        for date_col, symbol, color, name in (
            ("entry_date", "triangle-up", EXIT_GREEN, "trade entry"),
            ("exit_date", "triangle-down", ENTRY_ORANGE, "trade exit"),
        ):
            dates = pd.to_datetime(trade_log[date_col])
            marker_y = equity_curve.reindex(dates, method="ffill")
            fig.add_trace(
                go.Scatter(
                    x=list(dates), y=marker_y.to_numpy(), mode="markers",
                    marker=dict(symbol=symbol, size=10, color=color,
                                line=dict(width=1, color=PAPER_BG)),
                    name=name,
                    hovertemplate="%{x|%Y-%m-%d}<extra>" + name + "</extra>",
                ),
                row=1, col=1,
            )

    _apply_theme(fig, f"Equity curve & drawdown — {label}")
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return _write(fig, f"interactive_equity_{label}.html")


def plot_interactive_spread_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    ticker_a: str,
    ticker_b: str,
    signal_config: SignalConfig | None = None,
) -> Path:
    """Spread and z-score stacked on a shared x-axis with a range slider.

    Entry/exit/stop levels are shaded as horizontal regions on the z panel
    (not just lines) so the eye can see how much time the pair spends inside
    each regime, which is what actually determines trade frequency.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.5, 0.5],
    )
    fig.add_trace(
        go.Scatter(
            x=list(spread.index), y=spread.to_numpy(), mode="lines",
            line=dict(color=DATA_BLUE, width=1.7), name="spread",
            hovertemplate="%{y:.5f}<extra>spread</extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(zscore.index), y=zscore.to_numpy(), mode="lines",
            line=dict(color=MEDIAN_AMBER, width=1.7), name="z-score",
            hovertemplate="%{y:+.2f}<extra>z-score</extra>",
        ),
        row=2, col=1,
    )

    if signal_config is not None:
        entry, exit_, stop = signal_config.entry_z, signal_config.exit_z, signal_config.stop_z
        # Exit zone (mean reversion complete) in green; entry-to-stop zone
        # (position territory) in orange; the stop itself as a hard red line.
        fig.add_hrect(y0=-exit_, y1=exit_, fillcolor=EXIT_GREEN, opacity=0.10,
                      line_width=0, layer="below", row=2, col=1)
        for sign in (1, -1):
            fig.add_hrect(y0=sign * entry, y1=sign * stop, fillcolor=ENTRY_ORANGE,
                          opacity=0.12, line_width=0, layer="below", row=2, col=1)
            fig.add_hline(y=sign * stop, line=dict(color=STOP_RED, width=1, dash="dash"),
                          row=2, col=1)
        fig.add_annotation(
            text=f"bands: exit ±{exit_}, entry ±{entry}, stop ±{stop}",
            xref="paper", yref="paper", x=0.01, y=-0.13, showarrow=False,
            font=dict(size=11, color=INK_MUTED),
        )

    _apply_theme(fig, f"{ticker_a} / {ticker_b} — spread & rolling z-score", height=700)
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return _write(fig, f"interactive_spread_zscore_{ticker_a}_{ticker_b}.html")


def plot_pnl_distribution(
    pnl_per_path: np.ndarray,
    summary: dict | None = None,
    label: str = "pair",
) -> Path:
    """Histogram of simulated per-path strategy P&L with the distribution's
    key markers drawn on it - the chart's job is to show whether the P&L mass
    sits right of zero, so prob-of-profit is stated on the chart, not left to
    the reader to integrate by eye.
    """
    pnl = np.asarray(pnl_per_path, dtype=float)
    if summary is None:
        summary = summarize_pnl(pnl)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=pnl, nbinsx=60, marker=dict(color=BAND_RGBA.replace("0.16", "0.55"),
                                          line=dict(color=DATA_BLUE, width=1)),
            name="per-path P&L",
            hovertemplate="P&L %{x}<br>paths: %{y}<extra></extra>",
        )
    )
    for value, color, dash, name in (
        (summary["mean"], MEDIAN_AMBER, "solid", "mean"),
        (summary["median"], HISTORY_INK, "dash", "median"),
        (summary["p05"], STOP_RED, "dot", "5th pct"),
        (summary["p95"], EXIT_GREEN, "dot", "95th pct"),
    ):
        fig.add_vline(
            x=value, line=dict(color=color, width=1.5, dash=dash),
            annotation_text=f"{name}: {value:,.0f}", annotation_font_color=color,
            annotation_position="top",
        )
    fig.add_vline(x=0.0, line=dict(color=INK_MUTED, width=1))
    fig.add_annotation(
        text=(f"prob(profit) = {summary['prob_profit']:.1%}"
              f"<br>{summary['n_paths']} simulated paths"),
        xref="paper", yref="paper", x=0.02, y=0.95, xanchor="left", showarrow=False,
        font=dict(size=14, color=INK), align="left",
        bgcolor="rgba(29, 38, 54, 0.8)", bordercolor=GRID_COLOR, borderwidth=1,
    )

    _apply_theme(fig, f"Monte Carlo strategy P&L distribution — {label}", height=560)
    fig.update_xaxes(title_text="P&L per path ($, simplified spread-change convention)")
    fig.update_yaxes(title_text="Number of paths")
    return _write(fig, f"interactive_pnl_distribution_{label}.html")
