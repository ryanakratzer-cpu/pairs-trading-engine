"""Streamlit + Plotly dashboard for the pairs-trading engine.

Interactive front-end over the existing modules — it wires to the REAL package
layout (there is no src/ directory in this project):

    screening/cointegration.py  -> test_pair_cointegration  (ADF p, hedge ratio)
    signals/spread.py           -> build_spread, rolling_zscore, generate_signals
    backtest/simulator.py       -> PairBacktester            (equity curve, trades)
    backtest/metrics.py         -> compute_metrics           (net P&L, Sharpe, ...)
    data/loader.py              -> fetch_price_history        (yfinance + CSV cache)

Research/monitoring only — like every other entry point in this project it
computes signals and never places an order or talks to a broker.

Run it (after `py -m pip install streamlit`):

    streamlit run dashboard.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest.metrics import compute_metrics
from backtest.simulator import PairBacktestConfig, PairBacktester
from data.loader import align_and_clean, fetch_price_history
from screening.cointegration import test_pair_cointegration
from screening.focus_book import FOCUS_BOOK
from screening.universe import default_universe
from signals.spread import SignalConfig, build_spread, generate_signals, rolling_zscore

# Chart thresholds per the dashboard spec: entry at Z = +/-2.0 (red), exit at
# Z = 0.0 (green). NOTE: the engine's SignalConfig exits at |z| <= 0.5 by
# default, so the backtest below still uses the engine's real bands; these two
# constants only drive the visual guide lines the spec asked for.
ENTRY_Z = 2.0
EXIT_Z = 0.0

# Executing-signal events worth logging (everything that isn't "do nothing").
SIGNAL_EVENTS = ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD", "EXIT", "STOP_LOSS", "TIME_EXIT")

COLOR_Z = "#5aa2f0"
COLOR_ENTRY = "#e05c5c"
COLOR_EXIT = "#4fbf7e"
COLOR_GRID = "#2a3346"
COLOR_BENCH = "#c9a227"  # SPY benchmark line — distinct from the pair blue

# Append-only session log written by the daily tracker (run_daily_tracker.py).
# The dashboard only ever READS this file; the tracker owns it.
TRACK_RECORD_PATH = Path(__file__).resolve().parent / "outputs" / "daily_performance.csv"
TRACK_RECORD_COMMAND = "py run_daily_tracker.py"
# NOTE on the two return columns (see reporting/daily_performance.py):
#   strategy_return_pct = what WE earned (position-aware; exactly 0 when flat)
#   spread_move_pct     = what the SPREAD did (position-agnostic DIAGNOSTIC)
# Every performance/benchmark number on this tab uses strategy_return_pct.
# spread_move_pct is only ever shown as a separately-labelled diagnostic.
TRACK_RECORD_COLUMNS = [
    "date", "ticker_a", "ticker_b", "hedge_ratio", "adf_pvalue", "is_cointegrated",
    "half_life_days", "open_z", "close_z", "signal", "position",
    "strategy_return_pct", "spread_move_pct", "pair_pnl_usd",
    "ticker_a_return_pct", "ticker_b_return_pct", "spy_return_pct",
    "excess_return_pct", "logged_at",
]

LOOKBACK_CHOICES = {
    "6 months (~126d)": 180,
    "1 year (~252d)": 365,
    "2 years (~500d)": 730,
    "3 years (~750d)": 1095,
    "Full history (~900d)": 900,
}


def _position_label(position: int) -> str:
    if position > 0:
        return "Long spread (long A / short B)"
    if position < 0:
        return "Short spread (short A / long B)"
    return "Flat"


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker_a: str, ticker_b: str, lookback_days: int) -> pd.DataFrame:
    """Fetch and clean the two-ticker price panel. Cached so widget changes that
    don't alter these three inputs don't refetch. fetch_price_history also keeps
    its own on-disk CSV cache, so first load is the only slow one."""
    end = datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    raw = fetch_price_history([ticker_a, ticker_b], start=start, end=end)
    prices, _dropped = align_and_clean(raw)
    return prices


def zscore_figure(zscore: pd.Series, ticker_a: str, ticker_b: str) -> go.Figure:
    """Z-score timeline with entry (+/-2.0, red) and exit (0.0, green) guides."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(zscore.index), y=zscore.to_numpy(), mode="lines",
            line=dict(color=COLOR_Z, width=1.8), name="Z-score",
            hovertemplate="%{x|%Y-%m-%d}<br>z = %{y:+.2f}<extra></extra>",
        )
    )
    for y, color, label in (
        (ENTRY_Z, COLOR_ENTRY, f"entry +{ENTRY_Z}"),
        (-ENTRY_Z, COLOR_ENTRY, f"entry -{ENTRY_Z}"),
        (EXIT_Z, COLOR_EXIT, f"exit {EXIT_Z}"),
    ):
        fig.add_hline(
            y=y, line=dict(color=color, width=1.4, dash="dash"),
            annotation_text=label, annotation_position="right",
            annotation_font_color=color,
        )
    fig.update_layout(
        template="plotly_dark",
        title=f"{ticker_a} / {ticker_b} — rolling Z-score of the spread",
        height=460, margin=dict(l=60, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", showlegend=False,
    )
    fig.update_xaxes(title_text="Date", gridcolor=COLOR_GRID)
    fig.update_yaxes(title_text="Z-score", gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    return fig


def load_track_record() -> pd.DataFrame:
    """Read the tracker's append-only session log DEFENSIVELY.

    Returns an empty DataFrame when the file is missing, empty, unreadable or
    malformed — the dashboard must never crash just because the daily tracker
    hasn't run yet (or is mid-write). No tracker module is imported here on
    purpose: the CSV is the only contract between the two.
    """
    try:
        if not TRACK_RECORD_PATH.exists() or TRACK_RECORD_PATH.stat().st_size == 0:
            return pd.DataFrame(columns=TRACK_RECORD_COLUMNS)
        df = pd.read_csv(TRACK_RECORD_PATH)
    except Exception:  # noqa: BLE001 — any parse/IO failure degrades to "no data yet"
        return pd.DataFrame(columns=TRACK_RECORD_COLUMNS)

    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=TRACK_RECORD_COLUMNS)

    for col in TRACK_RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric = ["hedge_ratio", "adf_pvalue", "half_life_days", "open_z", "close_z",
               "position", "strategy_return_pct", "spread_move_pct", "pair_pnl_usd",
               "ticker_a_return_pct", "ticker_b_return_pct", "spy_return_pct",
               "excess_return_pct"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=TRACK_RECORD_COLUMNS)

    df["pair"] = df["ticker_a"].astype(str) + "/" + df["ticker_b"].astype(str)
    # A flat row earns nothing, whatever the spread did — never fall back to the
    # spread move here, that was exactly the bug this schema split fixed.
    df["strategy_return_pct"] = df["strategy_return_pct"].fillna(
        df["position"].fillna(0) * df["spread_move_pct"]
    )
    # excess = what WE earned − SPY; recompute where the tracker left it blank.
    df["excess_return_pct"] = df["excess_return_pct"].fillna(
        df["strategy_return_pct"] - df["spy_return_pct"]
    )
    return df.sort_values(["date", "pair"]).reset_index(drop=True)


def _sessions_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a (possibly multi-pair) log into one row per session date.

    Across the whole book a session's return is the equal-weight average of that
    day's POSITION-AWARE strategy returns; SPY is the same for every pair so its
    mean is just that day's benchmark return. A session is ACTIVE when at least
    one tracked pair actually held a position.
    """
    df = df.copy()
    df["is_active"] = df["position"].fillna(0) != 0
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            strategy_return_pct=("strategy_return_pct", "mean"),
            spread_move_pct=("spread_move_pct", "mean"),
            spy_return_pct=("spy_return_pct", "mean"),
            pair_pnl_usd=("pair_pnl_usd", "sum"),
            n_active=("is_active", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["is_active"] = daily["n_active"] > 0
    daily["excess_return_pct"] = daily["strategy_return_pct"] - daily["spy_return_pct"]
    # Compound the daily percentage returns into cumulative growth (in %).
    daily["cum_strategy_pct"] = (
        (1 + daily["strategy_return_pct"].fillna(0.0) / 100).cumprod() - 1
    ) * 100
    daily["cum_spy_pct"] = ((1 + daily["spy_return_pct"].fillna(0.0) / 100).cumprod() - 1) * 100
    return daily


def cumulative_return_figure(daily: pd.DataFrame, scope_label: str) -> go.Figure:
    """Cumulative STRATEGY return (position-aware) vs cumulative SPY return."""
    fig = go.Figure()
    for col, color, name in (
        ("cum_strategy_pct", COLOR_Z, "Strategy (cumulative, position-aware)"),
        ("cum_spy_pct", COLOR_BENCH, "SPY benchmark (cumulative)"),
    ):
        fig.add_trace(
            go.Scatter(
                x=list(daily["date"]), y=daily[col].to_numpy(), mode="lines+markers",
                line=dict(color=color, width=1.8), marker=dict(size=5, color=color),
                name=name,
                hovertemplate="%{x|%Y-%m-%d}<br>" + name + " = %{y:+.2f}%<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line=dict(color=COLOR_GRID, width=1.2, dash="dash"))
    fig.update_layout(
        template="plotly_dark",
        title=f"{scope_label} — cumulative return vs SPY ({len(daily)} recorded sessions)",
        height=460, margin=dict(l=60, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title_text="Session date", gridcolor=COLOR_GRID)
    fig.update_yaxes(title_text="Cumulative return (%)", gridcolor=COLOR_GRID,
                     zerolinecolor=COLOR_GRID)
    return fig


def excess_return_figure(daily: pd.DataFrame, scope_label: str) -> go.Figure:
    """Per-session excess return (strategy − SPY), green when the book beat the market."""
    excess = daily["excess_return_pct"].fillna(0.0)
    colors = [COLOR_EXIT if v >= 0 else COLOR_ENTRY for v in excess]
    fig = go.Figure(
        go.Bar(
            x=list(daily["date"]), y=excess.to_numpy(), marker_color=colors,
            name="Excess return",
            hovertemplate="%{x|%Y-%m-%d}<br>excess = %{y:+.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line=dict(color=COLOR_GRID, width=1.4))
    fig.update_layout(
        template="plotly_dark",
        title=f"{scope_label} — daily excess return vs SPY (strategy − SPY)",
        height=360, margin=dict(l=60, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", showlegend=False,
    )
    fig.update_xaxes(title_text="Session date", gridcolor=COLOR_GRID)
    fig.update_yaxes(title_text="Excess return (%)", gridcolor=COLOR_GRID,
                     zerolinecolor=COLOR_GRID)
    return fig


def render_track_record(ticker_a: str, ticker_b: str) -> None:
    """Multi-day track record tab — accumulating record of tracker sessions."""
    st.subheader("Track record (multi-day)")
    st.caption(
        "Accumulating session-by-session record written by the daily tracker. "
        "Research only — this is a log of computed signals, not executed orders."
    )

    record = load_track_record()
    if record.empty:
        st.info(
            "No track record yet — the daily tracker hasn't written any sessions.\n\n"
            f"Run it from the project root to log today's session:\n\n"
            f"```\n{TRACK_RECORD_COMMAND}\n```\n\n"
            f"It appends one row per pair per session to `outputs/daily_performance.csv`, "
            "and this tab fills in as the history builds."
        )
        return

    # ---- Scope filter: whole book, or a single pair ---------------------------
    pairs = sorted(record["pair"].dropna().unique().tolist())
    selected_pair = f"{ticker_a}/{ticker_b}"
    options = ["All pairs (whole book)", *pairs]
    default_index = options.index(selected_pair) if selected_pair in pairs else 0
    scope = st.selectbox(
        "View", options, index=default_index,
        help="Defaults to the sidebar-selected pair when it appears in the log.",
    )
    scoped = record if scope == options[0] else record[record["pair"] == scope]
    scope_label = "Whole book" if scope == options[0] else scope

    if scoped.empty:
        st.info(f"No recorded sessions for {scope} yet.")
        return

    daily = _sessions_by_date(scoped)

    # ---- Headline cards -------------------------------------------------------
    # Everything below is POSITION-AWARE: a session where nothing was held
    # returns 0%, not whatever the spread happened to do.
    n_rows = len(scoped)
    cum_strategy = float(daily["cum_strategy_pct"].iloc[-1])
    cum_spy = float(daily["cum_spy_pct"].iloc[-1])
    total_pnl = float(scoped["pair_pnl_usd"].fillna(0.0).sum())
    graded = daily["excess_return_pct"].dropna()
    n_beat = int((graded > 0).sum())
    pct_beat = (n_beat / len(graded)) if len(graded) else 0.0
    avg_excess = float(graded.mean()) if len(graded) else 0.0
    n_active = int(daily["is_active"].sum())
    n_flat = int(len(daily) - n_active)

    t1, t2, t3, t4, t5, t6 = st.columns(6)
    t1.metric("Sessions recorded", f"{n_rows:,}",
              help=f"Rows logged for {scope_label} across {len(daily)} distinct dates.")
    t2.metric("Active / flat sessions", f"{n_active} / {n_flat}",
              help="Sessions where at least one tracked pair held a position, vs "
                   "sessions the book sat out entirely. Flat sessions earn exactly "
                   "0% by construction.")
    t3.metric("Cumulative P&L", f"${total_pnl:,.0f}",
              help="Sum of per-session position-aware P&L in the log.")
    t4.metric("Cumulative return", f"{cum_strategy:+.2f}%",
              delta=f"{cum_strategy - cum_spy:+.2f}% vs SPY",
              help="Compounded position-aware session returns — agrees in sign with "
                   f"cumulative P&L. SPY over the same sessions: {cum_spy:+.2f}%.")
    t5.metric("Days beating the market", f"{n_beat} / {len(graded)}",
              delta=f"{pct_beat:.0%} hit rate",
              delta_color="normal" if pct_beat >= 0.5 else "inverse",
              help="Sessions where what WE earned exceeded SPY. Flat days only beat "
                   "a market that fell.")
    t6.metric("Avg daily excess", f"{avg_excess:+.2f}%",
              help="Mean of (strategy return − SPY return) per session.")

    if n_flat:
        st.caption(
            f"⚑ {n_flat} of {len(daily)} recorded session(s) were FLAT — the book held "
            "nothing all day and therefore earned exactly 0%. Only the "
            f"{n_active} active session(s) reflect the strategy actually trading."
        )
    if len(daily) < 5:
        st.caption(
            "Only a handful of sessions so far — treat these numbers as directional. "
            "Consistency is only judgeable once the log has a few weeks in it."
        )

    # ---- Charts ---------------------------------------------------------------
    st.plotly_chart(cumulative_return_figure(daily, scope_label), use_container_width=True)
    st.plotly_chart(excess_return_figure(daily, scope_label), use_container_width=True)

    # ---- Session table (most recent first) ------------------------------------
    st.subheader("Recorded sessions")
    recent = scoped.sort_values("date", ascending=False)
    table = pd.DataFrame({
        "Date": [d.date() for d in recent["date"]],
        "Pair": recent["pair"].to_numpy(),
        "Signal": recent["signal"].fillna("—").astype(str).str.replace("_", " ").str.title().to_numpy(),
        "Close Z": [f"{z:+.2f}" if pd.notna(z) else "—" for z in recent["close_z"]],
        "Position": [_position_label(int(p)) if pd.notna(p) else "—" for p in recent["position"]],
        "Our return": [f"{v:+.2f}%" if pd.notna(v) else "—" for v in recent["strategy_return_pct"]],
        "SPY return": [f"{v:+.2f}%" if pd.notna(v) else "—" for v in recent["spy_return_pct"]],
        "Excess": [f"{v:+.2f}%" if pd.notna(v) else "—" for v in recent["excess_return_pct"]],
        "P&L": [f"${v:,.0f}" if pd.notna(v) else "—" for v in recent["pair_pnl_usd"]],
        # Diagnostic only — what the SPREAD did, regardless of what we held.
        "Spread move (diag.)": [
            f"{v:+.2f}%" if pd.notna(v) else "—" for v in recent["spread_move_pct"]
        ],
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(
        f"Source: `{TRACK_RECORD_PATH.name}` — appended by `{TRACK_RECORD_COMMAND}`. "
        "**Our return** is position-aware (0% on flat days) and is the only column "
        "benchmarked against SPY. **Spread move** is a diagnostic showing what the "
        "spread did regardless of whether we were in it."
    )


def main() -> None:
    st.set_page_config(page_title="Pairs Trading Dashboard", page_icon="📈", layout="wide")
    st.title("📈 Pairs Trading Engine — live dashboard")
    st.caption("Cointegration-based statistical arbitrage. Research/monitoring only — no orders are ever placed.")

    # ---- Sidebar: pair + lookback selection -----------------------------------
    st.sidebar.header("Configuration")

    universe = default_universe()
    focus_labels = {p.label: p.key for p in FOCUS_BOOK}
    quick = st.sidebar.selectbox(
        "Quick-pick (focus book)", ["— custom —", *focus_labels.keys()],
        help="Persistence-selected pairs from screening/focus_book.py",
    )
    if quick in focus_labels:
        default_a, default_b = focus_labels[quick]
    else:
        default_a, default_b = "GDX", "GLD"

    ticker_a = st.sidebar.selectbox("Ticker A", universe, index=universe.index(default_a))
    ticker_b = st.sidebar.selectbox("Ticker B", universe, index=universe.index(default_b))

    lookback_label = st.sidebar.select_slider(
        "Historical lookback window", options=list(LOOKBACK_CHOICES.keys()),
        value="3 years (~750d)",
    )
    lookback_days = LOOKBACK_CHOICES[lookback_label]

    # ---- Tabs: point-in-time analysis vs the accumulating multi-day record ----
    # The sidebar above is shared by both tabs. The track record is rendered
    # first so that a data/fetch failure in the pair analysis (which calls
    # st.stop()) can never blank out the history tab.
    tab_pair, tab_record = st.tabs(["Pair analysis", "Track record (multi-day)"])
    with tab_record:
        render_track_record(ticker_a, ticker_b)
    with tab_pair:
        render_pair_analysis(ticker_a, ticker_b, lookback_days)


def render_pair_analysis(ticker_a: str, ticker_b: str, lookback_days: int) -> None:
    """Point-in-time analysis of a single pair (the original dashboard body)."""
    if ticker_a == ticker_b:
        st.warning("Pick two different tickers to form a pair.")
        st.stop()

    # ---- Load + analyze -------------------------------------------------------
    try:
        with st.spinner(f"Loading {ticker_a}/{ticker_b}…"):
            prices = load_prices(ticker_a, ticker_b, lookback_days)
    except Exception as exc:  # noqa: BLE001 — surface any fetch/clean failure in the UI
        st.error(f"Could not load prices for {ticker_a}/{ticker_b}: {exc}")
        st.stop()

    if ticker_a not in prices.columns or ticker_b not in prices.columns or len(prices) < 40:
        st.error(f"Not enough overlapping price history for {ticker_a}/{ticker_b} in this window.")
        st.stop()

    cfg = SignalConfig()
    eg = test_pair_cointegration(prices[ticker_a], prices[ticker_b], ticker_a=ticker_a, ticker_b=ticker_b)
    spread = build_spread(prices[ticker_a], prices[ticker_b], eg.hedge_ratio)
    zscore = rolling_zscore(spread, cfg.zscore_window).dropna()
    signals = generate_signals(zscore, cfg)
    current_z = float(zscore.iloc[-1]) if len(zscore) else float("nan")

    # Full backtest needs more than the re-cointegration window of bars; on short
    # lookbacks show signals/z only and mark P&L unavailable rather than erroring.
    bt_config = PairBacktestConfig()
    net_pnl = None
    trade_log = pd.DataFrame()
    metrics: dict = {}
    if len(prices) > bt_config.recheck_window_days:
        bt = PairBacktester(bt_config).run(prices, [(ticker_a, ticker_b)])
        equity = bt["equity_curve"]
        trade_log = bt["trade_log"]
        metrics = compute_metrics(equity, trade_log)
        net_pnl = float(equity.iloc[-1] - bt_config.initial_capital)

    # ---- Metric cards ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "ADF p-value", f"{eg.adf_pvalue:.4f}",
        delta="cointegrated" if eg.is_cointegrated else "not cointegrated",
        delta_color="normal" if eg.is_cointegrated else "inverse",
        help="Engle-Granger ADF test on the spread; < 0.05 ⇒ cointegrated.",
    )
    c2.metric("Hedge ratio (β)", f"{eg.hedge_ratio:.3f}", help="Spread = log(A) − β·log(B).")
    c3.metric("Current Z-score", f"{current_z:+.2f}", help="Latest rolling z-score of the spread.")
    if net_pnl is None:
        c4.metric("Net P&L", "—", help=f"Backtest needs > {bt_config.recheck_window_days} trading days; widen the lookback.")
    else:
        c4.metric(
            "Net P&L", f"${net_pnl:,.0f}",
            delta=f"{metrics.get('total_return', 0.0):.2%} return",
            help="Realized + marked P&L of the single-pair backtest over this window.",
        )

    # ---- Performance metrics row ----------------------------------------------
    st.subheader("Performance metrics")
    pf = metrics.get("profit_factor") if metrics else None
    perf = [
        ("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}" if metrics else "—",
         "Annualized, risk-free rate 0, 252 trading days."),
        ("Annualized return", f"{metrics['annualized_return']:.1%}" if metrics else "—", None),
        ("Annualized volatility", f"{metrics['annualized_vol']:.1%}" if metrics else "—", None),
        ("Profit factor", ("∞" if pf == float("inf") else f"{pf:.2f}") if metrics else "—",
         "Gross profit / gross loss across closed trades."),
        ("Max drawdown", f"{metrics['max_drawdown']:.2%}" if metrics else "—", None),
        ("Half-life (days)", f"{eg.half_life_days:.1f}" if eg.half_life_days else "—",
         "Spread mean-reversion speed (ln 2 / θ)."),
    ]
    for col, (label, value, help_text) in zip(st.columns(len(perf)), perf):
        col.metric(label, value, help=help_text)
    if not metrics:
        st.caption("Sharpe / return / drawdown need the full backtest — widen the lookback window.")

    # ---- Z-score chart --------------------------------------------------------
    st.plotly_chart(zscore_figure(zscore, ticker_a, ticker_b), use_container_width=True)

    # ---- Open trades + executing signals --------------------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("Open position")
        last_position = int(signals["position"].iloc[-1]) if len(signals) else 0
        if last_position == 0:
            st.info("No open position — the pair is currently flat.")
        else:
            # Walk back to the entry bar of the currently-open position.
            positions = signals["position"].to_numpy()
            entry_idx = len(positions) - 1
            while entry_idx > 0 and positions[entry_idx - 1] == last_position:
                entry_idx -= 1
            entry_date = signals.index[entry_idx]
            st.dataframe(
                pd.DataFrame(
                    [{
                        "Pair": f"{ticker_a}/{ticker_b}",
                        "Direction": _position_label(last_position),
                        "Entry date": entry_date.date(),
                        "Bars held": len(signals) - entry_idx,
                        "Entry Z": f"{float(signals['zscore'].iloc[entry_idx]):+.2f}",
                        "Current Z": f"{current_z:+.2f}",
                    }]
                ),
                hide_index=True, use_container_width=True,
            )

    with right:
        st.subheader("Executing signals (most recent)")
        events = signals[signals["event"].isin(SIGNAL_EVENTS)].copy()
        if events.empty:
            st.info("No entry/exit signals fired in this window.")
        else:
            events = events.tail(15).iloc[::-1]
            table = pd.DataFrame({
                "Date": [d.date() for d in events.index],
                "Signal": events["event"].str.replace("_", " ").str.title().to_numpy(),
                "Z-score": [f"{z:+.2f}" for z in events["zscore"]],
                "Position": [_position_label(p) for p in events["position"]],
            })
            st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Executed trades (backtest log)")
    if trade_log.empty:
        st.info("No closed trades in this window (widen the lookback to run the full backtest).")
    else:
        log = trade_log.copy()
        log["direction"] = log["position"].map(_position_label)
        log["pnl"] = log["pnl"].map(lambda v: f"${v:,.0f}")
        for col in ("entry_date", "exit_date"):
            log[col] = pd.to_datetime(log[col]).dt.date
        display_cols = ["ticker_a", "ticker_b", "direction", "entry_date", "exit_date",
                        "holding_days", "pnl", "exit_reason"]
        st.dataframe(log[display_cols], hide_index=True, use_container_width=True)
        if metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades", metrics["n_trades"])
            m2.metric("Win rate", f"{metrics['win_rate']:.0%}")
            m3.metric("Avg win", f"${metrics['avg_win']:,.0f}")
            m4.metric("Avg loss", f"${metrics['avg_loss']:,.0f}")


if __name__ == "__main__":
    main()
