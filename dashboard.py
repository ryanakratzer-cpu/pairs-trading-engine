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
            m3.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
            m4.metric("Max drawdown", f"{metrics['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
