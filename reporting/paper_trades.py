"""Live paper-trading log: pre-registered predictions, graded honestly after.

WHY this exists and why it is APPEND-ONLY: the point of this log is to find out
whether discretionary, council-driven calls actually work in real time. That
only means something if the prediction and its reasoning are committed BEFORE
the outcome is known and are never touched afterwards. Every integrity
property here exists to make hindsight-editing impossible:

  * rows are only ever APPENDED — no function in this module updates or deletes
    an existing row, and grading happens ON READ (never written back);
  * `opened_at_utc` is stamped at write time, and `rationale` (why we took the
    trade) is captured in the same row as the entry price, so a thesis cannot
    be quietly rewritten after the fact;
  * `model_signal` records what the SYSTEMATIC engine said at that same moment,
    so a discretionary call can never take credit for the model's discipline —
    and `model_agrees` makes disagreement explicit and countable;
  * a losing trade stays in the log exactly as written. Deleting or re-scoring
    a bad call defeats the entire purpose.

SIMULATED ONLY. Like every other module in this project this places no orders
and has no broker connectivity — it records beliefs and marks them to market.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data.loader import fetch_price_history

DEFAULT_PAPER_TRADE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "paper_trades.csv"

# Append-only schema. Every write reindexes to this, so a later code change
# cannot silently shift values into the wrong column.
PAPER_TRADE_COLUMNS = [
    "trade_id",          # stable id linking an OPEN row to its CLOSE row
    "action",            # OPEN | CLOSE
    "timestamp_utc",     # when the belief was committed (not when it was graded)
    "session_date",      # trading session the decision belongs to
    "ticker_a",
    "ticker_b",
    "direction",         # LONG_SPREAD (long A / short B) | SHORT_SPREAD | FLAT
    "conviction",        # low | medium | high — stated up front, graded later
    "hedge_ratio",
    "price_a",
    "price_b",
    "zscore",
    "notional",
    "model_signal",      # what the SYSTEMATIC engine said at this same moment
    "model_agrees",      # did the discretionary call match the model?
    "rationale",         # WHY — committed at entry, never edited
    "council_summary",   # the council's verdict in one line
]

OPEN_ACTION = "OPEN"
CLOSE_ACTION = "CLOSE"
DIRECTIONS = ("LONG_SPREAD", "SHORT_SPREAD", "FLAT")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_decision(
    trade_id: str,
    action: str,
    ticker_a: str,
    ticker_b: str,
    direction: str,
    conviction: str,
    hedge_ratio: float,
    price_a: float,
    price_b: float,
    zscore: float,
    rationale: str,
    council_summary: str,
    model_signal: str,
    session_date: str | None = None,
    notional: float = 10_000.0,
    path: Path | str = DEFAULT_PAPER_TRADE_PATH,
) -> int:
    """Append ONE decision row. Returns the number of rows written (always 1).

    There is deliberately no update/delete counterpart: a decision, once
    committed, is part of the record. Grading is done on read by
    `mark_to_market` / `grade_closed_trades`, which never write back.
    """
    if action not in (OPEN_ACTION, CLOSE_ACTION):
        raise ValueError(f"action must be {OPEN_ACTION} or {CLOSE_ACTION}, got {action!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    path = Path(path)
    row = {
        "trade_id": trade_id,
        "action": action,
        "timestamp_utc": _utc_now(),
        "session_date": session_date or datetime.now().strftime("%Y-%m-%d"),
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "direction": direction,
        "conviction": conviction,
        "hedge_ratio": hedge_ratio,
        "price_a": price_a,
        "price_b": price_b,
        "zscore": zscore,
        "notional": notional,
        "model_signal": model_signal,
        "model_agrees": _model_agrees(direction, model_signal),
        "rationale": rationale,
        "council_summary": council_summary,
    }
    frame = pd.DataFrame([row]).reindex(columns=PAPER_TRADE_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
    return 1


def _model_agrees(direction: str, model_signal: str) -> bool:
    """Did the discretionary call match what the systematic engine said?

    Kept explicit (and stored) so the log can answer a question the equity
    curve cannot: when we overrode the model, were we right to?
    """
    model_signal = (model_signal or "").upper()
    if direction == "LONG_SPREAD":
        return model_signal == "ENTER_LONG_SPREAD"
    if direction == "SHORT_SPREAD":
        return model_signal == "ENTER_SHORT_SPREAD"
    # FLAT agrees with anything that isn't an entry instruction.
    return model_signal not in ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD")


def load_trades(path: Path | str = DEFAULT_PAPER_TRADE_PATH) -> pd.DataFrame:
    """Read the log defensively; returns an empty frame when absent/unreadable."""
    path = Path(path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame(columns=PAPER_TRADE_COLUMNS)
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001 — an unreadable log must not crash a runner
        return pd.DataFrame(columns=PAPER_TRADE_COLUMNS)
    for col in PAPER_TRADE_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame


def _pair_pnl(direction: str, hedge_ratio: float, entry_a: float, entry_b: float,
              now_a: float, now_b: float, notional: float) -> float:
    """Dollar P&L of a dollar-neutral spread position, log-spread convention.

    position = +1 for LONG_SPREAD (long A / short hedge_ratio*B), -1 for SHORT.
    A FLAT row earns exactly 0 — never whatever the spread happened to do.
    """
    if direction == "FLAT":
        return 0.0
    position = 1 if direction == "LONG_SPREAD" else -1
    entry_spread = np.log(entry_a) - hedge_ratio * np.log(entry_b)
    now_spread = np.log(now_a) - hedge_ratio * np.log(now_b)
    return float(position * (now_spread - entry_spread) * notional)


def mark_to_market(
    path: Path | str = DEFAULT_PAPER_TRADE_PATH,
    price_fetcher: Callable[..., pd.DataFrame] = fetch_price_history,
    as_of: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Mark every OPEN trade that has no matching CLOSE to current prices.

    GRADE-ON-READ: this never writes to the log. `price_fetcher` is injectable
    so tests need no network. Returns (open positions frame, summary dict).
    """
    trades = load_trades(path)
    if trades.empty:
        return pd.DataFrame(), {"n_open": 0, "total_unrealized_pnl": 0.0}

    closed_ids = set(trades.loc[trades["action"] == CLOSE_ACTION, "trade_id"])
    opens = trades[(trades["action"] == OPEN_ACTION) & (~trades["trade_id"].isin(closed_ids))]
    opens = opens[opens["direction"] != "FLAT"]
    if opens.empty:
        return pd.DataFrame(), {"n_open": 0, "total_unrealized_pnl": 0.0}

    end = as_of or datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    tickers = sorted({t for t in opens["ticker_a"]} | {t for t in opens["ticker_b"]})
    try:
        prices = price_fetcher(tickers, start=start, end=end)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(), {"n_open": len(opens), "total_unrealized_pnl": None,
                                "error": "price fetch failed"}

    rows = []
    for trade in opens.itertuples():
        if trade.ticker_a not in prices.columns or trade.ticker_b not in prices.columns:
            continue
        now_a = float(prices[trade.ticker_a].dropna().iloc[-1])
        now_b = float(prices[trade.ticker_b].dropna().iloc[-1])
        pnl = _pair_pnl(trade.direction, float(trade.hedge_ratio), float(trade.price_a),
                        float(trade.price_b), now_a, now_b, float(trade.notional))
        rows.append({
            "trade_id": trade.trade_id,
            "pair": f"{trade.ticker_a}/{trade.ticker_b}",
            "direction": trade.direction,
            "conviction": trade.conviction,
            "opened_at": trade.timestamp_utc,
            "entry_z": trade.zscore,
            "entry_price_a": trade.price_a,
            "entry_price_b": trade.price_b,
            "current_price_a": now_a,
            "current_price_b": now_b,
            "unrealized_pnl": pnl,
            "unrealized_pct": 100.0 * pnl / float(trade.notional),
            "model_agreed": trade.model_agrees,
        })
    frame = pd.DataFrame(rows)
    summary = {
        "n_open": len(frame),
        "total_unrealized_pnl": float(frame["unrealized_pnl"].sum()) if len(frame) else 0.0,
        "n_winning": int((frame["unrealized_pnl"] > 0).sum()) if len(frame) else 0,
        "n_losing": int((frame["unrealized_pnl"] < 0).sum()) if len(frame) else 0,
    }
    return frame, summary


def grade_closed_trades(path: Path | str = DEFAULT_PAPER_TRADE_PATH) -> tuple[pd.DataFrame, dict]:
    """Score every completed OPEN->CLOSE round trip. Read-only, like everything
    else here: a losing trade is reported as a loss, not quietly re-scored.

    Returns (per-trade frame, summary with hit_rate, total_pnl, and the
    override record — how we did when we DISAGREED with the systematic model).
    """
    trades = load_trades(path)
    if trades.empty:
        return pd.DataFrame(), {"n_closed": 0, "hit_rate": None, "total_pnl": 0.0}

    opens = trades[trades["action"] == OPEN_ACTION].set_index("trade_id", drop=False)
    closes = trades[trades["action"] == CLOSE_ACTION].set_index("trade_id", drop=False)
    rows = []
    for trade_id, close in closes.iterrows():
        if trade_id not in opens.index:
            continue
        opened = opens.loc[trade_id]
        if isinstance(opened, pd.DataFrame):  # duplicate ids: take the first open
            opened = opened.iloc[0]
        pnl = _pair_pnl(
            str(opened["direction"]), float(opened["hedge_ratio"]),
            float(opened["price_a"]), float(opened["price_b"]),
            float(close["price_a"]), float(close["price_b"]), float(opened["notional"]),
        )
        rows.append({
            "trade_id": trade_id,
            "pair": f"{opened['ticker_a']}/{opened['ticker_b']}",
            "direction": opened["direction"],
            "conviction": opened["conviction"],
            "opened_at": opened["timestamp_utc"],
            "closed_at": close["timestamp_utc"],
            "entry_z": opened["zscore"],
            "exit_z": close["zscore"],
            "pnl": pnl,
            "pnl_pct": 100.0 * pnl / float(opened["notional"]),
            "correct": bool(pnl > 0),
            "model_agreed": bool(opened["model_agrees"]),
            "rationale": opened["rationale"],
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {"n_closed": 0, "hit_rate": None, "total_pnl": 0.0}

    overrides = frame[~frame["model_agreed"]]
    summary = {
        "n_closed": len(frame),
        "hit_rate": float(frame["correct"].mean()),
        "total_pnl": float(frame["pnl"].sum()),
        "avg_win": float(frame.loc[frame["pnl"] > 0, "pnl"].mean()) if (frame["pnl"] > 0).any() else 0.0,
        "avg_loss": float(frame.loc[frame["pnl"] < 0, "pnl"].mean()) if (frame["pnl"] < 0).any() else 0.0,
        # The question a discretionary log exists to answer:
        "n_overrides": len(overrides),
        "override_hit_rate": float(overrides["correct"].mean()) if len(overrides) else None,
        "override_pnl": float(overrides["pnl"].sum()) if len(overrides) else 0.0,
    }
    return frame, summary
