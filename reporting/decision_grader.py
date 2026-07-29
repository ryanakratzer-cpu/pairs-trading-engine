"""Decision grader: was the engine RIGHT on the days it actually acted?

WHY this exists: headline backtest metrics (total return, Sharpe, win rate)
summarize the equity curve, not the *decisions*. The question this module
answers is narrower and more useful for judging trust: on the specific days
|z| crossed the entry band and the state machine took a position, was that
call correct — and when it stopped out, was the stop justified?

Two lenses, deliberately kept separate because they disagree:

  (i) OUTCOME — did the resulting round-trip trade make money, net of the
      backtest's transaction costs and slippage? This is what a P&L statement
      would say. It is the lens that pays the bills, but it conflates the
      signal's quality with the cost model and the position sizing.

 (ii) THESIS — did the spread actually mean-revert after the entry?
          sign(z_entry) * (z_exit - z_entry) < 0  ==>  converged
      This is exactly reporting/journal.py's grading convention (there applied
      to a fixed forward horizon; here applied over the trade's own life), and
      it isolates whether the mean-reversion *prediction* was right regardless
      of whether the edge survived costs.

A trade can converge and still lose to costs; it can stop out and still have
been directionally sane. Where the lenses disagree is the informative part —
persistent "thesis right, outcome wrong" means the signal works but the cost
model eats it; persistent "thesis wrong, outcome right" means the P&L is luck.

Exit decisions are graded on what happened AFTER them, over a forward window:
  STOP_LOSS  — JUSTIFIED if the spread kept diverging (the stop saved money),
               TOO_EARLY if it snapped back right after the stop.
  TIME_EXIT  — JUSTIFIED if abandoning the thesis avoided further divergence,
               PREMATURE if the spread converged just after we gave up.
  EXIT       — CONVERGED if |z| really was back inside the exit band. This is
               near-structural (the state machine only fires EXIT on that
               condition), so it is a consistency check, not evidence.

Signal research only: nothing here places an order. `price_fetcher` is
injectable so the whole module can be exercised with no network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from backtest.simulator import PairBacktestConfig, PairBacktester
from data.loader import align_and_clean, fetch_price_history
from signals.spread import SignalConfig

ENTRY_EVENTS = ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD")
CLOSING_EVENTS = ("EXIT", "STOP_LOSS", "TIME_EXIT")
DECISION_EVENTS = ENTRY_EVENTS + CLOSING_EVENTS

# Exit label used when a position was still open on the pair's last bar — the
# simulator's forced liquidation. Not a decision the engine made, so these are
# reported but excluded from the exit-quality rates.
END_OF_SAMPLE = "END_OF_SAMPLE"

# Verdict vocabulary (strings, not enums, so the DataFrame prints readably).
THESIS_CONVERGED = "CONVERGED"
THESIS_DIVERGED = "DIVERGED"
OUTCOME_PROFIT = "PROFIT"
OUTCOME_LOSS = "LOSS"
EXIT_JUSTIFIED = "JUSTIFIED"
EXIT_TOO_EARLY = "TOO_EARLY"
EXIT_PREMATURE = "PREMATURE"
EXIT_CONVERGED = "CONVERGED"
EXIT_NOT_CONVERGED = "NOT_CONVERGED"
NOT_GRADED = "NOT_GRADED"

DEFAULT_FORWARD_BARS = 10
DEFAULT_LOOKBACK_DAYS = 900


@dataclass
class DecisionGradeResult:
    """Everything the grader produces for one price panel / pair list."""

    decisions: pd.DataFrame
    summary: dict
    per_pair: pd.DataFrame
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def entries(self) -> pd.DataFrame:
        if self.decisions.empty:
            return self.decisions
        return self.decisions[self.decisions["event"].isin(ENTRY_EVENTS)]

    @property
    def exits(self) -> pd.DataFrame:
        if self.decisions.empty:
            return self.decisions
        return self.decisions[self.decisions["event"].isin(CLOSING_EVENTS)]


def _forward_z(zscore: pd.Series, position: int, forward_bars: int) -> float:
    """z-score `forward_bars` bars after `position`, clipped to the series end.

    Returns NaN when the window lands on a NaN z (warm-up / gap), which the
    callers treat as ungradable rather than guessing.
    """
    target = min(position + forward_bars, len(zscore) - 1)
    if target <= position:
        return float("nan")
    value = zscore.iloc[target]
    return float(value) if pd.notna(value) else float("nan")


def _signed_move(z_reference: float, z_from: float, z_to: float) -> float:
    """sign(z_reference) * (z_to - z_from).

    Negative = the spread moved toward its mean from the side the decision was
    made on (convergence); positive = it pushed further out (divergence). Same
    convention as reporting/journal.py's spread_change_z.
    """
    if any(pd.isna(v) for v in (z_reference, z_from, z_to)):
        return float("nan")
    return float(np.sign(z_reference) * (z_to - z_from))


def _grade_pair(
    pair_key: tuple[str, str],
    pair_data: dict,
    trades_by_entry: dict,
    signal_config: SignalConfig,
    forward_bars: int,
) -> list[dict]:
    """Walk one pair's signal series and emit a row per decision event."""
    exit_z = signal_config.exit_z
    signals = pair_data["signals"]
    zscore = signals["zscore"]
    ticker_a, ticker_b = pair_key
    label = f"{ticker_a}/{ticker_b}"
    rows: list[dict] = []

    open_entry: dict | None = None
    open_entry_row: dict | None = None

    for position_idx, (date, signal_row) in enumerate(signals.iterrows()):
        event = signal_row["event"]
        if event not in DECISION_EVENTS:
            continue
        z_now = float(signal_row["zscore"]) if pd.notna(signal_row["zscore"]) else float("nan")

        base = {
            "date": date,
            "pair": label,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "event": event,
            "zscore": z_now,
        }

        if event in ENTRY_EVENTS:
            trade = trades_by_entry.get((ticker_a, ticker_b, pd.Timestamp(date)))
            row = {
                **base,
                "direction": "LONG_SPREAD" if event == "ENTER_LONG_SPREAD" else "SHORT_SPREAD",
                "taken": trade is not None,
                "exit_date": pd.NaT,
                "exit_event": None,
                "z_exit": float("nan"),
                "thesis_move": float("nan"),
                "thesis_verdict": NOT_GRADED,
                "pnl": trade["pnl"] if trade is not None else float("nan"),
                "outcome_verdict": NOT_GRADED,
                "holding_days": trade["holding_days"] if trade is not None else float("nan"),
                "forward_z": float("nan"),
                "forward_move": float("nan"),
                "exit_verdict": NOT_GRADED,
                "lenses_agree": pd.NA,
                # Entry fired at a |z| already past the stop band. The state
                # machine checks entry before stop, so these open a position
                # that is instantly eligible to be stopped out — a structural
                # quirk, not a real call. Flagged because the tiny move back
                # toward the band scores as CONVERGED on the thesis lens while
                # reliably losing money.
                "entry_beyond_stop": (
                    bool(abs(z_now) >= signal_config.stop_z) if pd.notna(z_now) else False
                ),
                "overshoot_exit": False,
            }
            # The trade P&L is known immediately from the log; the thesis lens
            # needs the matching close, filled in when we reach it below.
            if trade is not None and pd.notna(trade["pnl"]):
                row["outcome_verdict"] = OUTCOME_PROFIT if trade["pnl"] > 0 else OUTCOME_LOSS
            rows.append(row)
            open_entry = {"z": z_now, "idx": position_idx, "date": date}
            open_entry_row = row
            continue

        # --- closing events -------------------------------------------------
        forward_z = _forward_z(zscore, position_idx, forward_bars)
        forward_move = _signed_move(z_now, z_now, forward_z)
        row = {
            **base,
            "direction": None,
            "taken": open_entry_row["taken"] if open_entry_row is not None else False,
            "exit_date": date,
            "exit_event": event,
            "z_exit": z_now,
            "thesis_move": float("nan"),
            "thesis_verdict": NOT_GRADED,
            "pnl": open_entry_row["pnl"] if open_entry_row is not None else float("nan"),
            "outcome_verdict": (
                open_entry_row["outcome_verdict"] if open_entry_row is not None else NOT_GRADED
            ),
            "holding_days": (
                open_entry_row["holding_days"] if open_entry_row is not None else float("nan")
            ),
            "forward_z": forward_z,
            "forward_move": forward_move,
            "exit_verdict": NOT_GRADED,
            "lenses_agree": pd.NA,
            "entry_beyond_stop": False,
            "overshoot_exit": False,
        }

        if event == "EXIT":
            # Near-structural: the state machine fires EXIT precisely when |z|
            # re-enters the band. Graded anyway as a consistency check.
            row["exit_verdict"] = (
                EXIT_CONVERGED if pd.notna(z_now) and abs(z_now) <= exit_z else EXIT_NOT_CONVERGED
            )
        elif event == "STOP_LOSS":
            if pd.notna(forward_move):
                # Kept diverging after the stop => the stop saved money.
                row["exit_verdict"] = EXIT_JUSTIFIED if forward_move > 0 else EXIT_TOO_EARLY
        elif event == "TIME_EXIT":
            if pd.notna(forward_move):
                row["exit_verdict"] = EXIT_JUSTIFIED if forward_move > 0 else EXIT_PREMATURE

        # Close out the thesis lens on the entry this exit belongs to.
        if open_entry is not None and open_entry_row is not None:
            _close_thesis(
                open_entry_row, open_entry["z"], z_now, date, event, signal_config.entry_z
            )
            open_entry = None
            open_entry_row = None

        rows.append(row)

    # A position still open on the last bar: the simulator force-liquidates it
    # (END_OF_SAMPLE). Grade the thesis against the final z, but do not emit a
    # decision row — the engine never chose to exit there.
    if open_entry is not None and open_entry_row is not None:
        last_valid = zscore.dropna()
        if not last_valid.empty:
            _close_thesis(
                open_entry_row,
                open_entry["z"],
                float(last_valid.iloc[-1]),
                last_valid.index[-1],
                END_OF_SAMPLE,
                signal_config.entry_z,
            )

    return rows


def _close_thesis(
    entry_row: dict,
    z_entry: float,
    z_exit: float,
    exit_date,
    exit_event: str,
    entry_z: float,
) -> None:
    """Fill the thesis lens (and lens agreement) on an entry row, in place."""
    entry_row["exit_date"] = exit_date
    entry_row["exit_event"] = exit_event
    entry_row["z_exit"] = z_exit
    move = _signed_move(z_entry, z_entry, z_exit)
    entry_row["thesis_move"] = move
    if pd.notna(move):
        entry_row["thesis_verdict"] = THESIS_CONVERGED if move < 0 else THESIS_DIVERGED
    # The thesis lens's blind spot: a spread that rockets THROUGH the mean and
    # out the far side scores as "converged" (the signed move is negative) even
    # though the position was carried through a violent adverse swing. Flagged
    # so the headline convergence rate can be discounted honestly.
    if pd.notna(z_entry) and pd.notna(z_exit):
        entry_row["overshoot_exit"] = bool(
            np.sign(z_exit) != np.sign(z_entry) and abs(z_exit) > entry_z
        )
    outcome = entry_row["outcome_verdict"]
    if entry_row["thesis_verdict"] != NOT_GRADED and outcome in (OUTCOME_PROFIT, OUTCOME_LOSS):
        thesis_right = entry_row["thesis_verdict"] == THESIS_CONVERGED
        outcome_right = outcome == OUTCOME_PROFIT
        entry_row["lenses_agree"] = bool(thesis_right == outcome_right)


def _rate(mask_true: int, total: int) -> float | None:
    return float(mask_true) / total if total else None


def _mean_or_none(values: pd.Series) -> float | None:
    values = values.dropna()
    return float(values.mean()) if len(values) else None


def _summarize(decisions: pd.DataFrame) -> dict:
    """Headline hit-rates over a decision DataFrame (any subset works)."""
    summary = {
        "n_decisions": int(len(decisions)),
        "n_entries": 0,
        "n_entries_taken": 0,
        "n_exits": 0,
        "n_entries_graded_outcome": 0,
        "n_entries_graded_thesis": 0,
        "entry_hit_rate_outcome": None,
        "entry_hit_rate_thesis": None,
        "entry_hit_rate_thesis_strict": None,
        "n_entries_beyond_stop": 0,
        "n_overshoot_exits": 0,
        "n_lens_comparable": 0,
        "n_lens_disagreements": 0,
        "lens_agreement_rate": None,
        "n_stop_losses": 0,
        "stop_loss_justified_rate": None,
        "n_time_exits": 0,
        "time_exit_justified_rate": None,
        "n_clean_exits": 0,
        "clean_exit_converged_rate": None,
        "avg_pnl_correct_thesis": None,
        "avg_pnl_incorrect_thesis": None,
        "avg_pnl_winner": None,
        "avg_pnl_loser": None,
        "total_pnl": None,
    }
    if decisions.empty:
        return summary

    entries = decisions[decisions["event"].isin(ENTRY_EVENTS)]
    exits = decisions[decisions["event"].isin(CLOSING_EVENTS)]
    summary["n_entries"] = int(len(entries))
    summary["n_entries_taken"] = int(entries["taken"].sum()) if len(entries) else 0
    summary["n_exits"] = int(len(exits))

    outcome_graded = entries[entries["outcome_verdict"].isin((OUTCOME_PROFIT, OUTCOME_LOSS))]
    summary["n_entries_graded_outcome"] = int(len(outcome_graded))
    summary["entry_hit_rate_outcome"] = _rate(
        int((outcome_graded["outcome_verdict"] == OUTCOME_PROFIT).sum()), len(outcome_graded)
    )

    thesis_graded = entries[entries["thesis_verdict"].isin((THESIS_CONVERGED, THESIS_DIVERGED))]
    summary["n_entries_graded_thesis"] = int(len(thesis_graded))
    summary["entry_hit_rate_thesis"] = _rate(
        int((thesis_graded["thesis_verdict"] == THESIS_CONVERGED).sum()), len(thesis_graded)
    )

    # The two structural distortions of the thesis lens, and the rate with both
    # discounted: an entry only counts as right if the spread converged AND it
    # did not overshoot out the far side AND it was not an instant-stop entry.
    if "entry_beyond_stop" in entries.columns:
        summary["n_entries_beyond_stop"] = int(entries["entry_beyond_stop"].fillna(False).sum())
    if "overshoot_exit" in entries.columns:
        summary["n_overshoot_exits"] = int(entries["overshoot_exit"].fillna(False).sum())
    if len(thesis_graded) and {"entry_beyond_stop", "overshoot_exit"} <= set(entries.columns):
        strict_ok = (
            (thesis_graded["thesis_verdict"] == THESIS_CONVERGED)
            & ~thesis_graded["overshoot_exit"].fillna(False)
            & ~thesis_graded["entry_beyond_stop"].fillna(False)
        )
        summary["entry_hit_rate_thesis_strict"] = float(strict_ok.mean())

    comparable = entries[entries["lenses_agree"].notna()]
    summary["n_lens_comparable"] = int(len(comparable))
    if len(comparable):
        agree = comparable["lenses_agree"].astype(bool)
        summary["n_lens_disagreements"] = int((~agree).sum())
        summary["lens_agreement_rate"] = float(agree.mean())

    stops = exits[exits["event"] == "STOP_LOSS"]
    stops_graded = stops[stops["exit_verdict"].isin((EXIT_JUSTIFIED, EXIT_TOO_EARLY))]
    summary["n_stop_losses"] = int(len(stops))
    summary["stop_loss_justified_rate"] = _rate(
        int((stops_graded["exit_verdict"] == EXIT_JUSTIFIED).sum()), len(stops_graded)
    )

    time_exits = exits[exits["event"] == "TIME_EXIT"]
    time_graded = time_exits[time_exits["exit_verdict"].isin((EXIT_JUSTIFIED, EXIT_PREMATURE))]
    summary["n_time_exits"] = int(len(time_exits))
    summary["time_exit_justified_rate"] = _rate(
        int((time_graded["exit_verdict"] == EXIT_JUSTIFIED).sum()), len(time_graded)
    )

    clean = exits[exits["event"] == "EXIT"]
    summary["n_clean_exits"] = int(len(clean))
    summary["clean_exit_converged_rate"] = _rate(
        int((clean["exit_verdict"] == EXIT_CONVERGED).sum()), len(clean)
    )

    with_pnl = entries[entries["pnl"].notna()]
    summary["total_pnl"] = float(with_pnl["pnl"].sum()) if len(with_pnl) else None
    summary["avg_pnl_correct_thesis"] = _mean_or_none(
        with_pnl.loc[with_pnl["thesis_verdict"] == THESIS_CONVERGED, "pnl"]
    )
    summary["avg_pnl_incorrect_thesis"] = _mean_or_none(
        with_pnl.loc[with_pnl["thesis_verdict"] == THESIS_DIVERGED, "pnl"]
    )
    summary["avg_pnl_winner"] = _mean_or_none(with_pnl.loc[with_pnl["pnl"] > 0, "pnl"])
    summary["avg_pnl_loser"] = _mean_or_none(with_pnl.loc[with_pnl["pnl"] <= 0, "pnl"])
    return summary


def _per_pair_breakdown(decisions: pd.DataFrame) -> pd.DataFrame:
    """One summary row per pair — where the aggregate hit-rate actually comes from."""
    if decisions.empty:
        return pd.DataFrame()
    rows = []
    for pair, group in decisions.groupby("pair", sort=True):
        pair_summary = _summarize(group)
        rows.append(
            {
                "pair": pair,
                "n_entries": pair_summary["n_entries"],
                "n_taken": pair_summary["n_entries_taken"],
                "hit_rate_outcome": pair_summary["entry_hit_rate_outcome"],
                "hit_rate_thesis": pair_summary["entry_hit_rate_thesis"],
                "n_stop_losses": pair_summary["n_stop_losses"],
                "stop_justified_rate": pair_summary["stop_loss_justified_rate"],
                "n_disagreements": pair_summary["n_lens_disagreements"],
                "total_pnl": pair_summary["total_pnl"],
            }
        )
    return pd.DataFrame(rows)


def grade_decisions(
    price_panel: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    config: PairBacktestConfig | None = None,
    entries_allowed: pd.Series | None = None,
    forward_bars: int = DEFAULT_FORWARD_BARS,
) -> DecisionGradeResult:
    """Grade every decision the engine made on `price_panel` for `pairs`.

    Runs the same PairBacktester the rest of the project uses, then walks each
    pair's signal series and grades each ENTER / EXIT / STOP_LOSS / TIME_EXIT.
    `forward_bars` is how far after an exit we look to judge whether the exit
    was right.

    Note on `taken`: the state machine emits an entry signal per pair, but the
    portfolio caps concurrent positions, so a signalled entry is not always a
    filled trade. The outcome lens only covers filled trades (they are the only
    ones with P&L); the thesis lens covers every signalled entry.
    """
    config = config or PairBacktestConfig()
    result = PairBacktester(config).run(price_panel, list(pairs), entries_allowed=entries_allowed)
    trade_log = result["trade_log"]

    trades_by_entry: dict = {}
    if not trade_log.empty:
        for trade in trade_log.to_dict("records"):
            key = (trade["ticker_a"], trade["ticker_b"], pd.Timestamp(trade["entry_date"]))
            trades_by_entry[key] = trade

    rows: list[dict] = []
    for pair_key, pair_data in result["per_pair"].items():
        rows.extend(
            _grade_pair(
                pair_key,
                pair_data,
                trades_by_entry,
                signal_config=config.signal_config,
                forward_bars=forward_bars,
            )
        )

    decisions = pd.DataFrame(rows)
    if not decisions.empty:
        decisions = decisions.sort_values(["date", "pair"]).reset_index(drop=True)

    return DecisionGradeResult(
        decisions=decisions,
        summary=_summarize(decisions),
        per_pair=_per_pair_breakdown(decisions),
        trade_log=trade_log,
    )


def grade_signal_frame(
    ticker_a: str,
    ticker_b: str,
    signals: pd.DataFrame,
    trade_log: pd.DataFrame | None = None,
    signal_config: SignalConfig | None = None,
    forward_bars: int = DEFAULT_FORWARD_BARS,
) -> pd.DataFrame:
    """Grade one pair's already-computed signal frame (columns zscore/event).

    The seam between "run the backtest" and "grade what it decided". Exposed so
    a caller — or a test — can grade a hand-built signal path directly, with no
    prices, no backtest and no network.
    """
    trades_by_entry: dict = {}
    if trade_log is not None and not trade_log.empty:
        for trade in trade_log.to_dict("records"):
            key = (trade["ticker_a"], trade["ticker_b"], pd.Timestamp(trade["entry_date"]))
            trades_by_entry[key] = trade
    rows = _grade_pair(
        (ticker_a, ticker_b),
        {"signals": signals},
        trades_by_entry,
        signal_config=signal_config or SignalConfig(),
        forward_bars=forward_bars,
    )
    return pd.DataFrame(rows)


def summarize_decisions(decisions: pd.DataFrame) -> dict:
    """Public wrapper: hit-rate math over an arbitrary decision DataFrame.

    Exposed so a hand-built decision set can be summarized directly (tests do
    exactly this), and so a caller can re-summarize a filtered slice.
    """
    return _summarize(decisions)


def grade_focus_book(
    pairs: Sequence[tuple[str, str]] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    config: PairBacktestConfig | None = None,
    forward_bars: int = DEFAULT_FORWARD_BARS,
    price_fetcher: Callable[..., pd.DataFrame] = fetch_price_history,
    end: str | None = None,
) -> DecisionGradeResult:
    """Fetch prices for `pairs` (default: the focus book) and grade them.

    `price_fetcher` is injectable so this path is testable with no network.
    No entry gate is applied here on purpose: the point is to grade the
    decisions the signal engine itself made, not the macro overlay's vetoes.
    """
    from screening.focus_book import focus_pairs  # local: keeps import graph flat

    pairs = list(pairs) if pairs is not None else focus_pairs()
    tickers = sorted({t for pair in pairs for t in pair})
    end = end or datetime.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp(end) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    raw = price_fetcher(tickers, start=start, end=end)
    prices, _dropped = align_and_clean(raw)
    return grade_decisions(prices, pairs, config=config, forward_bars=forward_bars)


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _fmt_money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.0f}"


def format_report(result: DecisionGradeResult, show_tables: bool = True) -> str:
    """Human-readable decision report, with the sample-size caveats attached."""
    summary = result.summary
    lines: list[str] = []
    lines.append("=== Decision grade — were the engine's calls right? ===\n")

    if result.decisions.empty:
        lines.append("No decision events at all: the engine never crossed an entry band on this")
        lines.append("panel. Nothing to grade — that is an answer too (it trades very rarely).")
        return "\n".join(lines)

    lines.append(
        f"{summary['n_decisions']} decision events "
        f"({summary['n_entries']} entries, {summary['n_entries_taken']} of them actually filled; "
        f"{summary['n_exits']} exits)\n"
    )

    lines.append("ENTRY HIT-RATE — two lenses")
    lines.append(
        f"  by OUTCOME (round-trip P&L net of costs): {_fmt_rate(summary['entry_hit_rate_outcome'])}"
        f"  (n={summary['n_entries_graded_outcome']})"
    )
    lines.append(
        f"  by THESIS  (spread actually mean-reverted): {_fmt_rate(summary['entry_hit_rate_thesis'])}"
        f"  (n={summary['n_entries_graded_thesis']})"
    )
    lines.append(
        f"  by THESIS, DISCOUNTED (converged, no overshoot, not an instant-stop entry): "
        f"{_fmt_rate(summary['entry_hit_rate_thesis_strict'])}"
    )
    lines.append(
        f"  lenses agree on {_fmt_rate(summary['lens_agreement_rate'])} of comparable entries "
        f"({summary['n_lens_disagreements']}/{summary['n_lens_comparable']} disagree)"
    )
    lines.append("")

    lines.append("WHY THE RAW THESIS RATE IS FLATTERED")
    lines.append(
        f"  {summary['n_overshoot_exits']} entries 'converged' by shooting THROUGH the mean and out"
    )
    lines.append(
        "    the far side — the signed move is negative, but the position was carried"
    )
    lines.append("    through a violent adverse swing. Scored right; felt nothing like right.")
    lines.append(
        f"  {summary['n_entries_beyond_stop']} entries fired at |z| already past the stop band. The state"
    )
    lines.append(
        "    machine checks entry before stop, so these open a position that is instantly"
    )
    lines.append("    eligible to be stopped out. Their small bounce scores CONVERGED and loses money.")
    lines.append("")

    lines.append("EXIT QUALITY")
    lines.append(
        f"  stop-losses: {summary['n_stop_losses']} fired, "
        f"{_fmt_rate(summary['stop_loss_justified_rate'])} justified "
        f"(spread kept diverging afterwards)"
    )
    lines.append(
        f"  time exits:  {summary['n_time_exits']} fired, "
        f"{_fmt_rate(summary['time_exit_justified_rate'])} justified"
    )
    lines.append(
        f"  clean exits: {summary['n_clean_exits']} fired, "
        f"{_fmt_rate(summary['clean_exit_converged_rate'])} genuinely inside the exit band "
        f"(consistency check, near-structural)"
    )
    lines.append("")

    lines.append("P&L OF RIGHT vs WRONG CALLS")
    lines.append(f"  avg P&L when the thesis was right: {_fmt_money(summary['avg_pnl_correct_thesis'])}")
    lines.append(f"  avg P&L when the thesis was wrong: {_fmt_money(summary['avg_pnl_incorrect_thesis'])}")
    lines.append(f"  avg winner {_fmt_money(summary['avg_pnl_winner'])} / "
                 f"avg loser {_fmt_money(summary['avg_pnl_loser'])} / "
                 f"total {_fmt_money(summary['total_pnl'])}")
    lines.append("")

    if show_tables and not result.per_pair.empty:
        lines.append("PER-PAIR BREAKDOWN")
        table = result.per_pair.copy()
        for column in ("hit_rate_outcome", "hit_rate_thesis", "stop_justified_rate"):
            table[column] = table[column].map(_fmt_rate)
        table["total_pnl"] = table["total_pnl"].map(_fmt_money)
        lines.append(table.to_string(index=False))
        lines.append("")

    disagreements = result.entries
    if show_tables and not disagreements.empty:
        disagreements = disagreements[disagreements["lenses_agree"] == False]  # noqa: E712
        if not disagreements.empty:
            lines.append("WHERE THE LENSES DISAGREE (the informative rows)")
            columns = ["date", "pair", "event", "zscore", "z_exit", "thesis_move",
                       "thesis_verdict", "pnl", "outcome_verdict", "exit_event"]
            lines.append(disagreements[columns].to_string(index=False))
            lines.append("")

    lines.extend(_caveats(summary))
    return "\n".join(lines)


def _caveats(summary: dict) -> list[str]:
    """The honest warnings. Loud, and never suppressed."""
    n_thesis = summary["n_entries_graded_thesis"]
    lines = [
        "-" * 72,
        "READ THIS BEFORE TRUSTING ANY NUMBER ABOVE",
        "-" * 72,
        f"1. SAMPLE SIZE. {n_thesis} graded entries is a tiny sample. The engine trades",
        "   rarely by design (|z| has to clear the entry band), so a single trade moves",
        "   the hit-rate by whole percentage points. Treat these as anecdotes with",
        "   arithmetic attached, not as a measured edge.",
        "2. IN-SAMPLE. These are backtest decisions on data the pair selection already",
        "   saw. The focus book was chosen BECAUSE these pairs looked cointegrated over",
        "   this history. That is circular — it is not out-of-sample evidence, and it is",
        "   not a forward test. reporting/journal.py is the un-fudgeable forward record;",
        "   this module is diagnostic only.",
        "3. NO CONFIDENCE INTERVALS. With samples this small, a 60% hit-rate and a 40%",
        "   hit-rate are usually not distinguishable from each other, or from a coin.",
        "4. LENS DISAGREEMENT IS THE POINT. 'Thesis right, outcome wrong' means the",
        "   signal called it but costs ate the edge. 'Thesis wrong, outcome right' means",
        "   the P&L was luck. Read the two rates together, never one alone.",
        "5. Signal research only. Nothing here places an order.",
    ]
    if n_thesis and n_thesis < 20:
        lines.append(
            f"   !! With n={n_thesis}, no hit-rate here is statistically meaningful. !!"
        )
    return lines
