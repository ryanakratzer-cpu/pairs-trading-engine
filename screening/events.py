"""Event-exclusion windows: block new pair entries around scheduled macro
events whose outcomes move both legs violently and asymmetrically — FOMC
decision days and US federal elections.

Rationale (see the project's regime research notes): policy-uncertainty
spikes concentrate exactly the regime-shift risk that breaks cointegration,
and these events are known YEARS in advance — excluding them costs almost
nothing in trading days (~18/year) and removes the most predictable
volatility clusters. Like the regime stress mask, an event mask gates NEW
entries only; open positions ride through and exit on their own terms.

Dates are the scheduled decision/election days, hardcoded because they are
small, public, and fixed — no data fetch to fail. Extend the lists when the
Fed publishes the next calendar year.
"""

from __future__ import annotations

import pandas as pd

# Second (decision) day of each scheduled FOMC meeting.
FOMC_DECISION_DATES: list[str] = [
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

US_ELECTION_DATES: list[str] = [
    "2024-11-05",  # presidential
    "2026-11-03",  # midterms
]


def event_exclusion_mask(
    index: pd.Index,
    event_dates: list[str] | None = None,
    days_before: int = 1,
    days_after: int = 1,
) -> pd.Series:
    """Boolean series over `index`: True = entries allowed, False = inside an
    event window (compatible with generate_signals' `tradeable` and
    PairBacktester.run's `entries_allowed`).

    The window is [event - days_before, event + days_after] in CALENDAR days,
    intersected with the trading index — so a Monday FOMC blackout correctly
    reaches back to Friday rather than skipping the weekend gap.
    """
    if event_dates is None:
        event_dates = FOMC_DECISION_DATES + US_ELECTION_DATES
    mask = pd.Series(True, index=index)
    if len(index) == 0:
        return mask
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    for raw in event_dates:
        event = pd.Timestamp(raw)
        blocked = (dates >= event - pd.Timedelta(days=days_before)) & (
            dates <= event + pd.Timedelta(days=days_after)
        )
        mask[blocked] = False
    return mask
