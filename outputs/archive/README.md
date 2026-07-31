# Archived track record

`daily_performance_openclose_pre_2026-07-31.csv` — sessions 2026-07-28..07-30.

Archived, not deleted, on 2026-07-31. Two reasons it cannot be appended to:

1. **Different measurement convention.** Those rows measure OPEN->CLOSE. The
   tracker now measures CLOSE-TO-CLOSE, because the strategy holds positions
   overnight and the open->close window discarded the overnight gap — which on
   2026-07-30 was the entire day's P&L (+$252 real vs -$43 booked).
2. **Different book.** They cover the pre-2026-07-30 focus book (ABT/MRK,
   ALL/TRV, DUK/SO, COST/PEP, COP/SLB), four of whose members were removed once
   proper Engle-Granger p-values showed they were not cointegrated on the
   fitted window (two also had negative betas).

**Deliberately NOT backfilled.** Recomputing 07-28..07-30 under the new book
would be look-ahead: that book was selected using data through 07-29, so
"its" performance over those same days is hindsight, not track record. The
live record for the new book starts fresh at 2026-07-31.
