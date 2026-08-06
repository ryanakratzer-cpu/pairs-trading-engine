# Focus-book refresh — 2026-07-23

**Proposal only.** This report re-evaluates the fixed focus book (`screening/focus_book.py`) against fresh walk-forward persistence evidence. It does not modify the book, place orders, or call a broker. Membership remains a human/evidence decision — use this as the auditable monthly trail.

- Lookback: 900 days, top-N book size: 5
- Current members flagged DROP? (this run alone): **5/5**
- Challengers that would newly enter: **2**
- **Governance recommendation: 0/5 members REPLACE** (sustained, screen-passing drift across consecutive refreshes)

Ranking key: formation-passes (desc), then median holdout p-value (asc), then full-window ADF p-value (asc). Structural near-twins and sub-5-day half-lives are excluded; the proposed book is deduplicated to one pair per sector.

## Current book — status vs fresh evidence

| pair | sector | rank | form_passes | holdout_surv | med_holdout_p | adf_p | half_life_d | passes_screen | in_top_N | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABT/MRK | healthcare | 3 | 4 | 1 | 0.204 | 0.006 | 16.7 | no | yes | DROP? |
| ALL/TRV | insurance | 2 | 4 | 0 | 0.176 | 0.000 | 11.7 | yes | no | DROP? |
| DUK/SO | utilities | 4 | 3 | 0 | 0.160 | 0.002 | 16.4 | no | yes | DROP? |
| COST/PEP | consumer_staples | 5 | 3 | 1 | 0.289 | 0.048 | 30.3 | no | yes | DROP? |
| COP/SLB | energy | 10 | 3 | 0 | 0.788 | 0.008 | 17.4 | no | no | DROP? |

`verdict = KEEP` when the member still lands in the proposed one-per-sector top-N AND still passes the screen; otherwise `DROP?` flags it for human review. (This column reacts to a SINGLE run — see the governance recommendation below for the hysteresis-filtered action.)

## Governance recommendation (hysteresis — the action to actually take)

A member is only recommended REPLACE once the SAME challenger has out-ranked it AND passed the screen for enough consecutive monthly refreshes; otherwise WATCH (drift noted, not yet acted on) or KEEP. This is what prevents churning the book on one noisy run.

| pair | sector | action | challenger | challenger_passes_screen | consecutive_runs |
| --- | --- | --- | --- | --- | --- |
| ABT/MRK | healthcare | KEEP | - | no | 0 |
| ALL/TRV | insurance | WATCH | AIG/PRU | no | 0 |
| DUK/SO | utilities | KEEP | - | no | 0 |
| COST/PEP | consumer_staples | KEEP | - | no | 0 |
| COP/SLB | energy | KEEP | - | no | 0 |

## Proposed book — what the fresh evidence would build today

| rank | pair | sector_key | form_passes | holdout_surv | med_holdout_p | adf_p | half_life_d | passes_screen | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | AIG/PRU | insurance | 5 | 0 | 0.202 | 0.002 | 15.7 | no | NEW |
| 3 | ABT/MRK | healthcare | 4 | 1 | 0.204 | 0.006 | 16.7 | no | current |
| 4 | DUK/SO | utilities | 3 | 0 | 0.160 | 0.002 | 16.4 | no | current |
| 5 | COST/PEP | consumer_staples | 3 | 1 | 0.289 | 0.048 | 30.3 | no | current |
| 6 | C/GS | banks_financials | 3 | 0 | 0.422 | 0.035 | 24.5 | no | NEW |

## Challengers — proposed pairs not currently in the book

| pair | sector_key | rank | passes_screen | form_passes | holdout_surv | med_holdout_p | adf_p | half_life_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AIG/PRU | insurance | 1 | no | 5 | 0 | 0.202 | 0.002 | 15.7 |
| C/GS | banks_financials | 6 | no | 3 | 0 | 0.422 | 0.035 | 24.5 |
