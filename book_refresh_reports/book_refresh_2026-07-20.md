# Focus-book refresh — 2026-07-20

**Proposal only.** This report re-evaluates the fixed focus book (`screening/focus_book.py`) against fresh walk-forward persistence evidence. It does not modify the book, place orders, or call a broker. Membership remains a human/evidence decision — use this as the auditable monthly trail.

- Lookback: 900 days, top-N book size: 5
- Current members flagged DROP?: **5/5**
- Challengers that would newly enter: **2**

Ranking key: formation-passes (desc), then median holdout p-value (asc), then full-window ADF p-value (asc). Structural near-twins and sub-5-day half-lives are excluded; the proposed book is deduplicated to one pair per sector.

## Current book — status vs fresh evidence

| pair | sector | rank | form_passes | holdout_surv | med_holdout_p | adf_p | half_life_d | passes_screen | in_top_N | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ABT/MRK | healthcare | 3 | 4 | 1 | 0.205 | 0.005 | 16.5 | no | yes | DROP? |
| ALL/TRV | insurance | 1 | 4 | 1 | 0.116 | 0.014 | 11.9 | no | yes | DROP? |
| DUK/SO | utilities | 4 | 4 | 0 | 0.250 | 0.006 | 18.0 | no | no | DROP? |
| COST/PEP | consumer_staples | 10 | 3 | 1 | 0.376 | 0.056 | - | no | yes | DROP? |
| COP/SLB | energy | 15 | 3 | 0 | 0.731 | 0.007 | 17.1 | no | no | DROP? |

`verdict = KEEP` when the member still lands in the proposed one-per-sector top-N AND still passes the screen; otherwise `DROP?` flags it for human review.

## Proposed book — what the fresh evidence would build today

| rank | pair | sector_key | form_passes | holdout_surv | med_holdout_p | adf_p | half_life_d | passes_screen | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ALL/TRV | insurance | 4 | 1 | 0.116 | 0.014 | 11.9 | no | current |
| 2 | D/XLU | utilities | 4 | 0 | 0.154 | 0.700 | - | no | NEW |
| 3 | ABT/MRK | healthcare | 4 | 1 | 0.205 | 0.005 | 16.5 | no | current |
| 8 | LUV/UAL | airlines | 3 | 0 | 0.193 | 0.314 | - | no | NEW |
| 10 | COST/PEP | consumer_staples | 3 | 1 | 0.376 | 0.056 | - | no | current |

## Challengers — proposed pairs not currently in the book

| pair | sector_key | rank | passes_screen | form_passes | holdout_surv | med_holdout_p | adf_p | half_life_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D/XLU | utilities | 2 | no | 4 | 0 | 0.154 | 0.700 | - |
| LUV/UAL | airlines | 8 | no | 3 | 0 | 0.193 | 0.314 | - |
