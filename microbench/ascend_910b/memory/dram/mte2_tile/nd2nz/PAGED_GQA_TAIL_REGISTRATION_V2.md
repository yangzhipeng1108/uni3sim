# 910B paged-GQA tail-page ND2NZ registration, v2

Supersedes `PAGED_GQA_TAIL_REGISTRATION.md` (v1). v1's campaign was rejected against its own criteria and its artifact was never published; v1 stays in the tree as the record of that rejection. **v1's criteria are not retroactively relaxed** — this is a new design with its own frozen criteria, not a re-scoring of v1's data. v1's measured data is not reused for any acceptance decision here.

## Why the measurement is reopened (design-level reasons only)

Neither reason refers to any value v1 measured, and neither refers to the cell v1 was rejected on.

**A. The instrument's resolvable-row lower bound was never determined.** v1 registered a 1-row cell; the instrument refused it outright with a validation error, and v1 withdrew the cell. That withdrawal recorded *that* a bound exists but never located it. This is load-bearing rather than cosmetic: a decode step appends one token, so a request's last block-table page carries `(context + 1) mod 128` rows, which is approximately uniform over 1..127 across a workload. Without the bound, the fraction of production tails that lie inside the instrument's domain is unknown, and the unsupported remainder cannot be stated honestly. v1 could not answer "which production tails are measurable at all", so its cell set could not be justified as covering the production distribution.

**B. v1's acceptance statistic is not invariant in the number of trials.** v1 required `(max(per-process medians) - min(per-process medians)) / median(per-process medians) <= 5%`. The expectation of a max-minus-min range grows monotonically with the number of samples, so pairing it with a fixed 5% threshold makes the threshold's meaning depend on the trial count — an implementation detail (v1 used 4) rather than a property of the mechanism. A criterion whose strictness silently changes with n cannot certify a measurement domain. This is a defect in how the criterion was constructed, independent of any result it produced.

## Design

Two arms, one campaign, both at the D128/HKV8 paged K/V descriptor geometry already covered by `paged_gqa_128` (no new descriptor class, no new operator, no page-table term).

**Arm 1 — resolution characterization.** `(rows, columns, source_row_stride_elements)` = `(4,128,1024)`, `(8,128,1024)`, `(16,128,1024)`, `(32,128,1024)`. Purpose: locate the instrument's minimum resolvable row count as a mechanism property, by recording for each cell whether the instrument's own validation accepts or refuses it. This arm's output is a **domain boundary**, not a duration to be consumed by Forward. Rows are chosen as a geometric ladder spanning the gap between the refused 1-row point and the measured tails, fixed before any data is opened.

**Arm 2 — production tails.** `(61,128,1024)`, `(81,128,1024)`, `(127,128,1024)`, plus stride peers `(61,128,128)` and `(61,128,4096)`. These are the tail widths production decode actually produces (61 at 700 computed tokens, 81 at 2000, with 127 as the upper endpoint of the tail range) plus the contiguous and wider-stride controls used for the full page. Defined by the `(context + 1) mod 128` arithmetic, never by an observed duration, MAPE or residual.

Active AIC groups `(22, 23, 24)` for both arms: production decode grids land on 22 and 23 as well as 24, and `_strided_active_aic_group_is_measured` is already False at several of them.

**Trials: 8 per cell** (v1 used 4). Stated here because the acceptance statistic below is defined to be comparable across trial counts, so the trial count is a precision choice rather than a hidden threshold modifier.

## Acceptance criteria (frozen before any data is opened)

1. The runner's own per-cell `validation` field must read `pass` for every row of Arm 2. For Arm 1, a refusal is a **valid result** (it locates the bound) and does not fail the campaign; an Arm 1 cell is recorded as accepted or refused, nothing more.
2. Arm 2 stderr must be empty.
3. **Dispersion, on a statistic invariant in n:** for each Arm 2 cell, the relative median absolute deviation of the per-process medians, `MAD / median`, must be `<= 3%`, where MAD is the median of `|x_i - median(x)|`. MAD is a fixed-quantile estimator, so its expectation does not grow with the number of processes, and the same 3% means the same thing at 4 trials or at 8. This replaces v1's range-based statistic for reason B; it is not a relaxation of it, because the two statistics are not comparable — a range threshold and a MAD threshold at the same nominal percentage are different strictnesses by construction, which is exactly why v1's number is not carried over.
4. **Peer ordering:** at the 61-row width, the stride-128 and stride-4096 peers must bracket the stride-1024 value monotonically, in the same order the `paged_gqa_128` peers do at 128 rows.
5. **Bound consistency:** every Arm 2 cell must lie at or above the resolvable bound located by Arm 1. If Arm 1 places the bound above any Arm 2 row, that Arm 2 cell is withdrawn from publication as out-of-domain rather than being published.

Failure of criteria 1-4 on Arm 2 rejects the campaign with no resampling, no production change, no HWSpec change and no Forward evaluation.

## Publication boundary

A passing campaign is evidence for this descriptor's service at these row counts and for the domain bound only. It does **not** by itself license publishing the paged stride in Forward. Making it load-bearing additionally requires relaxing `measured_paged_kv_stride` clause 2 (`flash_attn.py:1394-1417`) to admit a short final page — a source-semantics change that must be argued from the CANN source, not from this measurement. Two independent fail-open holes also remain: `formula.py:3832` (`_gqa_group_fits_finite_stream`, at most 160 drained D128 descriptors per worker, measured to fail from about B=32 at 2000-token contexts) and `flash_attn.py:1946` (`_has_graph_padding`, which drops the per-AIC program on every padded decode step). Those must be closed together with this measurement, or in an order where each step is separately verifiable against the measured per-step device windows.

Tails below the resolvable bound located by Arm 1 stay **unsupported** and must be reported as such rather than interpolated or extrapolated from the measured widths.

## Hardware record to complete at run time

Host `<capture-host-0118b599d5e7>`, container `<ascend-container>`, one idle device, serial (one process per card), deployed commit, PID, and output path under `profiler_output/microbench/ascend_910b/`. On-card readings must be compared against canonical on the same instrument, given the recorded 11-25% instrument offset.
