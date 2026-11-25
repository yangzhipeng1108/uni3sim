# 910B paged-GQA tail-page ND2NZ registration, v3 (exhaustive domain)

Supersedes v2 for **domain coverage only**. v2's data, criteria and verdict stand unchanged and are not re-scored; its artifact (`aic_mte2_nd2nz_paged_gqa_tail_page_v2_cann_8_5_1.json`, commit `d3c5130af`) remains published and valid for the widths it measured. v3 does not relax any v2 criterion.

## Why reopened (design-level)

v2 passed every criterion it froze, so nothing here is a re-scoring of a rejected result. The design-level defect is **sampling density against the structure v2 itself measured**.

v2 measured the service at seven row counts and found it **non-smooth**: at stride 1024, 24 groups, mode 0 the local slope is 1.347 ns/row over 61→81 but 0.862 over 81→127, and 4.031 over the 61→81 segment when taken against 81 — a factor of 4.7 between adjacent segments. A two-parameter fixed-overhead-plus-per-row form leaves a maximum residual of 17.6% of the median (16.5% in mode 1), and a 16-row Cube-alignment reading fails outright (rows 4, 8 and 16 all align to 16 yet measure 133.061, 138.379 and 149.671 ns).

Consequently v2's domain cannot be extended to production by interpolation, and v2's own registration forbids it. Production decode tails are `(context + 1) mod 128`, approximately uniform over 1..127, so a three-width artifact reaches roughly 2.4% of decode steps. The binding constraint is domain coverage, not mechanism correctness.

## Design: exhaustive, so that no interpolation or functional form is needed

The decisive choice is to **measure every row count the production domain contains** rather than sample and interpolate. The tail range is finite and small: v1 established that 1 row is refused by the instrument and v2's Arm 1 established the resolvable bound lies in (1, 4], so the reachable domain is exactly rows **4 through 127**, 124 values. Measuring all of them removes interpolation, removes any fitted functional form, and removes the non-smoothness problem entirely — a non-smooth function is fine when every point in the domain is measured.

This is exhaustive mechanism measurement, not a fit. No parameter is estimated from the data; each row count carries its own measured service.

**Main arm.** `rows` = every integer 4..127; `columns` = 128; `source_row_stride_elements` = 1024; `active_aic_groups` = 24; both issue modes; **4 trials** per cell. Trial count is reduced from v2's 8 because the acceptance statistic below is invariant in trial count, so trials buy precision rather than strictness, and 124 row counts at 8 trials would not fit one idle-card window.

**Group-independence arm.** `rows` = 4, 61, 127; `columns` = 128; `stride` = 1024; `active_aic_groups` = 22 and 23; both modes; 4 trials. Purpose: verify the main arm's group-24 service transfers to the other production grids, since production decode lands on 22 and 23 as well. If it does not transfer, the published domain is restricted to group 24 and the others are reported unsupported — it is not rescued by scaling.

**Amendment, made before any v3 duration was opened.** The binary refuses a batch of more than 128 cells (`a batch cannot contain more than 128 cells`); 124 rows across 2 issue modes is 248. The main arm is therefore issued as two runs, rows 4..65 and rows 66..127, whose union is exactly the registered domain. This changes how the cells are issued, not which cells are measured, nor any criterion or tolerance. It was found by the instrument rejecting the first batch, so no duration data existed when the split was made.

## Acceptance criteria (frozen before any data is opened)

1. The runner's own `validation` field reads `pass` for every main-arm row. A refusal at any row 4..127 is **not** a campaign failure: that row is recorded as out-of-domain and withdrawn from publication, which sharpens the bound v2 left as an interval.
2. stderr empty on both arms.
3. Dispersion, same estimator and tolerance as v2 so the two artifacts remain comparable: for each cell, relative median absolute deviation of the per-process medians, `MAD / median <= 3%`.
4. **Monotonicity in rows** (structural invariant, not a fitted shape): the published service must be non-decreasing in row count at fixed stride, group and mode, allowing a tolerance equal to each point's own measured dispersion. Copying more occupied rows at identical stride cannot take less time; a violation beyond dispersion means the instrument or the geometry is not doing what the design assumes, and rejects the campaign.
5. **Group transfer:** for each of rows 4, 61, 127 the group-22 and group-23 services must agree with group 24 within 5%. Rows failing this are published for group 24 only, with 22 and 23 reported unsupported at those rows.

Failure of criteria 2, 3 or 4 rejects the campaign with no resampling, no production change, no HWSpec change and no Forward evaluation. Criteria 1 and 5 narrow the published domain rather than rejecting.

## Publication boundary

Unchanged from v2 and restated because exhaustive coverage does not by itself make the service load-bearing. Publishing it still does not license publishing the paged KV stride in Forward: `measured_paged_kv_stride` clause 2 (`flash_attn.py:1394-1417`) must be argued from CANN source to admit a short final page, and two independent fail-open holes remain — `formula.py:3832` (at most 160 drained D128 descriptors per worker, measured to fail from about B=32 at 2000-token contexts) and `flash_attn.py:1946` (`_has_graph_padding` drops the per-AIC program on every padded decode step). What v3 changes is that, once those three are addressed, the measured domain will cover essentially all decode tails instead of 2.4% of them.

Rows 1 to 3 remain **unsupported** and must be reported as such, never interpolated from row 4.

## Hardware record to complete at run time

Host `<capture-host-0118b599d5e7>`, container `<ascend-container>`, one idle device, serial (one process per card), deployed commit, PID, output path under `profiler_output/microbench/ascend_910b/`. On-card readings compared against canonical on the same instrument, given the recorded 11-25% instrument offset.

## Verdict: REJECT (recorded after the campaign, criteria unchanged)

The campaign ran to completion and is rejected by criterion 4. Criteria 1, 2, 3 and 5 pass: validation 992/992, no hard-error lines, worst relative MAD of per-process medians 2.16% against the 3% tolerance over 248 cells, and group transfer 12/12 within 2.27% of group 24. Criterion 4 fails with **51 monotonicity violations beyond each point's dispersion** — the trend rises (133.4 → 322.2 ns over rows 4..127) but local dips exceed tolerance repeatedly, e.g. rows 10→11 drops 1.363 ns against a 0.349 ns tolerance, 22→23 drops 2.566 against 0.606, and 26→27 drops 1.763 against 0.227.

Per this registration, a criterion-4 failure rejects the campaign with no resampling, no production change, no HWSpec change and no Forward evaluation. **Nothing from v3 is published.** The group-transfer result is clean but belongs to a rejected campaign and stands as an observation only.

Two findings are carried forward rather than acted on:

1. **The service is not monotone in occupied rows.** The raw sequence is a rising trend carrying 1–2.5 ns jitter (`4:133.4 5:135.8 6:136.6 7:136.5 8:138.4 9:144.2 10:146.3 11:144.9 …`), so an exhaustive per-row table would encode that jitter as mechanism. Alignment is not the right independent variable either: grouping by 8-, 16- and 32-row alignment leaves within-class spread at 15%, 19% and 28% of the median.
2. **Criterion 4's tolerance was the wrong dispersion.** It used each point's own MAD, which measures within-cell reproducibility (0.2–0.7 ns here); adjacent rows are different cells measured at different batch positions, whose mutual comparability is necessarily looser. This flaw is deliberately not fixed and the data not re-scored — adjusting a criterion after seeing the data it failed is precisely what the zero-fit red line forbids.

**This direction stops here.** v1 rejected, v2 passed but proved domain-insufficient (three measured widths against a tail distribution roughly uniform over 1..127), v3 rejected. Three campaigns, three evaluations that did not authorise a production change — GOAL's threshold for stopping a direction and recording the evidence instead. Any future attempt requires a different independent variable (neither raw rows nor 8/16/32-aligned rows) and a monotonicity tolerance derived from between-cell comparability, both frozen before data, and neither justified by the cell or criterion that failed here.
