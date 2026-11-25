# 910B marginal collective cost registration (compute-interleaved HCCL)

Authorized by the maintainer (2026-09-27) to measure the collectives TP2 needs with an independent mechanism microbenchmark, not from serving durations, and to publish the result as a production profile for acceptance. This document is the pre-data registration required by that authorization: design and re-measurement criteria, committed before any data is opened.

## What is missing and why this probe

`src/kernel/launch_config/ascend_910b/__init__.py` currently carries round development placeholders — `small_startup_us`, `ll128_startup_us`, `simple_startup_us` all 60.0, and the three `*_phase_fixed_us` all 0.2 — and its own comment states the reason they are placeholders rather than measurements.

A standalone sweep already exists (`microbench/ascend_910b/references/hccl_fullmesh_collective_startup_cann_8_5_1.json`): one collective costs **127–137 µs back-to-back** from 4 KiB to 128 KiB, flat across both protocol boundaries, and device-bound. Charging that figure is nevertheless wrong, for a reason that is physical rather than numerical: a decode step runs on one device serially, so the step's device time cannot exceed the measured token cadence, and 126 µs per collective drives the predicted decode step past that bound on both TP=2 models at QPS 1 — on qwen3-30b-a3b the collectives alone would consume 61% of the step budget.

What the model needs is the **marginal** cost of a collective issued into a stream that already contains compute. The same sweep bounds it from the other side: its isolated pattern costs **220 µs** against 127 µs back-to-back, so roughly 93 µs of the standalone figure is launch and rank-skew that surrounding compute absorbs. The existing comment names the gap exactly — "measuring that marginal cost needs a compute-plus-collective probe that does not exist yet" — and no such probe exists in `microbench/ascend_910b/communication/collectives/`.

This registration is that probe.

## Design

Two ranks, one per device, HCCL backend over HCCS. Synthetic compute and a constant-filled payload; workload-independent participation. No production operator, model shape, kernel or Serving duration takes part, and nothing emitted here is selected by comparing a Midlayer prediction against a production measurement.

**The measurement is a matched A/B on the same compute chain.**

- **Arm A (compute only):** a chain of `chain_length` identical synthetic matmuls on device, bounded by one device-event pair.
- **Arm B (compute + collective):** the *same* chain, with one `all_reduce` of `message_bytes` inserted after every `insert_every`-th matmul, bounded the same way.

Marginal cost per collective is `(T_B − T_A) / collective_count`. Because both arms issue byte-identical compute, the difference isolates what the collective adds *on top of* a stream that is already busy — which is the quantity the model charges and the quantity the standalone sweep cannot provide.

**Cells.** `message_bytes` ∈ {4 KiB, 16 KiB, 64 KiB, 143 KiB, 256 KiB, 512 KiB}: the range TP2 decode issues, with 143 KiB included because it is what qwen3-32b at hidden 5120, batch 14, bf16 actually all-reduces. `insert_every` ∈ {1, 4}: two compute densities, so the absorbed fraction is observed rather than assumed. `chain_length` fixed at 64 matmuls; matmul shape fixed at 2048×2048×2048 bf16, chosen to keep the device busy for well over one collective's standalone cost so that absorption can occur at all. 9 trials per cell, each a fresh process.

**Amendment: rank combination rule.** The registration above omitted how the two ranks' marginal costs combine, and a harness smoke at non-registered parameters (chain length 8, 1 trial, run only to prove HCCL initialises and the A/B timing works) made the omission visible: the two ranks report materially different marginal costs, as they must, since in an `all_reduce` whichever rank arrives first waits for the other. The rule is therefore fixed here, before the registered campaign opens any data, and it follows from the physics rather than from which number is smaller: **the published marginal cost is the maximum over ranks**, because a step cannot complete before its slower rank. Both ranks' values are recorded so the skew is visible rather than averaged away.

## Acceptance criteria (frozen before any data is opened)

1. **Arm equality by construction:** every cell records `chain_length`, matmul shape and dtype for both arms, and the recorded values must be identical between A and B. A mismatch rejects the cell.
2. **Non-negativity:** `T_B ≥ T_A` beyond the combined dispersion of the two arms. Adding collectives to a stream cannot make it faster; a violation means the probe is not measuring what it claims and rejects the campaign.
3. **Upper bound against the standalone sweep:** the marginal cost must be **≤ 137 µs**, the upper end of the published back-to-back standalone figure. A marginal cost above the standalone cost is self-contradictory — inserting into a busy stream cannot cost more than issuing alone — and would mean the A/B is contaminated. This is a bound taken from an already-published measurement, not a fitted target.
4. **Dispersion:** for each cell, relative median absolute deviation of the per-process medians of the marginal cost, `MAD / median ≤ 5%`. The tolerance is looser than the 3% used for single-descriptor copies because a difference of two timed chains carries both arms' noise.
5. **Monotonicity in message size** at fixed `insert_every`, within each point's dispersion: a larger all_reduce payload cannot complete faster.
6. **Absorption is reported, not assumed:** the ratio of marginal cost to the 127 µs standalone back-to-back figure is recorded per cell. No threshold is attached — this is the physical quantity the profile needs, and constraining it in advance would be assuming the answer.

Failure of criteria 2, 3, 4 or 5 rejects the campaign with no resampling, no production change, no HWSpec change and no Forward evaluation.

## Publication boundary

A passing campaign licenses replacing the six development placeholders with measured marginal values **for 910B only**, with units (µs per collective), the physical meaning (marginal device cost of one HCCL all_reduce inserted into a compute-occupied stream over HCCS, 2 ranks), and full provenance. It does not license touching any other hardware: `A800` and `C500` must remain bit-identical, which is verified by the 30-cell comparison, not asserted.

It also does not change any cell's `accuracy_claim_status`. That status is set by `sharegpt_compact_evaluation.py:2990` from the *backend name* (`tensor_collective_backend == "protobridge"` with `tensor_parallel_size > 1`), independently of profile provenance, so the six TP2 cells stay `unsupported_development_only` no matter how well the collective is measured. Changing that predicate would be an evaluation-convention change and is out of scope here. The measurement's value is that it moves TP2 pricing by mechanism; the labelling question is separate and belongs to whoever owns the convention.

## Hardware record to complete at run time

Host `<capture-host-0118b599d5e7>`, container `<ascend-container>`, devices 0 and 1 idle, serial (no other capture on either card), deployed commit, PIDs, output path under `profiler_output/microbench/ascend_910b/`. On-card readings compared against canonical on the same instrument, given the recorded 11–25% instrument offset.

## Verdict: REJECT (recorded after the campaign, criteria unchanged)

The campaign ran to completion on devices 0 and 1 with no errors and both ranks' records written. It is **rejected by criterion 5**.

| message bytes | insert_every | arm A | arm B | marginal r0 | marginal r1 | published (max) | absorbed vs 127 µs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 096 | 1 | 4116.3 | 11174.7 | 110.288 | 111.791 | 111.791 | 0.120 |
| 16 384 | 1 | 4111.0 | 11306.2 | 112.425 | 108.790 | 112.425 | 0.115 |
| 65 536 | 1 | 4110.1 | 11361.3 | 113.300 | 109.922 | 113.300 | 0.108 |
| 146 432 | 1 | 4108.4 | 10653.1 | 102.260 | 97.701 | 102.260 | 0.195 |
| 262 144 | 1 | 4102.3 | 10198.9 | 95.260 | 92.908 | 95.260 | 0.250 |
| 524 288 | 1 | 4113.6 | 11961.7 | 122.627 | 121.476 | 122.627 | 0.034 |
| 4 096 | 4 | 4115.2 | 4570.8 | 28.475 | 24.572 | 28.475 | 0.776 |
| 16 384 | 4 | 4111.8 | 4682.2 | 35.655 | 28.305 | 35.655 | 0.719 |
| 65 536 | 4 | 4103.2 | 4746.5 | 40.209 | 33.226 | 40.209 | 0.683 |
| 146 432 | 4 | 4113.0 | 4822.1 | 44.315 | 39.365 | 44.315 | 0.651 |
| 262 144 | 4 | 4106.9 | 5017.5 | 56.915 | 49.290 | 56.915 | 0.552 |
| 524 288 | 4 | 4106.0 | 5209.1 | 68.940 | 61.624 | 68.940 | 0.457 |

Criteria 1, 2 and 3 pass: arms record identical compute, `T_B ≥ T_A` on every cell and both ranks, and every marginal value is below the 137 µs standalone bound. **Criterion 5 fails at `insert_every = 1`**: the published marginal runs 111.8, 112.4, 113.3, **102.3, 95.3**, 122.6 µs across increasing payload, which is not non-decreasing. At `insert_every = 4` it is monotone (28.5 → 68.9 µs).

Per this registration a criterion-5 failure rejects the campaign with no resampling, no production change, no HWSpec change and no Forward evaluation. **Nothing is published and the six placeholders are unchanged.**

**Criterion 4 could not be evaluated, and that is an implementation defect against this registration.** The design says "9 trials per cell, each a fresh process", so criterion 4 is the relative MAD of *per-process* medians. The runner instead executes all 9 trials inside one process, so the data carries one median per rank and no per-process distribution exists to take a MAD over. The criterion is therefore neither passed nor failed — it is unevaluable on this data, which is a failure of the implementation to match its own registration rather than a property of the mechanism.

**The rejection carries the more important finding.** Absorption depends on **collective density**, not only on payload size. At `insert_every = 1` the stream is saturated with collectives, there is almost no compute left to hide them, and the marginal cost approaches the standalone 127 µs figure (absorbed fraction 0.03–0.25). At `insert_every = 4` absorption is real (0.46–0.78) and the size dependence becomes clean and monotone. The non-monotonicity at high density is consistent with that regime being synchronisation-bound rather than bandwidth-bound.

Consequently **a single marginal value per (protocol, message size) does not exist**: the quantity is a function of the surrounding compute density as well. The model's parameter shape here — three `*_startup_us` plus three `*_phase_fixed_us` — has no dimension in which density can be expressed. So the placeholders are not merely wrongly valued; **the parameter shape cannot represent the mechanism that was measured**. Replacing the six constants with any single measured number, however well sourced, would encode one density as if it were universal.

Any future attempt needs a parameterisation that carries density (or an equivalent occupancy term) before measurement is worth repeating, plus a runner that actually issues one process per trial as registered. Neither is justified by the criterion that failed here, and both must be frozen before data.
