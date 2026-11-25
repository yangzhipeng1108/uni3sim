# H200 typed register mechanisms

This probe prepares independent physical evidence for the typed-service gap in `VectorPredictor.roofline_without_launch`: `dsv32_moe_route` and `vllm_plain_greedy_sample` at `stage=argmax` currently reject complete pricing. It contains no routing, argmax, sorting, reduction, model tensors, or production timing. Removing this gap alone cannot price all 600 routing and 8 argmax calls recorded per original GLM phase: nonlinear work and the complete selection program also remain unresolved. The maximum end-to-end improvement cannot yet be quantified; this experiment establishes physical units before any rate or speedup claim.

## Measured object and domain

| Primitive | One source bundle per active lane | Synthetic input |
|---|---|---|
| `f32_gt` | FP32 ordered greater-than plus FP32 select | Two finite register states around zero |
| `f32_eq` | FP32 equality plus FP32 select | Two finite register states, one equal to pivot |
| `f32_nan` | FP32 NaN predicate plus FP32 select | Finite/qNaN register states |
| `s64_lt` | Signed full-width 64-bit comparison plus 64-bit select | Two-word states and runtime two-word pivot |
| `u64_gt` | Unsigned full-width 64-bit comparison plus 64-bit select | Two-word states and runtime two-word pivot |
| `shuffle32` | One full-warp b32 XOR-distance-1 shuffle | Distinct lane/chain integer labels |

All 32 lanes participate. One warp bundle means 32 lane bundles, and is neither one scalar operation nor one SASS instruction. Compare and select form one dependent bundle: their measured rate cannot be published as a pure predicate rate, added to another select rate, or directly substituted for a lexicographic `(value, index)` pair comparator. FP32 streams do not characterize arbitrary value distributions, subnormal handling, or all exceptional cases. One and four independent register chains expose dependency and concurrency differences; neither establishes saturation by itself. There are 31 unrolled bundles per chain per outer iteration, with loop/control overhead retained. No baseline subtraction or production-duration correction is applied.

Each warp records `clock64` begin/end and its SM ID. The runner groups by `(sample, smid)` and reports `sum(warp_bundles)/(max(end)-min(begin))` in **warp bundles per SM cycle**, together with individual **SM cycles per warp bundle**. Absolute timestamps from different SMs are never compared. SM ID space is validated with `%nsmid`, since 132 active SMs does not imply dense IDs. These are observed stream rates including scheduling gaps, loop overhead and imbalance, not automatically peak throughput or isolated latency. No rate is measured by offline preparation.

## Offline preparation

Run from this worktree, choosing a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' python microbench/nvidia_h200/compute/typed_register_primitives_runner.py \
  --prepare-only --nvcc /usr/local/cuda/bin/nvcc \
  --cuobjdump /usr/local/cuda/bin/cuobjdump \
  --baseline-commit "$(git rev-parse HEAD)" \
  --output experiment_logs/evaluation/operator_support/typed_register_preparation/offline
```

Preparation compiles SM90 machine code and compute90 PTX without executing either. A separate `g++ -DMIDLAYER_TYPED_CPU_ONLY` executable, which does not link CUDA, executes 4,613 CPU reference checks covering the synthetic period-two streams, odd/even loop lengths, NaN, signed zero and signed/unsigned distinctions. CPU reference success does not validate device inline PTX or hardware results. The future GPU path checks every lane and chain against this reference after timing.

The static gate requires all 12 primitive/chain variants, two clock-read sites, no timed global/local/shared memory operations, and zero stack/local/shared storage. FP32 variants must retain at least 31 predicates per chain; 64-bit variants at least 62 integer predicate sites per chain; shuffle variants exactly 31 shuffles per chain. Full timed SASS, resources and histograms are saved for review. These are preservation checks, not an automatic proof of a service rate. An initial input with identical low words was correctly rejected because the compiler hoisted part of the 64-bit comparison; distinct low words now retain the full two-word stream without weakening the gate.

`preparation.json` uses `h200_typed_register_preparation_v1`: build commands, source/binary hashes and compiler version, runner/helper hashes, CPU binary hash/version/check receipt, cuobjdump version, exact checkout baseline and dirty entries, domain/unit declarations and per-variant SASS/resources. `GPU_executed=false`, `measured_rate=null`, `Forward_binding_approved=false`, and `production_duration_consumed=false` are explicit. Compiler logs, PTX, SASS and resource dump are adjacent files. No output is imported by production Forward.

## Proposed hardware execution, pending authorization

The maintainer must assign one idle full H200 SM90/132-SM device, owner, execution path and approved environment. No fixed GPU index, old container or remote path is implied by this runner. Reserve roughly 10–15 minutes for initial validation and capture, with a 15-minute review point; actual host/process overhead is unmeasured. Record capture purpose, source identity, and retention in the capture manifest. A CLI execution switch does not grant hardware authorization.

1. Repeat offline preparation in the actual execution toolchain and review SASS preservation/resources. The CUDA12.2 offline binary is not evidence of a selected production provider or compiler.
2. Run `--execute --smoke-only --device <owned-index>` with the same required compiler/baseline arguments and a fresh registered output directory. This is 12 cells, one sample each, one warp, 127 outer iterations. Check actual GPU outputs, device identity, timers and clock gate before scaling.
3. Only after smoke succeeds, run a separate `--execute` capture. Fixed coverage is 6 primitives × 2 chain counts × 3 block sizes (32/128/256) × 3 grids (1/132/528) × 2 loop lengths (127/511): 216 cells, three samples and three rotated trials. Cells intentionally share a single device and execute serially to avoid contaminating the mechanism. No workload durations choose cells or parameters.
4. The existing `clock_gated_capture.py` supplies warmup, idle-device checks, 50-ms clock logs, its documented 1980-MHz gate and at most four attempts per trial. Rejected artifacts remain visible. Failure of clock gating or output/static validation is a failed capture, not a reason to alter the rate.
5. Review the register/warp/block/loop axes for dependency effects and a reproducible saturation region. If none exists, retain the measured domains and uncertainty; do not select a convenient maximum or tune against production error. Any later binding needs a separately reviewed primitive/program mapping and applicable-domain proof.

Device probe buffers peak below 5 MB, excluding CUDA context/driver allocations; warmup uses a separate 4-KB sink. Full capture emits 1,856,088 accepted warp CSV rows, with repeated attempts potentially quadrupling raw storage. Plan up to 2 GB host/disk headroom for CSV, JSON and rejected attempts, and follow the mandatory pullback/notification rule at 200 MB on remote tasks; these are planning estimates, not observed usage. Smoke output is tiny. Host validation and copies are outside clock intervals.

CSV fields are `sample,primitive,chains,threads,grid,outer,warp_index,smid,smid_space,begin_cycle,end_cycle,warp_bundles,checksum,validation_status`; `warp_bundles=outer*31*chains`. Each CSV is fail-closed on row count, identity, duplicate warp, configuration, timer or output errors. `observations.json` uses `h200_typed_register_observation_v1`, retaining device identity, preparation reference, clock verdicts/rejections, smoke/full scope, raw paths and per-SM summaries with physical units. `Forward_binding_approved` remains false even after capture.

## What this can and cannot unblock

With authorized measurements and a reviewed mapping, this provides scoped evidence for FP32 predicate/select, full-width integer compare/select and b32 shuffle bundles, instead of borrowing FP32 FLOP or INT32 ALU rates without proof. It does not determine CUDA `tanhf` value-dependent polynomial versus EX2/RCP branch counts, prove unbiased block/warp selection equivalence, establish the complete pair-choice/reduction program, bind actual production launches, or validate a sampler/provider. Inputs or proven bounds remain necessary where work depends on values; a bound is not an accepted point prediction. Normal whole-worker residency/reserves, normal available-KV/QPS16, native MTP and W1–W7 acceptance remain independent open gates. No HWSpec rate is changed by this preparation or its output.
