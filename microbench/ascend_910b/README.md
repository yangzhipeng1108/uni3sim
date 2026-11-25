# Ascend 910B microbenchmarks

These independent microbenchmarks isolate individual Ascend 910B hardware mechanisms and provide reproducible evidence for `HWSpec`, launch selection and Kernel models. They must not run complete production operators or derive parameters from production/evaluation kernel duration.

General zero-fit, machine-access and hardware-task rules are in the project modeling and evidence requirements. Parameter eligibility for Forward is defined in section 10 of [`docs/kernel_performance_model.md`](../../docs/kernel_performance_model.md).

## Directory map

| Path | Measurement |
|---|---|
| `compute/vector_pipe/` | UB-resident Vector opcode/request service |
| `memory/dram/bandwidth/` | GM/HBM streaming bandwidth sweep |
| `memory/dram/latency/` | dependent GM/L2 pointer chase |
| `memory/dram/mte2_tile/` | Basic GM→UB MTE2 tile service |
| `memory/dram/mte2_tile/{finite_drain,gemm_source_order,grouped_pair,l1_program,nd2nz}/` | MTE2 finite-drain, GEMM source-order, operand-pair, L1-program and ND2NZ mechanisms |
| `memory/dram/mte3_tile/` | UB→GM MTE3 tile service |
| `memory/dram/aiv_workspace_transfer/` | CANN FIA Vec1/Vec2 short workspace-transfer primitives, ordered chains and primary-AIV completion |
| `memory/fixpipe/` | Independent FixPipe service candidate |
| `operators/gemm/single_core_split_k_tile/` | MatMulV3 key-65632 resident MTE1/Cube/source-control tile service |
| `operators/flash_attn/` | FlashAttention tiling/metadata launch facts |
| `operators/fused_moe/` | Fused-MoE tiling and scatter/MTE2 mechanisms |
| `operators/norm/` | ReduceSum and Norm reduction mechanisms |
| `operators/swiglu/` | SwiGLU tiling facts |
| `operators/vector/softmax/` | UB softmax-vector mechanism experiments |
| `operators/vector/elementwise_control_lifecycle/` | 48-AIV elementwise DAG/TBuf/mutex source-control lifecycle candidate |
| `operators/vector/elementwise_single_tile_program/` | Complete 47-AIV single-tile elementwise 2R1W memory/control worker span |
| `runtime/cross_domain/` | AIC↔AIV cross-domain handoff |
| `runtime/fia_edge_handoff/` | Independent completion latency for three FIA AIC↔AIV CrossCore dependency edges |
| `runtime/fia_program_span/` | Complete program-span candidate for the CANN FIA nonempty three-slot controller and three handoffs |
| `runtime/launch/` | AIC/AIV/mixed empty tasks and minimal nonempty AIV task |
| `references/` | Compact, repository-auditable references and hardware material |

`runtime/launch/fallback_dispatch_runner.py` independently measures host wall service for repeated `aten.reshape` metadata nodes on synthetic NPU tensors. A chain-length sweep estimates per-node cost, retaining a 512-node holdout; it neither reads Serving duration nor reproduces a production graph. Results for both physical TP2 lanes and the source occurrence contract are in `references/compiled_fx_reshape_metadata_cann_8_5_1.json`. Production consumes this evidence only for the declared Ascend Qwen3 MoE TP2 graph structure.


The same fixed sweep in `runtime/launch/compiled_fx_metadata_runner.py` also captures source-authored `aten.view` and `aten.empty.memory_format` metadata nodes in dense fallback graphs. Independent results for both lanes and Qwen3/Llama occurrence contracts are in `references/compiled_fx_fallback_metadata_cann_8_5_1.json`. The dense contract requires layer-0 plus repeated layer-1 with conservation of the total attention split-child count; MoE graphs do not consume it.

`runtime/launch/compiled_fx_executor_runner.py` measures host service for synthetic opaque compiled-FX backend entries. Independent device 0/1 results and the Llama native-composite RoPE helper occurrence contract are in `references/compiled_fx_rotary_fallback_dispatch_cann_8_5_1.json`. Each source `RotaryEmbedding.forward_native` group incurs one `rotary-cos-sin-gather` anchor; `rotary-*` children retain ordinary eager-submit service. Registered or fused RoPE paths do not create this fallback entry.

`runtime/launch/model_runner_outer_prepare_runner.py` measures host-only source service for `NPUModelRunner` outer input, attention-metadata binding and logits-index preparation, without executing a model or NPU operator. The source snapshot, ranked/unranked token-embedding/LM-head anchor contract and independent container artifacts are in `references/model_runner_outer_prepare_cann_8_5_1.json`. Production events remain graph-external during replay and fail closed if either paired anchor is missing.

`runtime/launch/graph_replay_sync_runner.py` measures the `torch.npu.current_stream().synchronize()` API baseline after independent synthetic NPU work and verifies pending-device barrier behavior. Device 0/1 results and the vLLM-Ascend `ACLGraphWrapper` one-sync-per-replay occurrence contract are in `references/graph_replay_sync_cann_8_5_1.json`. Forward consumes only source-declared barrier semantics: wait for queued device predecessors, then charge the existing graph-replay enqueue service. Serving residuals and pending-work duration are not fitted into constants.

Each compilable leaf maintains its own `CMakeLists.txt`. Runner/report arguments are documented by each script's `--help`. Build artifacts belong in the leaf's ignored `build/` or an external temporary directory and must not be committed.

## Evidence currently consumed by Production Forward

Only the published mechanisms below are consumed by production. Reference/HWSpec files own values, units, domains and provenance; this README does not duplicate them.

| Mechanism | Canonical evidence | Consumer |
|---|---|---|
| NORM serialized Sqrt/Duplicate/Div request service | `references/vector_norm_request_cann_8_5_1.json` | Typed Vector service in `configs/hardware/ascend_910b.yaml` |
| Cast/Muls/Adds/Mul serialized Vector service | `references/vector_serialized_services_extended_cann_8_5_1.json` | Vector affine service in `configs/hardware/ascend_910b.yaml` |
| AddRMSNorm key-30 ReduceSumFP32 service | `references/add_rmsnorm_reduce_sum_fp32_cann_8_5_1.json` | Vector affine service in `configs/hardware/ascend_910b.yaml` |
| AddRMSNorm key-30 32-AIV H2048/H2560 input-pair completion | `references/aiv_add_rmsnorm_row_group_input_pair_completion_32_workers_cann_8_5_1.json` | Per-row key-30 `input_load`, taking the maximum with same-stage byte-capacity time |
| Vector-to-MTE2 fence service | `references/vector_to_mte2_fence_cann_8_5_1.json` | CANN AddRMSNorm key-30 per-row source DAG |
| RMSNorm first DataCopyPad MTE2→Vector completion | `references/aiv_datacopypad_mte2_to_vector_completion_{32,43,47}_workers_cann_8_5_1.json` | Exact AIV admission domain of the first CANN key-0 row-group gamma load |
| AIC/AIV/mixed empty-task floor | `runtime/launch/` retained capture | Scheduling LUT in `configs/hardware/ascend_910b.yaml` |
| minimal non-empty AIV task floor | `runtime/launch/` retained capture/report | `aiv_nonempty_task_latency_model` |
| ACL graph replay pre-replay synchronization barrier | `references/graph_replay_sync_cann_8_5_1.json` | Ascend `RuntimeGraphSegmentDescriptor.sync_before_replay` device-timeline barrier |
| FIA three-handoff program span | `references/fia_cross_domain_three_handoff_cann_8_5_1.json` and retained report | `fia_cross_domain_three_handoff` |
| Vector ReduceSum request latency | `operators/norm/` retained capture | `vector_reduce_sum_latency_lut` |
| CANN tiling/metadata facts | `references/*tiling*.json` and corresponding capture runner | Ascend launch package |
| AIC serialized global scalar read | `references/aic_scalar_global_get_value_cann_8_5_1.json` | paged SplitFuse QK/PV descriptor address dependency |
| GroupedMatmul K2880 base-M128 per-output-tile completion | `references/grouped_matmul_k2880_output_tile_completion_cann_8_5_1.json` | Finite-stream lower bound for each CANN MoE `SetSingleShape`/`IterateAll` |

DRAM latency/bandwidth, MTE2/MTE3, FixPipe, softmax and other operator experiments are candidates or diagnostic tools by default. They are not published unless explicitly referenced by current reference/HWSpec files, and predictors must not read them silently.

## Capture contract

A publishable capture must:

1. Use an explicit canonical-checkout commit or an audited source snapshot.
2. Connect through `ssh <capture-host-0118b599d5e7>`, execute in the designated engineering copy inside `<ascend-container>`, and verify the target NPU is idle before starting.
3. Isolate one physical mechanism, keeping unrelated initialization, host synchronization and complete operators outside the timed region.
4. Retain source, binary/compiler/CANN identities, command, device telemetry, per-trial raw data, numerical validation and exit status.
5. Freeze sweep, trial/order and publication gates in advance; do not choose points, thresholds or formulas after seeing results.
6. Mark failures explicitly unpublishable, without cherry-picking cells, building per-shape LUTs or reverting to old parameters.
7. Confirm child processes have exited and device state has recovered after the benchmark.

Write all raw artifacts under:

```text
profiler_output/microbench/ascend_910b/<run-id>/
```

This untracked directory retains complete raw provenance. Compact production evidence belongs in `references/` or `configs/hardware/ascend_910b.yaml`, explicitly bound to the raw-report identity.

## Publication flow

```text
source/contract freeze
        ↓
isolated hardware capture
        ↓
report validates raw + provenance
        ↓
candidate reference
        ↓
consumer/source audit
        ↓
HWSpec or launch package publication
        ↓
repository tests and evaluation
```

Hardware tasks produce only raw data and candidates; candidates never enter Forward automatically. Before publication, verify the consumer's dtype, operation count, task/worker mapping, timer domain and applicability exactly match the benchmark. If that mapping cannot be established, parameters remain unavailable and the predictor reports a limitation or fails loudly.

Historical runs, failure conclusions, checksums and one-time troubleshooting belong in retained raw data, Git history and corresponding reports. They are not maintained here as ongoing design contracts.

## FIA nonempty controller program span

`runtime/fia_program_span/` reproduces the fixed CANN 8.5.1 regular-GQA FIA three-slot `RunInfo` task cache, `CreateTask`/`ShouldExecuteTask`/`ExecuteTask` controller and three MM1→Vec1→MM2→Vec2 cross-domain handoffs. Timing starts before task-cache initialization and ends when the last valid Vec2 actor-local body returns. All four stages perform only checksum-retained minimal scalar work, keeping source branches nonempty without reproducing complete Attention arithmetic, tensor traffic or a production kernel. The frozen calibration sweep is the Cartesian product of active AIC groups `{1,8,16,24}` and tasks per group `{1,2,3,10,11}`. Preregistered holdouts use only 22-group single-task, `20×3+2×2` and `11×11+11×10` tail topologies. Before each measured cell, one untimed 24-group/11-task scheduler preamble and two exact-cell warmups remove preceding-grid state from the host-global mixed-task scheduler. Each of three independent runs uses six forward/reverse position-balanced rounds; every cell must have cross-run span and position-order error ≤20% to qualify for Forward.

Results have units `ns_per_complete_aic_group_program_span` and may only replace the old empty-ring span as an end-to-end lower bound on the source-bound main DAG. They must not be added to the existing task DAG or mixed-task outer envelope. The report freezes local/shared depth-survival formulas and the exact active-group/task-count domain in advance, independently checking intermediate counts and 22-group tail holdouts. Even after mechanism gates pass, HWSpec publication requires demonstrating improvement in the target decode bucket without regressions in other FlashAttention cases.

The three-run CANN 8.5.1 nonempty campaign passed cross-run and validation gates, but two position-balance cells exceeded the frozen 20% limit. The span is therefore unpublishable and must not enter Forward; stable cells must not be cherry-picked into services, parameters or LUTs. `references/fia_nonempty_program_span_negative_diagnostic_cann_8_5_1.json` retains auditable negative-diagnostic identity and gate conclusions only. The published empty source-bound FIA three-handoff service excludes nonempty actor bodies and cannot be extrapolated into this nonempty controller span.

## FIA cross-domain edge completion

`runtime/fia_edge_handoff/` executes C1→V1, V1→C2 and C2→V2 CrossCore dependency edges in fixed CANN 8.5.1 order within `KERNEL_TYPE_MIX_AIC_1_2`. Device-global 50 MHz `GetSystemCycle` measures producer-to-consumer timestamp differences. The timed region excludes Attention arithmetic, tensor transfers and production operators; actors write timestamps and complete-ring elapsed time only after all dependency edges finish. The frozen v2 campaign covers active AIC groups `{1,8,11,12,16,18,19,21,22,23,24}` and task counts `{4,8,16,32,64,92,128}`, directly retaining the first three completions and later steady completions. Each of three independent captures uses six forward/reverse position-balanced rounds. Per-edge completion service is eligible only if provenance, cross-run, cross-task-count, order, held-out-run and complete-ring lower-bound gates all pass. It is not complete FIA-task or production-kernel duration.

The first v1 capture had steady cross-run span ≤2.6%, but reducing three-slot ring fill to a single startup reached 38.5% APE on holdout `B24/N4/C2→V2`. The two V1→C2 senders also violated the assumed lane symmetry. All v1 results are therefore unpublishable; `references/fia_cross_domain_edge_completion_v1_negative_diagnostic_cann_8_5_1.json` freezes identity and failed gates. No subset may enter Forward.

After v2 directly retained the first three completions, the complete-ring lower-bound gate passed. Per-edge latency still varied substantially with task count and execution order at the same active-group count: maximum task-count span 162.9%, balanced-order P95 72.0%, independent-validation maximum APE 116.2%. An individual handoff is therefore not a stable service reusable independently of mixed-core scheduler state. `references/fia_cross_domain_edge_completion_v2_negative_diagnostic_cann_8_5_1.json` records this counterevidence; no topology cell or stable subset may be published to Forward.

## MatMulV3 large-M diagnostic boundary

Four launch-only captures from `operators/gemm/tiling_query_runner.py`, together with CANN 8.5.1 static source, identify two legacy large-effective-M mechanisms: key 65632 is an AIC:AIV=1:2 single-core Split-K GM-to-L1 mixed kernel; key 65536 is an AIC-only base kernel. Captured `Mc2MatmulV3TilingData` encodes L2 tiles and run-info after the 50-word `TCubeTiling`, without an arch35 Stream-K tail. The M4096 capture establishes key-65632 `singleCore=(512,2816,384)`, `base=(128,128,128)`, depth `(9,6)`, step `(3,1)`, L2 tile `(8,8,4,4,calOrder=0)` and `4*M*N + 20,972,032 bytes` workspace. Source shows each AIC exclusively owns a complete M×N macro tile and serially accumulates FP32 workspace in K=384 chunks; its two paired AIVs cast/store each completed inner-M stripe. There is no cross-AIC Stream-K tail reduction. Aligned stripes have only the one-way AIC `NotifyEvent<PIPE_FIX>(5)` → AIV `WaitFlagDevLocal(5)` edge, without per-stripe AIV→AIC completion. M macro tile, inner-M stripe and N macro tile uniquely determine workspace offsets, so consecutive stripes do not overwrite the same region. Source rejects both `reverse_aiv_to_aic_completion_per_stripe` and `workspace_tile_overwrite_requires_release`; neither justifies adding completion waits. Remaining cluster error is mechanism-level WONTFIX. The key-65536 source-order generic LRU still fails to reproduce retained non-duration traffic/cache counters, so L2 residency correction remains partial unsupported. Complete capture identity, source/binary SHA256, payload ABI, candidate rulings and LRU/counter gaps are in `references/matmul_v3_large_m_diagnostic_cann_8_5_1.json`.

`operators/gemm/single_core_split_k_tile/` further isolates the complete resident `128×128×{384,256}` local K-tile/tail program of each key-65632 `AMatMulB`: inputs reach L1 before timing; the timed region retains `SetSingleShape`, A1/B1 tensor binding, `Iterate`, M-to-FixPipe handoff, one FP32 workspace-tile store, `End` and ping/pong L0C lifecycle. It excludes GM→L1, AIV, production outer loops and complete GEMM. The frozen sweep uses `{64,256,1024,4096}` iterations per K extent on 24 AICs. Publication requires three forward/reverse runs passing positive-difference and 10% cross-run stability gates.

The `long`, `k5120-long` and `k13824-long` profiles of `memory/dram/mte2_tile/gemm_source_order/aic_mte2_gemm_region_runner.py` replay complete discrete worker schedules for key-65536 large regions `256×112, K8192`, `256×54, K5120` and `256×20, K13824`, respectively. They independently measure MTE2, MTE2→MTE1 and MTE2→MTE1→Cube spans. MTE2-only spans do not exceed the corresponding unified analytical pipeline. The latter two probes actively drain producer/consumer queues and cannot serve as production MatmulImpl completion floors. Exact programs, three-run results, raw/source/binary identity and rejection boundaries are in `references/matmul_v3_large_m_diagnostic_cann_8_5_1.json`.

The same directory's `k1536_m16_n224` profile isolates an asymmetric finite source program on 23 AICs: one A1 `16×1536` descriptor and six B1 `224×256` descriptors, retaining MTE1/Cube consumers. MTE2-only span does not establish missing service; drained MTE1/Cube completion does not represent production overlap. This cell is validation-only negative evidence; see `references/aic_matmul_v2_k1536_narrow_pipeline_completion_cann_8_5_1.json`.

## AddRMSNorm Split-D launch probe

`operators/norm/add_rmsnorm_launch_{application,interposer,report}.*` performs four fixed launch-only probes for BF16, H=16384 and T=1/64/256/1024. At the `aclnnAddRmsNorm` execute boundary, the interposer captures `aclrtLaunchKernelWithHostArgs` host arguments, placeholder map, runtime kernel name and `block_dim`, then returns success without submitting the target device kernel. The report strictly decodes 80-byte `AddRMSNormTilingData` through `AddRmsNorm` runtime reflection, rejecting non-key-31 kernels, layout changes and inconsistent repeated executions. It reads no duration, counters, utilization or evaluation error.

Before execution, verify device availability under the root contract and bind the physical device explicitly. Use `operators/flash_attn/tiling_reflection_runner.cpp` to generate a reflection TSV for the `libophost_nn.so` `AddRmsNorm` registration key, compile the interposer, run the application with `ASCEND_RT_VISIBLE_DEVICES=<physical-id>` and `LD_PRELOAD=<interposer>`, then invoke the report. Application/report `--help` and interposer source constants define the arguments. Raw artifacts must be written under `profiler_output/microbench/ascend_910b/<run-id>/`; compact validated discrete facts are in `references/add_rmsnorm_splitd_launch_cann_8_5_1.json`.

## AddRMSNorm SingleN input-pair completion probe

`operators/norm/add_rmsnorm_input_pair_{kernel,main,runner,report}.*` isolates the two equal-size initial GM→UB `DataCopyPad` requests and their two `MTE2_V` completion events in the CANN 8.5.1 AddRmsNorm SingleN BF16 path. Every AIV worker issues two requests with the same `payload_bytes`; the frozen sweep permits only exact `(block_dim,payload_bytes)` matches in `{1,2} × {4096,5120,8192,10240,16384}`. Results are ns per two-request launch completion and cannot be split into two additive independent requests. The probe neither executes full AddRMSNorm nor consumes production/evaluation duration. Production also performs a first-input BF16→FP32 Cast between the waits, so this remains a pair-completion mechanism candidate, not connected to predictor or HWSpec.

Publishable evidence requires three distinct run IDs, each with 120 position-balanced trials. For each trial/worker the report computes `request_interval(pair_order=1)-control_interval(pair_order=0)`, then selects the slowest worker in the launch. Within-run parity replication and corresponding-cell cross-run median spread must be ≤20%. The qualifying CANN 8.5.1 reference, raw/source/binary SHA256, exact units and lookup domain are in `references/aiv_add_rmsnorm_single_n_input_pair_completion_cann_8_5_1.json`.

`operators/norm/add_rmsnorm_row_group_input_pair/report.py` publishes only exact key-30, 32-AIV, BF16 H2048/H2560 cells with 4096/5120 bytes per input. It audits these cells from the original 10-cell position-balanced CSV, rejects prefiltered CSVs, and applies the parity gate across two complete schedules in each of three independent runs. Other payloads in the same campaign that fail reproducibility gates do not enter the reference or Forward.

## FIA AIV workspace transfer

`memory/dram/aiv_workspace_transfer/` isolates short workspace transfers in CANN 8.5.1 `FiaBlockVecNonQuant::DealBmm1ResBaseBlock` and `DealBmm2ResBaseBlock`. Vec1 cells measure 4096 B GM→UB, 2048 B UB→GM and their ordered chain; Vec2 cells measure 256 B state reads, 256 B state writes and their ordered read/write chain with `MTE3_MTE2` event/wait. Every cell retains the real `KERNEL_TYPE_MIX_AIC_1_2` topology. Under the D64, M=1 source division, the primary AIV executes Vec1/Vec2; the secondary retains `vecDealM=0`, with host validation that it does not execute the main data chain. The probe executes no Attention, MatMul or production workload.

In an audited 910B source snapshot, first record the snapshot SHA256 computed by the runner:

```bash
python microbench/ascend_910b/memory/dram/aiv_workspace_transfer/aiv_workspace_transfer_runner.py snapshot
```

Scan devices from lower to higher IDs under project modeling and evidence requirements, confirm an idle card, then explicitly supply the actual physical/logical device ID, complete source commit and snapshot. The output directory must be under `profiler_output/.../ascend_910b/` and must not already exist:

```bash
python microbench/ascend_910b/memory/dram/aiv_workspace_transfer/aiv_workspace_transfer_runner.py run --run-id <run-id> --source-commit <40-hex-commit> --expected-snapshot-sha256 <64-hex-sha256> --physical-device-id <physical-id> --output-dir profiler_output/microbench/ascend_910b/<run-id>
python microbench/ascend_910b/memory/dram/aiv_workspace_transfer/aiv_workspace_transfer_report.py --capture-dir profiler_output/microbench/ascend_910b/<run-id>
```

The runner configures/builds inside the capture directory using only the frozen, verified source snapshot; external binaries are rejected. It exposes the selected physical card as container logical device 0 and records 910B2 model, board ID and PCI identity. The runner produces `manifest.json` and `raw.csv`; the report produces `result.json` and detached `candidate.json`. Candidate units are `ns_per_actor_task_completion`. They may only supply a completion lower bound for an independent short-transfer chain, combined with aggregate-bandwidth time using `max`; they must not be summed across independent AIV actors and never enter Forward automatically.

## RoPE first-MTE completion probe

`memory/dram/latency/first_mte_completion_runner.py --campaign rope_key24_q_slice_edges` isolates key-24 4096 B Q-slice request-completion edges: direction 2 is `MTE2_V` after GM→UB `DataCopy`; direction 3 is `MTE3_V` after UB→GM `DataCopy`. The frozen sweep includes only `(block_dim,tile_bytes)` unloaded control `(1,4096)` and representative concurrency geometries `(32,4096)`, `(43,4096)`, `(47,4096)`, reversing request/control order for each. It does not execute production RoPE. The complete key-22 8192 B Q slice remains explicitly partial in this campaign; independent `(1,8192)` control and `(16,8192)` representative probes are needed. Direction 3 at 256 B is not production-representative.

`--campaign rope_key22_first_q_slice_edges` independently isolates the first complete Q slice in the key-22 single-UB program: CANN 8.5.1 ATB source fixes the BF16 slice at 4096 elements, or 8192 B. The base sweep includes unloaded control `(1,8192)` and maximum key-22 evaluation-domain concurrency `(16,8192)`. `rope_key22_active4_replication_edges` and `rope_key22_active8_replication_edges` add exact 4-worker and 8-worker admission geometries, respectively. All use directions 2/3 and reverse request/control order for each cell. Evidence applies only to each worker's first Q slice; it must not extend to later Q/K slices, unmeasured 2-worker geometry or key-24 ping-pong semantics.

Stability validation requires three separate runner invocations with distinct run IDs and three separate stdout CSVs. More schedule cycles within one run do not replace independent replication:

```bash
python microbench/ascend_910b/memory/dram/latency/first_mte_completion_runner.py --binary <absolute-first_mte_completion-binary> --device-id <physical-device-id> --run-id rope-key24-q-slice-01 --campaign rope_key24_q_slice_edges > profiler_output/microbench/ascend_910b/rope-key24-q-slice-01/raw.csv 2> profiler_output/microbench/ascend_910b/rope-key24-q-slice-01/runner.log
python microbench/ascend_910b/memory/dram/latency/first_mte_completion_runner.py --binary <absolute-first_mte_completion-binary> --device-id <physical-device-id> --run-id rope-key24-q-slice-02 --campaign rope_key24_q_slice_edges > profiler_output/microbench/ascend_910b/rope-key24-q-slice-02/raw.csv 2> profiler_output/microbench/ascend_910b/rope-key24-q-slice-02/runner.log
python microbench/ascend_910b/memory/dram/latency/first_mte_completion_runner.py --binary <absolute-first_mte_completion-binary> --device-id <physical-device-id> --run-id rope-key24-q-slice-03 --campaign rope_key24_q_slice_edges > profiler_output/microbench/ascend_910b/rope-key24-q-slice-03/raw.csv 2> profiler_output/microbench/ascend_910b/rope-key24-q-slice-03/runner.log
```

Key-22 campaigns also require three distinct run IDs and at least one complete balanced schedule cycle per run. Run `first_mte_completion_report.py --campaign <campaign> <run1.csv> <run2.csv> <run3.csv>` for independent-replication gates. Qualifying CANN 8.5.1 results, raw/source/binary SHA256 and exact lookup boundaries are recorded in the corresponding `references/aiv_rope_key22_first_q_slice_*_cann_8_5_1.json`.

## SwiGLU T1024 queue-edge probe

`memory/dram/latency/first_mte_completion_runner.py --campaign swiglu_t1024_queue_edges` isolates two depth-one TQue completion edges in CANN 8.5.1 BF16 SwiGLU: direction 4 is `MTE2_V` after GM→UB `DataCopy`; direction 5 is UB→GM `DataCopy` after Vector→MTE3 handoff. The static tiler's dominant T1024 base tile is `19 × 512 × 2 = 19456 B`, so the frozen `(block_dim,tile_bytes)` sweep includes only unloaded control `(1,19456)`, `(24,19456)`, `(48,19456)` and source-valid single-buffer payload control `(48,26624)`, reversing request/control order for every cell. This campaign does not execute complete SwiGLU or read production duration.

Publishable evidence requires three distinct run IDs, with at least one complete balanced schedule cycle each. Every run must pass schema, cell/order balance, pair-order replication and positive-completion gates; corresponding-cell cross-run median spread must be ≤20%. This reference is an independent mechanism candidate, not automatically consumed by the production pipeline.

The three-run CANN 8.5.1 result passing every gate, with raw/source/binary SHA256, is in `references/aiv_swiglu_t1024_queue_edge_completion_cann_8_5_1.json`. Lookup requires exact worker count, direction and payload; production is not yet connected.

## ReshapeAndCache key-200 equal-chain probe

`operators/kv_cache/single_tbuf_lifecycle/single_tbuf_lifecycle_runner.py` isolates one complete `TPipe` scope with one `VECCALC` `TBuf` initialization. The frozen sweep covers worker counts `{1,2,4,8,16,48}` and exact 1024/2048-byte CANN key-200 token buffers plus the 110080-byte CANN BF16 elementwise key-210020000 two-tile allocation (`ubFormer=5504`, compile-time DAG footprint 20 bytes/element). Each interval repeats 64 independent scopes to resolve the 20 ns device counter; `single_tbuf_lifecycle_report.py` uses the long-minus-short interval and requires three independent runs plus payload and concurrency invariance. The published service is a workload-independent per-AIV lifecycle and excludes slot mapping, copies, token loops, cache layout, arithmetic and complete production duration.

`operators/vector/gather_v3_two_tque_lifecycle_kernel.cpp` replaces only the device kernel in the preceding frozen harness to isolate the exact CANN 8.5.1 GatherV3 key-10330 queue initialization: one depth-two 512-byte `TQueBind<VECIN,VECOUT,1>` plus one depth-two 8-byte `TQue<VECIN,1>` in a single `TPipe` scope. The unchanged payload field is only an invariance label because both allocations are compile-time constants. Three independent runs must cover worker counts `{1,2,4,8,16,48}` and all three labels, and the existing report must pass before the exact one-lifecycle service can be used. The probe excludes queue access, copies, scalar index decode, Gather arithmetic and complete operator duration; the published evidence and exact source/snapshot identity are in `references/aiv_gather_v3_two_tque_lifecycle_cann_8_5_1.json`.

`operators/kv_cache/reshape_cache_equal_chain_runner.py` isolates the CANN 8.5.1 `ReshapeAndCache` key-200 equal-head copy chain. Each AIV uses the same `TBuf` for sequential K and V GM→UB→GM transfers; each chain retains `MTE2_MTE3` and `MTE3_MTE2` hard completion edges. It excludes slot mapping, token loops, host tiler, cache layout and complete production-operator duration. The frozen sweep includes only unloaded block1 control and the block48 source representative with highest error contribution, using 1024 B and 2048 B payloads and reversing request/control order.

Publishable evidence requires three distinct run IDs. Each must pass schema, 64-trial balanced cell order, pair-order replication and positive-completion gates; corresponding-cell cross-run median spread must be ≤20%. Verify the physical card is idle, explicitly supply device ID, and save stdout/stderr only under `profiler_output/`:

```bash
python microbench/ascend_910b/operators/kv_cache/reshape_cache_equal_chain_runner.py --binary <absolute-reshape-cache-equal-chain-binary> --device-id <physical-device-id> --run-id <unique-run-id> > profiler_output/microbench/ascend_910b/<run-id>/raw.csv 2> profiler_output/microbench/ascend_910b/<run-id>/runner.log
python microbench/ascend_910b/operators/kv_cache/reshape_cache_equal_chain_report.py <run-1.csv> <run-2.csv> <run-3.csv>
```

The three-run result passing all gates, with raw/source/binary SHA256, is in `references/reshape_cache_equal_chain_completion_cann_8_5_1.json`. Production consumes its exact service only for key 200, worker counts 1 or 48, and individual K/V payloads 1024 B or 2048 B. Other domains remain partial, without interpolation.

The `equal_midworkers_warm` campaign first executes exactly 16 launches on the same mechanism buffer, then measures the two-copy hard-event chain's device interval inside launch 17 to isolate cache-steady runtime service. Warmup launches are outside the timed region. The probe still excludes TPipe/TBuf initialization, slot mapping, token loops, cache layout and complete production duration. Its frozen domain is worker counts `{2,4,8,16}`, payloads `{1024,2048}` and one token per worker. Forward consumes only exact worker/payload cells, without interpolation/extrapolation or reuse for repeated-token workers.

```bash
python microbench/ascend_910b/operators/kv_cache/reshape_cache_equal_chain_runner.py --binary <absolute-reshape-cache-equal-chain-binary> --device-id <physical-device-id> --run-id <unique-run-id> --campaign equal_midworkers_warm --warmup-launches 16 > profiler_output/microbench/ascend_910b/<run-id>/raw.csv 2> profiler_output/microbench/ascend_910b/<run-id>/runner.log
python microbench/ascend_910b/operators/kv_cache/reshape_cache_equal_chain_report.py --campaign equal_midworkers_warm --warmup-launches 16 <run-1.csv> <run-2.csv> <run-3.csv>
```

Exact services, stability gates, source/binary identities and raw SHA256 from the three independent formal runs are in `references/reshape_cache_equal_chain_warm_completion_cann_8_5_1.json`.

## Elementwise source-control lifecycle probe

`operators/vector/elementwise_control_lifecycle/` isolates control overhead of the common CANN 8.5.1 BF16 Add/Mul scheduler on exactly 48 AIVs, without executing Add/Mul tensor payloads. Frozen variants are `control_baseline` with the common tile loop and terminal `PIPE_ALL` fence; `dag_tbuf` adds one 384 B `VECCALC` TBuf, seven-node traversal and 12 buffer-ID lookups per tile; `full_lifecycle` also adds `PIPE_MTE2/V/MTE3 = 2/9/1` mutex lock/unlock operations per tile. Tile-count classes are fixed at `{18,29,30,36,58,59}`: `{18,36,59}` are affine-calibration cells only, and `{29,30,58}` are holdouts frozen before capture.

`operators/vector/elementwise_single_tile_program/` retains the common CANN BF16 Add/Mul 47-AIV single-tile memory/control program: 46 workers handle 1408 elements each, the tail worker handles 768, executing a pair of GM→UB copies, three casts, `Duplicate` in place of production Add/Mul, one UB→GM copy and a terminal `PIPE_ALL` fence. Forward uses only the three-run median of the maximum worker body as a contention lower bound for this exact source domain; production arithmetic remains conserved by independent Vector demand. Complete production/evaluation duration is not used.

Capture requires three independent run IDs, each with six forward/reverse position-balanced rounds. The report requires corresponding-cell cross-run median spread and position-balance error ≤20%, holdout MAPE ≤15%, maximum APE ≤25%, and positive paired deltas and `startup_ns + tile_count * per_tile_ns` parameters. If any gate fails, parameters stay `null` and `mechanism_publishable=false`. Qualifying candidate units are `ns_per_48_aiv_source_control_program_completion`; later integration may only use `max(existing_48_aiv_tile_pipeline_span, source_control_candidate)`, without addition or per-shape LUTs. This directory generates no canonical reference and does not automatically connect to Forward.
