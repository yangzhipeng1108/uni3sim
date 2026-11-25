# WP11 (wp11_moe_fp8) report: H200 FP8 block-quantised Triton fused-MoE

Branch `h200-port-wp11_moe_fp8`, baseline `89f80345b0321681caa7a86049475f84f0e0cc52`. Hardware: the capture host GPUs 0 and 3 only. Remote run root `<public-root>/h200_microbench/wp11_moe_fp8/` (`src/` execution copies, `runs/`, `logs/`, `eval/`). Local raw copies (gitignored): `profiler_output/microbench/nvidia_h200/wp11_fp8_tile_pipeline_r2/`, `profiler_output/microbench/nvidia_h200/wp11_fp8_tile_wave_refill_r2/`; evaluation records (gitignored): `experiment_logs/h200_wp11_moe_fp8/`.

Status: all five hard gates pass (evidence below). Goal completed.

## Gate table

| Gate | Target | Actual | Status |
|---|---|---|---|
| G1 | source-derived launch | The H200 selector (`src/kernel/launch_config/nvidia_h200/fused_moe_launch.py`, owned via `_OWN_PACKAGE_SELECTORS["nvidia_h200"]`) reproduces grid, block, tile, GROUP_SIZE_M/warps/stages (bundled tuned config) and dynamic shared memory of all 14 distinct in-scope geometries (4 pure-decode windows x W1/W2 + 3 prefill step sizes x W1/W2). Registers match 12/14; the two naive-assignment W2 launches run 97 registers vs the 100 recorded in the WP0 compile metadata (source reason below; residency unchanged at 4 CTAs/SM). No cluster launch (Triton non-cluster). | Pass |
| G2 | zero-fit provenance | Every new HWSpec parameter is from two new H200 mechanism probes (synthetic FP8 block sparse-row K loop; multi-wave L2-resident refill), clock-gated at 1980 MHz (56/56 processes accepted), with scripted runner + capture driver + derivation and `references/*fp8*_sm90.json`, or a static source/compile/spec fact; the reused early-return tail is an existing H200 reference. No production kernel or serving duration sets, selects or tunes any value. | Pass |
| G3 | serving path | In the H200 DSV3-L32 TP4 midlayer simulation every FUSED_MOE interval is priced by `vllm_triton_fused_moe_fp8_block128_{naive,aligned}_{w1,w2}` (step dump `sim_by_kind_algorithm_us`, all 5 compare runs); no A800 `triton_fused_moe_w1/w2` interval remains. | Pass |
| G4 | evaluation reported | Per-geometry predicted vs real per-call table (14 rows, before/after) and family tables before/after for C=1/4/16/64 and prefill; extraction script `scripts/evaluation/serving/extract_h200_fused_moe_trace_calls.py`, comparison `scripts/evaluation/serving/compare_h200_fused_moe_fp8_calls.py`. | Pass |
| G5 | A800 unchanged + tests | 8 A800 kernel-dataset fused_moe cases (47 physical kernels) give byte-identical semantic evaluation output before (baseline tree) and after (sha256 `edaa0d4c...` both); `python scripts/run_tests.py`: 4101 passed. | Pass |

## What was built

- Launch (`src/kernel/launch_config/nvidia_h200/fused_moe_launch.py`): for W1/W2 FUSED_MOE with BF16 compute dtype, FP8 routed weights, `weight_group_size` 0, EP1, E=256, top-8, hidden 7168 and N/rank in {512, 256}, the selector resolves the bundled `E=256,N=<N>,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` config by vLLM's nearest-key rule (ranges and Triton 3.6 compile facts restated from `microbench/nvidia_h200/references/fused_moe_tuned_config_vllm_0_24_0_h200.json`), applies `_prepare_expert_assignment`'s naive predicate (`tokens*topk*4 <= E`), and derives `EM` (`numel + E*(BM-1)` capped at `numel*BM` when `numel < E`, and at `A.size(0)*top_k*BM` when `A.size(0) < BM`; naive `EM = tokens*topk*BM`), `grid = cdiv(EM,BM)*cdiv(N_proj,BN)`, `block = 32*num_warps`, `BLOCK_SIZE_K = min(BK,128)`, the W2 router-weight marker, dynamic shared memory and warps/stages. Every other H200 FUSED_MOE request (BF16 routed weights, other E/H/N, EP>1, the INT4 `routed_experts` aggregate) keeps the projected vLLM 0.11 rule, restated verbatim because hardware packages cannot import one another; tests pin it equal to the A800 selector.
- HWSpec: `scheduling.grouped_gemm_fp8_block_tile_pipelines` in `configs/hardware/nvidia_h200.yaml`, one row per compiled FP8 tile program (7 rows: BM16 w4 s3/s4/s5, BM16 w8 s4, BM32 w4 s3, BM64 w4 s3/s4), normalized by `_normalize_grouped_gemm_fp8_block_tile_pipelines` in `src/kernel/hardware_spec.py` (exact fields, content-addressed references, values must equal the reference, `pipeline_depth = num_stages - 1`, threads = 32 x warps, residency cross-checked against SM register/shared-memory/thread capacities).
- Predictor (`src/kernel/predictors/fused_moe.py`, `_fp8_block_model`): source-derived FP8 traffic (1-byte A/B, FP32 per-token-group A scales and per-block B scales per K block, FP32 router weight for W2, BF16 masked output; naive: one M tile per route with one valid row; aligned: `ceil(rows/BM)` tiles per expert; DRAM = unique compulsory bytes), Tensor work at the HWSpec FP8/primary rate ratio, dequantization FMA work (`acc += dot*a_s*b_s`), and the existing `GpuGroupedGemmPipelineEnvelopeDemand` resident-wave K-pipeline keyed to the FP8 row via the new optional `fp8_block_program` field (`src/kernel/model_contract.py`).
- Formula (`src/kernel/formula.py`, `_resolve_gpu_grouped_gemm_pipeline_envelope`): FP8 rows take their coefficients from the named row and skip the BF16-only long-K and sparse-refill special cases; every resident wave after the first costs `max(K_blocks x max(local, DRAM round), wave_refill + K_blocks x wave_k_round_interval, K_blocks x DRAM round)`. BF16 rows (`fp8_block_program = None`) are unchanged.
- Probes: `microbench/nvidia_h200/operators/fused_moe/sm90_fp8_block_tile_pipeline_{runner.py,capture.sh,reference.py}` -> `references/fp8_block_grouped_gemm_tile_pipeline_sm90.json`; `microbench/nvidia_h200/operators/fused_moe/sm90_fp8_block_tile_wave_refill_{runner.py,capture.sh,reference.py}` -> `references/fp8_block_grouped_gemm_tile_wave_refill_sm90.json`.
- Tests: `tests/kernel/test_h200_fp8_fused_moe.py` (trace geometry reproduction, naive/aligned predicate, projected-rule equality for out-of-domain and `routed_experts`, FP8 template use, fail-closed for unmeasured programs / A800, content-addressed rows).
- Evaluation: `scripts/evaluation/serving/extract_h200_fused_moe_trace_calls.py` (stdlib; run on the remote CSVs), `scripts/evaluation/serving/compare_h200_fused_moe_fp8_calls.py`, and a `sim_by_kind_algorithm_us` grouping added to `compare_h200_decode_kernel_families.py` (reporting only).

## Kernel source skeleton (vLLM 0.24.0 `fused_moe_kernel`, `use_fp8_w8a8`, block [128,128])

Per program: map `pid` to `(pid_m, pid_n)` with GROUP_SIZE_M swizzle; early return if `pid_m*BM >= num_tokens_post_padded`; token rows from `sorted_token_ids` (aligned) or `pid_m` with one valid row (naive, constexpr `naive_block_assignment`); K loop over `cdiv(K,BK)` blocks: masked FP8 A load, FP8 B panel load (N wraps, not masked), A scale load (masked per row), B block scale load, `acc += tl.dot(a,b) * a_scale[:,None] * b_scale[None,:]`, Triton software pipelining with `num_stages - 1` operand buffers in shared memory (shared memory = (S-1) x (A+B stage) + scratch, e.g. 37376 = 2 x 18432 + 512); epilogue: W2 router-weight multiply, cast BF16, masked store to the token rows.

## G1: launch derivation vs real geometry

Real geometry: rank-0 (device 0) `fused_moe_kernel` rows of `dsv3l32_tp4_pure_decode_uniform_r2/window_{1..4}` (C=1/4/16/64) and `dsv3l32_tp4_prefill_uniform/window_1` (steps of one, three and four 7373-token prompts, identified from `EM`), extracted by `extract_h200_fused_moe_trace_calls.py` (W1/W2 by source-order pairing: W2 grid = 7 x W1 grid; 0 unpaired launches in every window). The older `vllm_0_24_0_dsv3_l32_tp4_launch_geometry_sm90.json` (earlier mixed decode+chunked-prefill windows) contains the same decode grids plus three chunked-prefill W1/W2 pairs (8600/60200, 9624/67368, 25032/175224; 236/237 registers, 0.099 MB); the selector yields exactly these pairs with config d26bcbdc1237b406 for chunk sizes of 6577-6584, 7601-7608 and 23009-23016 tokens respectively (the chunk token counts are not in the trace, so these rows are checked for existence only).

| window | M tokens | proj | real grid / block / regs / dyn smem (MB) | launches | selector algorithm (config id) | selector grid / block / tile / GSM / warps / stages / smem B / regs | result |
|---|---:|---|---|---:|---|---|---|
| C=1 | 1 | W1 | 64 / 128 / 100 / 0.037 | 7888 | `vllm_triton_fused_moe_fp8_block128_naive_w1` (cb47dc9924056440) | 64 / 128 / 16x128x128 / 1 / 4 / 3 / 37376 / 100 | match |
| C=1 | 1 | W2 | 448 / 128 / 97 / 0.037 | 7888 | `vllm_triton_fused_moe_fp8_block128_naive_w2` (cb47dc9924056440) | 448 / 128 / 16x128x128 / 1 / 4 / 3 / 37376 / 100 | match except regs (note) |
| C=4 | 4 | W1 | 256 / 128 / 100 / 0.037 | 6148 | `vllm_triton_fused_moe_fp8_block128_naive_w1` (0e0b351e3086258f) | 256 / 128 / 16x128x128 / 16 / 4 / 3 / 37376 / 100 | match |
| C=4 | 4 | W2 | 1792 / 128 / 97 / 0.037 | 6148 | `vllm_triton_fused_moe_fp8_block128_naive_w2` (0e0b351e3086258f) | 1792 / 128 / 16x128x128 / 16 / 4 / 3 / 37376 / 100 | match except regs (note) |
| C=16 | 16 | W1 | 1024 / 128 / 100 / 0.037 | 3248 | `vllm_triton_fused_moe_fp8_block128_aligned_w1` (cb47dc9924056440) | 1024 / 128 / 16x128x128 / 1 / 4 / 3 / 37376 / 100 | match |
| C=16 | 16 | W2 | 7168 / 128 / 100 / 0.037 | 3248 | `vllm_triton_fused_moe_fp8_block128_aligned_w2` (cb47dc9924056440) | 7168 / 128 / 16x128x128 / 1 / 4 / 3 / 37376 / 100 | match |
| C=64 | 64 | W1 | 2176 / 128 / 100 / 0.037 | 1537 | `vllm_triton_fused_moe_fp8_block128_aligned_w1` (cb47dc9924056440) | 2176 / 128 / 16x128x128 / 1 / 4 / 3 / 37376 / 100 | match |
| C=64 | 64 | W2 | 15232 / 128 / 100 / 0.037 | 1537 | `vllm_triton_fused_moe_fp8_block128_aligned_w2` (cb47dc9924056440) | 15232 / 128 / 16x128x128 / 1 / 4 / 3 / 37376 / 100 | match |
| prefill | 7373 | W1 | 9392 / 128 / 236 / 0.099 | 29 | `vllm_triton_fused_moe_fp8_block128_aligned_w1` (d26bcbdc1237b406) | 9392 / 128 / 64x128x128 / 16 / 4 / 4 / 98816 / 236 | match |
| prefill | 7373 | W2 | 65744 / 128 / 237 / 0.099 | 29 | `vllm_triton_fused_moe_fp8_block128_aligned_w2` (d26bcbdc1237b406) | 65744 / 128 / 64x128x128 / 16 / 4 / 4 / 98816 / 237 | match |
| prefill | 22119 | W1 | 24136 / 128 / 236 / 0.099 | 29 | `vllm_triton_fused_moe_fp8_block128_aligned_w1` (d26bcbdc1237b406) | 24136 / 128 / 64x128x128 / 16 / 4 / 4 / 98816 / 236 | match |
| prefill | 22119 | W2 | 168952 / 128 / 237 / 0.099 | 29 | `vllm_triton_fused_moe_fp8_block128_aligned_w2` (d26bcbdc1237b406) | 168952 / 128 / 64x128x128 / 16 / 4 / 4 / 98816 / 237 | match |
| prefill | 29492 | W1 | 31512 / 128 / 236 / 0.099 | 29 | `vllm_triton_fused_moe_fp8_block128_aligned_w1` (d26bcbdc1237b406) | 31512 / 128 / 64x128x128 / 16 / 4 / 4 / 98816 / 236 | match |
| prefill | 29492 | W2 | 220584 / 128 / 237 / 0.099 | 29 | `vllm_triton_fused_moe_fp8_block128_aligned_w2` (d26bcbdc1237b406) | 220584 / 128 / 64x128x128 / 16 / 4 / 4 / 98816 / 237 | match |

Register note: `naive_block_assignment` is a `tl.constexpr`, so the naive program (C <= 8) is a different compiled binary; the WP0 compile metadata was captured for the aligned specialization (100 registers). The trace shows 97 registers for naive W2 and 100 for naive W1. The selector carries no register parameter into Forward; the residency used by the template (4 CTAs/SM) is identical for 97 and 100 registers (both allocate 104 registers/thread per warp granule). Tile, GROUP_SIZE_M, warps and stages are not columns of the nsys trace; they follow from the tuned-config resolution and are consistent with the observed dynamic shared memory (37376 B for BM16 s3, 98816 B for BM64 s4). The simulated prefill step has 29,498 tokens (prompt lengths 7373..7376 in the probe driver) and launches grid 31520/220640; the per-geometry evaluation uses the real 29,492.

## G2: parameter provenance

New HWSpec rows (`configs/hardware/nvidia_h200.yaml`, `scheduling.grouped_gemm_fp8_block_tile_pipelines`):

| program | fixed_span_ns | local_k_round_interval_ns | wave_refill_ns | wave_k_round_interval_ns | K1024 held-out slope err (max) | pipeline_depth | resident CTAs/SM | regs W1/W2 (production) vs probe | dyn smem B (production = probe) | processes / min clock MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| bm16_bn128_bk128_w4_s3 | 6880.0 | 698.7 | 1645.3 | 1047.4 | 1.0% | 2 | 4 | 100/100 vs 98 | 37376 | 4+4 / 1980 |
| bm16_bn128_bk128_w8_s4 | 7008.0 | 592.0 | 2180.1 | 986.1 | 0.6% | 3 | 3 | 80/76 vs 80 | 55296 | 4+4 / 1980 |
| bm16_bn128_bk128_w4_s4 | 7040.0 | 600.0 | 1958.7 | 1042.5 | 1.0% | 3 | 4 | 100/100 vs 98 | 55808 | 4+4 / 1980 |
| bm16_bn128_bk128_w4_s5 | 7200.0 | 586.7 | 1942.6 | 827.6 | 1.0% | 4 | 3 | 100/100 vs 98 | 74240 | 4+4 / 1980 |
| bm32_bn128_bk128_w4_s3 | 7072.0 | 749.3 | 1521.6 | 1264.0 | 0.5% | 2 | 3 | 160/159 vs 162 | 41472 | 4+4 / 1980 |
| bm64_bn128_bk128_w4_s4 | 7392.0 | 605.3 | 2338.1 | 679.4 | 0.9% | 3 | 2 | 236/237 vs 200 | 98816 | 4+4 / 1980 |
| bm64_bn128_bk128_w4_s3 | 6986.7 | 704.0 | 2018.4 | 663.1 | 2.6% | 2 | 2 | 236/237 vs 200 | 74240 | 4+4 / 1980 |

| parameter | source | how derived |
|---|---|---|
| `fixed_span_ns`, `local_k_round_interval_ns` | H200 mechanism probe `sm90_fp8_block_tile_pipeline_runner.py` (synthetic sparse-row program with the vLLM 0.24 FP8 block K loop statement for statement; 8 streams; BM16/BM64 active rows 1/BM; 128-MiB polluter before every launch; K512/K2048 x N1024/N2048 = grid 64/128; 9 trials); reference `microbench/nvidia_h200/references/fp8_block_grouped_gemm_tile_pipeline_sm90.json` | Rule of the H200 BF16 BM16 pipeline reference, unchanged: per process and grid interval = (med K2048 - med K512)/12, fixed = med K512 - 4 x interval; component-wise maximum over 4 processes x 2 grids (`sm90_fp8_block_tile_pipeline_reference.py`) |
| `wave_refill_ns`, `wave_k_round_interval_ns` | H200 mechanism probe `sm90_fp8_block_tile_wave_refill_runner.py` (same K loop, operand streams wrapped mod 8 so operands are L2-resident, W = 1/2/4/8 full resident waves, K512/K1024/K2048, host launch covered by `torch.cuda._sleep`); reference `fp8_block_grouped_gemm_tile_wave_refill_sm90.json` | Per process: least-squares slope of the cell median over W at K512/K2048; interval = (slope2048 - slope512)/12, refill = slope512 - 4 x interval; maximum over 4 processes; K1024 slope held out (<= 2.6 %) (`sm90_fp8_block_tile_wave_refill_reference.py`) |
| `empty_tail_round_ns` (79.06) | existing H200 reference `bm16_grouped_gemm_empty_tail_sm90.json` (zero-work 128-thread CTA probe) | reused: the FP8 kernel keeps the same source early return before any load and launches 128 threads/CTA |
| `pipeline_depth` | static source/compiler fact | Triton `num_stages - 1` operand buffers; confirmed by the production and probe shared-memory footprints |
| `resident_ctas_per_sm_by_variant`, `registers_per_thread_by_variant`, `dynamic_shared_memory_bytes_per_cta`, `threads_per_cta` | static compile facts from `fused_moe_tuned_config_vllm_0_24_0_h200.json` (WP0, no timing) | residency re-derived by `HWSpec` from SM thread/warp/register/shared-memory capacities and must agree |
| launch tables (config ranges, tiles, GSM, warps, stages) | installed vLLM 0.24.0 source + bundled tuned JSON (WP0 reference) | nearest-key partition restated in the selector |
| FP8 operand widths, FP32 scale/router bytes, block 128 | vLLM source (`use_fp8_w8a8`, `per_token_group_quant` A scales, `[E, N/128, K/128]` B scales, `topk_weights` FP32) | static |
| FP8 Tensor rate | existing H200 yaml `tensor_core` variants (public 1978.9 vs 989.4 TFLOP/s dense) | ratio applied to express FP8 MMA work in primary-variant units |

Clock gate: every probe process sampled `nvidia-smi -i <gpu> --query-gpu=clocks.sm,utilization.gpu -lms 50` over its timed section and was accepted only if every sample read >= 1980 MHz. Tile pipeline run r2: 28/28 processes accepted, 0 rejected; wave refill run r2: 28/28 accepted, 0 rejected (all window minima 1980 MHz). Superseded runs kept on the remote for audit: `runs/tile_pipeline_r1` (first two programs only, older runner; 8/8 accepted, replaced by r2 with the extended runner), `runs/wave_refill_r1` (runner raised on 3-CTA/SM grids that are not a multiple of 8 panels; fixed, rerun as r2). No production kernel or serving duration was read by any runner or derivation; the trace durations below are evaluation only.

## G3/G4: family tables (rank 0, one step), before -> after

Commands (after, commit `be8bd9744`; independent batches run in parallel): `PYTHONHASHSEED=0 PYTHONPATH=. python scripts/evaluation/serving/compare_h200_decode_kernel_families.py --real-summary .../nsys_pure_decode_uniform_r2/decode_summary.json --batch {1,4,16,64} --prompt-tokens 7373 --method midlayer --phase decode --real-key C={n} --output experiment_logs/h200_wp11_moe_fp8/after/decode_c{n}.json` and `--phase prefill --batch 4 --real-key prefill --real-summary .../nsys_prefill_uniform/decode_summary.json`. Before = the baseline JSONs `decode_families_puredecode_midlayer_c{1,4,16,64}.json`, `prefill_families_midlayer_c4.json`. The compare script needs `scripts/evaluation/serving/probe_dsv3_decode_step_breakdown.py`, which is untracked in the main checkout; an untracked copy (scratch dir renamed to `/tmp/dsv3_decode_probe_wp11`) was used and is not committed.

Fused-MoE grouped GEMM family, us per step:

| case | real | before | after | before err | after err |
|---|---:|---:|---:|---:|---:|
| decode C=1 | 1447 | 3161 | 1742 | +118% | +20% |
| decode C=4 | 2817 | 4158 | 2845 | +48% | +1.0% |
| decode C=16 | 8077 | 9158 | 7836 | +13% | -3.0% |
| decode C=64 | 17229 | 21411 | 17960 | +24% | +4.2% |
| prefill (4 x 7373) | 188747 (window average of the 1-, 3- and 4-prompt steps); 276,740 for the 4-prompt step alone (29 x (4925.1 + 4617.8)) | 446096 | 258007 | +136% / +61% | +37% / -6.8% |

Algorithm split after (us/step): C=1 naive W1 1336 + naive W2 406; C=4 naive 1870 + 975; C=16 aligned 5198 + 2638; C=64 aligned 11947 + 6013; prefill aligned 139397 + 118610.

All families, real / before / after (us per step):

| family | C=1 | C=4 | C=16 | C=64 | prefill |
|---|---|---|---|---|---|
| GEMM | 2094 / 4277 / 4277 | 2995 / 4373 / 4373 | 2837 / 3518 / 3518 | 1971 / 3815 / 3815 | 67122 / 273815 / 273815 |
| fused MoE grouped GEMM | 1447 / 3161 / 1742 | 2817 / 4158 / 2845 | 8077 / 9158 / 7836 | 17229 / 21411 / 17960 | 188747 / 446096 / 258007 |
| MoE routing | 0 / 130 / 130 | 0 / 139 / 139 | 162 / 145 / 145 | 173 / 172 / 172 | 2614 / 28947 / 28947 |
| attention | 877 / 498 / 498 | 970 / 546 / 546 | 1734 / 1440 / 1440 | 7164 / 5006 / 5006 | 65210 / 856910 / 856910 |
| allreduce + norm | 580 / 846 / 846 | 666 / 792 / 792 | 745 / 894 / 894 | 1277 / 1268 / 1268 | 107466 / 170017 / 170017 |
| other | 742 / 43 / 43 | 737 / 45 / 45 | 768 / 48 / 48 | 1413 / 52 / 52 | 96339 / 8357 / 8357 |
| step (real wall) | 5570 / 8954 / 7536 | 7144 / 10054 / 8741 | 13635 / 15202 / 13881 | 29081 / 31724 / 28273 | 553411 / 1784143 / 1596054 |

## G4: per-geometry predicted vs real per call

Real = median over all rank-0 launches of that geometry in its window (count in the G1 table), from `experiment_logs/h200_wp11_moe_fp8/fused_moe_trace_calls.json` (remote `eval/fused_moe_trace_calls.json`). Prediction: `compare_h200_fused_moe_fp8_calls.py --before` builds the Serving-shaped FUSED_MOE spec (BF16 dtype, FP8 routed weights, EP1) with a seeded uniform top-8 routing of the window's token count (the deployment uses uniform routing), selects the launch through the public dispatcher, asserts the selected grid/block matches the real geometry, and calls `predict(..., "midlayer")`. Before = same spec with the projected vLLM 0.11 launch the baseline used (C=1 W1 reproduces the sim's 89.6 us).

| window | M | proj | grid | real median us (p10-p90) | before us / err | after us / err | after bottleneck |
|---|---:|---|---:|---|---|---|---|
| C=1 | 1 | W1 | 64 | 38.7 (38.1-39.5) | 89.6 / +131.6% | 46.1 / +19.1% | pipeline (local round) |
| C=1 | 1 | W2 | 448 | 11.1 (10.9-11.4) | 19.4 / +74.6% | 14.0 / +26.2% | pipeline (DRAM round) |
| C=4 | 4 | W1 | 256 | 63.0 (60.1-65.6) | 97.9 / +55.4% | 64.5 / +2.4% | pipeline |
| C=4 | 4 | W2 | 1792 | 34.2 (33.5-35.1) | 45.5 / +33.0% | 33.6 / -1.9% | pipeline |
| C=16 | 16 | W1 | 1024 | 183.2 (177.6-188.8) | 280.9 / +53.4% | 185.6 / +1.3% | pipeline |
| C=16 | 16 | W2 | 7168 | 95.4 (91.2-99.5) | 117.5 / +23.2% | 94.2 / -1.3% | pipeline |
| C=64 | 64 | W1 | 2176 | 390.8 (381.2-401.1) | 465.7 / +19.2% | 407.2 / +4.2% | pipeline |
| C=64 | 64 | W2 | 15232 | 202.2 (197.3-207.2) | 261.3 / +29.2% | 205.0 / +1.4% | pipeline |
| prefill | 7373 | W1 | 9392 | 1381.8 (1378.9-1385.7) | 1709.2 / +23.7% | 1293.4 / -6.4% | pipeline (later-wave refill) |
| prefill | 7373 | W2 | 65744 | 1322.9 (1318.3-1325.6) | 908.0 / -31.4% | 1112.1 / -15.9% | pipeline (later-wave refill) |
| prefill | 22119 | W1 | 24136 | 3748.0 (3735.3-3753.4) | 7810.5 / +108.4% | 3636.5 / -3.0% | l2 |
| prefill | 22119 | W2 | 168952 | 3538.2 (3533.5-3543.2) | 3094.9 / -12.5% | 3109.2 / -12.1% | pipeline (later-wave refill) |
| prefill | 29492 | W1 | 31512 | 4925.1 (4888.1-4930.3) | 11454.8 / +132.6% | 4817.3 / -2.2% | l2 |
| prefill | 29492 | W2 | 220584 | 4617.8 (4611.2-4625.4) | 4026.4 / -12.8% | 4105.2 / -11.1% | pipeline (later-wave refill) |

Development history relevant to the evaluation (no parameter was changed by it): the first end-to-end version (commit `914256e7e`, fixed span + K rounds only) predicted the prefill W2 geometries at -40..-45 % (L2-bound 2.77 ms vs 4.62 ms at M=29492) while every decode geometry was within +-5 % except C=1. The prefill W2 program has only 4 K blocks per CTA and 2 CTAs/SM, so each of its ~815 resident waves re-pays the CTA pipeline fill/drain that the single fixed span does not cover. That mechanism was then measured independently (multi-wave probe, L2-resident operands, K1024 held out) and added for FP8 rows only as a lower bound of later waves beside their DRAM round service (commit `be8bd9744`). Decode geometries are DRAM-round bound and unchanged by it; prefill W2 moved to -11..-16 %.

## G5: A800 unchanged + tests

- A800 spot check: `scripts/evaluation/kernel/run_kernel_evaluation.py --mode forward --hardware-id nvidia_a800 --dataset <subset> --jobs 4 --semantic-output <json>` on a subset of 8 canonical A800 fused_moe cases (`FUSEDMOE_TOK1_E256_TOPK8_H7168_INT2048`, `TOK64_E256_TOPK8_H7168_INT2048`, `TOK4096_E256_TOPK8_H7168_INT2048`, `TOK1_E8_TOPK2_H4096_INT14336`, `TOK16384_E8_TOPK2_H4096_INT14336`, `TOK1024_E32_TOPK4_H2880_INT2880`, `TOK1_E128_TOPK8_H2048_INT768`, `TOK32768_E128_TOPK8_H2048_INT768`; 47 physical kernels incl. W1/W2), run in a `git archive` export of the baseline `89f80345b` and in this branch at `be8bd9744`. Both semantic outputs have sha256 `edaa0d4ccd9b71b68b1cdf19583da82cc1b9babbefd4ab8ba5010cf252b562dc` (byte-identical; subset and outputs in `experiment_logs/h200_wp11_moe_fp8/`). Case totals (forward, ns) e.g. TOK1_E256 453113, TOK64_E256 11466065, TOK4096_E256 43551788, identical before/after.
- `python scripts/run_tests.py` (48 workers): `4101 passed in 73.68s` at `be8bd9744`.
- Out-of-domain H200 launches (BF16 routed weights, E8/E128, GLM-like E256/H6144, INT4 `routed_experts`) are pinned equal to the A800 selector by tests.

## Remaining errors and evidence

- C=1 W1 +19 %, W2 +26 % (46.1 vs 38.7 us, 14.0 vs 11.1 us). The single-wave prediction is `fixed_span (6.88 us) + K rounds`; for W2 (4 K blocks, 448 CTAs) the DRAM round service dominates the rounds and the fixed span is a large share. The fixed span comes from CUDA-event timing of a sub-wave probe launch and may include event/launch gaps that the nsys kernel duration excludes, and the template adds the fixed span serially although part of the pipeline fill overlaps the DRAM stream. No independent measurement separating those was made; reported, not tuned.
- Prefill W2 -11..-16 %: the per-wave refill is measured with contiguous probe output tiles; production W2 stores 64x128 BF16 tiles to scattered token rows (3.4 GB per call at M=29492) and multiplies router weights in the epilogue, which the probe's epilogue does not reproduce. Not measured; reported.
- Prefill W1 is L2-resource bound (generic H200 L2 read service 4454.6 B/clk) within -2..-6 %.
- Family prefill comparison: the baseline method compares the simulated 4-prompt step with the real window average over 1-, 3- and 4-prompt steps (188.7 ms); against the matching 4-prompt step (276.7 ms) the MoE family is -6.8 %.
- MoE routing (`moe_align`/`count_and_sort`, 130 us at C=1 in sim vs 0 real) is out of scope: vLLM 0.24 skips `moe_align_block_size` in the naive assignment (C <= 8), which the projected A800 recipe does not represent.

## Decisions needed from the maintainer

1. Out-of-domain H200 FUSED_MOE requests keep the projected vLLM 0.11 launch by restating the A800 selector body (generic W1/W2 rule, bundled-key guard, INT4 `routed_experts` Marlin rule) inside the H200 package, because `_OWN_PACKAGE_SELECTORS` routes the whole `select_fused_moe_launch` function and packages cannot import each other. A test pins the two equal. wp12 (GEMM) faces the same choice; a dispatcher-level "domain predicate" would avoid the duplication if preferred.
2. The Serving FP8 contract records both per-tensor and 128x128 block scaling as `weight_group_size = 0`. The selector therefore enters the FP8 block path from the enumerated (E=256, top-8, H=7168, N in {512, 256}) domain, which the model contract admits only for DeepSeek-V3 (block [128,128] required by its schema). A scale-granularity attribute would make this explicit.
3. `scripts/evaluation/serving/probe_dsv3_decode_step_breakdown.py`, imported by the tracked `compare_h200_decode_kernel_families.py`, is untracked in the main checkout; this branch used an untracked copy and does not commit it.
4. Shared files touched in self-contained blocks for merging with wp10/wp12: `select_launch_config._OWN_PACKAGE_SELECTORS` (one entry), `hardware_spec.py` (one scheduling key, one normalizer, one residency check), `model_contract.GpuGroupedGemmPipelineEnvelopeDemand.fp8_block_program`, `formula._resolve_gpu_grouped_gemm_pipeline_envelope` (FP8-row branch), `nvidia_h200.yaml` (one scheduling list).

## Commits

`521120abf` probe (first two programs), `14166864a` probe extended to all 7 programs, `914256e7e` launch + HWSpec + predictor + evaluation scripts, `1aea28103` wave refill probe, `a17d8049f` wave probe fix + derivation, `be8bd9744` later-wave refill in FP8 rows, plus the report commit. Not pushed.
