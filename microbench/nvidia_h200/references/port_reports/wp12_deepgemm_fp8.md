# WP12 report: H200 FP8 block-scaled dense GEMMs (DeepGEMM / TRT-LLM FP8 blockscale)

Worktree `<project-workspace>/h200_port/wp12_deepgemm_fp8`, branch `h200-port-wp12_deepgemm_fp8`, baseline `89f80345b`. GPU 1 on the capture host only. Every real duration below is evaluation only; no production or serving duration sets, selects or tunes any parameter.

## Gate table

| Gate | Target | Actual | Status |
|---|---|---|---|
| G1 | source-derived launch | All 52 distinct real in-scope geometry groups (4 pure-decode windows x 7 shapes, prefill window 3 token counts x 8 shapes) reproduced exactly by the new selector: kernel, tile, stages, cluster/multicast (incl. `kIsTMAMulticastOnA`), grid, block, D-swizzle, dynamic smem (nsys MB), registers (launch-bound budget). `scale_1x128_kernel` grid 8 x 132 / block 256 from source; its 28 registers are a compile fact, not derived. | Pass |
| G2 | zero-fit provenance | 7 new HWSpec parameters, all from the H200 MMA-free TMA-ring probe (3 clock-gated trials, 6/6 captures accepted, 0 rejected, every busy sample 1980 MHz); every launch value is a source fact; FP8 tensor rate is the existing public-spec HWSpec value. | Pass |
| G3 | serving path | In all five G3 runs every block-FP8 projection is priced by `GEMM/gemm/trtllm_fp8_blockscale_swapab_gemm` (C=1/4/16) or `GEMM/gemm/deepgemm_sm90_fp8_gemm_1d2d` (C=64, prefill) in `sim_by_kind_algorithm_us`; the only remaining `cublas_analytical_gemm` entries are the BF16 LM head, MLA batched GEMMs and router (out of scope). | Pass |
| G4 | evaluation reported | Per-geometry table (35 rows) and family tables before/after below; extraction script `scripts/evaluation/serving/extract_h200_fp8_gemm_real_calls.py`, comparison script `scripts/evaluation/serving/compare_h200_fp8_gemm_calls.py`. | Pass |
| G5 | A800 unchanged + tests | 12 A800 kernel-dataset GEMM cases predict bit-identically on baseline and HEAD; `python scripts/run_tests.py`: 4096 passed. | Pass |

## What was built

- Lowering (`src/serving/kernel_lowering/lower_workload.py`): for a DeepSeek-V3 step whose checkpoint is block-FP8 (the MLP linears carry the model-resolved `fp8` dtype; `model_config` admits DSV3 FP8 only as 128x128 e4m3 blocks with dynamic 1x128 activations), the dense projections `fused_qkv_a_proj, q_b_proj, kv_b_proj, o_proj, gate_up_proj, down_proj` get the launch-selection attribute `weight_quantization = fp8_e4m3_block128x128_act1x128_dynamic`. No prediction attribute changes. Note: the shared Kimi/GLM builder labels the MLA projections `bf16`; the real trace runs them as DeepGEMM FP8 (N=2112/6144/7168/8192 kernels), so the lowering reads the checkpoint format from the MLP linears instead of that label.
- Contract module `src/kernel/launch_config/block_fp8_gemm.py` (attribute value, algorithm names, `scale_1x128` launch facts), importable by lowering, launch packages and predictors.
- Dispatcher (`select_launch_config.py`): main-agent contract adopted (`owns_<function>` routing, same `_launch_function` as paper-plot `dd4aee05f`); `_OWN_PACKAGE_SELECTORS["nvidia_h200"] = {("gemm_launch", "select_gemm_launch")}`.
- H200 launch (`src/kernel/launch_config/nvidia_h200/gemm_launch.py`): `owns_select_gemm_launch` = declared block-FP8 linear; `select_gemm_launch` has no rule outside that domain (no A800 rule copied). It reproduces vLLM 0.24 `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`: `M < 32` -> FlashInfer/TRT-LLM `fp8_gemm_run` = `scale_1x128_kernel<<<num_sms*8,256>>>` + `fp8_gemm_kernel_swapAB` with `jit::get_best_gemm_config(N, M, K, 1, sms, false, true)`; `M >= 32` -> DeepGEMM `fp8_gemm_nt` -> `sm90_fp8_gemm_1d2d_impl` with `get_best_config<SM90ArchSpec>` of DeepGEMM `891d57b4` (the commit vLLM 0.24.0 pins; its `include/` is byte-identical to the installed `vllm/third_party/deep_gemm/include`). Eligibility `N % 64 == 0`, `K % 128 == 0` fails loud otherwise.
- Predictor (`src/kernel/predictors/gemm_block_fp8.py`, dispatched from `GEMMPredictor.midlayer/roofline`): source-structured template (persistent grid, TMA producer ring, per-k-block wgmma, TMA tile store). One-wave launches (every decode launch): `fill + kblocks x max(stage interval, FP8 k-block time)`, stage interval = p-norm of the per-SM ring stage service and the chip DRAM stream of the `ceil(N/tile_n)` distinct weight boxes per stage round. Multi-wave launches (prefill): launch shell + `max(waves x kblocks x FP8 k-block time at the public rate, compulsory reads / TMA DRAM stream + BF16 writes / tile-store rate)` + final tile-store drain. Swap-AB calls add `scale_1x128` priced at the measured launch span floor of grid 1056 (bounded by its bytes). Launch span floor composes by max.
- HWSpec section `scheduling.block_fp8_gemm_tma_ring_service` (`src/kernel/hardware_spec.py`, `configs/hardware/nvidia_h200.yaml`), reference `microbench/nvidia_h200/references/fp8_block_gemm_tma_ring_sm90.json`.
- Probes (`microbench/nvidia_h200/operators/gemm/`): `fp8_tma_operand_stream.cu` (MMA-free TMA ring; launch shell, fill, per-stage service vs streaming CTAs/box rows/stages; nsys-timed), `fp8_wgmma_kblock_issue.cu` (not published, see below), runner `fp8_gemm_mechanisms_runner.py` (clock-gated via `compute/clock_gated_capture.py`), derivation `fp8_gemm_mechanisms_reference.py`.

## G1 launch derivation

Real geometry: nsys `cuda_gpu_trace` of the pure-decode r2 windows (C=1/4/16/64, device 0) and the prefill window, extracted by `scripts/evaluation/serving/extract_h200_fp8_gemm_real_calls.py` (output `experiment_logs/h200_wp12_deepgemm_fp8/real_fp8_gemm_calls.json`; 1D2D M recovered from the `per_token_group_quant` launch that feeds each call: `M = GrdX * BlkX/16 / (K/128)`). The prefill window turns out to contain steps of 7373, 22119 and 29492 tokens. Registers: `min(255, 65536/threads)` rounded down to 8 (`__launch_bounds__(threads, 1)`). Tile columns in GEMM coordinates (tokens x output channels).

| window | M | N x K | real kernel (alg, tile M, tile N, stages, cluster) | selector | grid real/sel | block real/sel | regs real/launch-bound | smem MB real/sel | other | match |
|---|---:|---|---|---|---|---|---|---|---|---|
| C=1,4,16 | 1,4,16 | 1024x7168, 2112x7168, 7168x4096, 7168x4608, 9216x7168 | swapAB, 16, 128, 8, 1 | same | 132/132 | 384/384 | 168/168 | 0.153/0.153 | | yes (15 rows) |
| C=1,4,16 | 1,4,16 | 6144x1536 | swapAB, 16, 128, 8, 1 | same | 132/132 | 384/384 | 168/168 | 0.153/0.153 | | yes (3) |
| C=1,4,16 | 1,4,16 | 7168x512 | swapAB, 16, 128, 8, 1 | same | 132/132 | 384/384 | 168/168 | 0.153/0.153 | | yes (3) |
| C=64 | 64 | 1024x7168, 2112x7168 | 1d2d, 64, 16, 16, 1 | same | 132/132 | 256/256 | 248/248 | 0.170/0.170 | swzD 32, mcA 0 | yes (2) |
| C=64 | 64 | 6144x1536 | 1d2d, 64, 48, 15, 1 | same | 132/132 | 256/256 | 248/248 | 0.225/0.225 | swzD 32, mcA 0 | yes |
| C=64 | 64 | 7168x512, 7168x4096, 7168x4608 | 1d2d, 64, 64, 13, 1 | same | 132/132 | 256/256 | 248/248 | 0.225/0.225 | swzD 128, mcA 0 | yes (3) |
| C=64 | 64 | 9216x7168 | 1d2d, 64, 80, 11, 1 | same | 132/132 | 256/256 | 248/248 | 0.217/0.217 | swzD 32, mcA 0 | yes |
| prefill | 7373, 22119, 29492 | all 8 shapes (incl. kv_b 8192x512) | 1d2d, 256, 128, 3, 2 | same | 132/132 | 384/384 | 168/168 | 0.216-0.217 / same | swzD 128, mcA 1 | yes (24) |
| C=1,4,16 | - | `scale_1x128_kernel` (every swapAB call) | grid 1056, block 256, 28 regs, 0 smem | grid 8 x 132, block 256 (source) | | | 28 = compile fact | | | geometry yes; regs not derived |

The full 52-row table (one row per window/shape/M, appendix A) is generated from the extraction JSON and `select_block_fp8_gemm_config` (helper copy `experiment_logs/h200_wp12_deepgemm_fp8/g1_table.py`): all 52 match. The earlier mixed-window reference (`vllm_0_24_0_dsv3_l32_tp4_launch_geometry_sm90.json`) also contains 1D2D BLOCK_M=256/BLOCK_N=112 geometries whose M is not recorded there; the selector emits exactly that configuration for 1024x7168 at M in 6145-13312 (2048 values), so they are consistent with the same heuristic but were not checked call by call.

## G2 parameter provenance

| parameter (HWSpec `block_fp8_gemm_tma_ring_service`) | value | source |
|---|---:|---|
| `ring.stage_floor_ns` | 106.5 ns | TMA-ring probe surfaces: grid search of `(floor^p + (CTAs x box bytes / bw)^p)^(1/p)` on the steady per-stage slopes of 52 (geometry, CTA-count) lines; probe data only |
| `ring.dram_stream_bytes_per_ns` | 4175 B/ns | same fit |
| `ring.combination_power` | 3 | same fit (grid {1,2,3,4,6,8}) |
| `ring.fill_base_ns`, `fill_per_store_byte_ns` | 2719.3 ns, 0.0533 ns/B | least squares over the same lines' intercepts vs per-CTA tile-store bytes |
| `launch_shell_ns` | 1568 ns | median of 4 store-free kblocks=0 shells (spread 96 ns) |
| `full_chip_tile_store_bytes_per_ns` | 2079.5 B/ns | (132 x 64 KiB) / (shell with 64 KiB store - store-free shell), same geometry |
| FP8 tensor rate | 1892.8 FLOP/clk/subcore | existing HWSpec public H200 spec (not new) |
| launch span floors | existing | `small_kernel_span_floor` / `cuda_graph_small_node_span` (not new) |
| tile/stage/cluster/threads/smem, `scale_1x128` grid/threads, M<32 switch, N%64/K%128 eligibility | - | vLLM 0.24.0, FlashInfer 0.6.12 TRT-LLM `jit_utils.cuh` / `fp8_blockscale_gemm_kernel.cuh`, DeepGEMM `891d57b4` `heuristics/sm90.hpp`, `impls/sm90_fp8_gemm_1d2d.hpp` |

Probe capture: runner `microbench/nvidia_h200/operators/gemm/fp8_gemm_mechanisms_runner.py --device 1`, raw `<public-root>/h200_microbench/wp12_deepgemm_fp8/mech_run2/fp8_gemm_mechanisms_raw.json` (local copy `profiler_output/microbench/nvidia_h200/wp12_fp8_gemm_mechanisms_run2/`), 272 ring configurations x 120 launches (20 discarded) x 3 trials, nsys kernel durations; source/binary SHA256 in the reference. Fit quality on the probe itself: median |stage error| 3.4 %, max 20.5 % (swap-AB ring at 17 streaming CTAs: measured 141 ns, form 112 ns), max fill deviation 0.72 us.

Not published: the FP8 wgmma k-block probe (`fp8_wgmma_kblock_issue.cu`). Its isolated shared-memory-operand throughput reached 4.3-5.7 PFLOP/s with a fixed operand buffer and 2.5-3.1 PFLOP/s when the operands rotate over 4-6 stage buffers (smoke variants under `<public-root>/h200_microbench/wp12_deepgemm_fp8/smoke/dbg*.cu`, results = correct 128.0 accumulators), i.e. above the public dense FP8 rate (1.98 PFLOP/s) and dependent on operand reuse, so it does not isolate the production tensor mechanism. The template uses the public rate; this is recorded in the reference (`wgmma_probe_not_published`).

## G3 serving path

`scripts/evaluation/serving/compare_h200_decode_kernel_families.py` (now self-contained, cherry-picked `b332baefc`; adds `sim_by_kind_algorithm_us`) for `--phase decode --batch 1/4/16/64 --real-key C=<n>` and `--phase prefill --batch 4 --real-key prefill`. Rank-0 GEMM time by kind/variant/algorithm after the change:

| run | block-FP8 algorithm (us/step) | other GEMM entries (out of scope) |
|---|---|---|
| C=1 | `GEMM/gemm/trtllm_fp8_blockscale_swapab_gemm` 1453 | `batched_gemm` 1458, `moe_router` 469, `gemv` (LM head) 102 |
| C=4 | `trtllm_fp8_blockscale_swapab_gemm` 1453 | `batched_gemm` 1516, `moe_router` 479, LM head `cublas_analytical_gemm` 107 |
| C=16 | `trtllm_fp8_blockscale_swapab_gemm` 1453 | `batched_gemm` 528, `moe_router` 519, LM head 107 |
| C=64 | `GEMM/gemm/deepgemm_sm90_fp8_gemm_1d2d` 1270 | `moe_router` 681, `batched_gemm` 296, LM head 108 |
| prefill | `deepgemm_sm90_fp8_gemm_1d2d` 83959 | `moe_router` 6464, LM head 107 |

Outputs: `experiment_logs/h200_wp12_deepgemm_fp8/after/{decode_c1,decode_c4,decode_c16,decode_c64,prefill_c4}.json`.

## G4 per-geometry predicted vs real

Real = median per-call nsys duration over the window (GEMM kernel + the `scale_1x128_kernel` of the same call for swap-AB). Sim = the serving interval of that KernelSpec in the G3 step (`compare_h200_fp8_gemm_calls.py`, outputs `experiment_logs/h200_wp12_deepgemm_fp8/after/calls_*.json`). Prefill compares the simulated 29,498-token step with the real 29,492-token calls. `aux` = real kernel on the shared-expert side stream, concurrent with the routed fused-MoE.

| window | M | N x K | alg | tile, stages, cluster | predicted us (groups; bound) | real us GEMM + scale = total | calls, stream | error |
|---|---:|---|---|---|---|---|---|---:|
| C=1 | 1 | 1024x7168 | swapAB | 16x128, 8, 1 | 10.17 (scale 1.22, fill 2.94, stages 6.01; ring) | 10.81 + 1.70 = 12.51 | 7888, aux | -18.7% |
| C=1 | 1 | 2112x7168 | swapAB | 16x128, 8, 1 | 10.57 (scale 1.22, fill 2.94, stages 6.42; ring) | 10.72 + 1.60 = 12.32 | 8704, main | -14.2% |
| C=1 | 1 | 6144x1536 | swapAB | 16x128, 8, 1 | 6.54 (scale 1.22, fill 2.94, stages 2.39) | 5.22 + 1.54 = 6.75 | 8704, main | -3.1% |
| C=1 | 1 | 7168x512 | swapAB | 16x128, 8, 1 | 5.06 (scale 1.22, fill 2.94, stages 0.91) | 4.26 + 1.76 = 6.02 | 7888, aux | -15.8% |
| C=1 | 1 | 7168x4096 | swapAB | 16x128, 8, 1 | 11.44 (scale 1.22, fill 2.94, stages 7.29) | 10.37 + 1.63 = 12.00 | 8704, main | -4.6% |
| C=1 | 1 | 7168x4608 | swapAB | 16x128, 8, 1 | 12.35 (scale 1.22, fill 2.94, stages 8.20) | 10.88 + 1.50 = 12.38 | 816, main | -0.2% |
| C=1 | 1 | 9216x7168 | swapAB | 16x128, 8, 1 | 20.25 (scale 1.22, fill 2.94, stages 16.10) | 20.16 + 1.54 = 21.70 | 816, main | -6.6% |
| C=4 | 4 | 1024x7168 | swapAB | 16x128, 8, 1 | 10.17 | 10.94 + 2.30 = 13.25 | 6148, aux | -23.2% |
| C=4 | 4 | 2112x7168 | swapAB | 16x128, 8, 1 | 10.57 | 10.75 + 1.70 = 12.45 | 6784, main | -15.1% |
| C=4 | 4 | 6144x1536 | swapAB | 16x128, 8, 1 | 6.54 | 5.25 + 1.54 = 6.78 | 6784, main | -3.6% |
| C=4 | 4 | 7168x512 | swapAB | 16x128, 8, 1 | 5.06 | 29.86 + 2.18 = 32.03 | 6148, aux | -84.2% |
| C=4 | 4 | 7168x4096 | swapAB | 16x128, 8, 1 | 11.44 | 10.50 + 1.70 = 12.19 | 6784, main | -6.1% |
| C=4 | 4 | 7168x4608 | swapAB | 16x128, 8, 1 | 12.35 | 10.97 + 1.60 = 12.57 | 636, main | -1.8% |
| C=4 | 4 | 9216x7168 | swapAB | 16x128, 8, 1 | 20.25 | 20.86 + 1.63 = 22.50 | 636, main | -10.0% |
| C=16 | 16 | 1024x7168 | swapAB | 16x128, 8, 1 | 10.17 | 11.10 + 3.87 = 14.98 | 3248, aux | -32.1% |
| C=16 | 16 | 2112x7168 | swapAB | 16x128, 8, 1 | 10.57 | 10.82 + 1.79 = 12.61 | 3584, main | -16.2% |
| C=16 | 16 | 6144x1536 | swapAB | 16x128, 8, 1 | 6.54 | 5.41 + 1.66 = 7.07 | 3584, main | -7.5% |
| C=16 | 16 | 7168x512 | swapAB | 16x128, 8, 1 | 5.06 | 19.42 + 2.27 = 21.69 | 3248, aux | -76.7% |
| C=16 | 16 | 7168x4096 | swapAB | 16x128, 8, 1 | 11.44 | 10.49 + 1.76 = 12.25 | 3584, main | -6.6% |
| C=16 | 16 | 7168x4608 | swapAB | 16x128, 8, 1 | 12.35 | 11.07 + 1.63 = 12.70 | 336, main | -2.8% |
| C=16 | 16 | 9216x7168 | swapAB | 16x128, 8, 1 | 20.25 | 20.13 + 1.73 = 21.86 | 336, main | -7.3% |
| C=64 | 64 | 1024x7168 | 1d2d | 64x16, 16, 1 | 8.84 (fill 2.83, stages 6.01; ring) | 9.15 | 1537, aux | -3.4% |
| C=64 | 64 | 2112x7168 | 1d2d | 64x16, 16, 1 | 9.21 (fill 2.83, stages 6.38) | 9.18 | 1696, main | +0.3% |
| C=64 | 64 | 6144x1536 | 1d2d | 64x48, 15, 1 | 5.44 (fill 3.05, stages 2.39) | 5.82 | 1696, main | -6.7% |
| C=64 | 64 | 7168x512 | 1d2d | 64x64, 13, 1 | 4.07 (fill 3.16, stages 0.91) | 3.97 | 1537, aux | +2.5% |
| C=64 | 64 | 7168x4096 | 1d2d | 64x64, 13, 1 | 10.45 (fill 3.16, stages 7.29) | 11.14 | 1696, main | -6.2% |
| C=64 | 64 | 7168x4608 | 1d2d | 64x64, 13, 1 | 11.36 (fill 3.16, stages 8.20) | 12.13 | 159, main | -6.4% |
| C=64 | 64 | 9216x7168 | 1d2d | 64x80, 11, 1 | 19.47 (fill 3.27, stages 16.21) | 22.72 | 159, main | -14.3% |
| prefill | 29498 | 1024x7168 | 1d2d | 256x128, 3, 2 | 256.4 (tensor 250.7, dram 83.0, shell 1.5, drain 4.2; tensor) | 289.8 | 29, main | -11.5% |
| prefill | 29498 | 2112x7168 | 1d2d | 256x128, 3, 2 | 475.8 (tensor 470.0) | 558.0 | 32, main | -14.7% |
| prefill | 29498 | 6144x1536 | 1d2d | 256x128, 3, 2 | 294.5 (tensor 288.7, dram 187.7) | 375.8 | 32, main | -21.6% |
| prefill | 29498 | 7168x512 | 1d2d | 256x128, 3, 2 | 213.7 (dram 207.9, tensor 111.9; dram) | 208.8 | 29, main | +2.3% |
| prefill | 29498 | 7168x4096 | 1d2d | 256x128, 3, 2 | 901.1 (tensor 895.3) | 1040.1 | 32, main | -13.4% |
| prefill | 29498 | 7168x4608 | 1d2d | 256x128, 3, 2 | 1013.0 (tensor 1007.2) | 1156.4 | 3, main | -12.4% |
| prefill | 29498 | 8192x512 | 1d2d | 256x128, 3, 2 | 242.9 (dram 237.1; dram) | 223.5 | 32, main | +8.6% |
| prefill | 29498 | 9216x7168 | 1d2d | 256x128, 3, 2 | 2011.3 (tensor 2005.5) | 2281.7 | 3, main | -11.9% |

Before this package every one of these calls was `cublas_analytical_gemm` BF16 (e.g. C=1 o_proj 24.3 us vs 12.0 real; prefill 7168x4096 3121 us vs 1040 real).

### Family tables before/after (same compare script; before = baseline `89f80345b`)

Decode steps (rank 0, us per step; `gemm` group = all GEMM KernelSpecs vs real `dense_fp8_gemm + gemm_other`):

| C | real gemm | sim before | sim after | real step wall | sim step before | sim step after |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2094 | 4277 | 3483 | 5570 | 8954 | 8160 |
| 4 | 2995 | 4373 | 3555 | 7144 | 10054 | 9236 |
| 16 | 2837 | 3518 | 2607 | 13635 | 15202 | 14291 |
| 64 | 1971 | 3815 | 2355 | 29081 | 31724 | 30264 |

In-scope part only (real `dense_fp8_gemm` family = DeepGEMM/swapAB + `scale_1x128` kernels; sim = block-FP8 algorithms): C=1 1652 vs 1453, C=4 2520 vs 1453, C=16 2267 vs 1453, C=64 1329 vs 1270 (before: `GEMM/gemm` 2248 / 2378 / 2471 / 2837 BF16). The remaining sim `gemm` excess at C=1/4 is the out-of-scope MLA `batched_gemm` (1458/1516 us, A800 BF16 analytical tiles vs real nvjet) and router.

Prefill step (us): real `gemm` 67,122 / sim before 273,815 / sim after 90,530; step 553,411 / 1,784,143 / 1,600,858. The real prefill window mixes steps of 7373, 22119 and 29492 tokens (dense FP8 per step: 25,438 / 71,820 / 95,089 us; family value 64,158 = their mean), while the simulated step is the 29,498-token step; like-for-like the in-scope dense FP8 is 83,959 sim vs 95,089 real (-11.7 %).

## G5 A800 unchanged + tests

- A800 spot check (`experiment_logs/h200_wp12_deepgemm_fp8/a800_spot.py`: every 20th A800 kernel-dataset GEMM case through `evaluation.kernel.evaluation_runner.predict_forward_op_total`), run on a `git archive` of baseline `89f80345b` and on HEAD: 12/12 cases bit-identical (e.g. `GEMM_B1_M1_K13824_N5120` 99218.63027204019 ns both, `GEMM_B1_M8192_K14336_N4096` 4078713.332820169 ns both, `GEMM_B4_M4096_K28672_N8192` 31988502.425717074 ns both, `GEMM_B8_M4096_K8192_N8192`, `GEMM_B512_M1_K8192_N28672`, ...; `a800_before.json` / `a800_after2.json`). Unit test `test_weight_quantization_does_not_change_a800_launch` asserts the A800 launch is identical with and without the declaration.
- `python scripts/run_tests.py`: `4096 passed in 80.21s` (new tests: `tests/kernel/launch_config/test_nvidia_h200_block_fp8_gemm.py` 13, `tests/serving/test_workload_and_lowering.py::test_dsv3_fp8_dense_projections_lower_to_h200_block_fp8_gemms`).

## Remaining errors and their evidence (not tuned)

1. Shared-expert GEMMs (1024x7168, 7168x512) run on a side CUDA stream (stream 2486/2482/2478/2466 vs main 19) concurrently with the routed fused-MoE; their real medians inflate with batch (7168x512: 4.26 -> 29.86 -> 19.42 us at C=1/4/16, p10 4.1-7.5, p90 38-54 us) while the same geometry on the main stream at C=64 matches (+2.5 %). This is cross-stream contention, not the GEMM mechanism; the step timeline decides whether it is exposed.
2. 2112x7168 (17 streaming CTAs) -14..-16 %: the probe itself shows a knee the geometry-free ring form misses at 17 CTAs (measured 141 ns/stage, form 112 ns, -20 %; reference residuals). The other swap-AB shapes are -0.2..-10 %.
3. `scale_1x128_kernel`: predicted 1.22 us (measured span floor at grid 1056) vs 1.5-1.8 us real on the main stream; not modeled beyond the floor.
4. Prefill long-K shapes -12..-15 %, 6144x1536 -22 %: busiest-CTA tensor time at the public dense FP8 rate (real kernels reach ~1.7 PFLOP/s = 80 % of it); per-k-block promotion/`wgmma.wait` and per-tile epilogue costs are not modeled, and the isolated wgmma probe could not measure them (see G2). 6144x1536 sits where tensor (289 us) and write stream (188 us) overlap imperfectly.
5. C=64 9216x7168 -14 %: one math warp-group with BLOCK_N=80 promotion; the ring bound alone is reported.

## Remote run directories (the capture host, GPU 1)

- `<public-root>/h200_microbench/wp12_deepgemm_fp8/mech_run2/` published probe capture (nsys reports, clock logs, raw JSON); `mech_run1/` first capture (superseded coverage; same probes).
- `<public-root>/h200_microbench/wp12_deepgemm_fp8/eval/` real per-call extraction (evaluation only).
- `<public-root>/h200_microbench/wp12_deepgemm_fp8/smoke/` build smoke and wgmma operand-rotation checks.
- `<public-root>/h200_microbench/wp12_deepgemm_fp8/repo_mb/` probe/runner copy executed on the host.

## Notes for the maintainer

- Merge: `_OWN_PACKAGE_SELECTORS["nvidia_h200"]` gains `("gemm_launch", "select_gemm_launch")`; `_launch_function` is the paper-plot `dd4aee05f` version (reimplemented, not cherry-picked: that commit also touches the wp11 fused-MoE module). `compare_h200_decode_kernel_families.py` = `b332baefc` + `simulate_step` split + per-process scratch dir + `sim_by_kind_algorithm_us`.
- `configs/hardware/nvidia_h200.yaml` changed, so any policy that pins its SHA (e.g. `configs/runtime_memory_policy/inferencex/deepseek_v3_l32_nvidia_h200_aggregate_tp4_ep1_runtime_memory.json`) needs rebinding after the three packages merge; the compare scripts rebind at runtime.
- The DSV3 MLA projection `weight_dtype="bf16"` labels in `src/serving/workload/build_workload.py` (shared Kimi/GLM builder, outside this package) do not describe the DSV3 FP8 checkpoint; the lowering now declares block-FP8 for them from the checkpoint format. Fixing the label at the builder is a separate decision.

## Appendix A: full G1 table

| window | M | N x K | real kernel (alg, tile M(tokens), tile N, stages, cluster) | selector | grid real/sel | block real/sel | regs real/launch-bound | smem MB real/sel | other | match |
|---|---:|---|---|---|---|---|---|---|---|---|
| C=1 | 1 | 1024x7168 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=1 | 1 | 2112x7168 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=1 | 1 | 6144x1536 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=1 | 1 | 7168x512 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=1 | 1 | 7168x4096 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=1 | 1 | 7168x4608 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=1 | 1 | 9216x7168 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=4 | 4 | 1024x7168 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=4 | 4 | 2112x7168 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=4 | 4 | 6144x1536 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=4 | 4 | 7168x512 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=4 | 4 | 7168x4096 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=4 | 4 | 7168x4608 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=4 | 4 | 9216x7168 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=16 | 16 | 1024x7168 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=16 | 16 | 2112x7168 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=16 | 16 | 6144x1536 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=16 | 16 | 7168x512 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=16 | 16 | 7168x4096 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=16 | 16 | 7168x4608 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=16 | 16 | 9216x7168 | ('swapAB', 16, 128, 8, 1) | ('swapAB', 16, 128, 8, 1) | 132/132 | 384/384 | 168/168 | 0.153/0.153 |  | yes |
| C=64 | 64 | 1024x7168 | ('1d2d', 64, 16, 16, 1) | ('1d2d', 64, 16, 16, 1) | 132/132 | 256/256 | 248/248 | 0.170/0.170 | swzD real 32/sel 32; mcA real 0/sel 0 | yes |
| C=64 | 64 | 2112x7168 | ('1d2d', 64, 16, 16, 1) | ('1d2d', 64, 16, 16, 1) | 132/132 | 256/256 | 248/248 | 0.170/0.170 | swzD real 32/sel 32; mcA real 0/sel 0 | yes |
| C=64 | 64 | 6144x1536 | ('1d2d', 64, 48, 15, 1) | ('1d2d', 64, 48, 15, 1) | 132/132 | 256/256 | 248/248 | 0.225/0.225 | swzD real 32/sel 32; mcA real 0/sel 0 | yes |
| C=64 | 64 | 7168x512 | ('1d2d', 64, 64, 13, 1) | ('1d2d', 64, 64, 13, 1) | 132/132 | 256/256 | 248/248 | 0.225/0.225 | swzD real 128/sel 128; mcA real 0/sel 0 | yes |
| C=64 | 64 | 7168x4096 | ('1d2d', 64, 64, 13, 1) | ('1d2d', 64, 64, 13, 1) | 132/132 | 256/256 | 248/248 | 0.225/0.225 | swzD real 128/sel 128; mcA real 0/sel 0 | yes |
| C=64 | 64 | 7168x4608 | ('1d2d', 64, 64, 13, 1) | ('1d2d', 64, 64, 13, 1) | 132/132 | 256/256 | 248/248 | 0.225/0.225 | swzD real 128/sel 128; mcA real 0/sel 0 | yes |
| C=64 | 64 | 9216x7168 | ('1d2d', 64, 80, 11, 1) | ('1d2d', 64, 80, 11, 1) | 132/132 | 256/256 | 248/248 | 0.217/0.217 | swzD real 32/sel 32; mcA real 0/sel 0 | yes |
| prefill | 29492 | 1024x7168 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.217/0.217 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 7373 | 1024x7168 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.217/0.217 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 22119 | 1024x7168 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.217/0.217 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 29492 | 2112x7168 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.217/0.217 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 7373 | 2112x7168 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.217/0.217 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 22119 | 2112x7168 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.217/0.217 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 29492 | 6144x1536 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 7373 | 6144x1536 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 22119 | 6144x1536 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 29492 | 7168x512 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 7373 | 7168x512 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 22119 | 7168x512 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 29492 | 7168x4096 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 7373 | 7168x4096 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 22119 | 7168x4096 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 29492 | 7168x4608 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 7373 | 7168x4608 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 22119 | 7168x4608 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 29492 | 8192x512 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 7373 | 8192x512 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 22119 | 8192x512 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.216/0.216 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 29492 | 9216x7168 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.217/0.217 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 7373 | 9216x7168 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.217/0.217 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
| prefill | 22119 | 9216x7168 | ('1d2d', 256, 128, 3, 2) | ('1d2d', 256, 128, 3, 2) | 132/132 | 384/384 | 168/168 | 0.217/0.217 | swzD real 128/sel 128; mcA real 1/sel 1 | yes |
All 52 rows match.
