# WP10 report: H200 FA3 MLA attention through a source-derived Midlayer template

Status: **checkpoint (Goal not completed)**. G1, G2, G4 and G5 pass. G3 passes for the FA3 main kernels (prefill and decode) and the split combine, which carry 99.95% of the real attention-family time in scope; the per-call `prepare_varlen_num_blocks` setup kernel has a derived launch but no priced KernelSpec, because emitting one needs changes to the serving lowering and KernelSpec contract, which are outside this package's owner list (decision requested below).

Branch `h200-port-wp10_fa3_mla`, baseline `89f80345b`. Source: vllm-flash-attention `803020a8fa15407871341d41eba4919ade2ee1ee` (the `GIT_TAG` in vLLM v0.24.0 `cmake/external_projects/vllm_flash_attn.cmake`). The `hopper/` files are copied to `<public-root>/src/vllm-flash-attention-803020a8/` (with `PROVENANCE.txt`, `SHA256SUMS`, and the v0.24.0 cmake file beside it). github.com was unreachable from the capture environment at fetch time, so the files were fetched on the local host and copied over.

## Gate table

| Gate | Target | Actual | Status |
|---|---|---|---|
| G1 | source-derived launch | Every distinct FA3 geometry in the 4 pure-decode windows and the prefill window matches: kernel template (every template argument), tile, stages, grid, block, cluster and registers (168). Decode smem matches 0.232 MB. The combine grids (4,4,1)/(20,4,1)/(80,4,1)/(320,4,1) are derived from the varlen tile scheduler. The prepare launch (32/128 threads) is derived. Not derived: prefill smem (the source sum is 213.4 kB against the recorded 0.216 MB; 2.6 kB is unexplained) and the combine/prepare register counts (compiler facts, used as recorded) | Pass |
| G2 | zero-fit provenance | New HWSpec section `scheduling.fa3_mla_mechanisms`, built only from three independent H200 probes (WGMMA gemm service, online-softmax step, paged cp.async two-stage ring). All are clock-gated: 12 accepted captures, 0 rejected, every busy sample at 1980 MHz. The remaining inputs are static source or deployment facts, or H200 references that already existed. No FA duration is used | Pass |
| G3 | serving path | Step dumps (`sim_by_algorithm_us`) show `FLASH_ATTN/mla_decode/vllm_fa3_sm90_mla_decode`, `FLASH_ATTN/combine/vllm_fa3_sm90_combine` and `FLASH_ATTN/mla_prefill/vllm_fa3_sm90_mla_prefill`; no A800 algorithm is left for these kernels. `prepare_varlen_num_blocks` is not priced (0.02-0.04% of the real step): waived by the maintainer under the documented low-impact approximation boundary | Pass (waiver) |
| G4 | evaluation reported | Per-geometry and family tables before and after are below. Extraction: `scripts/evaluation/serving/extract_h200_attention_kernel_calls.py` and `compare_h200_fa3_geometries.py` | Pass |
| G5 | A800 unchanged + tests | 9 A800 FLASH_ATTN kernel-dataset cases (3 MLA, 2 decode, 2 prefill, 2 mixed) predict bit-identically before and after, re-checked after the graph-fact follow-up. `python scripts/run_tests.py` on the follow-up (merged onto paper-plot 31490dad4): 4132 passed | Pass |

## Kernel execution structure (source -> template)

- **Prefill** `run_mha_fwd_<90, bf16, 192, 128>`. `tile_size_fwd_sm90(192,128,causal)` gives 128 x 128 with MmaPV_is_RS and IntraWGOverlap. Two MMA warpgroups plus a one-warp TMA producer (384 threads), a 2-stage K/V ring, and the varlen dynamic persistent scheduler (grid = 132, LPT order, head sections sized for L2). With kHeadDim > 128 there is no inter-warpgroup scheduler barrier. Both warpgroups wait on the same K/V stage, so they run in lockstep. Per KV block each warpgroup issues QK (m64n128k16 SS x 12) and then PV of the previous block (m64n128k16 RS x 8). It waits for QK and runs one online-softmax step while PV drains. Template: `II = QK(2 WG) + max(PV(2 WG), softmax(2 WG))`. The first and last iterations are priced as `QK + softmax` and `PV`. Tiles are list-scheduled over 132 CTAs in source order, with an L2 K/V-stream lower bound per iteration and one DRAM latency for the first load.
- **Decode** `run_mha_fwd_<90, bf16, 64, 512, Split, PagedKVNonTMA, PackGQA>` (q_pe/k_pe 64, qv/v 512). The tile is 64 x 64 with no intra-WG overlap and LargeHeadDimV, so the second MMA warpgroup only runs PV. A 128-thread cp.async paged producer fills a 2-stage ring. vLLM passes `num_splits = 32` (`flash_attn_max_num_splits_for_cuda_graph`) under full CUDA graphs, and the prepare kernel's dynamic split rule sets the splits per request. Per KV block the softmax warpgroup runs QK+QV (m64n64k16 SS x 36) alone, then one softmax step, then both warpgroups run m64n256k16 PV x 4 concurrently. That consumer occupancy (3073 cycles) is the stage hold at which the paged ring was measured. The ring's steady initiation interval and first-block latency at the active-CTA count come from exact lookup.
- **Combine** `FlashAttnFwdCombine<(8,128), log2 splits, 256>`, varlen scheduler STANDARD or LINEARIZE. Priced by its dependent L2 rounds: split count, then LSE plus 3 prefetched stages, then one round per 3 further splits. Waves come from occupancy.
- **Prepare** `prepare_varlen_num_blocks_kernel<NumWarps>`, grid `ceil(b/992)`, block `32 * pow2(ceil(b/31))`. Launch derived; not priced.

## G1: launch derivation (real geometry from the r2 pure-decode windows and the prefill window, rank 0)

Template columns are (stages, tile M/N/K, hdimV, causal, varlen, PagedKVNonTMA, HasQv, MmaPV_is_RS, IntraWGOverlap, PackGQA, Split; scheduler kBlockM, kBlockN, MMA threads, producer threads, Split, PackGQA, LPT).

| Geometry | Real (nsys) | Selector (`src/kernel/launch_config/nvidia_h200/flash_attn_launch.py`) | Match |
|---|---|---|---|
| decode main, C=1/4/16/64 | `FlashAttnFwdSm90<2, 64x64x64, 512, 0,1,1,1,0,0,1,1>`, scheduler `<64,64,256,128,1,1,LPT 0>`; grid (132,1,1), block (384,1,1), cluster 1, 168 regs, 0.232 MB | `vllm_fa3_sm90_mla_decode`: tile q64/kv64/qk64/v512, depth 2, splits 32; grid (132,1,1), block (384,1,1), cluster (1,1,1). 168 = floor(65536/384) rounded down to 8 (`__launch_bounds__(384,1)`). Source smem sum ~231.7 kB | yes |
| combine, C=1 | `FlashAttnFwdCombine<(8,128), 5, 256, 1, 0, 1>`, grid (4,4,1), block 256, 46 regs, 0.018 MB | `vllm_fa3_sm90_combine` (4,4,1) (STANDARD), block 256; log2 max splits 5 because splits = 32; smem 17.4 kB | yes (regs recorded) |
| combine, C=4 / 16 / 64 | grid (20,4,1) / (80,4,1) / (320,4,1) | LINEARIZE: ceil(total_q*32/8)+b = 20 / 80 / 320, y = 512/128 = 4 | yes |
| decode prepare, C=1/4/16 | `<1,false>` (1,1,1) x 32 | `select_fa3_prepare_varlen_launch(b)` (1,1,1) x 32 | yes (not priced) |
| decode prepare, C=64 | `<4,false>` (1,1,1) x 128 | ceil(64/31)=3, switched to 4 warps = 128 | yes (not priced) |
| prefill main, 1/3/4 x 7373 tokens | `FlashAttnFwdSm90<2, 128x128x192, 128, 1,1,0,0,1,1,0,0>`, scheduler `<128,128,256,32,0,0,LPT 1>`; grid 132, block 384, 168 regs, 0.216 MB | `vllm_fa3_sm90_mla_prefill`: tile q128/kv128/qk192/v128, depth 2, splits 1 (heuristic: total_mblocks >= 0.8 x 132); grid 132, block 384, cluster 1, 168 regs | yes except smem (source 213.4 kB) |
| prefill prepare | `<1,false>` (1,1,1) x 32, once per layer | (1,1,1) x 32 | yes (not priced) |

Owned domains (`owns_select_target_mla_launch` and `owns_select_flash_attn_launch`, following the paper-plot dd4aee05f dispatcher contract, reimplemented here) cover three request classes:

- fresh causal MLA prefill with qk 192 / v 128 that the heuristic runs unsplit;
- absorbed MLA decode with latent 512 + rope 64, one query per request and at most 64 heads;
- that decode's split combine.

Every other request, including chunked-context prefill and non-MLA FLASH_ATTN, keeps the projected A800 package. The decode split bound needs vLLM's full-CUDA-graph setting, which the target-model workload does not declare. The selector therefore uses the audited deployment facts (`FULL_AND_PIECEWISE`, `max_cudagraph_capture_size` 256, from the real server log). An explicit `graph_runtime_mode` attribute overrides them (`NONE` switches to the FA3 heuristic).

## G2: parameter provenance

| Parameter (HWSpec `scheduling.fa3_mla_mechanisms` -> `microbench/nvidia_h200/references/fa3_mla_mechanisms_sm90.json`) | Value used | Source |
|---|---|---|
| WGMMA gemm cycles, `ss_n128_kmajor` k12, 2 WG (prefill QK) | 1535.8 cyc | `operators/flash_attn/fa3_wgmma_service.cu` (generated by `fa3_wgmma_service_gen.py`): wgmma.mma_async BF16->FP32, 128B-swizzle descriptors, fence/commit/wait_group 0 per gemm, clock64 |
| WGMMA `rs_n128_mnmajor` k8, 2 WG (prefill PV) | 1024.1 cyc | same probe |
| WGMMA `ss_n64_kmajor` k36, 1 WG (decode QK+QV) | 1328.1 cyc | same probe (`--plan decode_chain`) |
| WGMMA `rs_n256_mnmajor` k4, 2 WG (decode PV) | 1024.1 cyc | same probe |
| Online-softmax step 2x32 scores / 2x32 O, 2 WG (prefill) | 1782.6 cyc | `fa3_softmax_step.cu` (FA3 `softmax.h` max_get_scale + online_softmax + bf16 pack + rescale_o; mode 1 minus mode-0 baseline) |
| Online-softmax step 2x16 / 2x64, 1 WG (decode) | 720.6 cyc | same probe |
| Decode paged ring first-block latency and steady II at hold 3073, grids 1..132 | e.g. g=32: 5158 / 3403; g=128: 9946 / 4485 cyc | `fa3_paged_kv_stream.cu`: 128-thread cp.async.cg producer with the PagedKVManager pattern (page-table load + 4 K + 32 V 16-B copies per thread per 64-row block, 16-token pages), 2-stage mbarrier ring, consumers hold each stage for 3073 cycles, L2 flushed per trial. The hold is the derived decode chain (1328.1 + 720.6 + 1024.1 = 3072.8), checked by the predictor (fails closed if it drifts) |
| Measured SM clock (cycles <-> ns) | 1980 MHz | clock gate of the three captures |
| L2 read capacity, DRAM/L2 latency, CTA dispatch cost, span floor, occupancy limits | existing H200 references | `l2_read_bandwidth_sm90`, `pointer_chase_latency_sm90`, `cta_dispatch_cost_sm90`, `small_kernel_span_floor_sm90`, scheduling fields (reused, not re-measured) |
| Tiles, stages, warpgroups, splits heuristic, dynamic-split 1.1 margin (float32), 32 MiB L2 head sections, combine stages/threads, CUDA-graph split bound 32 | static | vllm-flash-attention 803020a8 `hopper/` and vLLM 0.24.0 `flashattn_mla.py` / `config/attention.py` |
| Full CUDA graphs + capture size 256 | deployment fact | real server log `.../dsv3l32_tp4_pure_decode_uniform_r2/server.log` (non-duration) |
| Combine 46 regs/thread | launch observation | recorded Reg/Trd (non-duration) |

Clock gate: busy nvidia-smi samples (50 ms) >= 1980 MHz. Captures `mech_r1` (7 accepted / 0 rejected, 917 busy samples), `mech_k36` (1/0) and `mech_ring3073` (4/0); every busy sample reads 1980 MHz. Runner: `operators/flash_attn/fa3_mechanism_runner.py` (plans `base`, `decode_chain`, `decode_ring`). Derivation: `fa3_mechanism_reference.py`. Remote raw data: `<public-root>/h200_microbench/wp10_fa3_mla/{mech_r1,mech_k36,mech_ring3073}`.

The prefill template was checked against its mechanisms. II = 1535.8 + max(1024.1, 1782.6) = 3318 cycles. The L2 K/V bound is 132 x 80 KiB / 4454.6 B per cycle = 2427 cycles, which does not bind. Pure tensor time would be 2560 cycles.

## G4: per-geometry predicted vs real (per call)

Real values are rank-0 per-call medians over each window, grouped by launch and by the step token count (grid X of the preceding `concat_and_cache_mla`). The prefill window turned out to hold three different steps (4, 3 and 1 fresh prompts), not four prompts in every step. Predicted values come from `midlayer` at the real request shape. The decode windows carry no per-call KV lengths, so decode is predicted at the first and last decode position (KV 7396..8396). Before = the A800 path.

| Geometry | Real median us (calls) | Before (A800 path, sim step) | After predicted us | Signed error after |
|---|---:|---:|---:|---:|
| decode main C=1 | 19.87 (8704) | Triton grouped MLA 7.53 | 9.15 .. 10.97 | -54% .. -45% |
| decode combine C=1 | 5.86 | Triton stage2 7.14 | 1.79 | -69% |
| decode main C=4 | 21.70 (6784) | 8.35 | 12.62 .. 14.30 | -42% .. -34% |
| decode combine C=4 | 7.01 | 7.84 | 1.66 .. 1.52 | -76% .. -78% |
| decode main C=16 | 46.69 (3584) | 33.22 | 38.36 .. 42.89 | -18% .. -8% |
| decode combine C=16 | 5.34 | 10.75 | 1.22 | -77% |
| decode main C=64 | 213.66 (1696) | 132.98 | 135.77 .. 153.89 | -36% .. -28% |
| decode combine C=64 | 8.00 | 22.40 | 1.22 | -85% |
| prefill main 1 x 7373 | 741.10 (32) | - | 708.86 | -4.4% |
| prefill main 3 x 7373 | 2213.10 (32) | - | 2113.24 | -4.5% |
| prefill main 4 x 7373 | 2944.17 (32) | FA2 26,759.65 | 2816.01 | -4.4% |

Output file: `experiment_logs/h200_dsv3_l32_alignment/wp10_fa3_mla/fa3_geometries.json`.

### Family tables (rank 0 us per step: real / before / after)

`compare_h200_decode_kernel_families.py` is the paper-plot b332baefc self-contained version, plus the `sim_by_algorithm_us` and `sim_flash_attn_calls` outputs. `before` = baseline tree 89f80345b, `after` = this branch. JSONs are in `experiment_logs/h200_dsv3_l32_alignment/wp10_fa3_mla/{before,after}_*.json`.

| Group | C=1 | C=4 | C=16 | C=64 | prefill (4 x 7373) |
|---|---|---|---|---|---|
| attention (+KV write) | 877 / 498 / 378 | 970 / 546 / 485 | 1734 / 1440 / 1299 | 7164 / 5006 / 4490 | 65210 / 856910 / 90713 |
| gemm | 2094 / 4277 / 4277 | 2995 / 4373 / 4373 | 2837 / 3518 / 3518 | 1971 / 3815 / 3815 | 67122 / 273815 / 273815 |
| moe_grouped_gemm | 1447 / 3161 / 3161 | 2817 / 4158 / 4158 | 8077 / 9158 / 9158 | 17229 / 21411 / 21411 | 188747 / 446096 / 446096 |
| moe_routing | 0 / 130 / 130 | 0 / 139 / 139 | 162 / 145 / 145 | 173 / 172 / 172 | 2614 / 28947 / 28947 |
| allreduce_norm | 580 / 846 / 846 | 666 / 792 / 792 | 745 / 894 / 894 | 1277 / 1268 / 1268 | 107466 / 170017 / 170017 |
| other | 742 / 43 / 43 | 737 / 45 / 45 | 768 / 48 / 48 | 1413 / 52 / 52 | 96339 / 8357 / 8357 |
| step (real wall / before / after) | 5570 / 8954 / 8835 | 7144 / 10054 / 9993 | 13635 / 15202 / 15062 | 29081 / 31724 / 31208 | 553411 / 1784143 / 1017946 |

The prefill attention row compares one simulated 4-prompt step (32 x 2816 us) with the real window average over 4-, 3- and 1-prompt steps. At equal geometry the error is -4.4% (table above), so the +39% family delta reflects the mixed step composition of the real window, not the template.

## Remaining errors and evidence (not tuned)

1. **Decode fixed per-call cost.** An evaluation-only isolated run of the real kernel (`fa3_decode_eval.py`, GPU 2, CUDA-graph replay, not clock-gated, durations never fed back) with one CTA per request (`num_splits=1`, batch 1) gives a per-block slope of 1.574 us = 3117 cycles over 128 -> 256 blocks. The template's consumer chain is 3073 cycles (-1.4%). With 64 CTAs (batch 64, `num_splits=1`) the real rate is 1.81 us per block against 1.80 us from the ring lookup at g=64. The per-block mechanisms therefore hold. What is missing is a fixed cost of about 12.5 us per main+combine call in isolation at C=1 (intercept of the 32-split slope), against about 6 us in the template. Candidates, none measured: Q/Qv PackGQA cp.async load and scheduler-metadata reads before the first block, the fp32 O_partial epilogue (64 KB per tile), and combine LSE/logsumexp work. No independent mechanism measurement is available yet, so the gap is reported rather than priced.
2. **Serving vs isolated decode at C=64.** In isolation the real main+combine takes 145.8 us (contiguous pages) and 152.8 us (pages randomly scattered over a 0.77 GiB layer pool; `fa3_decode_layout_eval.py`). The model gives 139 us. The serving trace shows 221.7 us. Page scatter explains about 5%. The serving telemetry shows 1980 MHz whenever the GPU is busy, so the SM clock does not explain the rest either. The remaining +45% serving-context slowdown is unexplained.
3. **Combine.** FA3 launches the combine with programmatic dependent launch. The main kernel triggers `launch_dependent_grids` before its last epilogue, and the combine waits in `griddepcontrol.wait`, so its nsys duration includes overlap with the main kernel's tail. The template prices only the combine's own L2 rounds (1.2-1.8 us) against real 5.3-8.0 us. That is -130 to -220 us per step, 2-4% of the decode step.
4. **Prefill** is -4.4% at every geometry. Unpriced: causal-diagonal masking iterations, the epilogue O store, and the LPT/L2-section details beyond list scheduling.

## Follow-up: deployment CUDA-graph facts (main-agent ruling 2)

The earlier selector fell back to hard-coded "audited deployment" constants when the graph facts were missing. The ruling rejected that as a default for a missing key, so it is replaced as follows.

- `build_workload` attaches the two config-level facts vLLM's FlashAttnMLA metadata builder reads to every target MLA workload node: `full_cuda_graphs` (`cudagraph_mode.has_full_cudagraphs()`, i.e. `FULL` in the configured uniform-decode or non-uniform dispatch) and `max_cudagraph_capture_size` (max capture size, 0 when graphs are disabled).
- `_target_mla_attention_specs` carries both facts on the MLA KernelSpec. `kernel_spec.py` makes them required for `mla_prefill` / `mla_decode`, validates their types, and requires a positive size when full graphs are on.
- The FA3 decode selector reads both facts and fails loud if either is absent. The `_AUDITED_DEPLOYMENT_*` constants are removed.
- Both H200 scripts (`compare_h200_decode_kernel_families.py`, `run_h200_dsv3_l32_tp4_closedloop_sim.py`) now pass `GRAPH_ARGS`:
  - `--graph-mode FULL_AND_PIECEWISE`;
  - `--graph-capture-sizes 1 2 4 8 16 24 ... 248 256`, the exact `cudagraph_capture_sizes` of the vLLM 0.24.0 server log (`nsys/dsv3l32_tp4_pure_decode_uniform_r2/server.log` and `logs/serve_dsv3_l32_uniform.log`);
  - `--graph-piecewise-eager-workload-kinds flash_attn kv_update`, because the log's `splitting_ops` contain `vllm::unified_mla_attention_with_output` and `vllm::unified_mla_kv_cache_update`.
- Limitation of the piecewise-eager setting: vLLM's MLA attention op also contains the decode `W_UK_T` / `W_UV` bmm and the prefill `kv_b_proj`, which are LINEAR workload nodes here, so a kind-level list cannot mark them eager.
- `build_workload` used to reject any enabled graph execution for target models (`kimi_k2`, `glm_moe_dsa`, `deepseek_v3`). It now rejects only steps the runtime would pad to a capture bucket; that is the part that is not source-closed. Target-model nodes still carry no per-step graph token domain, as before.

### Before/after (before = paper-plot 31490dad4 with all three templates and graph NONE; after = this follow-up with the graph config)

| Case | Step real wall / before / after us | Attention real / before / after | FA3 decode splits before / after |
|---|---|---|---|
| decode C=1 | 5570 / 6622 / 6622 | 877 / 378 / 378 | 32 / 32 |
| decode C=4 | 7144 / 7862 / 7862 | 970 / 485 / 485 | 32 / 32 |
| decode C=16 | 13635 / 12830 / **blocked** | 1734 / 1299 / - | 32 / - |
| decode C=64 | 29081 / 26297 / **blocked** | 7164 / 4490 / - | 32 / - |
| prefill 4 x 7373 | 553411 / 646572 / 646572 | 65210 / 90713 / 90713 | n/a |

Other families at C=1/C=4/prefill are unchanged: gemm 3483/3555/90530, moe_grouped_gemm 1742/2845/258007, allreduce_norm 846/792/170017. Steps that run (a bucket-sized decode, or a prefill above 256 tokens, which runs eagerly) predict identically, because no H200 host-launch profile is configured and target-model nodes carry no graph domain. The only thing the configuration feeds is the FA3 split bound, which already equalled the old constant.

**Blocked:** with the real graph config, the C=16 and C=64 simulations (and the closed-loop sweep) fail at `build_workload` with "Serving graph padding is not source-closed for target-model MLA/DSA workload emission". The staggered chunked-prefill admissions create decode or mixed steps whose token counts are not capture sizes (for example 5 -> 8). Supporting them means emitting padded target-model workloads: graph token domains on every target node, and padded MLA requests with zero KV length in the FA3 decode, prepare and combine. That is a serving feature outside this package's authorization. Decision requested below.

## Target-model CUDA-graph padding (main-agent ruling: option a)

The ruling authorized option (a). Target-model workloads now emit the vLLM 0.24 padded CUDA-graph step, and the guard that refused padded steps is gone. Sources: `gpu_model_runner.py` `_build_attention_metadata` / `execute_model` and `cudagraph_dispatcher.py` `_create_padded_batch_descriptor`.

Emission (`_apply_target_graph_token_domains` in `build_workload.py`) runs on steps whose runtime mode is FULL or PIECEWISE. Every node inside the model forward gets the generic `graph_logical_tokens`, `graph_model_input_tokens` and `graph_runtime_mode` fields, and its token-shaped fields are scaled to the padded rows (fail loud unless they are integral multiples of the logical tokens). Scaled fields by node:

- embedding gather: rows, elements, index_count;
- rmsnorm: rows;
- dense and router GEMM: m;
- rope: seq_len, elements;
- MLA KV-cache write (vLLM always uses the padded slot mapping when the KV update is separate): seq_len, elements;
- silu: tok, elements;
- add: elements;
- fused MoE: tokens (routing counts stay logical, as in the generic contract);
- tensor-parallel collectives: bytes.

Nodes that are not scaled:

- `lm_head` and its all-gather consume `hidden_states[logits_indices]` outside the captured model, so they stay logical and eager.
- The MLA op (`unified_mla_attention_with_output`) follows `pad_attn = (mode == FULL)`:
  - Under FULL, the decode absorption bmms run on the padded rows. The MLA decode keeps its logical requests plus the replay domain: the padded requests have query length 0 and seq_len 0 (`seq_lens[num_reqs:] = 0`, flat `query_start_loc`), so the FA3 prepare kernel gives them zero tiles.
  - Under PIECEWISE, the op runs eagerly on unpadded metadata. The absorption bmms, prefill `kv_b_proj` and attention stay logical with no domain. This is the op-boundary eager marking, done through the MLA-op positions (`mla_decode_q_down`, `mla_decode_v_up`, `kv_b_proj`) with no schema change.
- Node kinds with no padded-row contract (DSA indexer, the int4 routed-experts target variant, MTP) fail loud with "cannot represent CUDA-graph padding". The GLM-5.2 test covers this. The A800 target-MLA selector fails loud on a padded domain.

Lowering: the target specs reuse the generic contract (physical launch, logical prediction, and `model_input_shape`), and the batched-GEMM prediction strides follow the logical rows.

- The FA3 decode KernelSpec may carry the FULL replay domain. The kernel-spec schema validates it, so FULL decodes use the measured graph-node span floor.
- The FA3 combine grid is sized from the padded request count and padded total_q: 3 real requests padded to 4 give grid (20,4,1), where the logical batch would give (15,4,1). Padded CTAs return at their seqlen check and are priced only through the grid-size CTA dispatch cost.
- The prepare kernel's launch would also use the padded batch; it stays unpriced under the waiver.

Deployment fact: vLLM clamps padded requests to `max_num_seqs` (128 in the server log). The deployment's decode steps never select a capture size above 128, so the clamp never binds. This is recorded in code, not checked.

### Compare set with the real graph config (before = 31490dad4, graph NONE; after = this branch, FULL_AND_PIECEWISE)

| Case | Step real / before / after us | attention real / before / after | gemm before / after | MoE routing before / after | allreduce_norm before / after |
|---|---|---|---|---|---|
| decode C=1 | 5570 / 6622 / 6620 | 877 / 378 / 378 | 3483 / 3466 | 130 / 141 | 846 / 849 |
| decode C=4 | 7144 / 7862 / 7851 | 970 / 485 / 485 | 3555 / 3539 | 139 / 144 | 792 / 793 |
| decode C=16 | 13635 / 12830 / 12808 | 1734 / 1299 / 1296 | 2607 / 2591 | 145 / 147 | 894 / 891 |
| decode C=64 | 29081 / 26297 / 26293 | 7164 / 4490 / 4487 | 2355 / 2355 | 172 / 172 | 1268 / 1268 |
| prefill 4 x 7373 | 553411 / 646572 / 646572 | 65210 / 90713 / 90713 | 90530 / 90530 | 28947 / 28947 | 170017 / 170017 |

The small shifts come from FULL-replayed small kernels taking the graph-node span floor instead of the standalone floor. The FA3 decode keeps 32 splits: the config facts give the same bound the removed constant gave. The prefill above 256 tokens runs eagerly and is unchanged.

### Closed-loop sweep (8K/1K uniform, TPOT/TTFT p50; `compare_h200_dsv3_l32_real_vs_sim.py`)

`midlayer_try1` is the earlier A800-template baseline. `midlayer_all3_graph` has all three WP templates (FA3, FP8 MoE, block-FP8 GEMM) with the real graph config.

| C | real TPOT ms | try1 | all3+graph | real TTFT ms | try1 | all3+graph |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5.42 | 8.96 (+65%) | 6.62 (+22%) | 239 | 386 (+62%) | 162 (-32%) |
| 2 | 6.22 | 10.36 (+67%) | 6.97 (+12%) | 258 | 429 (+66%) | 172 (-33%) |
| 4 | 7.70 | 11.36 (+48%) | 8.42 (+9%) | 268 | 426 (+59%) | 172 (-36%) |
| 8 | 10.79 | 16.43 (+52%) | 10.72 (-1%) | 268 | 437 (+63%) | 174 (-35%) |
| 16 | 16.71 | 23.99 (+44%) | 15.91 (-5%) | 278 | 431 (+55%) | 174 (-38%) |
| 32 | 26.51 | 37.57 (+42%) | 24.63 (-7%) | 291 | 439 (+51%) | 177 (-39%) |
| 64 | 41.87 | 59.20 (+41%) | 39.49 (-6%) | 336 | 450 (+34%) | 184 (-45%) |

Throughput per GPU error goes from -31..-41% (try1) to -18..+6% (x axis). TTFT is now 32-45% too fast. This is reported, not investigated here. The single prefill step (4 fresh prompts) is still +17% in the compare set, so the TTFT sign flip points at the closed-loop prefill scheduling (chunked 8K prompts interleaved with decode), not at the prefill kernels.

Artifacts: `experiment_logs/h200_dsv3_l32_alignment/wp10_fa3_mla/graph_padding/`, `experiment_logs/h200_dsv3_l32_tp4_closedloop_sim/midlayer_all3_graph/` and `experiment_logs/h200_dsv3_l32_alignment/dsv3l32_tp4_8k1k_uniform__midlayer_try1__midlayer_all3_graph.{json,png}`, all gitignored in this worktree.

## Decode fixed-cost follow-up (main-agent ruling 3)

Not probed within this budget; still reported. Quantified gap:

- **One CTA per request (`num_splits=1`), isolated.** The fixed cost per call is about 7.3 us against 1.4 us in the template (a 5.9 us gap).
- **C=1 in serving (32 splits, about 4 blocks per tile).** The measured call takes 19.9 us against 9.2 us predicted.

Mechanism candidates for a future probe:

- **Kernel prologue** of a 384-thread, 232 KB-smem CTA: barrier init, `setmaxnreg`, TMA descriptor prefetch.
- **Scheduler metadata.** The dependent global reads of `num_splits_dynamic` / `prepare_seqlen_q` and the tile-semaphore atomic.
- **Q and Qv load.** The PackGQA cp.async load of Q (32 x 64) and Qv (32 x 512), which is not part of the ring's first-block latency.
- **Epilogue.** The split fp32 O_partial store (32 x 512 x 4 B per tile) plus LSE, and store completion before exit.
- **Combine overlap.** The combine's PDL wait overlaps the main kernel's tail.

No isolated production-kernel run sets any parameter.

## Decisions needed from the maintainer

1. prepare_varlen: waived (ruling 1).
2. Target-model graph padding: implemented (ruling on option a). The remaining target-model kinds without a padded contract (DSA indexer, int4 routed experts, MTP) fail loud.
3. Merge notes: this work touches `build_workload.py` (graph facts, `_apply_target_graph_token_domains`), `lower_workload.py` (target specs use the generic graph contract; MLA decode/combine domain), `kernel_spec.py` (required MLA graph facts, optional FULL decode domain), the H200 FA3 selector (graph facts, padded combine grid), one fail-loud guard in the A800 target-MLA selector, both H200 scripts (`GRAPH_ARGS`), and tests.

## Files

- `src/kernel/launch_config/nvidia_h200/flash_attn_launch.py` (new: selectors, owns predicates, split heuristic, prepare launch)
- `src/kernel/launch_config/select_launch_config.py` (owned selectors; owns-routed dispatcher)
- `src/kernel/predictors/flash_attn.py` (new wp10 FA3 block plus two dispatch hooks: `midlayer`, `_model_uncached`)
- `src/kernel/hardware_spec.py`, `configs/hardware/nvidia_h200.yaml` (`scheduling.fa3_mla_mechanisms`)
- `microbench/nvidia_h200/operators/flash_attn/fa3_*` (probes, generator, runner, derivation) and `microbench/nvidia_h200/references/fa3_mla_mechanisms_sm90.json`
- `scripts/evaluation/serving/extract_h200_attention_kernel_calls.py`, `compare_h200_fa3_geometries.py`, `compare_h200_decode_kernel_families.py`
- `tests/kernel/launch_config/test_nvidia_h200_fa3_mla_launch.py`
- Evaluation records (gitignored `experiment_logs/`): `experiment_logs/h200_dsv3_l32_alignment/wp10_fa3_mla/` (extracted real calls, geometry table, before/after family JSONs, A800 spot check and its script, isolated-run scripts and JSONs)
- Remote: `<public-root>/h200_microbench/wp10_fa3_mla/` (`mech_r1`, `mech_k36`, `mech_ring3073`, `eval/`), `<public-root>/src/vllm-flash-attention-803020a8/`

## Commands

- Tests: `python scripts/run_tests.py` -> 4098 passed (82 s).
- G3/G4: `PYTHONHASHSEED=0 PYTHONPATH=. python scripts/evaluation/serving/compare_h200_decode_kernel_families.py --real-summary .../nsys_pure_decode_uniform_r2/decode_summary.json --batch {1,4,16,64} --prompt-tokens 7373 --method midlayer --phase decode --real-key C={n} --output ...`, and `--phase prefill --batch 4 --real-key prefill --real-summary .../nsys_prefill_uniform/decode_summary.json`.
- Real per-call extraction (on the capture host): `python3 extract_h200_attention_kernel_calls.py --device 0 --output attention_calls_r2_prefill.json C=1:<r2>/window_1_cuda_gpu_trace.csv ... prefill:<prefill>/window_1_cuda_gpu_trace.csv`, then `PYTHONPATH=. python scripts/evaluation/serving/compare_h200_fa3_geometries.py --real-calls ... --output ...`.
- G5: `experiment_logs/.../wp10_fa3_mla/a800_spot.py` run with `PYTHONPATH` set to the baseline export and to this branch; 9/9 identical.
