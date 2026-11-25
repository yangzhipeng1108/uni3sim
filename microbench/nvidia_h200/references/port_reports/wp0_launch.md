# WP0 launch facts — H200 port report

Branch `h200-port-wp0_launch`, baseline `eb6e819d7`. Hardware `nvidia_h200` (the capture host GPUs 0 and 3). Stack: vLLM 0.24.0 (+cu129), torch 2.11.0+cu129, Triton 3.6.0, cuBLASLt 12.9.1.4 (`cublasLtGetVersion()` = 120901) from `<public-root>/tools/cuda-12.9`. No kernel duration was recorded, read or used anywhere in this package; nothing here is timed, so the clock-logging protocol does not apply (no timed capture, 0 accepted / 0 rejected).

## Gates

| Gate | Target | Actual | Status |
|---|---|---|---|
| G1 GEMM LUT | Candidate H200 LUT + manifest in the A800 schema covering the same shape domain; row count and schema check | `profiler_output/launch_data/nvidia_h200/gemm_launch_selection.{csv.zst,manifest.json}`: 140,765 rows (A800: 140,765), selection keys identical and in the same order, CSV header identical to A800 schema v3, manifest key set identical; all rows pass the A800 v3 row contract (`src.kernel.launch_config.gemm_launch_selection._parse_csv(..., "nvidia_a800")`). 140,752 physical operations queried, all rank-0 launches captured and replaced by the no-op with CUPTI proof. | Pass (candidate; canonical publishing and an `nvidia_h200` schema entry in `src` are main-agent work) |
| G2 MoE configs | H200 vLLM 0.24 config table for the listed shapes + compile metadata | `microbench/nvidia_h200/references/fused_moe_tuned_config_vllm_0_24_0_h200.json`: per-M (1..32768) selection ranges for TP4 (N=512) and TP8 (N=256), fp8_w8a8 block [128,128] and BF16, resolved by vLLM's `try_get_optimal_moe_config` on the H200; 28 distinct kernel configs, each compiled for W1 and W2 (56 binaries) with registers, spills, shared memory, num_warps, num_stages and resident CTAs/SM | Pass |
| G3 launch geometry | Every distinct kernel in the 4 windows with geometry and operator label; no duration column anywhere | `microbench/nvidia_h200/references/vllm_0_24_0_dsv3_l32_tp4_launch_geometry_sm90.json`: 103 distinct kernel names, 381 distinct geometry tuples, 2,473,696 kernel launches over 4 windows x 4 devices, every name labelled (26 operator labels). The extractor indexes only GrdX..BlkZ, Reg/Trd, StcSMem, DymSMem, Device, Name and refuses to write if an extracted section contains a timing token; `grep -i duration` on the reference hits only the two `false` boundary flags | Pass |
| G4 reproducible | Scripts regenerate each deliverable | Commands below | Pass |

## Deliverables and regeneration

| Deliverable | Script(s) | Command |
|---|---|---|
| GEMM domain (A800 CSV keys -> operations) | `microbench/nvidia_h200/operators/gemm/cublaslt_heuristic_runner.py domain` | `python cublaslt_heuristic_runner.py domain --a800-csv launch_data/nvidia_a800/gemm_launch_selection.csv.zst --a800-manifest launch_data/nvidia_a800/gemm_launch_selection.manifest.json --output profiler_output/microbench/nvidia_h200/wp0_gemm_launch/gemm_launch_domain.json` (local, the configured Python environment) |
| GEMM capture (remote) | `cublaslt_heuristic_runner.py capture` + `cublaslt_heuristic_query.cpp` | on the capture host in `<public-root>/h200_microbench/wp0_launch`: `python3 cublaslt_heuristic_runner.py capture --domain gemm_launch_domain.json --devices 0,3 --output-dir full_20260925` |
| GEMM candidate LUT | `microbench/nvidia_h200/operators/gemm/build_gemm_launch_selection_candidate.py` | `python build_gemm_launch_selection_candidate.py --domain .../gemm_launch_domain.json --facts .../rank0_launch_facts.jsonl.gz --summary .../candidate_summary.json` |
| MoE config table raw (remote) | `microbench/nvidia_h200/operators/fused_moe/fused_moe_tuned_config_vllm_0_24_0_runner.py` | `CUDA_VISIBLE_DEVICES=0 <vllm-0.24-python> fused_moe_tuned_config_vllm_0_24_0_runner.py --output moe_configs_20260925.json` (cache locations are caller-configured) |
| MoE reference | `microbench/nvidia_h200/operators/fused_moe/derive_fused_moe_tuned_config_reference.py` | `python derive_fused_moe_tuned_config_reference.py --raw profiler_output/microbench/nvidia_h200/wp0_gemm_launch/moe_configs_20260925.json --remote-raw the capture host:<public-root>/h200_microbench/wp0_launch/moe_configs_20260925.json` |
| Launch geometry reference (remote, stdlib python3) | `microbench/nvidia_h200/operators/launch_geometry/nsys_trace_launch_geometry.py` | `python3 nsys_trace_launch_geometry.py --trace-csv <nsys dir>/window_{1,2,3,4}_cuda_gpu_trace.csv --output geometry.json` |

Remote raw: `<public-root>/h200_microbench/wp0_launch/` (`full_20260925/` with the full ordered heuristic lists `raw_results_gpu{0,3}.jsonl.gz`, compact facts, candidate summary and the compiled query; `pilot_2000/`; `moe_configs_20260925.json`; `geometry.json`). Local copies (gitignored): `profiler_output/microbench/nvidia_h200/wp0_gemm_launch/`.

## G1 GEMM LUT details

| Item | A800 | H200 |
|---|---|---|
| Runner / query | `microbench/nvidia_a800/operators/gemm/cublaslt_heuristic_{runner.py,query.cpp}` | `microbench/nvidia_h200/operators/gemm/cublaslt_heuristic_{runner.py,query.cpp}` (sha256 `d8af35de...`, `f0466cfe...`, recorded in the candidate summary) |
| cuBLASLt | 120205 (CUDA 12.2) | 120901 (CUDA 12.9.1.4) |
| Selector contract | rank-0, 32 MiB workspace | unchanged |
| Rows / physical operations | 140,765 / 140,752 | 140,765 / 140,752 (domain = exact key set of the A800 CSV, i.e. ShareGPT workload domain + A800 kernel-dataset GEMM launches; `domain_sha256` identical) |
| Kernel family at rank 0 | ampere_* / CUTLASS / gemvx | 100 % Hopper nvjet (`nvjet_tst_*` 140,544 ops, `nvjet_hsh_*` 208 fp16 ops); no gemv kernel, the 5 `variant=gemv` lm_head m=1 keys launch a tiled nvjet GEMM (key kept, tile columns filled) |
| Stages enum | concrete `KxD` | always `64xAUTO`; pipeline_depth read from the launched nvjet symbol (`..._<K>x<STAGES>_...`), checked against TILE_ID, STAGES_ID K, CLUSTER_SHAPE_ID and the launch cluster attribute for every op (0 mismatches) |
| Split-K | per rank-0 | 958 ops split-K with `fp32_external` reduction (nvjet `splitK` main + `cublasLt::splitKreduce_kernel`), 139,794 direct |
| Grid form | one CTA per tile (+K) | 114,130 ops persistent grid (<= 132 CTAs, fewer than tiles x splits); others tiles x K-splits with a 2-D thread-block cluster (launch attribute id 4) |
| CTA raster columns | `cublaslt_m_group` / `cutlass_n_group` / `none` | `none`, log_group 0, serpentine false, k_slices = split_k for every row (no verified raster-word layout for nvjet; the cluster shape and v/h raster letter stay in the compact capture) |
| Pipeline depth distribution | — | 4: 110,606 rows; 5: 9,972; 3: 3,062; 6: 2,892; 16: 2,692; others < 2,000 |

Schema diff vs A800 manifest: identical key set and CSV header; value differences only `hardware_id` (`nvidia_h200`), `software_revision` (`cuda_12.9`), `hardware_revision` (contract projected from `configs/hardware/nvidia_h200.yaml`), and the content hashes / evidence sources / baseline / `generated_by`. `schema_version` 3 and the four raster launch fields are taken from the `nvidia_a800` schema template because `src.kernel.launch_config.gemm_launch_selection` currently maps `nvidia_h200` to the 6-field schema v2 and has no `cuda_12.9` software revision; the maintainer must add the H200 entry before `load_gemm_launch_selection_files(..., hardware_id="nvidia_h200")` can load it.

Porting changes in the query (all stated in its header): hardware_id `nvidia_h200`; no-op PTX `.target sm_90` / ISA 7.8; `cuLaunchKernelEx` attributes (cluster dimension) are recorded instead of rejected, the no-op is still launched attribute-free; tile/stages/cluster enum ids are emitted raw and named by the runner from the compiled `cublasLt.h` (the 12.9 enums outgrew the 12.2 hand tables); CLUSTER_SHAPE_ID added. Runner: device check `NVIDIA H200` / cc 9.0, idle requirement memory.used <= 16 MiB (idle H200 reports 4 MiB), devices 0 and 3 split round-robin (70,376 ops each), remote-runnable with stdlib python3.

## G2 MoE details

| Shape | Resolution | M ranges | Example (M=1) |
|---|---|---|---|
| TP4 fp8_w8a8 [128,128], E=256, N=512 | bundled `E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`, nearest tuned key | 15 | BM16 BN128 BK128 GSM1 w4 s3 |
| TP4 BF16, E=256, N=512 | `get_default_config` (no bundled BF16 E=256 H200 file) | 6 | BM16 BN64 BK128 GSM1 w4 s4 |
| TP8 fp8_w8a8 [128,128], E=256, N=256 | bundled `E=256,N=256,...` | 12 | BM16 BN128 BK128 GSM16 w4 s4 |
| TP8 BF16, E=256, N=256 | `get_default_config` | 6 | BM16 BN64 BK128 GSM1 w4 s4 |

Compile metadata (Triton 3.6, sm_90, cuOccupancy): fp8 BM16 configs 100 regs / 37,376 B (s3) or 55,808 B (s4) smem / 4 CTAs/SM; fp8 BM16 w8 s4 80 regs / 55,296 B / 3; fp8 BM32 160 regs / 41,472 B / 3; fp8 BM64 236-237 regs / 74,240-98,816 B / 2; BF16 BM128 BN128 w8 116 regs / 98,304 B / 2; no spills anywhere. Cross-check: the real TP4 decode trace's `fused_moe_kernel` launches show 128 threads, 97-100 regs and 0.037 MB dynamic smem, consistent with the fp8 BM16 s3 entries. The A800 reference keys (`schema_version`, `hardware_id`, `evidence_kind`, both flags, `deployment`, `source`, `bundled_bf16_config_domains`, `forward_scope`) are all present; `hardware_execution_performed` is true here because the kernels were compiled and launched (untimed) to read the binaries.

## G3 launch-geometry details

103 kernel names over the 4 windows on all 4 TP ranks. Operator labels (regex over the demangled symbol, `OPERATOR_RULES` in the extractor): DeepGEMM `sm90_fp8_gemm_1d2d_impl` (18) and `fp8_gemm_kernel_swapAB` (7), cuBLASLt nvjet BF16 GEMM (13) + splitKreduce (1), Triton `fused_moe_kernel` (1), MoE router GEMM (3), moe_align / count_and_sort (2), FA3 MLA `FlashAttnFwdSm90` (2) + `FlashAttnFwdCombine` (1) + prepare_varlen (2), FlashInfer trtllm allreduce fusion (1), NCCL (2), per_token_group_quant (2), TRT-LLM `scale_1x128` (1), MLA KV-cache write (1), act_and_mul (1), rmsnorm (4, Inductor-fused), rotary embedding and SiLU-mul inferred from Inductor fused-op names (3 + 2, labelled `*_inductor`), plus torch/Inductor elementwise, reduce, gather, sampler and softmax kernels. Shared memory is kept verbatim as the nsys CSV's MB string (0.001 MB resolution) because the CSV carries no exact bytes; exact bytes exist in the `.sqlite` next to the CSVs, which this package was not authorised to read.

## Config keys

No `configs/hardware/nvidia_h200.yaml` key is owned by WP0 (launch facts are versioned observations, not HWSpec parameters), so there is no yaml-key row to fill. One observation for the maintainer: the GEMM manifest's `launch_contract` projected from the current `configs/hardware/nvidia_h200.yaml` carries `l2_cache` = 52,428,800 B (50 MiB), while the Goal states the device reports 62,914,560 B.

## Blocked / partial

- CTA raster for nvjet kernels: not derivable from verified evidence (no sm_90 parameter-layout disassembly), encoded as `none`; partial.
- Canonical publication of the LUT and the `src` schema entry for `nvidia_h200` (schema v3 + `cuda_12.9`) are main-agent work.
- Exact shared-memory bytes for the production geometry need read access to the nsys `.sqlite` (or a re-export); not done.
