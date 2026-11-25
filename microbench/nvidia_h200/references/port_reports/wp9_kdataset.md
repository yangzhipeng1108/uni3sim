# WP9 — H200 kernel dataset candidate (evaluation ground truth)

> **2026-09-26: the WP9 candidates described below were deleted.** They were captured with 3–4 concurrent runner processes per GPU, and a waiting process was observed creating a CUDA context on the same GPU while another window was being measured, so the GPU was not idle during measurement. The data is being recaptured with exactly one process per GPU (vLLM 0.30) in the H200 kcampaign (`microbench/nvidia_h200/kcampaign/`).

Goal incomplete — checkpoint: four H200 candidates are captured, loaded and documented. G1, G2 and G5 pass. G3 is a declared subset (vector 288/296, flash_attn 330/334, with bounded reasons). G4 differs from the Goal's wording: the Goal describes the A800 device-Event protocol, but the A800 canonical ground truth being mirrored uses the NSYS NVTX protocol, which is what was captured here (evidence below). The maintainer has to rule on that difference and on the 12 excluded cases before this can be marked complete.

## Gate table

| Gate | Target | Actual | Status |
|---|---|---|---|
| G1 fused_moe | H200 candidate for the A800 fused_moe case list | 176/176 A800 case IDs, including all 16 E256/H7168 DeepSeek-R1 BF16 shapes (TOK 1..32768); FP8 w8a8 block: not applicable, the A800 runner has no FP8 path | Pass |
| G2 gemm | same | 221/221 | Pass |
| G3 vector + flash_attn | same, or a bounded blocked reason | vector 288/296, flash_attn 330/334; the 12 excluded cases and their reasons are listed below | Declared subset |
| G4 protocol parity | A800 protocol except hardware/stack coupling; clock logged; non-1980 MHz captures rejected | A800 canonical `nsys_nvtx_kernel_median` protocol reproduced. Every change is listed in the runner headers. Every window gates SM clock >=1980 MHz before the NVTX push and after the NVTX pop; 32 windows rejected; retained windows are all 1980/1980 MHz. A continuous `nvidia-smi -lms 50` log covered every campaign. | Pass under the NSYS reading; maintainer to confirm (see Protocol) |
| G5 schema | Loads with `load_kernel_dataset_document` | All 4 load. All 1015 entries parse under `parse_kernel_evaluation_timing` with the A800 kernel-timing contract. `nvidia_h200` has no entry in `evaluation/kernel/dataset.py::_KERNEL_EVALUATION_TIMING_SOURCE` (loader not changed) | Pass (integration note) |

## Candidates

Paths are local and gitignored, under `profiler_output/kernel_dataset/nvidia_h200/`. Each family has `<family>.json.zst`, `<family>.manifest.json` (per-case status, windows, spread, clock-rejected windows, stack versions) and `raw/<family>/cases/<case>/window_NN.launch.json` (compact per-window launch facts and telemetry).

| Family | A800 cases | H200 cases | Passing cases that needed retry rounds | Max retained 3-window spread | Clock-rejected windows | H200/A800 time p5 / p50 / p95 (information only) |
|---|---:|---:|---:|---:|---:|---|
| fused_moe | 176 | 176 | 6 | 4.56% | 12 | 0.124 / 0.295 / 0.453 |
| gemm | 221 | 221 | 27 | 4.99% | 9 | 0.259 / 0.337 / 0.466 |
| vector | 296 | 288 | 108 | 4.96% | 0 | 0.267 / 0.489 / 0.598 |
| flash_attn | 334 | 330 | 23 | 4.98% | 11 | 0.207 / 0.318 / 0.627 |

Excluded cases (bounded; no remaining path without clock locking or a looser gate):

- Power cap, 2 cases: `FLASHATTN_PAGED_PREFILL_B128_S2048_HQ32_HKV8_D128_C1` and `FLASHATTN_PAGED_PREFILL_B128_S4096_HQ32_HKV8_D128_C1`. All 6 windows (11 of 12 across the pair) ran below 1980 MHz. The runner's WARMUP=5 of this ~50–100 ms prefill trips the H200 SW power cap (event reason `0x4`), so SM clock is 1935–1965 MHz before and during the NVTX call. The container cannot lock clocks (`nvidia-smi -lgc` denied), so the >=1980 MHz rule cannot be met for these shapes.
- Single-call jitter above the 5% three-window limit after 4 rounds (up to 15 windows), 10 cases:
  - vector (8): `VECTOR_RMSNORM_T8_H2048_B1`, `VECTOR_RMSNORM_T1_H5120_B1`, `VECTOR_RMSNORM_T2_H16384_B1`, `VECTOR_ELEMENTWISE_T1_H8192_B1_{ADD,MUL}`, `VECTOR_KVCACHE_T4_KVH4D128_B1`, `VECTOR_KVCACHE_T4_KVH8D128_B1`, `VECTOR_KVCACHE_T8_KVH8D128_B1`.
  - flash_attn (2): `FLASHATTN_MLA_B16_S512_HQ64_HKV1_D512_C0`, `FLASHATTN_MLA_B1_S8192_HQ64_HKV1_D512_C0`.
  - Cause: these are 1.2–3 µs single kernels (FA3 MLA ~20 µs) with a bimodal ~200 ns (vector) / ~2–4 µs (MLA) run-to-run step. An A/B check (`runs/ab_query`, `runs/ab_noquery`) showed the step does not come from the pre-NVTX `nvidia-smi` gap.

Retry policy is the A800 practice (targeted re-capture of failing cases, "new three-window capture"). Each round adds 3 fresh process windows. A case keeps the newest three clock-accepted windows. At most 4 rounds (15 windows) are allowed. `manifest.cases[].window_count` records how many windows each case used.

## Protocol (G4) — why NSYS rather than device Events

The Goal says the A800 runners use "no-profiler device-event capture, steady-state call counts, L2-flush rings, power warm-up". Those are the `kernel_golden` Event modes (`scripts/kernel_dataset/nvidia_a800/kernel_golden_runner.py`). The published A800 ground truth does not use them. `kernel_dataset/nvidia_a800/*.json.zst` has `op_info.evaluation.timing.source = nsys_nvtx_kernel_median` for all 1027 entries, and `evaluation/kernel/dataset.py` fixes that source for `nvidia_a800` kernel granularity. The protocol is recorded in `experiment_logs/kernel_golden/nvidia_a800_device_golden_20260912/summary_manifest.json` (canonical checkout):

- `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --export=sqlite`;
- one exact public-op call inside a unique NVTX range;
- CUDA device kernel start→end;
- 3 independent process windows, keeping the median whole-trace window;
- 5% three-window limit;
- no explicit eviction.

The H200 candidates follow that protocol so they are comparable case for case, kernel for kernel, with the same schema. This is a worker decision that overrides the Goal's inference. The maintainer can overrule it; if so, the Event modes would need an H200 port of the L2-coupled contracts in `nvidia_a800_event_capture.py`, which is not done here.

Changes versus A800 (hardware/stack coupling only; each is stated in the runner or module header):

- Runner path and NVTX marker prefix: `scripts/kernel_dataset/nvidia_h200/<family>_runner.py`, `H200_KERNEL::<family>::<case>`. Runner arguments are the A800 job commands from `dataset_pipeline/kernel_dataset/nvidia_a800.py`, including the canonical GEMM dtype. FusedMoE passes the canonical `tok_per_expert` routing. `FA_SERVING_*` args are rebuilt from the canonical Attr.
- `H200CaptureSession` (`dataset_pipeline/kernel_dataset/nvidia_h200_capture.py`):
  - device identity check (NVIDIA H200, 132 SMs, 60 MiB L2);
  - BF16 4096³×1000 GEMM power warm-up (A800 Vector power-warm-up constants), then an idle wait until nvidia-smi reports 1980 MHz, placed before the runner's own warm-up;
  - SM-clock query before the NVTX push and after the NVTX pop; a window is rejected unless both are >=1980 MHz;
  - exclusive GPU lock taken before the first CUDA call, so staged processes never overlap on the GPU.
  - The A800 ran at locked 1140 MHz; the H200 cannot be locked.
- Optional pre-NVTX sweep minimum: 480 MiB (8×60 MiB). The canonical protocol leaves it at 0.
- Only the NVTX mode is ported. The Event/ring/graph modes are not.
- H200 has no Nsight Compute. Each kernel record carries NSYS/CUPTI launch facts under the A800 NCU field names:
  - `Metadata`: `Kernel Name`, `Grid Size`, `Block Size`, `CC=9.0`;
  - `Sections.Launch Statistics.Header`: `Registers Per Thread`, `Static/Dynamic Shared Memory Per Block`, `Shared Memory Configuration Size`, `Grid Size`, `Block Size`, `Threads`.
  - NCU counter sections are absent.
- FusedMoE `Attr.moe_tiling` is replaced by the H200 Triton config and grid (`try_get_optimal_moe_config` on H200). All other Attr fields are the A800 values.
- `FA_SERVING_*` cases are A800 op-granularity entries. On H200 they are kernel granularity, like the rest of the family.

## vLLM 0.11 (A800) → 0.24 (H200) op mapping

| Family / layout | A800 call | H200 call | Evidence |
|---|---|---|---|
| gemm | `torch.nn.Linear` (cuBLAS) | same; torch 2.11, cuBLAS 12.9.1.4 | — |
| fused_moe | `fused_experts(..., inplace=False, activation="silu")` | `fused_experts(..., activation=MoEActivation.SILU)` (0.24 removed `inplace`, takes an enum); same Triton `fused_moe_kernel` path | 0.24 signature |
| vector rope / silu / kvcache / elementwise | `forward_cuda` / `ops.reshape_and_cache_flash` / `a+b`, `a*b` | unchanged; same vLLM CUDA kernels | kernel names in candidate |
| vector rmsnorm / add_rmsnorm | `RMSNorm.forward_cuda` → `ops.rms_norm` / `ops.fused_add_rms_norm` | 0.24 `forward_cuda` returns `forward_native`, an 8–11 kernel eager ATen chain, so the runner calls `ops.rms_norm(empty_like(x), ...)` / `ops.fused_add_rms_norm(...)` directly (the 0.11 `forward_cuda` body). Construction runs under `set_current_vllm_config(VllmConfig())` | `vllm/model_executor/layers/layernorm.py:96-112` |
| flash_attn paged / paged_varlen / chunked_prefill | vLLM FA2 `flash_attn_varlen_func(block_table, num_splits=1)` | FA3 `flash_attn_varlen_func(block_table, scheduler_metadata=get_scheduler_metadata(...), fa_version=3, num_splits=0)`, i.e. the eager `FlashAttentionImpl` call | `v1/attention/backends/flash_attn.py:448,870-895` |
| flash_attn contiguous decode (forced SP) | Dao `flash_attn_with_kvcache(num_splits=SP)` | Dao package absent → FA3 varlen over dense K/V (`cu_seqlens_k`), `num_splits=SP` | — |
| flash_attn contiguous prefill | Dao `flash_attn_func` | FA3 varlen over dense K/V, `num_splits=0` | — |
| flash_attn mla | Ampere Triton `decode_attention_fwd(num_kv_splits=4)` | `FLASH_ATTN_MLA` decode, the first SM90 MLA backend: FA3 `flash_attn_varlen_func(q=q_pe, k=k_pe_cache, v=kv_c_cache, q_v=q_nope, causal=True, num_splits=0, scheduler_metadata)`, block 16 | `platforms/cuda.py:136-141`, `backends/mla/flashattn_mla.py:346-365` |

Caveat, to inform the maintainer's choice of evaluation target: a torch.compile H200 deployment may fuse vector ops into Inductor kernels. The candidate measures the vLLM CUDA custom kernels, the same kernel family A800 measured.

## Stack versions and clock

- Stack: torch 2.11.0+cu129, triton 3.6.0, vLLM 0.24.0+cu129, nvidia-cublas-cu12 12.9.1.4, flashinfer 0.6.12 (not used), NSYS 2025.6.1, driver 575.57.08, NVIDIA H200 GPU 1, `CUDA_VISIBLE_DEVICES=1`.
- Clock:
  - windows accepted / rejected: fused_moe 534/12, gemm 771/9, vector 1503/0, flash_attn 1114/11;
  - retained windows: 1980/1980 MHz only;
  - continuous 50 ms logs: remote `runs/main/<family>/clock_*.csv`.

## G5 load check output

`python scripts/kernel_dataset/nvidia_h200/build_dataset.py check --family <f>`:

```
fused_moe  cases=176 load_kernel_dataset_document=ok timing_schema_under_nvidia_a800_contract=ok nvidia_h200_in_evaluation_timing_contract=false
gemm       cases=221 ... ok ... ok ... false
vector     cases=288 ... ok ... ok ... false
flash_attn cases=330 ... ok ... ok ... false
```

For evaluation to consume the candidates, the maintainer needs to add `nvidia_h200: nsys_nvtx_kernel_median` to the evaluation and writer timing-source maps. The loader itself needs no change.

## Reproduce

Local steps:

1. Build plans: `python scripts/kernel_dataset/nvidia_h200/build_dataset.py plans --family <f> --output profiler_output/kernel_dataset/nvidia_h200/plans/<f>.json`.
2. Copy the code snapshot and plans to the remote.

Remote steps, from `<public-root>/h200_microbench/wp9_kdataset/`:

1. `launch.sh <family> main --windows 3 --concurrency N`. This wraps `scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py` with the vllm024 python, `pydeps` (zstandard, pandas), the shared GPU lock and GPU 1.
2. Retries: `--case-ids <file> --window-ids 4,5,6` and so on.

Back on the local side:

1. Copy back `*.launch.json`.
2. Assemble: `python scripts/kernel_dataset/nvidia_h200/build_dataset.py assemble --family <f> --results-root profiler_output/kernel_dataset/nvidia_h200/raw/<f> [--allow-subset]`.
3. Summarize: `python microbench/nvidia_h200/references/port_reports/summarize_wp9_candidates.py`.

Remote artefacts:

- `runs/main/<family>/cases/<case>/window_NN.{nsys-rep,launch.json,telemetry.json}`: 1.1 GB, kept on the remote.
- `logs/*.log`.
- `runs/main/vector_contaminated_forward_native/`: 16 early rmsnorm windows captured while the 0.24 `forward_native` path was live. They were moved out and re-captured.
- `runs/ab_*`: the pre-NVTX query A/B check.
