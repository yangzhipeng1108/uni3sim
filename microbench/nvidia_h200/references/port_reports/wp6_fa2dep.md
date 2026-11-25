# WP6 H200 report: FA2 dependency / combine / spill / Oaccum parameters

Package `wp6_fa2dep`, branch `h200-port-wp6_fa2dep`, baseline `8e7ae7db1`. All captures ran on the capture host GPU 2 (NVIDIA H200, 132 SMs, driver 575.57.08) from probe sources at commit `7b6b4c46aaeb8173a44677a8ee48c9105f901f52`. Remote run root: `<public-root>/h200_microbench/wp6_fa2dep/repo/profiler_output/microbench/nvidia_h200/<run_id>/` (raw CSVs stay there; local copies of the small derived files are under `profiler_output/microbench/nvidia_h200/<run_id>/`, gitignored).

Every probe is an owner copy of the A800 probe under `microbench/nvidia_h200/<same subpath>` and still models the FA2 2.8.3 **sm80 source program** (option A: identical model, H200 data). Note: the H200 vLLM 0.24 deployment actually runs FA3 for MLA attention; these parameters describe the FA2 program only.

All values are SM cycles in the clock64 same-CTA envelope domain (cycle counts are the published unit, as on A800). Where ns are needed, convert with the measured, clock-gated H200 SM clock of 1980 MHz (ns = cycles / 1.98); never 1.14 or 1.41 GHz.

## Gate table

| Gate | Target | Actual | Status |
|---|---|---|---|
| G1 coverage | Every row has an H200 value or a bounded blocked reason with the alternative tried | All 6 yaml keys (E, E2, F x2, G, H) have measured H200 values. Two validation-only sub-checks are blocked with a bounded reason and alternative: the Oaccum NCU cache-tier audit (ncu not installed; replaced by a clock64 DRAM/L2 separation gate, passed, ratio 3.27) and the local-spill pinned flash_attn 2.8.3 sm80 production cubin audit (A800 artifact not on H200; the probe's own sm_90 stack/site geometry is audited against the same contract and matches exactly). Combine split counts the FA2 heuristic can choose with 132 SMs but the unchanged probe cannot represent (D128 33..128, D64 17..128) are recorded as out of probe scope (see G). | Pass |
| G2 same mechanism | Owner copies differ only in hardware coupling, listed in header comments | `diff microbench/nvidia_a800/<p>.cu microbench/nvidia_h200/<p>.cu` shows only: sm_80->sm_90 / 108->132 device checks and %smid bounds, grids 108/216->132/264, device name A800->H200, Oaccum one-CTA/SM shared-memory guard 96->128 KiB (228 KiB/SM on H200). Every change is listed in the file header comments; runner/report headers list their changes (build/capture split, idle 4 MiB baseline, clock gate, blocked NCU audit, schema names). | Pass |
| G3 reproducible | Scripted runner + derivation per reference | Each probe has `*_runner.py build` (local CUDA 12.2 nvcc, same toolchain as A800) and `capture` (remote), a reporter (scalar/resident/combine/oaccum) or in-runner derivation (spill) that follows the A800 selection rule unchanged, and `microbench/nvidia_h200/operators/flash_attn/fa2_dependency_reference_publisher.py` that writes all six references from the derived reports. Remote driver used: `run_all.sh` (steps listed below). | Pass |
| G4 clock-gated | Clock logged, rejects recorded | 50 ms `nvidia-smi` logger (line-buffered) ran during every capture; a cell is rejected when any busy or live-context sample is below 1980 MHz and is re-run. Accepted / rejected-then-rerun cells: scalar 288/1, resident 126/0, combine A800 set 927/6, combine extension 1899/119, Oaccum 486/0, spill d64 2/0 captures, spill d128 2/0 captures. All retained busy samples 1980 MHz (min = max = 1980). Recorded in each reference `h200_provenance.clock_gate` and each capture manifest. | Pass |
| G5 schema | Same keys as A800 references, hardware_id nvidia_h200 | Key check (script in this report's verification section): every H200 reference contains all keys of its A800 counterpart (A800 reference JSON for the resident overlap; A800 report.json for scalar / combine / Oaccum; the A800 runner's report schema for the spill probe, whose A800 report is not on disk) with 0 missing keys, `hardware_id: nvidia_h200`, `production_or_evaluation_duration_consumed: false`; the only additions are `h200_provenance` and `h200_yaml_values`. The resident keys named `*_216_to_108` are kept byte-identical and hold the H200 264/132 ratios (stated in the reference). | Pass |

Goal status: all hard gates pass (Completed).

## Parameter table

| yaml key path | A800 value | H200 value | unit | reference path | derivation | status |
|---|---|---|---|---|---|---|
| `register_reduction_dependency_cycles` (E) | 4 geometries x 7 stages, e.g. (4,16): max 205.5, local_sum 139.53, scale_exp2_sum 649.42, barrier 53.36, quad_sum 80.54, predicate 174.25, bf16 10.74 | 4 geometries x 7 stages, e.g. (4,16): max 206.07, local_sum 77.06, scale_exp2_sum 684.77, barrier 44.67, quad_sum 82.06, predicate 172.83, bf16 11.17 (full table below) | cycles per 128-thread CTA stage | `microbench/nvidia_h200/references/fa2_scalar_dependency_sm90.json` | trials 0-1: median over 6 cells (2 trials x grids 1/32/132) of per-cell median clock64 cycles/repeat minus matched timer baseline; trial 2 frozen validation (max APE <= 0.50% for all non-conversion stages, <= 2.19% for bf16 conversion) | measured |
| `register_reduction_dependency_cycles.entries[4,16].two_resident_demand_factors` (E2) | max 0.6775, barrier 0.5271, quad_sum 0.5068, scale_exp2_sum 0.8255, predicate 0.7279, bf16 0.5896 | max 0.7215, barrier 0.5020, quad_sum 0.5122, scale_exp2_sum 0.7752, predicate 0.7700, bf16 0.6091 (resident_ctas_per_sm 2) | dimensionless per-CTA demand factor | `microbench/nvidia_h200/references/fa2_resident_dependency_overlap_sm90.json` | max over 3 trials of median CUDA-event span(264)/span(132), divided by 2; all six A800 publication gates pass (barrier/quad span ratio max 1.030, CTA-cycle ratio max 1.037, spreads <= 0.089) | measured |
| `fa2_local_spill_dependency_cycles.spill_round_cycles` (F, D64 chunked) | 2195.0 | 2194.0 | cycles per 128-thread CTA spill round | `microbench/nvidia_h200/references/fa2_local_spill_round_d64_chunked_sm90.json` | median over 9 trials of CTA-median clock64 cycles, local-spill minus register-control, rounds=1 (grid 132; grid 264 gives 2195, agreement within 1 tick); probe sm_90 static contract 56 B stack, 13 STL / 13 LDL (matches) | measured (production-cubin audit not re-run; see G1) |
| `fa2_dense_d128_causal_local_spill_dependency_cycles.spill_round_cycles` (F, D128 causal) | 8197.0 | 8197.0 | cycles per 128-thread CTA local-stack round | `microbench/nvidia_h200/references/fa2_local_spill_round_d128_causal_sm90.json` | same as above (grid 264 gives 8196); probe sm_90 static contract 176 B stack, 51 STL / 53 LDL (matches) | measured (production-cubin audit not re-run; see G1) |
| `fa2_combine_dependency_cycles` (G) | reduction widths 2..32, conversion, 14 split fragments (table in yaml) | reduction max/sum: w2 41.85/42.86, w4 63.85/64.85, w8 89.86/89.86, w16 118.87/118.87, w32 147.87/147.87; bf16 output conversion 29.73 (both geometries); 14 A800-table split fragments + 32 heuristic-extension split fragments (tables below) | cycles per 128-thread CTA stage | `microbench/nvidia_h200/references/fa2_combine_dependency_sm90.json` | trials 0-1: median over 6 cells (2 trials x grids 1/32/132) of baseline-corrected cell medians; trial 2 frozen validation (consumed stages 5-8 max APE <= 0.48%, reduction <= 0.41%); independent extension capture reproduces reduction entries within 0.02 cycles | measured (split counts beyond probe representability out of scope, see below) |
| `fa2_oaccum_load_use_cycles` (H) | l2_warm_cg first 495.63 / additional 269.56; dram_cg first 1020.24 / additional 711.44 | l2_warm_cg first 399.10 / additional 287.61; dram_cg first 1305.97 / additional 963.35 (model first_plus_additional_load_interval_v1 for both tiers, same as A800) | SM cycles per 128-thread CTA iteration | `microbench/nvidia_h200/references/fa2_oaccum_ldg128_load_use_sm90.json` | non-NCU clock64 path unchanged: (LDG128+4FMA minus 4FMA-only)/repeats; invariant model rejected (spread 46%/53%), segmented fit over trials 0-1 (54 cells/tier), trial 2 validation MAPE / max APE 3.27% / 17.73% (L2) and 6.47% / 17.29% (DRAM), within the A800 15% / 25% thresholds | measured (NCU audit blocked, alternative gate passed) |

### E: H200 values for `register_reduction_dependency_cycles.entries` (software_revision `cuda_12.2_sm90`)

| rows | score values | fp32_max_quad4 | fp32_local_sum | fp32_scale_exp2_sum | cta_barrier | fp32_quad4_sum | integer_column_predicate | fp32_to_bf16_conversion |
|---|---|---|---|---|---|---|---|---|
| 2 | 32 | 200.0625 | 137.0625 | 664.1171875 | 44.671875 | 72.0625 | 205.328125 | 11.2109375 |
| 2 | 64 | 328.0625 | 265.0625 | 1269.453125 | 44.671875 | 72.0625 | 398.4609375 | 20.3125 |
| 4 | 16 | 206.06640625 | 77.0625 | 684.765625 | 44.671875 | 82.0625 | 172.828125 | 11.171875 |
| 4 | 32 | 334.0625 | 141.0625 | 1274.8671875 | 44.671875 | 82.0625 | 333.4609375 | 20.3125 |

E2 for the other two measured geometries (not attached in the A800 yaml, available in the reference): (2,32) max 0.7159, barrier 0.5123, quad_sum 0.5000, scale_exp2_sum 0.7564, predicate 0.7961, bf16 0.6134; (4,32) max 0.7891, barrier 0.5082, quad_sum 0.5149, scale_exp2_sum 0.8136, predicate 0.8428, bf16 0.6256.

### G: H200 `source_fragment_entries` (cycles)

The A800 table (split choices of the A800 D64/D128 FA set), unchanged geometry:

| block_m/head_dim | num_splits | max_splits | slse_publish_transpose | post_max_exp_local_sum | post_sum_log | scale_exp_smem_barrier |
|---|---|---|---|---|---|---|
| 8/64 | 2 | 2 | 47.5156 | 54.3906 | 120.0469 | 166.7969 |
| 8/64 | 4 | 4 | 47.5234 | 54.3906 | 120.0469 | 97.875 |
| 8/64 | 8 | 8 | 48.2344 | 54.3906 | 120.0547 | 101.8984 |
| 4/128 | 2 | 2 | 47.5234 | 54.3906 | 120.0547 | 166.7891 |
| 4/128 | 3 | 4 | 47.4922 | 54.3828 | 120.0469 | 167.8516 |
| 4/128 | 4 | 4 | 47.5 | 54.3906 | 120.0547 | 167.8438 |
| 4/128 | 5 | 8 | 47.5312 | 54.3906 | 120.0547 | 171.8438 |
| 4/128 | 6 | 8 | 47.5156 | 54.3906 | 120.0469 | 171.8516 |
| 4/128 | 7 | 8 | 47.5234 | 54.3906 | 120.0469 | 171.8594 |
| 4/128 | 8 | 8 | 47.5312 | 54.3945 | 120.0469 | 99.8672 |
| 4/128 | 9 | 16 | 48.2812 | 54.3906 | 120.0469 | 175.875 |
| 4/128 | 11 | 16 | 48.2773 | 54.3906 | 120.0469 | 175.8828 |
| 4/128 | 16 | 16 | 48.2891 | 54.3984 | 120.0547 | 101.9102 |
| 4/128 | 22 | 32 | 47.5312 | 54.3906 | 120.0469 | 186.0625 |

Heuristic extension: FA2 `num_splits_heuristic` with 132 SMs (launch slots 264, max_splits 128) can choose every split count 1..128 (enumerated over batch*heads*mblocks < 0.8*264 and n_blocks 1..1199). Measured in a separate capture (`h200_fa2_combine_dependency_h200ext_20260925_r1`, same probe, `--fragment-set h200_heuristic_extension`): all other D64 splits 2..16 and D128 splits 2..32. Split 1 has no combine kernel.

| block_m/head_dim | num_splits | max_splits | slse_publish_transpose | post_max_exp_local_sum | post_sum_log | scale_exp_smem_barrier |
|---|---|---|---|---|---|---|
| 8/64 | 3 | 4 | 47.5312 | 54.3906 | 120.0547 | 169.8594 |
| 8/64 | 5 | 8 | 48.2422 | 54.3906 | 120.0547 | 175.8906 |
| 8/64 | 6 | 8 | 48.2422 | 54.3906 | 120.0547 | 175.8906 |
| 8/64 | 7 | 8 | 48.2344 | 54.3906 | 120.0469 | 175.8828 |
| 8/64 | 9 | 16 | 49.5078 | 54.3984 | 120.0547 | 186.0703 |
| 8/64 | 10 | 16 | 49.5039 | 54.3906 | 120.0508 | 186.0664 |
| 8/64 | 11 | 16 | 49.5078 | 54.3906 | 120.0547 | 186.0703 |
| 8/64 | 12 | 16 | 49.5 | 54.3906 | 120.0469 | 186.0625 |
| 8/64 | 13 | 16 | 49.5 | 54.3906 | 120.0469 | 186.0625 |
| 8/64 | 14 | 16 | 49.5078 | 54.3906 | 120.0547 | 186.0703 |
| 8/64 | 15 | 16 | 49.5 | 54.3945 | 120.0469 | 186.0703 |
| 8/64 | 16 | 16 | 49.5078 | 54.3906 | 120.0547 | 106.1797 |
| 4/128 | 10 | 16 | 48.2656 | 54.3906 | 120.0469 | 175.8828 |
| 4/128 | 12 | 16 | 48.2734 | 54.3984 | 120.0469 | 175.8906 |
| 4/128 | 13 | 16 | 48.2734 | 54.375 | 120.0469 | 175.8828 |
| 4/128 | 14 | 16 | 48.2734 | 54.3906 | 120.0469 | 175.8984 |
| 4/128 | 15 | 16 | 48.2812 | 54.3906 | 120.0469 | 175.9062 |
| 4/128 | 17 | 32 | 47.5391 | 54.3906 | 120.0547 | 186.0703 |
| 4/128 | 18 | 32 | 47.5391 | 54.3906 | 120.0547 | 186.0703 |
| 4/128 | 19 | 32 | 47.5312 | 54.3906 | 120.0469 | 186.0703 |
| 4/128 | 20 | 32 | 47.5312 | 54.3906 | 120.0469 | 186.0625 |
| 4/128 | 21 | 32 | 47.5312 | 54.3906 | 120.0469 | 186.0625 |
| 4/128 | 23 | 32 | 47.5312 | 54.3906 | 120.0469 | 186.0625 |
| 4/128 | 24 | 32 | 47.5391 | 54.3906 | 120.0547 | 186.0703 |
| 4/128 | 25 | 32 | 47.5234 | 54.3906 | 120.0547 | 186.0703 |
| 4/128 | 26 | 32 | 47.5391 | 54.3906 | 120.0547 | 186.0703 |
| 4/128 | 27 | 32 | 47.5312 | 54.3906 | 120.0469 | 186.0625 |
| 4/128 | 28 | 32 | 47.5312 | 54.3906 | 120.0469 | 186.0625 |
| 4/128 | 29 | 32 | 47.5312 | 54.3906 | 120.0469 | 186.0703 |
| 4/128 | 30 | 32 | 47.5312 | 54.3867 | 120.0469 | 186.0625 |
| 4/128 | 31 | 32 | 47.5312 | 54.3906 | 120.0469 | 186.0625 |
| 4/128 | 32 | 32 | 47.5312 | 54.3906 | 120.0391 | 106.1719 |

Out of probe scope (bounded): D128 splits 33..128 (kMaxSplits 64/128) and D64 splits 17..128 (kMaxSplits 32..128). The unchanged probe has `static_assert(MaxSplits <= 32)` and a mode-9 LSE buffer of grid*128 floats (D64 splits > 16 would read past it). Measuring them requires new kernel instantiations and a larger buffer, i.e. a change to the measured program beyond hardware coupling; this needs a main-agent decision. Alternative applied: the full representable range was measured, and the four consumed stages are flat in num_splits within each kMaxSplits class (slse/post_max/post_sum vary by at most 2.1 cycles across all 46 geometries), with scale_exp_smem_barrier rising by kMaxSplits class (D128: 166.8 / 167.8 / 171.9 / 175.9 / 186.1 for max_splits 2/4/8/16/32), except the full-class cells num_splits == max_splits (D64 4/8/16, D128 8/16/32) at 97.9..106.2.

## Hardware-coupling changes (G2 detail)

- `.cu`: device check sm_80/108 SMs -> sm_90/132 SMs; %smid bound 108 -> 132; resident grids 108/216 -> 132/264; spill device name "NVIDIA A800-SXM4-80GB" -> "NVIDIA H200" and SM count 132; Oaccum grids 1/32/108 -> 1/32/132 and one-CTA/SM guard 96 KiB -> 128 KiB (96 KiB would admit two CTAs on a 228 KiB SM). The resident 77824-byte pad still yields exactly two resident CTAs on H200 (checked at run time). L2 working sets unchanged (8 MiB warm set; 80 MiB eviction buffer > 60 MiB L2).
- Runners: device name, `-arch=sm_90 -cudart static` with the same CUDA 12.2 nvcc as A800 (binaries built locally, SASS/ptxas/stack audits with the local 12.2 cuobjdump), output root `nvidia_h200`, idle threshold memory.used <= 8 MiB (H200 idle reads 4 MiB), explicit `--baseline-commit` (remote copy is not a git checkout), clock gate. The resident A800 "nvidia-smi before == after clock" check is replaced by the clock gate, because the instantaneous reading outside a live context is the 345 MHz idle clock.
- Reports: grids, %smid bound, schema/hardware names; Oaccum CUDA-event plausibility converts with the gated 1980 MHz clock; NCU audit replaced by the blocked record plus the tier-separation gate; combine table chosen by the manifest `fragment_set`. Selection rules unchanged.

## Remote runs (GPU 2 only)

Driver: `repo/run_all.sh <commit>` (steps scalar, resident, spill64, spill128, oaccum, combine, combext), logs `<public-root>/h200_microbench/wp6_fa2dep/runs/run_scalar_r2.log` and `run_rest_r2.log`. Run ids: `h200_fa2_scalar_dependency_20260925_r1`, `h200_fa2_resident_dependency_20260925_r1`, `h200_local_spill_d64_chunked_20260925_r1`, `h200_local_spill_d128_causal_20260925_r1`, `h200_fa2_oaccum_ldg128_load_use_20260925_r1`, `h200_fa2_combine_dependency_a800set_20260925_r1`, `h200_fa2_combine_dependency_h200ext_20260925_r1`. A first launch (`run_all_r1.log`) failed before any cell because `nvidia-smi -f` block-buffers its log; the gate was switched to a line-buffered logger (commit `7b6b4c4`) and all captures were redone. No processes remain on GPU 2.

## Verification

- `python3 -m py_compile` on all new runners/reports/publisher; `git diff --check` clean; Oaccum reporter `--self-test` passes.
- G5 key check: for each H200 reference, `set(A800 keys) - set(H200 keys) == {}` and `set(H200 keys) - set(A800 keys) == {h200_provenance, h200_yaml_values}`.
- Not run: repository test suite (no `src/`/`configs/`/`tests/` change; microbench-only package).

## Notes for integration

- The yaml blocks' `source_sha256` / `report_sha256` / `production_*_sha256` fields must be replaced with the H200 values in each reference (`h200_provenance.files`, `report_sha256`); the spill production-binary hashes remain A800 provenance only.
- `software_revision` for these entries should read `cuda_12.2_sm90` (spill: FA2 2.8.3 source program, sm_90 probe build), as given in each reference's `h200_yaml_values`.
