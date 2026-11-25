# A800 LSU mechanism probes

## Scalar-half DRAM source probe

`scalar_half_dram_stream.cu` measures the source-limited read service of four-warp SM80 CTAs at one, two, and three resident CTAs per SM. Each issue group contains 64 independent default-cache/eviction-first scalar-half load pairs; the first stream is shared and cache-resident while the second traverses disjoint DRAM storage. The probe contains no GEMV output, dot product, reduction, or production timing.

Run only on an idle A800. Select the device with `CUDA_VISIBLE_DEVICES`, compile for `sm_80`, retain the compiler resource report and SASS, and store native and NCU output below `profiler_output/`.

## SiLU BF16 load-pair probe

`silu_bf16_load_pair.cu` isolates the two scalar BF16 global loads issued by the vLLM SiLU-and-multiply kernel. The control, single-load, and adjacent paired-load variants use the same 1024-thread indexing loop and sweep 108, 216, and 432 CTAs across the seven production widths. One native process emits nine replications of every cell, and each reported trial averages eight separately evicted and separately timed launches to reduce CUDA-event quantization without converting the workload into a warm-cache loop. Its audited cyclic schedule rotates every cell through run position and balances the three variants across local order positions; three separate captures use rotations 0, 1, and 2 with mutually disjoint position sets for each cell.

The probe is mechanism-only. It may establish the two adjacent load instructions, the 32-byte-sector request count, and the physical two-CTA-per-SM occupancy bound. Its durations, ratios, or workload cells must not become Forward parameters or a per-shape lookup. In particular, the physical dual-residency boundary begins at 216 CTAs on the 108-SM A800; the probe does not justify excluding the 256-CTA production launch.

Run only on an idle A800 and keep captures under `profiler_output/`:

```bash
python microbench/nvidia_a800/memory/lsu/silu_bf16_load_pair_runner.py --output profiler_output/microbench/nvidia_a800/silu_bf16_load_pair_run0 --device 0 --run-id run0 --rotation 0 --owner OWNER
python microbench/nvidia_a800/memory/lsu/silu_bf16_load_pair_runner.py --output profiler_output/microbench/nvidia_a800/silu_bf16_load_pair_run1 --device 0 --run-id run1 --rotation 1 --owner OWNER
python microbench/nvidia_a800/memory/lsu/silu_bf16_load_pair_runner.py --output profiler_output/microbench/nvidia_a800/silu_bf16_load_pair_run2 --device 0 --run-id run2 --rotation 2 --owner OWNER
python microbench/nvidia_a800/memory/lsu/silu_bf16_load_pair_report.py --output profiler_output/microbench/nvidia_a800/silu_bf16_load_pair_audit.json profiler_output/microbench/nvidia_a800/silu_bf16_load_pair_run0 profiler_output/microbench/nvidia_a800/silu_bf16_load_pair_run1 profiler_output/microbench/nvidia_a800/silu_bf16_load_pair_run2
```

The runner fails before launch unless the selected device is a 108-SM compute-capability-8.0 A800 with zero utilization, no compute process, and at most 8 MiB of driver-reserved memory. Each capture freezes and hashes the probe source, runner, report, binary, resource dump, probe SASS, production vLLM binary, selected production SM80 cubin, exact production SiLU function SASS, and raw CSV. NVCC's host ELF build IDs need not be byte-identical across captures, so every probe binary SHA is retained separately while the report requires identical source, resource contract, device SASS, and production SASS. The report requires exactly three distinct run IDs and rotations, revalidates every hash and schema field, proves the cyclic order, requires nine samples for each cell, limits each run's interquartile spread to 20%, and limits the range across the three run medians to 20%. The interquartile within-run gate is deliberate because CUDA event results for the shortest control cells are quantized in approximately 1 µs steps; raw samples remain available for inspection.

The versioned reference is `microbench/nvidia_a800/references/silu_bf16_load_pair_sm80.json`. Its `identity` records the exact captured source, runner, report, and binary artifacts, while `published_tool_identity` separately hashes the currently published source, runner, and report so control-only runner hardening does not rewrite capture provenance. It intentionally contains no durations or fitted values. Raw timings and the duration-bearing audit remain ignored under `profiler_output/`.

## CTA-wide dependent round completion

`per_thread_round_completion.cu` measures the unit a source loop body executes: one CTA-wide round in which every thread issues the declared number of global loads, with the round's result feeding the next round's address. Loads inside a round are independent and coalesced; rounds are strictly serialized. It exists to adjudicate whether a per-round pipeline stage should be priced by the round's bytes or by its request count, which the two predictions separate cleanly: byte-proportional requires `t(16 B) = 8 * t(2 B)`, request-floored requires `t(16 B) ~= t(2 B)` while the round stays far below the SM's byte capacity.

The declared matrix is access width 2, 4, 8 and 16 bytes per thread; 1 or 2 loads per thread per round; 32, 128, 256, 512 and 1024 threads per CTA; grid 1; and three residency regimes. `l2` reads a 16 MiB window, larger than the per-SM L1 and smaller than the 40 MiB device L2; `l1` reads a 64 KiB window the CTA retains in its own L1; `mixed` gives one stream each, which is the shape of a second source pass that rereads its own row while pulling a shared row it has not touched. Every stream's whole window is walked untimed before the timed section, and the emitted `prime_rounds` and `timed_wraps_a` columns make that residency claim auditable rather than assumed.

Timing reads `%globaltimer` inside the CTA, so a dynamic SM clock cannot corrupt the reported nanoseconds. The probe loads a zero-filled buffer and reports only its own elapsed time: no production kernel, tensor, or duration is consumed, and the loaded values are never used except to keep the next address dependent.

The load encoding is part of the contract, not an accident. `rms_norm_kernel` declares its input and weight `const __restrict__` and the production `_C.abi3.so` issues `LDG.E.U16.CONSTANT` for them, which a plain typed read through a `__restrict__` pointer reproduces exactly. `fused_add_rms_norm_kernel<bf16, 8>` updates input and residual in place and issues plain `LDG.E.128` for those two streams; nvcc withholds the read-only path only when the kernel also stores through the pointer, so that encoding requires a round carrying its store and is **outside this probe's declared domain**. Results for a 16-byte round therefore describe the read-only path and must be labelled as such wherever they are used.

An earlier revision used inline `ld.global.ca` asm, which emits the scoped `LDG.E.STRONG.SM` encoding rather than the weak `LDG.E` the production source compiles to. Both revisions were run; they agree within trial spread, so the two-stream serialization reported below is not an artifact of that encoding.

Build and inspect the host-only configuration contract:

```bash
/usr/local/cuda-12.2/bin/nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo \
  -o per_thread_round_completion per_thread_round_completion.cu
./per_thread_round_completion --describe
```

Run only after confirming the assigned A800 is idle:

```bash
CUDA_VISIBLE_DEVICES=1 ./per_thread_round_completion
```

Stdout is one CSV row per trial:

```text
residency,streams,width_bytes,threads,rounds,prime_rounds,timed_wraps_a,trial,elapsed_ns,round_ns,bytes_per_round,round_bytes_per_s
```

The versioned reference is `microbench/nvidia_a800/references/per_thread_round_completion_sm80.json`. It records median round completion per declared cell together with the trial spread; raw timings stay under `profiler_output/`.

The probe is mechanism-only. It establishes how a round's completion time depends on request count, access width and residency class. It must not be turned into a per-shape lookup keyed by a production kernel's hidden size, and its cells must not be selected or weighted by any production error.

### Store axis, and why the capture needs an idle device

The round shapes the two NORM kernels actually execute carry a store: `rms_norm` writes one 2-byte result per element in its second pass, and `fused_add_rms_norm` writes a 16-byte vector in each of its two passes. A round measured without that store is a lower bound on the production round, and the shortfall is not a constant — comparing the measured per-round slope across hidden sizes gives 0.81 for `rms_norm` and 0.37 for `fused_add_rms_norm`, so no single bridging factor exists and choosing one would be fitting.

The probe therefore declares a `stores` axis. A store is only defined for the two-stream round and writes the *other* stream's word, which keeps the buffer zero-filled while denying `ptxas` the "stores back exactly what it read" proof. That also reproduces the production encoding for free: with a store present, the written stream's load drops from `LDG.E.*.CONSTANT` to plain `LDG.E`, so the `W=16, streams=2, stores=1` cell emits `LDG.E.128` + `LDG.E.128.CONSTANT` + `STG.E.128` — the shape `fused_add_rms_norm_kernel<bf16, 8>` emits, except that production's second main stream is also plain because it too is written in the other pass.

`per_thread_round_completion_runner.py` refuses to capture unless the selected device is an idle 108-SM compute-capability-8.0 A800 with no compute processes, the same precondition the SiLU load-pair runner enforces. This is not conservatism. The probe occupies one CTA on one SM, so its L1-resident cells are private to that SM and reproduce exactly while other tenants run, but every L2-resident cell shares the device L2. A capture taken on 2026-09-23, while eight other processes held all eight A800s at 100%, read **534 ns** for the same `l2 / 1 load / W=2 / 1024 threads` cell an idle device reads at **196 ns** — with the per-cell trial spread still at 1.02, and with physically impossible orderings such as an 8-byte round completing faster than a 4-byte one. Sustained interference does not make these cells noisy; it makes them steadily wrong, so no post-hoc statistic separates them reliably. `per_thread_round_completion_report.py` still withholds cells whose trial spread exceeds its gate and records a `width_monotonicity_flag`, but those are audit aids, not the guard.

Consequently the store cells are **not yet published**: every cell at the 1024-thread geometry the production NORM kernels launch with was captured under that interference. `microbench/nvidia_a800/references/per_thread_round_completion_sm80.json` remains the earlier verified-idle capture, which has no store axis. The contended capture is retained only as evidence of the failure mode, under `profiler_output/`.
