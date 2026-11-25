# A800 block-reduction microbenchmark

`block_reduce_latency.cu` isolates the collective sequence used by the pinned vLLM BF16 vec8 Add-RMSNorm source once per row: CUB `BlockReduce<float, 1024>` with runtime valid-item count `blockDim.x`, one thread-local variance division plus `rsqrtf`, and the following block-wide broadcast barrier. Its leading seven kernel arguments mirror the production ABI, while the tensor streams stay outside the timed region. It is compiled into two otherwise identical binaries: one brackets the region only with `clock64` for per-CTA dependency cycles, while the other brackets it only with `%globaltimer` for cross-CTA/grid span. No interval contains the other timer domain, global-memory access, or a production-operator call. Host-side validation consumes every block result after the second timestamp.

The same mechanism also applies to the exact vLLM 0.11.0 BF16 scalar-width `rms_norm_kernel` in the tail-free 1024-thread domain. [`../../references/vllm_rmsnorm_v0_11_0_sm80.json`](../../references/vllm_rmsnorm_v0_11_0_sm80.json) pins the upstream source, installed extension and BF16 conversion header. That source instantiates the same CUB float/1024 reduction, runtime valid count, division/epsilon/`rsqrtf`, shared publication and barrier. The installed sm_80 RMSNorm kernel uses 24 registers/thread and 168 bytes of static shared memory/CTA, while the installed BF16 vec8 Add-RMSNorm kernel uses 32 registers/thread and the same 168 bytes of static shared memory/CTA. The different register counts still admit two 1024-thread CTAs per SM, and the Add-RMSNorm resources admit eight 256-thread CTAs per SM, so the occupancy boundaries used by the reduction services are unchanged. This transfers only the isolated reduction/publication mechanism; no RMSNorm production duration is consumed.

`block_reduce_latency_runner.py` freezes the production launch domain: block widths `256,1024` and grids `1,108,216,432,864,1728`. Before inspecting a GPU, it requires CUDA 12.8.93 with CUB 2.7.0, verifies the live vLLM extension and the retained production Add-RMSNorm SASS/CFG artifact, builds native-only `sm_80` code, and emits `static_equivalence.json`. The proof checks the two CUB shuffle trees, reduction/publication barriers, normal division/epsilon/RSQ path, exactly two reads of the selected timer, and absence of global traffic or the other timer domain inside each measured interval. Any identity or proof mismatch fails before device launch.

After that gate, the runner requires an idle A800 and records three independently ordered trials with 200 raw samples per cell. Each cell is bracketed by `nvidia-smi` identity, process and clock checks. One discarded single-block mechanism run after the idle/ownership preflight conditions the normal A800 idle-clock transition; only subsequent cells whose before/after clocks are identical enter raw. The capture manifest binds the exact Git commit, source, runner, reporter, CUDA/CUB tools, live production extension, retained artifact, candidate binary/SASS/static proof, raw CSV, device identity, clocks, driver, and timer domain. Production/evaluation kernel duration is not an input.

`block_reduce_latency_report.py` predeclares an occupancy-segmented physical decomposition for each discrete block width. `cuobjdump --dump-resource-usage` must report 24 registers/thread and 168 B static shared memory/CTA for the isolated reduction probe. Together with the A800 SM limits (65,536 registers, 163,840 B shared memory, 2,048 threads, and 32 blocks), the probe resources imply resident CTA limits $R_{256}=8$ and $R_{1024}=2$; the production BF16 vec8 Add-RMSNorm kernel's 32-register/thread and 168-byte footprint produces the same two limits. Trials 0--1 provide the single-block dependency latency $L_B$ from the `clock64` median at grid 1, resident service interval $S_{B,res}$ from the matching occupancy boundary, and queued service interval $S_{B,queue}$ from that boundary to the 16-CTA/SM endpoint. No timing data selects the breakpoint.

Every block records `%smid` only after its second timestamp; because CUDA blocks do not migrate, `max(end)-min(begin)` is comparable within one SM, and the report takes the median same-SM span before deriving service. Cycles are converted using the bracketed stable SM clock. The cross-SM globaltimer span is retained only as a diagnostic because its 1024 ns quantization is too coarse for this mechanism; it never derives a parameter. With $N=\max(1,\lceil G/108\rceil)$, trial 2 is frozen validation on same-SM multi-wave spans:

$$
T_{reduce}(G,B)=L_B+
\left(\min(N,R_B)-1\right)S_{B,res}+
\max(0,N-R_B)S_{B,queue}.
$$

This is a block-reduction mechanism contract, not a production NORM latency table. It may enter Forward only through a source-bound reduction phase and a resident-CTA worker schedule; it must not be added wholesale on top of the same reduction issue/SFU work.

Run the static gate after the benchmark source, runner and reporter have been committed. `--cuda-root` must be one assembled CUDA 12.8.93 prefix; PATH and CUDA 12.2 fallbacks are intentionally rejected:

```bash
python microbench/nvidia_a800/compute/reduction/block_reduce_latency_runner.py \
  --static-only \
  --run-id a800_block_reduce_static_<commit>_<date> \
  --cuda-root /path/to/cuda-12.8.93-prefix \
  --production-python \
    python \
  --production-artifact-dir \
    profiler_output/microbench/nvidia_a800/add_rmsnorm_sass_cfg_984de21_20260817
```

Only after that command passes and `nvidia-smi` proves the selected card idle:

```bash
python microbench/nvidia_a800/compute/reduction/block_reduce_latency_runner.py \
  --device-id 3 \
  --run-id a800_block_reduce_<commit>_<date> \
  --cuda-root /path/to/cuda-12.8.93-prefix \
  --production-python \
    python \
  --production-artifact-dir \
    profiler_output/microbench/nvidia_a800/add_rmsnorm_sass_cfg_984de21_20260817

python microbench/nvidia_a800/compute/reduction/block_reduce_latency_report.py \
  --input profiler_output/microbench/nvidia_a800/<run-id>/raw.csv \
  --manifest profiler_output/microbench/nvidia_a800/<run-id>/manifest.json \
  --output profiler_output/microbench/nvidia_a800/<run-id>/report.json
```

## Measurement

The following 2026-08-10 measurements predate the auditable runner. They are historical candidate evidence only: their raw samples, exact source/binary identity, live clocks, and postflight state were not retained, so they cannot populate HWSpec.

| Threads | 1 block | 1 block/SM | 2 blocks/SM | 4 blocks/SM | 8 blocks/SM |
|---:|---:|---:|---:|---:|---:|
| 256 | 567 cycles | 567 | 590 | 646 | 723 |
| 1024 | 630 cycles | 630 | 824 | 862 | 859 |

For one block, `smsp__inst_executed.sum` is 477 warp instructions at 256 threads and 1,430 at 1,024 threads. Those counters confirm that the existing static RMSNorm instruction contract already contains the reduction work; the elapsed collective interval must not be added wholesale or it would double count issue work.

The full collective interval contains instruction issue and SFU work already represented elsewhere in Forward. It therefore cannot be added as an operator correction. A passing report remains a candidate mechanism contract: publication additionally requires a source-bound resident-CTA schedule and a composition that does not double-count existing issue/SFU demand. FP16, generic Add-RMSNorm, the exceptional `FCHK` slow path, and blocks other than 256/1024 are outside this benchmark's validity domain. Standalone BF16 RMSNorm is supported only by the exact static identity and tail-free 1024-thread scope pinned above.
