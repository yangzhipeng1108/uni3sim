# C500 typed register mechanisms

This independent microbenchmark measures six synthetic register streams on MACA 3.7.2/C500 (104 AP, wave64): FP32 ordered greater-than, equality and NaN tests with FP32 select, signed INT64 less-than with 64-bit select, unsigned INT64 greater-than with 64-bit select, and a b32 XOR-distance1 wave shuffle. A bundle includes the compare and selected result; it is neither a pure predicate instruction nor a complete `(value,index)` argmax comparator. No model tensors, argmax, routing, reduction, sorting, timed global memory or production durations are involved. Nothing here changes HWSpec or approves Forward binding.

Each stream alternates between two known states. A runtime full-width pivot and read/write volatile register fences prevent the compiler from specializing a 64-bit comparison to a sign-only test or folding the recurrence. Half the INT64 lanes use candidate values immediately below/above a runtime pivot with equal high words, forcing low-word ordering; the other half use differing high words. There are 1 or 4 independent chains and 31 statically unrolled bundles per outer iteration. Host reference checks every lane and every chain after each launch, including NaN and signed/unsigned two-word boundary checks on CPU. The CPU reference validates the reference model only; GPU correctness requires a successful smoke and matrix receipt.

## Offline reference

```bash
python microbench/metax_c500/compute/vector/typed_register_primitives_runner.py \
  --self-test --output /tmp/c500_typed_cpu_reference
```

This compiles with g++ and `MIDLAYER_TYPED_CPU_ONLY`, without including/linking MACA or touching a device. Output is a CPU build log, JSON reference result and receipt. Use a new output directory for every invocation. Remove temporary output after retaining a reproducible receipt under the task's registered output directory.

## Authorized hardware execution

Confirm prescribed container/device ownership and idleness first. Compile in the task-owned remote directory; record the exact source snapshot and compiler version before execution. The runner does not select a card or perform the executor's preflight.

```bash
/opt/maca/mxgpu_llvm/bin/mxcc -x maca --maca-path=/opt/maca \
  -offload-arch=native -O3 -std=c++17 typed_register_primitives.cpp \
  -o typed_register_primitives
python typed_register_primitives_runner.py --smoke \
  --binary ./typed_register_primitives --device-id 0 --output RAW/smoke
python typed_register_primitives_runner.py --matrix --trial 1 \
  --binary ./typed_register_primitives --device-id 0 --output RAW/trial1
python typed_register_primitives_runner.py --matrix --trial 2 \
  --binary ./typed_register_primitives --device-id 0 --output RAW/trial2
python typed_register_primitives_runner.py --matrix --trial 3 \
  --binary ./typed_register_primitives --device-id 0 --output RAW/trial3
```

Smoke executes 12 small correctness cells: 6 primitives × chains 1/4, 64 threads, one block, outer=1, one sample. A matrix trial executes 72 cells: 6 primitives × chains 1/4 × geometries `(threads,grid)=(64,1),(64,104),(256,104)` × outer 127/1023, with 3 samples each. Matrix trials repeat the same declared synthetic workload; the trial field distinguishes captures, without tuning rates. Binary-level direct CLI is `--device-id ID --sample primitive chains threads grid outer samples`.

Capture the compiled device program/resources before treating measurements as physical evidence. Confirm all 12 primitive/chain programs retain 31 × chains typed compare/select bundles or b32 shuffle operations in the loop, the outer loop remains, and the timed region has no local/shared/global memory operations or spills. Compare compiler lowerings separately: a C++ compare/select bundle may contain multiple instructions, and results cannot automatically be assigned to individual predicate, select, INT64 word, or production argmax services. If the program differs, preserve raw evidence and refuse unsupported binding rather than adjusting a numeric factor.

## Clock and output semantics

Every raw CSV row contains hardware identity, explicit device ID, AP/wave counts and clock metadata, sample, primitive, chains, thread/grid/outer geometry, wave/block identity, begin/end cycle stamps, bundle count, checksum and pass status. Each begin/end pair belongs to the same wave in the same kernel invocation. MACA 3.7.2 does not expose a supported physical AP-ID intrinsic: `mxshmem_device_utils.h` marks `__smid()` unsupported and the mctlass SmId assembly is disabled. Therefore no physical AP ID is invented and no absolute stamps from different waves/APs are subtracted or merged.

`wave_bundles = outer × 31 × chains`; one wave bundle is a compare+select operation or one b32 XOR-distance1 shuffle executed across 64 active lanes, distinguished by primitive and the receipt unit. The receipt reports local elapsed cycles per wave bundle, reciprocal wave bundles per local cycle, and 64× that reciprocal in lane bundles per local cycle, with distribution extrema/median. These intervals include the bounded loop/timer overhead and contention of their declared geometry; they are observations, not an inferred pure instruction latency, AP peak issue rate, chip-wide aggregate throughput, or approved Forward rate. In particular, summing wave rates cannot establish device peak throughput without an independently justified scheduling/timer model. All GPU results remain `Forward_binding_approved=false`.

Raw captures belong under the registered `profiler_output/` campaign, not `experiment_logs/`. Preserve compiler/device/source identities and the source binary with the raw lifecycle; publish no canonical data from this worker.
