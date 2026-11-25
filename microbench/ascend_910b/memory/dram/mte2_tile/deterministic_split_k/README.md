# Deterministic Split-K MTE2 topology probe

This independent CANN 8.5.1 probe isolates the deterministic Split-K source's GM-to-A1/B1 MTE2 descriptor sequence together with the MTE1 consumer/event lifetime needed to free A1/B1 slots. It is not a GEMM and does not execute Cube, FixPipe, workspace, AIV reduction, output, or a production workload duration.

The fixed topology comes from `Mc2MatMulMultiCoreSplitKDivideL2cache` for the normalized BF16 rank-two `K=28672, N=8192` launch family: 24 AICs; two N slabs; 75 K chunks; three four-chunk workers and twenty-one three-chunk workers; a tail K chunk of 256; one A and thirty-two B CopyIn descriptors per chunk. The sweep covers every 16-row-aligned A tail from 16 through 128 while B retains the physical `N x K` transposed-view stride of 28672 elements. The source initializes `PRELOAD(4)` and introduces no AIV wait before its first two output slabs, both of which are preserved by the timed producer/consumer body. Each B descriptor is consumed through three `128x128` MTE1 pairs before its A1/B1 queue slot is freed; this establishes a runnable source-shaped queue lifetime without inserting an artificial per-descriptor MTE2 barrier.

The source binding is CANN 8.5.1's `opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h`, SHA256 `f88f88f9278f55cf2deeb34ea0c2dce0aac0d1f20525b7ab1bc71ad9f5201605`. Lines 1096--1110 bind the dynamic tiling payload to `MatmulImpl`; the CANN Matmul depth checks define A1/B1 queue multiplicity as `depth/(step*stepK)`, which gives one A1 slot and two B1 slots for this payload, while `dbL0A=dbL0B=2` gives the two L0 slots. Lines 1118--1145 initialize `PRELOAD(4)` for either traversal order; `iterateOrder=1` selects the outer-N/inner-M `mmNM` path, and lines 1237--1246 begin the AIV wait/synchronization only at `count > 1`. `SetSingleShape(mCoreUse, ...)` carries the semantic M tail into the ND2NZ source copy; the probe sweeps the hardware's 16-row transfer alignment.

Build in the agreed 910B container with the source snapshot copied from this worker commit:

```bash
ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 cmake -S . -B build -DSOC_VERSION=Ascend910B2
cmake --build build -j 1
```

Use a confirmed-idle NPU and capture raw CSV only under `profiler_output/`:

```bash
python aic_mte2_deterministic_split_k_runner.py --binary build/aic_mte2_deterministic_split_k --device-id 0 --owner kernel-accuracy-gemm --baseline de1feff54456efd907234a5c9100bd7e16540853 --run-id deterministic_split_k_mte2
```
