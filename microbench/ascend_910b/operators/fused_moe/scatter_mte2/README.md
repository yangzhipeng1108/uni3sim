# Ascend 910B scatter MTE2 mini-GEMM probe

This mechanism probe launches the same mini-GEMM scan on all 24 AICs. Every AIC independently visits every expert weight at the same address sequence, so it measures a replicated cache-service stream; it does not model or claim the production fused-MoE worker-to-expert assignment.

`BM_VAL`, `BN_VAL`, and `BK_VAL` are required positive build parameters. The executable accepts the same shape as runtime arguments only for an auditable CSV record and rejects any mismatch with the compiled shape. Each AIC writes a private `BM * BN` output region. `logical_bytes_per_aic_scan` counts one A tile and one B tile for every timed expert iteration. `replicated_logical_bytes_all_aic` multiplies that logical request count by 24; neither field claims physical DRAM traffic.

Configure and build one geometry with CANN 8.5.1, for example:

```bash
cmake -S . -B build -DBM_VAL=128 -DBN_VAL=128 -DBK_VAL=128
cmake --build build -j
```

Run with `experts N Mc K stride_kb logical_device_id`. `logical_device_id` is interpreted after any `ASCEND_RT_VISIBLE_DEVICES` remapping and is checked against the ACL-visible device count.

```bash
./build/mini_gemm_bench 128 128 128 128 2048 0
```

The stride must be large enough for one contiguous B matrix. The program emits exactly one CSV header and one data row after validating shape, allocation arithmetic, device visibility, and positive cycle counts.
