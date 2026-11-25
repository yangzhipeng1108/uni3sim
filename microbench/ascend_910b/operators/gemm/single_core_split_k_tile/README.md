# MatMulV3 single-core Split-K resident tile probe

This standalone CANN 8.5.1 probe isolates the AIC-local program used by legacy MatMulV3 tiling key 65632 for one BF16 `128 x 128 x 384` base tile. The timed loop preserves `SetSingleShape`, local A1/B1 tensor binding, `Iterate`, the M-to-FixPipe handoff, one FP32 workspace-tile store, `End`, and the ping/pong L0C lifetime. A and B are populated in L1 before timing. The probe therefore measures the resident MTE1/Cube/source-control/FixPipe completion service independently of GM-to-L1 traffic, paired-AIV cast/store, the production outer loops, and any complete production GEMM duration.

The frozen campaign uses all 24 AICs, the source-complete K384 tile and K256 tail, and iteration counts `64`, `256`, `1024`, and `4096`. Run three independent processes, alternating forward and reverse cell order. The service is only publishable when each cell is positive on every AIC, the differenced steady service is positive, and the three-run median spread is at most 10%.

Configure and build inside the agreed 910B container:

```bash
ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 cmake -S . -B build -DSOC_VERSION=Ascend910B2
cmake --build build -j 1
```

Bind one idle physical device with `ASCEND_RT_VISIBLE_DEVICES`, then run one or more counts:

```bash
ASCEND_RT_VISIBLE_DEVICES=4 ./build/single_core_split_k_tile --batch 384 64 384 256 384 1024 384 4096 256 64 256 256 256 1024 256 4096
```

The executable emits one CSV row per count. `max_cycles` is the launch completion span across the 24 AICs; `worker_cycles` preserves every worker result for stability auditing.
