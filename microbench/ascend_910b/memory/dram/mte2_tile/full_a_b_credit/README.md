# MatMulV2 full-A decode B1-credit probe

This independent Ascend 910B/CANN 8.5.1 mechanism probe reproduces only the GM→L1 ND2NZ descriptor order and B1 event-credit topology of the qwen3-4b decode MatMulV2 launches (persistent full-A, split_k=1), as structured by the open CANN `mat_mul_v3` AL1-full-load template `mat_mul_l1_full_load.h`. Each AIC loads one `16×K` bf16 A slab into a single resident A1 slot, then streams B as `(b_l1_k×64)×tile_N` slabs through two B1 slots. It compares a descriptor-only MTE2 drain (per-wave `PipeBarrier<PIPE_MTE2>`) against a credit-preserved trace with a two-slot pipeline and zero-work MTE2→MTE1 acknowledgements at the queue-defined credit-return points and no cross-wave barrier. It deliberately excludes MTE1 loads, Cube, FixPipe, workspace traffic, AIV work, production-sized loops, and all production/evaluation duration.

The frozen sweep scans 1, 2, 4, 8, 16 and 24 output tiles (waves) per AIC over the canonical launch geometries: qkv/gate_up/lm_head (K2560, slab 256×256, 10 slabs/wave, 24/19 AICs), o (K4096, slab 512×112, 8 slabs/wave, 23 AICs), and down (K9728, slab 256×112 `b_l1_k=4` capacity variant, 38 slabs/wave, 23 AICs — the production `b_l1_k=8` slot pair would exceed the 512-KiB L1 budget next to the 304-KiB A slab, so this cell is a recorded deviation, not an exact replica). A 4096-MiB cyclic source keeps every slab L2-cold: consecutive launches advance through the partition by `waves×n_slabs+32` tiles, any tile re-read within one launch is separated by at least a full per-AIC partition cycle, and every partition cycle streams ≥ 4 GiB in aggregate across AICs against a 192-MiB shared L2, so no timed tile can survive in L2 between reads. The probe may establish or reject the B1-credit/wave-ramp mechanism; it cannot be interpreted as a GEMM duration measurement.

Two measurement caveats are handled inside the kernel and host: (1) the AIC system counter does not tick during the first tens of microseconds after kernel entry, so the kernel first streams a fixed 32-slab dummy batch through MTE2 and discards it — this wake-up batch is offset `waves×n_slabs` tiles past the timed tiles so the timed section never re-reads what the wake-up just streamed (an earlier revision offset it by zero and read a spurious 154 GB/s per AIC from L2 residue); (2) `PROBE_DEBUG_DUMP=1` in the environment makes the host print per-launch wall time and per-block raw slots and downgrade validation failures to warnings, which is how the wake-up defect was caught (wall ≈ 98 µs against 8 µs of counted time).

Build inside the agreed `<ascend-container>` container after selecting an idle physical device:

```bash
ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 cmake -S . -B build -DSOC_VERSION=Ascend910B2
cmake --build build -j 1
```

Run only through the runner so hardware, baseline, device, process and source identity are recorded:

```bash
python aic_mte2_full_a_b_credit_runner.py --binary build/aic_mte2_full_a_b_credit --device-id 0 --owner kernel-accuracy --baseline <40-hex-commit> --run-id full_a_b_credit_r1 --cann-source /usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_l1_full_load.h
```
