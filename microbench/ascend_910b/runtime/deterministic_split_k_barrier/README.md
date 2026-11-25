# Deterministic Split-K collective barrier lifecycle

This independent Ascend 910B probe mirrors only the CANN 8.5.1 `Mc2MatMulMultiCoreSplitKDivideL2cache` control order for 24 AIC and 48 paired AIV actors: the AIC-to-AIV flag, all-AIV `SyncAll`, the return flag, and the all-AIC `SyncAll` before the second N slab. It intentionally excludes GEMM operands, Cube, FixPipe, workspace, reduction, and output stores.

`mode=0` is the source-matched two-slab control sequence. `mode=1` preserves the two cross-core flag edges and `PIPE_ALL` boundaries but omits `SyncAll`; it is a control used only to isolate collective completion, never a production schedule.
