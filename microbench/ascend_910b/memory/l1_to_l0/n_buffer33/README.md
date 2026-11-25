# CANN NBuffer33 L1-to-L0 pair

This independent mechanism benchmark measures a completed BF16 `128×128×128` non-transposed A1→A2/B1→B2 MTE1 pair using the CANN 8.5.1 `NBuffer33MatmulPolicy` source geometry. It allocates the source policy's three A1 buffers, two B1 buffers, and two L0A/L0B slots, populates L1 before timing, and cycles those slots in source ownership order. The timed interval excludes MTE2, Cube, FixPipe, workspace, output traffic, production GEMM execution, and any production duration.

Build it under the fixed CANN 8.5.1 toolkit, then run `aic_n_buffer33_mte1_pair --batch 24 20000`. Hardware raw output belongs under `profiler_output/` and may be used only as a source-compatible MTE1 service measurement.
