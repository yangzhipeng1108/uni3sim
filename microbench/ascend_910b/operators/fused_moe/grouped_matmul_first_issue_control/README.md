# GroupedMatmul controller-to-first-issue probe

This standalone CANN 8.5.1 AIC-only mechanism probe establishes the source dependency from the cumulative group-list controller to the first `MMCompute` call assigned to each AIC. It reproduces the source-defined group loop for the uniform BF16 CUMSUM program through `WaitPreTaskEnd`, `UpdateMnConfig`, `GlobalTensor<int64_t>::GetValue`, static `SetMNConfig`, the AIC ring, `MNBlockIdxCompute`, and `SetNextTaskStart`. It records immediately before the first `MMCompute` call; it does not claim to timestamp the later MTE/Cube issue inside `MMCompute`.

The probe does not call `MMCompute`, `IterateAll`, `VectorCompute`, or `SyncAll`; it reads no matrix operand, executes no MTE/Cube/FixPipe work, and consumes no production or evaluation duration. Its host driver accepts a generic non-negative group-row list, records all 24 AIC worker timestamps, and requires at least three fresh processes. It permits exactly one source pass per launch: repeating the group loop and averaging its timestamps would erase the one-pass cold/control service that a Forward invocation must retain.

Build in the agreed 910B container with `ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 cmake -S . -B build -DSOC_VERSION=Ascend910B2`, then `cmake --build build -j 8`. The runner remaps physical device 2, 6, or 7 to logical device zero.
