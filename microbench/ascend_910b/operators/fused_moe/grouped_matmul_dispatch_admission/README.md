# GroupedMatmul M16/M64 dispatch admission

This independent CANN 8.5.1 mechanism microbenchmark isolates the asynchronous client submission made by the BF16 `GMMCompute::MMCompute` path. It never invokes GroupedMatmul, routing, activation, reduction, or a complete FusedMoE operator.

Each of 24 AIC workers submits a finite sequence of identical M64/N256 tiles using `SetOrgShape`, `SetSingleShape`, `SetTensorA`, `SetTensorB`, and `IterateAll<false>`. The timestamp ends immediately when that non-synchronous client call returns. `PIPE_ALL` is deliberately issued only after every timestamp is recorded; its completion is not timed or consumed.

The accepted source programs are the captured CANN grouped-Matmul base-M16 and base-M64 programs. Both issue W1 K2048 and W2 K512 M/N tiles. M16 uses `base=(16,256,64)`, `A1/B1=(120,8)`, `stepKa/stepKb=(60,4)`, and `L0A/L0B/L0C=(2,2,1)`; M64 uses `base=(64,256,64)`, `A1/B1=(24,8)`, `stepKa/stepKb=(12,4)`, and the same L0 buffers. The probe accepts only finite issue depths `1,2,4,8,16,32`; it can establish an admission/back-pressure boundary, but cannot provide an operator duration or an analytical Forward service by itself.
