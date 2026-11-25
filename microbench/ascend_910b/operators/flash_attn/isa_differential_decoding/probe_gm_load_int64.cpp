// Differential probe: one dependent int64 scalar GM load per iteration.
// Compiling at two NREP values and diffing .text isolates that load's opcode.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

#ifndef NREP
#define NREP 8
#endif

extern "C" __global__ __aicore__ void dd_gm_load_int64(
        __gm__ int64_t* gm, __gm__ int64_t* out) {
    AscendC::GlobalTensor<int64_t> source;
    source.SetGlobalBuffer(gm);
    int64_t cursor = 0;
#pragma unroll
    for (int index = 0; index < NREP; ++index) {
        // The returned value is the next index, so the chain is serialized
        // and the loop body compiles to exactly one instruction.
        cursor = source.GetValue(cursor);
    }
    AscendC::GlobalTensor<int64_t> sink;
    sink.SetGlobalBuffer(out);
    sink.SetValue(0, cursor);
}
