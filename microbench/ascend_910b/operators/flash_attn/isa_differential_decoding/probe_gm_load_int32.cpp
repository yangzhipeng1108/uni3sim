// Differential probe: one int32 scalar GM load plus its index arithmetic.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

#ifndef NREP
#define NREP 8
#endif

extern "C" __global__ __aicore__ void dd_gm_load_int32(
        __gm__ int32_t* gm, __gm__ int64_t* out) {
    AscendC::GlobalTensor<int32_t> source;
    source.SetGlobalBuffer(gm);
    int64_t cursor = 0;
    int64_t accumulator = 0;
#pragma unroll
    for (int index = 0; index < NREP; ++index) {
        accumulator += static_cast<int64_t>(source.GetValue(cursor));
        cursor = (cursor + 7) & 255;
    }
    AscendC::GlobalTensor<int64_t> sink;
    sink.SetGlobalBuffer(out);
    sink.SetValue(0, accumulator);
}
