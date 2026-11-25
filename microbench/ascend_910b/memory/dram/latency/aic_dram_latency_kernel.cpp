// Ascend 910B AIC GlobalTensor<int32_t>::GetValue latency microbenchmark.
//
// One AIC follows a dependent pointer chain in GM. Each next address is the
// value returned by the preceding GetValue, so the timed loop exposes one
// scalar-memory completion at a time instead of measuring streaming bandwidth.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

extern "C" __global__ __aicore__ void dram_latency(
        __gm__ int32_t* buffer, __gm__ int64_t* output, int64_t hops) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    AscendC::GlobalTensor<int32_t> values;
    values.SetGlobalBuffer(buffer);

    int32_t index = 0;
    for (int64_t hop = 0; hop < 64; ++hop) {
        index = values.GetValue(index);
    }

    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t hop = 0; hop < hops; ++hop) {
        index = values.GetValue(index);
    }
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output);
    result.SetValue(0, end - start);
    result.SetValue(1, static_cast<int64_t>(index));
}
