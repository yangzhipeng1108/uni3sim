// Isolate the Ascend 910B AIV WholeReduceSum + vector-pipe barrier path.
// Input is initialized in UB before timing; the measured loop performs no
// GM/L2 transfer and does not replay a complete RMSNorm operator.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

extern "C" __global__ __aicore__ void whole_reduce_latency(
        __gm__ int64_t* out, int64_t mask, int64_t repeat_times,
        int64_t n_iters) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> input_buffer;
    pipe.InitBuffer(
        input_buffer,
        static_cast<size_t>(mask * repeat_times) * sizeof(float));
    auto input = input_buffer.Get<float>();
    AscendC::TBuf<AscendC::TPosition::VECOUT> output_buffer;
    pipe.InitBuffer(
        output_buffer, static_cast<size_t>(repeat_times) * 32);
    auto reduced = output_buffer.Get<float>();

    AscendC::Duplicate<float>(
        input, 1.0f, static_cast<uint32_t>(mask * repeat_times));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(
        reduced, input, static_cast<int32_t>(mask),
        static_cast<int32_t>(repeat_times), 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();

    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t iteration = 0; iteration < n_iters; ++iteration) {
        AscendC::WholeReduceSum<float, false>(
            reduced, input, static_cast<int32_t>(mask),
            static_cast<int32_t>(repeat_times), 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const int64_t end = AscendC::GetSystemCycle();
    out[0] = end - begin;
}
