// Isolate a complete UB-resident FP32 ReduceSum + vector-pipe barrier.
// Input initialization is outside the measured loop, so no GM/L2 transfer is
// included. The Level-2 ReduceSum count contract reduces the complete row.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kMinimumCount = 32;
constexpr int64_t kMaximumCount = 16384;

}  // namespace

extern "C" __global__ __aicore__ void reduce_sum_latency(
        __gm__ int64_t* gm_cycles, __gm__ float* gm_validation,
        int64_t count, int64_t n_iters) {
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(gm_cycles);
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    if (count < kMinimumCount || count > kMaximumCount
            || count % 32 != 0 || n_iters <= 0) {
        cycles.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    const size_t tensor_bytes = static_cast<size_t>(count) * sizeof(float);
    AscendC::TBuf<AscendC::TPosition::VECIN> input_buffer;
    pipe.InitBuffer(input_buffer, tensor_bytes);
    auto input = input_buffer.Get<float>();
    AscendC::TBuf<AscendC::TPosition::VECCALC> temporary_buffer;
    pipe.InitBuffer(temporary_buffer, tensor_bytes);
    auto temporary = temporary_buffer.Get<float>();
    AscendC::TBuf<AscendC::TPosition::VECOUT> output_buffer;
    pipe.InitBuffer(output_buffer, 64);
    auto reduced = output_buffer.Get<float>();

    AscendC::Duplicate<float>(
        input, 1.0f, static_cast<uint32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float>(
        reduced, input, temporary, static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();

    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t iteration = 0; iteration < n_iters; ++iteration) {
        AscendC::ReduceSum<float>(
            reduced, input, temporary, static_cast<int32_t>(count));
        AscendC::PipeBarrier<PIPE_V>();
    }
    const int64_t end = AscendC::GetSystemCycle();
    cycles.SetValue(0, end - begin);

    // Preserve one completed reduction result outside the timed region.  The
    // host verifies it against the sum of `count` ones, so an optimized-away,
    // truncated-mask, or otherwise invalid reduction fails loudly.
    AscendC::GlobalTensor<float> validation;
    validation.SetGlobalBuffer(gm_validation);
    validation.SetValue(0, reduced.GetValue(0));
}
