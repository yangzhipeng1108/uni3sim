// Isolate the CANN 8.5.1 RMSNorm key-1 per-row partial-sum handoff.
//
// ComputeFormerPrecision writes one FP32 partial into the first element of
// each 32-byte sumLocal slot, then performs V_S, N ordered GetValue/adds, and
// S_V.  This probe preserves only that group-terminal mechanism.  It does not
// copy production inputs, reduce a column tile, normalize, or write RMSNorm
// output.  The paired control has the same Vector write and event shell but
// accumulates the known scalar directly, so the positive difference isolates
// the sparse LocalTensor GetValue part of the otherwise matched add chain.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;
constexpr uint32_t kPartialStrideElements = 8;
constexpr uint32_t kMaximumPartials = 64;
constexpr uint32_t kGroupElements =
    kPartialStrideElements * kMaximumPartials;

__aicore__ inline float RunControl(
        const AscendC::LocalTensor<float>& partials,
        uint32_t partial_count, int64_t iterations) {
    float reduce_out = 0.0f;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const float value = static_cast<float>((iteration & 1) + 1);
        AscendC::Duplicate<float>(
            partials, value, partial_count * kPartialStrideElements);
        const event_t vector_to_scalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
        reduce_out = 0.0f;
        #pragma unroll 1
        for (uint32_t partial = 0; partial < partial_count; ++partial) {
            reduce_out += value;
        }
        const event_t scalar_to_vector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
    }
    return reduce_out;
}

__aicore__ inline float RunPartialSumHandoff(
        const AscendC::LocalTensor<float>& partials,
        uint32_t partial_count, int64_t iterations) {
    float reduce_out = 0.0f;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const float value = static_cast<float>((iteration & 1) + 1);
        AscendC::Duplicate<float>(
            partials, value, partial_count * kPartialStrideElements);
        const event_t vector_to_scalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vector_to_scalar);
        reduce_out = 0.0f;
        #pragma unroll 1
        for (uint32_t partial = 0; partial < partial_count; ++partial) {
            reduce_out += partials.GetValue(
                partial * kPartialStrideElements);
        }
        const event_t scalar_to_vector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalar_to_vector);
    }
    return reduce_out;
}

}  // namespace

extern "C" __global__ __aicore__ void rmsnorm_key1_partial_sum(
        __gm__ int64_t* output, int64_t partial_count,
        int64_t pair_order, int64_t iterations) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(
        output + block_index * kOutputValuesPerBlock);
    if (partial_count <= 0 || partial_count > kMaximumPartials
            || pair_order < 0 || pair_order > 1
            || iterations <= 0 || iterations % 2 != 0) {
        result.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partial_buffer;
    pipe.InitBuffer(partial_buffer, kGroupElements * sizeof(float));
    auto partials = partial_buffer.Get<float>();
    AscendC::Duplicate<float>(partials, 0.0f, kGroupElements);
    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t target_start = 0;
    int64_t target_end = 0;
    float control_sink = 0.0f;
    float target_sink = 0.0f;
    if (pair_order == 0) {
        control_start = AscendC::GetSystemCycle();
        control_sink = RunControl(
            partials, static_cast<uint32_t>(partial_count), iterations);
        control_end = AscendC::GetSystemCycle();
        target_start = AscendC::GetSystemCycle();
        target_sink = RunPartialSumHandoff(
            partials, static_cast<uint32_t>(partial_count), iterations);
        target_end = AscendC::GetSystemCycle();
    } else {
        target_start = AscendC::GetSystemCycle();
        target_sink = RunPartialSumHandoff(
            partials, static_cast<uint32_t>(partial_count), iterations);
        target_end = AscendC::GetSystemCycle();
        control_start = AscendC::GetSystemCycle();
        control_sink = RunControl(
            partials, static_cast<uint32_t>(partial_count), iterations);
        control_end = AscendC::GetSystemCycle();
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    const int64_t sink = static_cast<int64_t>(control_sink + target_sink);
    result.SetValue(0, control_end - control_start);
    result.SetValue(1, target_end - target_start);
    result.SetValue(2, sink);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
