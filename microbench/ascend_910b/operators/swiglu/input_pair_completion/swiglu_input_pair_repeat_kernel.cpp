// Measure a generic q2 A/B CopyIn pair at 4 and 8 KiB under fixed admission.
// Each size is repeated in the same AIV execution, in both phase orders, so
// launch overhead and slow device-state drift cannot create a byte slope.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

// Keep every worker result aligned to one 64-byte cache line; the explicit
// DCCI below must publish all recorded counters, even at high active widths.
constexpr int64_t kOutputValuesPerBlock = 8;
constexpr uint32_t kSmallPayloadBytes = 4096;
constexpr uint32_t kLargePayloadBytes = 8192;

template <int QueueDepth>
__aicore__ inline int64_t RunInputPair(
        AscendC::TQue<AscendC::QuePosition::VECIN, QueueDepth>& first_queue,
        AscendC::TQue<AscendC::QuePosition::VECIN, QueueDepth>& second_queue,
        AscendC::GlobalTensor<uint8_t> first_source,
        AscendC::GlobalTensor<uint8_t> second_source,
        uint32_t payload_bytes) {
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(payload_bytes / 32), 0, 0};
    auto first = first_queue.template AllocTensor<uint8_t>();
    AscendC::DataCopy(first, first_source, copy_params);
    first_queue.EnQue(first);
    auto second = second_queue.template AllocTensor<uint8_t>();
    AscendC::DataCopy(second, second_source, copy_params);
    second_queue.EnQue(second);
    auto first_ready = first_queue.template DeQue<uint8_t>();
    auto second_ready = second_queue.template DeQue<uint8_t>();
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t sink = (static_cast<int64_t>(first_ready.GetValue(0))
                          + static_cast<int64_t>(second_ready.GetValue(0)));
    first_queue.FreeTensor(first_ready);
    second_queue.FreeTensor(second_ready);
    return sink;
}

template <int QueueDepth>
__aicore__ inline int64_t RunRepeatedPair(
        AscendC::TQue<AscendC::QuePosition::VECIN, QueueDepth>& first_queue,
        AscendC::TQue<AscendC::QuePosition::VECIN, QueueDepth>& second_queue,
        AscendC::GlobalTensor<uint8_t> first_source,
        AscendC::GlobalTensor<uint8_t> second_source,
        uint32_t payload_bytes, int64_t repetitions, int64_t* sink) {
    const int64_t start = AscendC::GetSystemCycle();
    for (int64_t repetition = 0; repetition < repetitions; ++repetition) {
        *sink += RunInputPair(
            first_queue, second_queue, first_source, second_source,
            payload_bytes);
    }
    return AscendC::GetSystemCycle() - start;
}

}  // namespace

extern "C" __global__ __aicore__ void swiglu_input_pair_repeat(
        __gm__ uint8_t* first_source, __gm__ uint8_t* second_source,
        __gm__ int64_t* output, int64_t repetitions, int64_t phase_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    if (repetitions <= 0 || phase_order < 0 || phase_order > 1) {
        result.SetValue(0, -1);
        AscendC::DataCacheCleanAndInvalid<int64_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(result);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> first_queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> second_queue;
    pipe.InitBuffer(first_queue, 2, kLargePayloadBytes);
    pipe.InitBuffer(second_queue, 2, kLargePayloadBytes);
    AscendC::GlobalTensor<uint8_t> first_global;
    AscendC::GlobalTensor<uint8_t> second_global;
    first_global.SetGlobalBuffer(first_source + block_index * kLargePayloadBytes);
    second_global.SetGlobalBuffer(second_source + block_index * kLargePayloadBytes);
    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t sink = 0;
    int64_t small_cycles = 0;
    int64_t large_cycles = 0;
    if (phase_order == 0) {
        small_cycles = RunRepeatedPair(
            first_queue, second_queue, first_global, second_global,
            kSmallPayloadBytes, repetitions, &sink);
        large_cycles = RunRepeatedPair(
            first_queue, second_queue, first_global, second_global,
            kLargePayloadBytes, repetitions, &sink);
    } else {
        large_cycles = RunRepeatedPair(
            first_queue, second_queue, first_global, second_global,
            kLargePayloadBytes, repetitions, &sink);
        small_cycles = RunRepeatedPair(
            first_queue, second_queue, first_global, second_global,
            kSmallPayloadBytes, repetitions, &sink);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    result.SetValue(0, small_cycles);
    result.SetValue(1, large_cycles);
    result.SetValue(2, sink);
    result.SetValue(3, repetitions);
    result.SetValue(4, phase_order);
    AscendC::DataCacheCleanAndInvalid<int64_t,
        AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
