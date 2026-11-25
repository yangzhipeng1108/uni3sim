// Isolate first and steady visits of the q2 A/B CopyIn primitive used by the
// CANN 8.5.1 BF16 SwiGLU source loop.  This is intentionally not a SwiGLU
// operator: it retains only DataCopy/EnQue/DeQue/Free and its source-visible
// PIPE_ALL completion.  Each visit has a matched PIPE_ALL-only control.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

// The DCCI below publishes one cache line.  Keep all sampled fields on that
// line and align successive workers to a separate 128-byte result stride.
constexpr int64_t kOutputValuesPerBlock = 16;
constexpr uint32_t kPayloadBytes = 18432;

__aicore__ inline int64_t RunControl() {
    const int64_t start = AscendC::GetSystemCycle();
    AscendC::PipeBarrier<PIPE_ALL>();
    return AscendC::GetSystemCycle() - start;
}

template <int QueueDepth>
__aicore__ inline int64_t RunInputPair(
        AscendC::TQue<AscendC::QuePosition::VECIN, QueueDepth>& first_queue,
        AscendC::TQue<AscendC::QuePosition::VECIN, QueueDepth>& second_queue,
        AscendC::GlobalTensor<uint8_t> first_source,
        AscendC::GlobalTensor<uint8_t> second_source, int64_t* sink) {
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(kPayloadBytes / 32), 0, 0};
    const int64_t start = AscendC::GetSystemCycle();
    auto first = first_queue.template AllocTensor<uint8_t>();
    AscendC::DataCopy(first, first_source, copy_params);
    first_queue.EnQue(first);
    auto second = second_queue.template AllocTensor<uint8_t>();
    AscendC::DataCopy(second, second_source, copy_params);
    second_queue.EnQue(second);
    auto first_ready = first_queue.template DeQue<uint8_t>();
    auto second_ready = second_queue.template DeQue<uint8_t>();
    AscendC::PipeBarrier<PIPE_ALL>();
    *sink += static_cast<int64_t>(first_ready.GetValue(0))
        + static_cast<int64_t>(second_ready.GetValue(0));
    first_queue.FreeTensor(first_ready);
    second_queue.FreeTensor(second_ready);
    return AscendC::GetSystemCycle() - start;
}

}  // namespace

extern "C" __global__ __aicore__ void swiglu_input_pair_loop_edges(
        __gm__ uint8_t* first_source, __gm__ uint8_t* second_source,
        __gm__ int64_t* output, int64_t repetitions, int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    if ((repetitions != 1 && repetitions != 2)
            || pair_order < 0 || pair_order > 1) {
        result.SetValue(0, -1);
        AscendC::DataCacheCleanAndInvalid<int64_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(result);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> first_queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> second_queue;
    pipe.InitBuffer(first_queue, 2, kPayloadBytes);
    pipe.InitBuffer(second_queue, 2, kPayloadBytes);
    AscendC::GlobalTensor<uint8_t> first_global;
    AscendC::GlobalTensor<uint8_t> second_global;
    first_global.SetGlobalBuffer(first_source + block_index * kPayloadBytes);
    second_global.SetGlobalBuffer(second_source + block_index * kPayloadBytes);
    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t control_first = 0;
    int64_t control_steady = 0;
    int64_t request_first = 0;
    int64_t request_steady = 0;
    int64_t sink = 0;
    if (pair_order == 0) {
        control_first = RunControl();
        if (repetitions == 2) {
            control_steady = RunControl();
        }
        request_first = RunInputPair(
            first_queue, second_queue, first_global, second_global, &sink);
        if (repetitions == 2) {
            request_steady = RunInputPair(
                first_queue, second_queue, first_global, second_global,
                &sink);
        }
    } else {
        request_first = RunInputPair(
            first_queue, second_queue, first_global, second_global, &sink);
        if (repetitions == 2) {
            request_steady = RunInputPair(
                first_queue, second_queue, first_global, second_global,
                &sink);
        }
        control_first = RunControl();
        if (repetitions == 2) {
            control_steady = RunControl();
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    result.SetValue(0, control_first);
    result.SetValue(1, control_steady);
    result.SetValue(2, request_first);
    result.SetValue(3, request_steady);
    result.SetValue(4, sink);
    result.SetValue(5, repetitions);
    result.SetValue(6, pair_order);
    AscendC::DataCacheCleanAndInvalid<int64_t,
        AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
