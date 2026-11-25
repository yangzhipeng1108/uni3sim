// Isolate the two-input TQue completion envelope used by CANN 8.5.1
// SwiGLU CopyIn. The public source issues A DataCopy/EnQue followed by B
// DataCopy/EnQue before Compute dequeues either input. This probe retains
// only that source-ordered memory mechanism and executes no SwiGLU math.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumPayloadBytes = 32768;

__aicore__ inline void RunControl() {
    AscendC::PipeBarrier<PIPE_ALL>();
}

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
    const int64_t sink = (
        static_cast<int64_t>(first_ready.GetValue(0))
        + static_cast<int64_t>(second_ready.GetValue(0)));
    first_queue.FreeTensor(first_ready);
    second_queue.FreeTensor(second_ready);
    return sink;
}

}  // namespace

extern "C" __global__ __aicore__ void swiglu_input_pair_completion(
        __gm__ uint8_t* first_source, __gm__ uint8_t* second_source,
        __gm__ int64_t* output, int64_t payload_bytes,
        int64_t stride_bytes, int64_t queue_depth, int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    if (payload_bytes <= 0 || payload_bytes > kMaximumPayloadBytes
            || payload_bytes % 32 != 0 || stride_bytes < payload_bytes
            || (queue_depth != 1 && queue_depth != 2)
            || pair_order < 0 || pair_order > 1) {
        AscendC::GlobalTensor<int64_t> invalid;
        invalid.SetGlobalBuffer(
            output + block_index * kOutputValuesPerBlock);
        invalid.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> first_queue_depth1;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> second_queue_depth1;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> first_queue_depth2;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> second_queue_depth2;
    if (queue_depth == 1) {
        pipe.InitBuffer(first_queue_depth1, 1, payload_bytes);
        pipe.InitBuffer(second_queue_depth1, 1, payload_bytes);
    } else {
        pipe.InitBuffer(first_queue_depth2, 2, payload_bytes);
        pipe.InitBuffer(second_queue_depth2, 2, payload_bytes);
    }
    AscendC::GlobalTensor<uint8_t> first_global;
    AscendC::GlobalTensor<uint8_t> second_global;
    first_global.SetGlobalBuffer(
        first_source + block_index * stride_bytes);
    second_global.SetGlobalBuffer(
        second_source + block_index * stride_bytes);

    // Queue construction belongs to the separately published lifecycle
    // service. Drain it before both intervals.
    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t request_start = 0;
    int64_t request_end = 0;
    int64_t sink = 0;
    if (pair_order == 0) {
        control_start = AscendC::GetSystemCycle();
        RunControl();
        control_end = AscendC::GetSystemCycle();
        request_start = AscendC::GetSystemCycle();
        sink = queue_depth == 1
            ? RunInputPair(
                first_queue_depth1, second_queue_depth1,
                first_global, second_global,
                static_cast<uint32_t>(payload_bytes))
            : RunInputPair(
                first_queue_depth2, second_queue_depth2,
                first_global, second_global,
                static_cast<uint32_t>(payload_bytes));
        request_end = AscendC::GetSystemCycle();
    } else {
        request_start = AscendC::GetSystemCycle();
        sink = queue_depth == 1
            ? RunInputPair(
                first_queue_depth1, second_queue_depth1,
                first_global, second_global,
                static_cast<uint32_t>(payload_bytes))
            : RunInputPair(
                first_queue_depth2, second_queue_depth2,
                first_global, second_global,
                static_cast<uint32_t>(payload_bytes));
        request_end = AscendC::GetSystemCycle();
        control_start = AscendC::GetSystemCycle();
        RunControl();
        control_end = AscendC::GetSystemCycle();
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, sink);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
