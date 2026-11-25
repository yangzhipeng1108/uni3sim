// Isolate generic row-stride effects in the CANN q2 A/B CopyIn mechanism.
// The fixed campaign grid is deliberately independent of production shapes.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumRows = 4;
constexpr int64_t kMaximumRowBytes = 8192;
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
        uint32_t row_count, uint32_t row_bytes, uint32_t source_stride) {
    const AscendC::DataCopyParams copy_params = {
        static_cast<uint16_t>(row_count),
        static_cast<uint16_t>(row_bytes / 32),
        static_cast<uint16_t>((source_stride - row_bytes) / 32),
        0,
    };
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

extern "C" __global__ __aicore__ void swiglu_input_pair_geometry(
        __gm__ uint8_t* first_source, __gm__ uint8_t* second_source,
        __gm__ int64_t* output, int64_t row_count, int64_t row_bytes,
        int64_t source_stride, int64_t base_phase, int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    const int64_t payload_bytes = row_count * row_bytes;
    if (row_count <= 0 || row_count > kMaximumRows || row_bytes <= 0
            || row_bytes > kMaximumRowBytes || row_bytes % 32 != 0
            || payload_bytes > kMaximumPayloadBytes
            || source_stride < row_bytes || source_stride % 32 != 0
            || base_phase < 0 || base_phase % 32 != 0
            || pair_order < 0 || pair_order > 1) {
        AscendC::GlobalTensor<int64_t> invalid;
        invalid.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
        invalid.SetValue(0, -1);
        return;
    }
    const int64_t worker_span = (
        (row_count - 1) * source_stride + row_bytes);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> first_queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> second_queue;
    pipe.InitBuffer(first_queue, 2, payload_bytes);
    pipe.InitBuffer(second_queue, 2, payload_bytes);

    AscendC::GlobalTensor<uint8_t> first_global;
    AscendC::GlobalTensor<uint8_t> second_global;
    first_global.SetGlobalBuffer(
        first_source + base_phase + block_index * worker_span);
    second_global.SetGlobalBuffer(
        second_source + base_phase + block_index * worker_span);

    // TPipe/TQue construction belongs to the separate lifecycle mechanism.
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
        sink = RunInputPair(
            first_queue, second_queue, first_global, second_global,
            static_cast<uint32_t>(row_count),
            static_cast<uint32_t>(row_bytes),
            static_cast<uint32_t>(source_stride));
        request_end = AscendC::GetSystemCycle();
    } else {
        request_start = AscendC::GetSystemCycle();
        sink = RunInputPair(
            first_queue, second_queue, first_global, second_global,
            static_cast<uint32_t>(row_count),
            static_cast<uint32_t>(row_bytes),
            static_cast<uint32_t>(source_stride));
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
