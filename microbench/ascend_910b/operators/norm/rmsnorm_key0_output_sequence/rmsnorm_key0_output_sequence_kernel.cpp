// Isolate the source-ordered Y then rstd MTE3 completion sequence from the
// CANN 8.5.1 BF16 RMSNorm key-0 program.  The probe retains the two depth-one
// VECOUT queues and their exact DataCopyPad requests, but executes no RMSNorm
// arithmetic and consumes no production-operator duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumPayloadBytes = 16384;
constexpr int64_t kWorkerStrideBytes = 16384;
constexpr int64_t kRstdStrideBytes = 32;

template <bool CopyData>
__aicore__ inline void RunOutputSequence(
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1>& y_queue,
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1>& rstd_queue,
        AscendC::GlobalTensor<half> y_destination,
        AscendC::GlobalTensor<float> rstd_destination,
        uint32_t payload_bytes) {
    const AscendC::DataCopyParams y_copy = {
        1, static_cast<uint16_t>(payload_bytes), 0, 0};
    const AscendC::DataCopyParams rstd_copy = {1, 4, 0, 0};

    auto y_produced = y_queue.AllocTensor<half>();
    AscendC::Duplicate<half>(
        y_produced, static_cast<half>(1.0f), payload_bytes / sizeof(half));
    AscendC::PipeBarrier<PIPE_V>();
    y_queue.EnQue(y_produced);
    auto y_ready = y_queue.DeQue<half>();
    if constexpr (CopyData) {
        AscendC::DataCopyPad(y_destination, y_ready, y_copy);
    }
    y_queue.FreeTensor(y_ready);

    auto rstd_produced = rstd_queue.AllocTensor<float>();
    AscendC::Duplicate<float>(rstd_produced, 1.0f, 8);
    AscendC::PipeBarrier<PIPE_V>();
    rstd_queue.EnQue(rstd_produced);
    auto rstd_ready = rstd_queue.DeQue<float>();
    if constexpr (CopyData) {
        AscendC::DataCopyPad(rstd_destination, rstd_ready, rstd_copy);
    }
    rstd_queue.FreeTensor(rstd_ready);
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace

extern "C" __global__ __aicore__ void rmsnorm_key0_output_sequence(
        __gm__ uint8_t* y_destination, __gm__ uint8_t* rstd_destination,
        __gm__ int64_t* output, int64_t payload_bytes,
        int64_t pair_order) {
    const int64_t worker = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + worker * kOutputValuesPerBlock);
    if (payload_bytes <= 0 || payload_bytes > kMaximumPayloadBytes
            || payload_bytes % 32 != 0
            || pair_order < 0 || pair_order > 1) {
        result.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> y_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> rstd_queue;
    pipe.InitBuffer(y_queue, 1, static_cast<uint32_t>(payload_bytes));
    pipe.InitBuffer(rstd_queue, 1, 32);
    AscendC::GlobalTensor<half> y_global;
    AscendC::GlobalTensor<float> rstd_global;
    y_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(y_destination)
        + worker * kWorkerStrideBytes / sizeof(half));
    rstd_global.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(rstd_destination)
        + worker * kRstdStrideBytes / sizeof(float));
    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t request_start = 0;
    int64_t request_end = 0;
    if (pair_order == 0) {
        control_start = AscendC::GetSystemCycle();
        RunOutputSequence<false>(
            y_queue, rstd_queue, y_global, rstd_global,
            static_cast<uint32_t>(payload_bytes));
        control_end = AscendC::GetSystemCycle();
        request_start = AscendC::GetSystemCycle();
        RunOutputSequence<true>(
            y_queue, rstd_queue, y_global, rstd_global,
            static_cast<uint32_t>(payload_bytes));
        request_end = AscendC::GetSystemCycle();
    } else {
        request_start = AscendC::GetSystemCycle();
        RunOutputSequence<true>(
            y_queue, rstd_queue, y_global, rstd_global,
            static_cast<uint32_t>(payload_bytes));
        request_end = AscendC::GetSystemCycle();
        control_start = AscendC::GetSystemCycle();
        RunOutputSequence<false>(
            y_queue, rstd_queue, y_global, rstd_global,
            static_cast<uint32_t>(payload_bytes));
        control_end = AscendC::GetSystemCycle();
    }

    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, 1);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
