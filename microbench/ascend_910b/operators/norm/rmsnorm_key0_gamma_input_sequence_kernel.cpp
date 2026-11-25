// Isolate the two serial GM-to-UB completion edges at the start of the
// CANN 8.5.1 BF16 RMSNorm key-0 source program.  Production completes the
// gamma queue before it issues the first x request.  Two independent
// depth-one VECIN queues preserve that ordering without executing RMSNorm.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumPayloadBytes = 16384;

template <bool CopyData>
__aicore__ inline void RunQueueSequence(
        AscendC::TQue<AscendC::QuePosition::VECIN, 1>& gamma_queue,
        AscendC::TQue<AscendC::QuePosition::VECIN, 1>& input_queue,
        AscendC::GlobalTensor<uint8_t> gamma_source,
        AscendC::GlobalTensor<uint8_t> input_source,
        uint32_t payload_bytes,
        AscendC::LocalTensor<uint8_t>& gamma_ready,
        AscendC::LocalTensor<uint8_t>& input_ready) {
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(payload_bytes), 0, 0};
    AscendC::DataCopyPadParams pad_params;

    auto gamma_produced = gamma_queue.AllocTensor<uint8_t>();
    if constexpr (CopyData) {
        AscendC::DataCopyPad(
            gamma_produced, gamma_source, copy_params, pad_params);
    }
    gamma_queue.EnQue(gamma_produced);
    gamma_ready = gamma_queue.DeQue<uint8_t>();

    // This issue point is deliberately after gamma DeQue.  CANN's depth-one
    // TQue DeQue waits on the MTE2_V event emitted by EnQue, so the input
    // request cannot overlap the gamma completion.
    auto input_produced = input_queue.AllocTensor<uint8_t>();
    if constexpr (CopyData) {
        AscendC::DataCopyPad(
            input_produced, input_source, copy_params, pad_params);
    }
    input_queue.EnQue(input_produced);
    input_ready = input_queue.DeQue<uint8_t>();

    // Keep the system-counter boundary identical in request and control.
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace

extern "C" __global__ __aicore__ void rmsnorm_key0_gamma_input_sequence(
        __gm__ uint8_t* gamma_source, __gm__ uint8_t* input_source,
        __gm__ int64_t* output, int64_t payload_bytes,
        int64_t stride_bytes, int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    if (payload_bytes <= 0 || payload_bytes > kMaximumPayloadBytes
            || payload_bytes % 32 != 0 || stride_bytes < payload_bytes
            || pair_order < 0 || pair_order > 1) {
        result.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> gamma_queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    pipe.InitBuffer(
        gamma_queue, 1, static_cast<uint32_t>(payload_bytes));
    pipe.InitBuffer(
        input_queue, 1, static_cast<uint32_t>(payload_bytes));
    AscendC::GlobalTensor<uint8_t> gamma_global;
    AscendC::GlobalTensor<uint8_t> input_global;
    gamma_global.SetGlobalBuffer(
        gamma_source + block_index * stride_bytes);
    input_global.SetGlobalBuffer(
        input_source + block_index * stride_bytes);

    AscendC::PipeBarrier<PIPE_ALL>();
    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t request_start = 0;
    int64_t request_end = 0;
    int64_t sink = 0;
    if (pair_order == 0) {
        AscendC::LocalTensor<uint8_t> control_gamma;
        AscendC::LocalTensor<uint8_t> control_input;
        control_start = AscendC::GetSystemCycle();
        RunQueueSequence<false>(
            gamma_queue, input_queue, gamma_global, input_global,
            static_cast<uint32_t>(payload_bytes),
            control_gamma, control_input);
        control_end = AscendC::GetSystemCycle();
        gamma_queue.FreeTensor(control_gamma);
        input_queue.FreeTensor(control_input);
        AscendC::LocalTensor<uint8_t> request_gamma;
        AscendC::LocalTensor<uint8_t> request_input;
        request_start = AscendC::GetSystemCycle();
        RunQueueSequence<true>(
            gamma_queue, input_queue, gamma_global, input_global,
            static_cast<uint32_t>(payload_bytes),
            request_gamma, request_input);
        request_end = AscendC::GetSystemCycle();
        sink = (
            static_cast<int64_t>(request_gamma.GetValue(0))
            + static_cast<int64_t>(request_input.GetValue(0)));
        gamma_queue.FreeTensor(request_gamma);
        input_queue.FreeTensor(request_input);
    } else {
        AscendC::LocalTensor<uint8_t> request_gamma;
        AscendC::LocalTensor<uint8_t> request_input;
        request_start = AscendC::GetSystemCycle();
        RunQueueSequence<true>(
            gamma_queue, input_queue, gamma_global, input_global,
            static_cast<uint32_t>(payload_bytes),
            request_gamma, request_input);
        request_end = AscendC::GetSystemCycle();
        sink = (
            static_cast<int64_t>(request_gamma.GetValue(0))
            + static_cast<int64_t>(request_input.GetValue(0)));
        gamma_queue.FreeTensor(request_gamma);
        input_queue.FreeTensor(request_input);
        AscendC::LocalTensor<uint8_t> control_gamma;
        AscendC::LocalTensor<uint8_t> control_input;
        control_start = AscendC::GetSystemCycle();
        RunQueueSequence<false>(
            gamma_queue, input_queue, gamma_global, input_global,
            static_cast<uint32_t>(payload_bytes),
            control_gamma, control_input);
        control_end = AscendC::GetSystemCycle();
        gamma_queue.FreeTensor(control_gamma);
        input_queue.FreeTensor(control_input);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, sink);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
