// Isolate the two-request MTE2 completion envelope used by CANN 8.5.1
// AddRmsNorm SingleN.  The public source issues both equal-sized DataCopyPad
// requests and their MTE2_V events before waiting for either event.  This
// benchmark retains only that memory mechanism; it does not execute any
// normalization arithmetic or replay a production operator.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumPayloadBytes = 16384;

__aicore__ inline void RunControl() {
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunInputPair(
        AscendC::LocalTensor<uint8_t> first,
        AscendC::LocalTensor<uint8_t> second,
        AscendC::GlobalTensor<uint8_t> first_source,
        AscendC::GlobalTensor<uint8_t> second_source,
        uint32_t payload_bytes) {
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(payload_bytes), 0, 0};
    AscendC::DataCopyPadParams pad_params;
    AscendC::DataCopyPad(first, first_source, copy_params, pad_params);
    const event_t first_event = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(
            AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(first_event);
    AscendC::DataCopyPad(second, second_source, copy_params, pad_params);
    const event_t second_event = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(
            AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(second_event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(first_event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(second_event);
    // Match the single-request service benchmark's counter boundary: make
    // completion visible to the scalar system counter before recording the
    // interval.  No Vector arithmetic is present in this probe.
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace

extern "C" __global__ __aicore__ void add_rmsnorm_input_pair(
        __gm__ uint8_t* first_source, __gm__ uint8_t* second_source,
        __gm__ int64_t* output, int64_t payload_bytes,
        int64_t stride_bytes, int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    if (payload_bytes <= 0 || payload_bytes > kMaximumPayloadBytes
            || payload_bytes % 32 != 0 || stride_bytes < payload_bytes
            || pair_order < 0 || pair_order > 1) {
        AscendC::GlobalTensor<int64_t> invalid;
        invalid.SetGlobalBuffer(
            output + block_index * kOutputValuesPerBlock);
        invalid.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_buffer;
    pipe.InitBuffer(
        input_buffer, static_cast<uint32_t>(2 * payload_bytes));
    auto local = input_buffer.Get<uint8_t>();
    auto first = local[0];
    auto second = local[payload_bytes];
    AscendC::GlobalTensor<uint8_t> first_global;
    AscendC::GlobalTensor<uint8_t> second_global;
    first_global.SetGlobalBuffer(
        first_source + block_index * stride_bytes);
    second_global.SetGlobalBuffer(
        second_source + block_index * stride_bytes);

    // TBuf construction is a separate kernel-lifecycle mechanism.  Drain it
    // before either interval so the paired-copy service is invariant to the
    // control/request order rather than crediting UB initialization to the
    // first interval.
    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t request_start = 0;
    int64_t request_end = 0;
    if (pair_order == 0) {
        control_start = AscendC::GetSystemCycle();
        RunControl();
        control_end = AscendC::GetSystemCycle();
        request_start = AscendC::GetSystemCycle();
        RunInputPair(
            first, second, first_global, second_global,
            static_cast<uint32_t>(payload_bytes));
        request_end = AscendC::GetSystemCycle();
    } else {
        request_start = AscendC::GetSystemCycle();
        RunInputPair(
            first, second, first_global, second_global,
            static_cast<uint32_t>(payload_bytes));
        request_end = AscendC::GetSystemCycle();
        control_start = AscendC::GetSystemCycle();
        RunControl();
        control_end = AscendC::GetSystemCycle();
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    const int64_t sink = (
        static_cast<int64_t>(first.GetValue(0))
        + static_cast<int64_t>(second.GetValue(0)));
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, sink);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
