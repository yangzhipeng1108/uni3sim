// Isolate warmed repeated-row DataCopyPad completion edges from the CANN
// 8.5.1 BF16 RMSNorm key-0 queue program.  This probe does not execute
// reduction, reciprocal square root, gamma multiplication, or RMSNorm output.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kMte2Steady = 0;
constexpr int64_t kMte2Terminal = 1;
constexpr int64_t kMte3Steady = 2;
constexpr int64_t kMte3Terminal = 3;
constexpr int64_t kMinimumPayloadBytes = 4096;
constexpr int64_t kMaximumPayloadBytes = 10240;
constexpr int64_t kSteadyRequests = 8;
constexpr int64_t kTerminalRequests = 1;
constexpr int64_t kWorkerStrideBytes = 131072;
constexpr int64_t kOutputValuesPerBlock = 8;

__aicore__ inline void CompleteMte2ForVector(event_t event) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
}

__aicore__ inline void CompleteVectorForMte3(event_t event) {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
}

__aicore__ inline void CompleteMte3ForVector(event_t event) {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
}

__aicore__ inline void RunControl(int64_t request_count) {
    #pragma unroll 1
    for (int64_t request = 0; request < request_count; ++request) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline int64_t RunMte2Rows(
        AscendC::TQue<AscendC::QuePosition::VECIN, 1>& input_queue,
        AscendC::GlobalTensor<half> source, int64_t request_count,
        int64_t first_row, int64_t payload_bytes) {
    const int64_t payload_elements = payload_bytes / sizeof(half);
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(payload_bytes), 0, 0};
    AscendC::DataCopyPadParams pad_params;
    int64_t sink = 0;
    #pragma unroll 1
    for (int64_t request = 0; request < request_count; ++request) {
        const int64_t row = first_row + request;
        auto produced = input_queue.AllocTensor<half>();
        AscendC::DataCopyPad(
            produced, source[row * payload_elements], copy_params, pad_params);
        input_queue.EnQue(produced);
        auto consumed = input_queue.DeQue<half>();
        CompleteMte2ForVector(EVENT_ID0);
        sink += static_cast<int64_t>(consumed.GetValue(0));
        input_queue.FreeTensor(consumed);
    }
    return sink;
}

__aicore__ inline void RunMte3Rows(
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1>& output_queue,
        AscendC::GlobalTensor<half> destination, int64_t request_count,
        int64_t first_row, int64_t payload_bytes, bool initialize) {
    const int64_t payload_elements = payload_bytes / sizeof(half);
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(payload_bytes), 0, 0};
    #pragma unroll 1
    for (int64_t request = 0; request < request_count; ++request) {
        const int64_t row = first_row + request;
        auto produced = output_queue.AllocTensor<half>();
        if (initialize) {
            AscendC::Duplicate<half>(
                produced, static_cast<half>(1.0f), payload_elements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        CompleteVectorForMte3(EVENT_ID0);
        output_queue.EnQue(produced);
        auto consumed = output_queue.DeQue<half>();
        AscendC::DataCopyPad(
            destination[row * payload_elements], consumed, copy_params);
        output_queue.FreeTensor(consumed);
        CompleteMte3ForVector(EVENT_ID0);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void rmsnorm_key0_steady_edges(
        __gm__ half* source, __gm__ half* destination,
        __gm__ int64_t* output, int64_t mode, int64_t payload_bytes,
        int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    if (mode < kMte2Steady || mode > kMte3Terminal
            || payload_bytes < kMinimumPayloadBytes
            || payload_bytes > kMaximumPayloadBytes
            || payload_bytes % 32 != 0
            || pair_order < 0 || pair_order > 1) {
        result.SetValue(0, -1);
        return;
    }
    const int64_t request_count = (
        mode == kMte2Steady || mode == kMte3Steady
        ? kSteadyRequests : kTerminalRequests);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
    pipe.InitBuffer(input_queue, 1, payload_bytes);
    pipe.InitBuffer(output_queue, 1, payload_bytes);
    AscendC::GlobalTensor<half> global_source;
    AscendC::GlobalTensor<half> global_destination;
    global_source.SetGlobalBuffer(
        source + block_index * kWorkerStrideBytes / sizeof(half));
    global_destination.SetGlobalBuffer(
        destination + block_index * kWorkerStrideBytes / sizeof(half));

    int64_t sink = 0;
    if (mode == kMte2Steady || mode == kMte2Terminal) {
        sink += RunMte2Rows(
            input_queue, global_source, 1, 0, payload_bytes);
    } else {
        RunMte3Rows(
            output_queue, global_destination, 1, 0, payload_bytes, true);
    }

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t target_start = 0;
    int64_t target_end = 0;
    if (pair_order == 0) {
        control_start = AscendC::GetSystemCycle();
        RunControl(request_count);
        control_end = AscendC::GetSystemCycle();
    }
    target_start = AscendC::GetSystemCycle();
    if (mode == kMte2Steady || mode == kMte2Terminal) {
        sink += RunMte2Rows(
            input_queue, global_source, request_count, 1, payload_bytes);
    } else {
        RunMte3Rows(
            output_queue, global_destination, request_count, 1,
            payload_bytes, false);
        sink += request_count;
    }
    target_end = AscendC::GetSystemCycle();
    if (pair_order == 1) {
        control_start = AscendC::GetSystemCycle();
        RunControl(request_count);
        control_end = AscendC::GetSystemCycle();
    }

    result.SetValue(0, control_end - control_start);
    result.SetValue(1, target_end - target_start);
    result.SetValue(2, sink);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
