// Isolate the double-buffer memory and event edges present in the Ascend 910B
// RoPE key-24 grid-47 path. This is not a RoPE replay: it contains no position,
// trigonometric, cast, interleave, or production arithmetic.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kMte2First = 0;
constexpr int64_t kMte2Steady = 1;
constexpr int64_t kMte3Completion = 2;
constexpr int64_t kEventInitDrain = 3;
constexpr int64_t kSlotBytes = 4096;
constexpr int64_t kBufferBytes = 2 * kSlotBytes;
constexpr int64_t kWorkerStrideBytes = 131072;
constexpr int64_t kOutputValuesPerBlock = 8;
constexpr int64_t kEventReplications = 8;

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

__aicore__ inline void RunControl() {
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunMte2Requests(
        AscendC::LocalTensor<half> local,
        AscendC::GlobalTensor<half> source, int64_t payload_bytes,
        int64_t request_count, int64_t first_request) {
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(payload_bytes / 32), 0, 0};
    for (int64_t request = 0; request < request_count; ++request) {
        const int64_t sequence = first_request + request;
        auto slot = local[(sequence % 2) * kSlotBytes / sizeof(half)];
        AscendC::DataCopy(
            slot, source[sequence * kSlotBytes / sizeof(half)], copy_params);
        CompleteMte2ForVector(
            sequence % 2 == 0 ? EVENT_ID0 : EVENT_ID1);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunMte3Requests(
        AscendC::LocalTensor<half> local,
        AscendC::GlobalTensor<half> destination, int64_t payload_bytes,
        int64_t request_count) {
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(payload_bytes / 32), 0, 0};
    for (int64_t request = 0; request < request_count; ++request) {
        const event_t event = request % 2 == 0 ? EVENT_ID0 : EVENT_ID1;
        auto slot = local[(request % 2) * kSlotBytes / sizeof(half)];
        CompleteVectorForMte3(event);
        AscendC::DataCopy(
            destination[request * kSlotBytes / sizeof(half)], slot,
            copy_params);
        CompleteMte3ForVector(event);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunEventInitDrain() {
    #pragma unroll 1
    for (int64_t replication = 0;
            replication < kEventReplications; ++replication) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace

extern "C" __global__ __aicore__ void rope_key24_steady_edges(
        __gm__ half* source, __gm__ half* destination,
        __gm__ int64_t* output, int64_t mode, int64_t payload_bytes,
        int64_t request_count, int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    const bool valid_payload = (
        payload_bytes == 128 || payload_bytes == 2048
        || payload_bytes == 4096);
    const bool valid_steady_count = (
        request_count == 1 || request_count == 2
        || request_count == 4 || request_count == 8
        || request_count == 20 || request_count == 21);
    if (mode < kMte2First || mode > kEventInitDrain
            || pair_order < 0 || pair_order > 1
            || (mode == kMte2Steady
                && (payload_bytes != 4096 || !valid_steady_count))
            || ((mode == kMte2First || mode == kMte3Completion)
                && (!valid_payload || request_count != 1))
            || (mode == kEventInitDrain
                && (payload_bytes != 0
                    || request_count != 2 * kEventReplications))) {
        result.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kBufferBytes);
    auto local = buffer.Get<half>();
    AscendC::GlobalTensor<half> global_source;
    AscendC::GlobalTensor<half> global_destination;
    global_source.SetGlobalBuffer(
        source + block_index * kWorkerStrideBytes / sizeof(half));
    global_destination.SetGlobalBuffer(
        destination + block_index * kWorkerStrideBytes / sizeof(half));
    AscendC::Duplicate<half>(
        local, static_cast<half>(1.0f), kBufferBytes / sizeof(half));
    AscendC::PipeBarrier<PIPE_V>();

    if (mode == kMte2Steady) {
        RunMte2Requests(local, global_source, 4096, 1, 0);
    }

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t request_start = 0;
    int64_t request_end = 0;
    if (pair_order == 0) {
        control_start = AscendC::GetSystemCycle();
        RunControl();
        control_end = AscendC::GetSystemCycle();
    }
    request_start = AscendC::GetSystemCycle();
    if (mode == kMte2First) {
        RunMte2Requests(
            local, global_source, payload_bytes, request_count, 0);
    } else if (mode == kMte2Steady) {
        RunMte2Requests(
            local, global_source, payload_bytes, request_count, 1);
    } else if (mode == kMte3Completion) {
        RunMte3Requests(
            local, global_destination, payload_bytes, request_count);
    } else {
        RunEventInitDrain();
    }
    request_end = AscendC::GetSystemCycle();
    if (pair_order == 1) {
        control_start = AscendC::GetSystemCycle();
        RunControl();
        control_end = AscendC::GetSystemCycle();
    }

    int64_t sink = request_count;
    if (mode == kMte2First || mode == kMte2Steady) {
        sink += static_cast<int64_t>(local.GetValue(0));
    }
    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, sink);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
