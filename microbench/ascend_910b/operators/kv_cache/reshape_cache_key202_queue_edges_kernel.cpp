// Isolate the CANN 8.5.1 ReshapeAndCache key-202 two-slot queue edge.
//
// Source provenance is frozen by reshape_and_cache_cann_8_5_1.json. This is
// not a production operator replay: it has no slot mapping, token loop, host
// tiler, cache layout, or full-operator duration target. Each of the 48 AIVs
// issues three fixed K/V transfers. The first two consume the independent K
// and V queue slots; the timed third pair exposes MTE2->MTE3 completion and
// first-slot reuse without adding a K-to-V completion dependency.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kBufferSlots = 2;
constexpr int64_t kWorkerStrideBytes = 8192;
constexpr int64_t kOutputValuesPerBlock = 8;

using DoubleSlotQueue = AscendC::TQueBind<
    AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, kBufferSlots>;

__aicore__ inline void QueueTransfer(
        DoubleSlotQueue& queue, AscendC::GlobalTensor<half>& source,
        AscendC::GlobalTensor<half>& destination, int64_t offset,
        const AscendC::DataCopyParams& copy_params) {
    auto local = queue.AllocTensor<half>();
    AscendC::DataCopy(local, source[offset], copy_params);
    queue.EnQue(local);
    local = queue.DeQue<half>();
    AscendC::DataCopy(destination[offset], local, copy_params);
    queue.FreeTensor(local);
}

__aicore__ inline void PrimeBothSlots(
        DoubleSlotQueue& key_queue, DoubleSlotQueue& value_queue,
        AscendC::GlobalTensor<half>& key_source,
        AscendC::GlobalTensor<half>& value_source,
        AscendC::GlobalTensor<half>& key_destination,
        AscendC::GlobalTensor<half>& value_destination,
        int64_t payload_elements,
        const AscendC::DataCopyParams& copy_params) {
    QueueTransfer(key_queue, key_source, key_destination, 0, copy_params);
    QueueTransfer(value_queue, value_source, value_destination, 0, copy_params);
    QueueTransfer(
        key_queue, key_source, key_destination, payload_elements, copy_params);
    QueueTransfer(
        value_queue, value_source, value_destination, payload_elements,
        copy_params);
}

__aicore__ inline void ReuseFirstSlots(
        DoubleSlotQueue& key_queue, DoubleSlotQueue& value_queue,
        AscendC::GlobalTensor<half>& key_source,
        AscendC::GlobalTensor<half>& value_source,
        AscendC::GlobalTensor<half>& key_destination,
        AscendC::GlobalTensor<half>& value_destination,
        int64_t payload_elements,
        const AscendC::DataCopyParams& copy_params) {
    const int64_t reuse_offset = 2 * payload_elements;
    QueueTransfer(
        key_queue, key_source, key_destination, reuse_offset, copy_params);
    QueueTransfer(
        value_queue, value_source, value_destination, reuse_offset,
        copy_params);
}

}  // namespace

extern "C" __global__ __aicore__ void reshape_cache_key202_queue_edges(
        __gm__ half* key_source, __gm__ half* value_source,
        __gm__ half* key_destination, __gm__ half* value_destination,
        __gm__ int64_t* output, int64_t payload_bytes, int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    const int64_t worker_offset = (
        block_index * kWorkerStrideBytes / sizeof(half));
    const int64_t payload_elements = payload_bytes / sizeof(half);

    AscendC::TPipe pipe;
    DoubleSlotQueue key_queue;
    DoubleSlotQueue value_queue;
    pipe.InitBuffer(key_queue, kBufferSlots, payload_bytes);
    pipe.InitBuffer(value_queue, kBufferSlots, payload_bytes);

    AscendC::GlobalTensor<half> key_input;
    AscendC::GlobalTensor<half> value_input;
    AscendC::GlobalTensor<half> key_output;
    AscendC::GlobalTensor<half> value_output;
    key_input.SetGlobalBuffer(key_source + worker_offset);
    value_input.SetGlobalBuffer(value_source + worker_offset);
    key_output.SetGlobalBuffer(key_destination + worker_offset);
    value_output.SetGlobalBuffer(value_destination + worker_offset);
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(payload_bytes / 32), 0, 0};

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t request_start = 0;
    int64_t request_end = 0;
    if (pair_order == 0) {
        control_start = AscendC::GetSystemCycle();
        AscendC::PipeBarrier<PIPE_ALL>();
        control_end = AscendC::GetSystemCycle();
        PrimeBothSlots(
            key_queue, value_queue, key_input, value_input, key_output,
            value_output, payload_elements, copy_params);
        request_start = AscendC::GetSystemCycle();
        ReuseFirstSlots(
            key_queue, value_queue, key_input, value_input, key_output,
            value_output, payload_elements, copy_params);
        AscendC::PipeBarrier<PIPE_ALL>();
        request_end = AscendC::GetSystemCycle();
    } else {
        PrimeBothSlots(
            key_queue, value_queue, key_input, value_input, key_output,
            value_output, payload_elements, copy_params);
        request_start = AscendC::GetSystemCycle();
        ReuseFirstSlots(
            key_queue, value_queue, key_input, value_input, key_output,
            value_output, payload_elements, copy_params);
        AscendC::PipeBarrier<PIPE_ALL>();
        request_end = AscendC::GetSystemCycle();
        control_start = AscendC::GetSystemCycle();
        AscendC::PipeBarrier<PIPE_ALL>();
        control_end = AscendC::GetSystemCycle();
    }

    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, 6);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
