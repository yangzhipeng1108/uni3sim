// Isolate the repeated CANN 8.5.1 ReshapeAndCache key-202 K/V queue pair.
//
// This is not a production operator replay: it has no slot mapping, host
// tiler, cache layout, or production timing target. Each of 48 AIVs executes
// the source-defined maximum of 22 fixed 2048-byte K/V pairs through two
// independent two-slot TQueBind queues.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kBufferSlots = 2;
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

__aicore__ inline void RunPairs(
        DoubleSlotQueue& key_queue, DoubleSlotQueue& value_queue,
        AscendC::GlobalTensor<half>& key_source,
        AscendC::GlobalTensor<half>& value_source,
        AscendC::GlobalTensor<half>& key_destination,
        AscendC::GlobalTensor<half>& value_destination,
        int64_t block_offset, int64_t stride_elements,
        int64_t repeat_count, const AscendC::DataCopyParams& copy_params) {
    for (int64_t repeat = 0; repeat < repeat_count; ++repeat) {
        const int64_t offset = block_offset + repeat * stride_elements;
        QueueTransfer(
            key_queue, key_source, key_destination, offset, copy_params);
        QueueTransfer(
            value_queue, value_source, value_destination, offset, copy_params);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace

extern "C" __global__ __aicore__ void reshape_cache_key202_repeated_pair(
        __gm__ half* key_source, __gm__ half* value_source,
        __gm__ half* key_destination, __gm__ half* value_destination,
        __gm__ int64_t* output, int64_t payload_bytes,
        int64_t stride_bytes, int64_t repeat_count, int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    const int64_t stride_elements = stride_bytes / sizeof(half);
    const int64_t block_offset = block_index * repeat_count * stride_elements;

    AscendC::TPipe pipe;
    DoubleSlotQueue key_queue;
    DoubleSlotQueue value_queue;
    pipe.InitBuffer(key_queue, kBufferSlots, payload_bytes);
    pipe.InitBuffer(value_queue, kBufferSlots, payload_bytes);

    AscendC::GlobalTensor<half> key_input;
    AscendC::GlobalTensor<half> value_input;
    AscendC::GlobalTensor<half> key_output;
    AscendC::GlobalTensor<half> value_output;
    key_input.SetGlobalBuffer(key_source);
    value_input.SetGlobalBuffer(value_source);
    key_output.SetGlobalBuffer(key_destination);
    value_output.SetGlobalBuffer(value_destination);
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
        request_start = AscendC::GetSystemCycle();
        RunPairs(
            key_queue, value_queue, key_input, value_input,
            key_output, value_output, block_offset, stride_elements,
            repeat_count, copy_params);
        request_end = AscendC::GetSystemCycle();
    } else {
        request_start = AscendC::GetSystemCycle();
        RunPairs(
            key_queue, value_queue, key_input, value_input,
            key_output, value_output, block_offset, stride_elements,
            repeat_count, copy_params);
        request_end = AscendC::GetSystemCycle();
        control_start = AscendC::GetSystemCycle();
        AscendC::PipeBarrier<PIPE_ALL>();
        control_end = AscendC::GetSystemCycle();
    }

    const int64_t final_offset = (
        block_offset + (repeat_count - 1) * stride_elements);
    const int64_t sink = (
        static_cast<int64_t>(key_output.GetValue(final_offset))
        + static_cast<int64_t>(value_output.GetValue(final_offset)));
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, sink);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
