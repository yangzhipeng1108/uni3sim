// Isolate repeated CANN 8.5.1 ReshapeAndCache key-200 copy chains.
//
// This is not a production operator replay: it has no slot mapping, host
// tiler, cache layout, or production timing target. Each AIV executes a fixed
// count of source-shaped K-then-V GM->UB->GM chains through one reused TBuf.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;

__aicore__ inline void CopyChain(
        AscendC::GlobalTensor<half>& source,
        AscendC::LocalTensor<half>& local,
        AscendC::GlobalTensor<half>& destination,
        const AscendC::DataCopyParams& copy_params) {
    AscendC::DataCopy(local, source, copy_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
    AscendC::DataCopy(destination, local, copy_params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

}  // namespace

extern "C" __global__ __aicore__ void reshape_cache_equal_repeated_chain(
        __gm__ half* key_source, __gm__ half* value_source,
        __gm__ half* key_destination, __gm__ half* value_destination,
        __gm__ int64_t* output, int64_t payload_bytes,
        int64_t stride_bytes, int64_t repeat_count, int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();
    const int64_t stride_elements = stride_bytes / sizeof(half);
    const int64_t block_offset = block_index * repeat_count * stride_elements;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> token_buffer;
    pipe.InitBuffer(token_buffer, payload_bytes);
    auto local = token_buffer.Get<half>();
    AscendC::GlobalTensor<half> key_input;
    AscendC::GlobalTensor<half> value_input;
    AscendC::GlobalTensor<half> key_output;
    AscendC::GlobalTensor<half> value_output;
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(payload_bytes / 32), 0, 0};

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t request_start = 0;
    int64_t request_end = 0;
    if (pair_order == 0) {
        control_start = AscendC::GetSystemCycle();
        for (int64_t repeat = 0; repeat < repeat_count; ++repeat) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        control_end = AscendC::GetSystemCycle();
        request_start = AscendC::GetSystemCycle();
        for (int64_t repeat = 0; repeat < repeat_count; ++repeat) {
            const int64_t offset = block_offset + repeat * stride_elements;
            key_input.SetGlobalBuffer(key_source + offset);
            value_input.SetGlobalBuffer(value_source + offset);
            key_output.SetGlobalBuffer(key_destination + offset);
            value_output.SetGlobalBuffer(value_destination + offset);
            CopyChain(key_input, local, key_output, copy_params);
            CopyChain(value_input, local, value_output, copy_params);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        request_end = AscendC::GetSystemCycle();
    } else {
        request_start = AscendC::GetSystemCycle();
        for (int64_t repeat = 0; repeat < repeat_count; ++repeat) {
            const int64_t offset = block_offset + repeat * stride_elements;
            key_input.SetGlobalBuffer(key_source + offset);
            value_input.SetGlobalBuffer(value_source + offset);
            key_output.SetGlobalBuffer(key_destination + offset);
            value_output.SetGlobalBuffer(value_destination + offset);
            CopyChain(key_input, local, key_output, copy_params);
            CopyChain(value_input, local, value_output, copy_params);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        request_end = AscendC::GetSystemCycle();
        control_start = AscendC::GetSystemCycle();
        for (int64_t repeat = 0; repeat < repeat_count; ++repeat) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        control_end = AscendC::GetSystemCycle();
    }

    const int64_t final_offset = (
        block_offset + (repeat_count - 1) * stride_elements);
    key_output.SetGlobalBuffer(key_destination + final_offset);
    value_output.SetGlobalBuffer(value_destination + final_offset);
    const int64_t sink = (
        static_cast<int64_t>(key_output.GetValue(0))
        + static_cast<int64_t>(value_output.GetValue(0)));
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, sink);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
