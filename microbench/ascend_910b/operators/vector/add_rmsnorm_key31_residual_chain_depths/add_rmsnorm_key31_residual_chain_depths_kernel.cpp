// Independent depth sweep for the CANN 8.5.1 AddRMSNorm key-31 residual
// reload chain. It retains only DataCopyPad -> EnQue/DeQue -> BF16 Cast.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr uint32_t kTilesPerRow = 2;
constexpr uint32_t kNumColElements = 16384;
constexpr uint32_t kElementsPerRequest = 8192;
constexpr uint32_t kRequestBytes = kElementsPerRequest * sizeof(bfloat16_t);
constexpr uint32_t kMaximumRequests = 44;
constexpr uint32_t kWorkerStrideBytes = kMaximumRequests * kRequestBytes;
constexpr uint32_t kOutputValuesPerWorker = 8;

__aicore__ inline void PrepareControl(
        AscendC::TQue<AscendC::QuePosition::VECIN, 1>& queue) {
    auto local = queue.AllocTensor<bfloat16_t>();
    AscendC::Duplicate<bfloat16_t>(
        local, static_cast<bfloat16_t>(0.5f), kElementsPerRequest);
    AscendC::PipeBarrier<PIPE_V>();
    queue.FreeTensor(local);
}

__aicore__ inline void ControlChain(
        AscendC::TQue<AscendC::QuePosition::VECIN, 1>& queue,
        AscendC::LocalTensor<float>& fp32, uint32_t row_count) {
    for (uint32_t tile = 0; tile < kTilesPerRow; ++tile) {
        for (uint32_t row = 0; row < row_count; ++row) {
            auto produced = queue.AllocTensor<bfloat16_t>();
            queue.EnQue(produced);
            auto consumed = queue.DeQue<bfloat16_t>();
            AscendC::Cast(
                fp32, consumed, AscendC::RoundMode::CAST_NONE,
                kElementsPerRequest);
            AscendC::PipeBarrier<PIPE_V>();
            queue.FreeTensor(consumed);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RequestChain(
        AscendC::TQue<AscendC::QuePosition::VECIN, 1>& queue,
        AscendC::LocalTensor<float>& fp32,
        const AscendC::GlobalTensor<bfloat16_t>& source,
        uint32_t row_count) {
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(kRequestBytes), 0, 0};
    AscendC::DataCopyPadParams pad_params;
    for (uint32_t tile = 0; tile < kTilesPerRow; ++tile) {
        for (uint32_t row = 0; row < row_count; ++row) {
            const uint32_t offset = (
                row * kNumColElements + tile * kElementsPerRequest);
            auto produced = queue.AllocTensor<bfloat16_t>();
            AscendC::DataCopyPad(
                produced, source[offset], copy_params, pad_params);
            queue.EnQue(produced);
            auto consumed = queue.DeQue<bfloat16_t>();
            AscendC::Cast(
                fp32, consumed, AscendC::RoundMode::CAST_NONE,
                kElementsPerRequest);
            AscendC::PipeBarrier<PIPE_V>();
            queue.FreeTensor(consumed);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline float ReadVectorSink(
        AscendC::LocalTensor<float>& fp32) {
    const event_t event_v_s = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(event_v_s);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(event_v_s);
    return fp32.GetValue(0);
}

}  // namespace

extern "C" __global__ __aicore__ void add_rmsnorm_key31_residual_chain_depths(
        __gm__ bfloat16_t* source, __gm__ int64_t* output,
        uint32_t block_factor, uint32_t last_block_factor,
        int64_t pair_order) {
    const uint32_t worker = AscendC::GetBlockIdx();
    const uint32_t row_count = (
        worker + 1 == AscendC::GetBlockNum()
        ? last_block_factor : block_factor);
    const uint32_t request_count = row_count * kTilesPerRow;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_buffer;
    pipe.InitBuffer(input_queue, 1, kRequestBytes);
    pipe.InitBuffer(fp32_buffer, kElementsPerRequest * sizeof(float));
    auto fp32 = fp32_buffer.Get<float>();
    AscendC::GlobalTensor<bfloat16_t> global_source;
    global_source.SetGlobalBuffer(
        source + worker * (kWorkerStrideBytes / sizeof(bfloat16_t)));

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t request_start = 0;
    int64_t request_end = 0;
    float control_sink = 0.0f;
    float request_sink = 0.0f;
    if (pair_order == 0) {
        PrepareControl(input_queue);
        control_start = AscendC::GetSystemCycle();
        ControlChain(input_queue, fp32, row_count);
        control_end = AscendC::GetSystemCycle();
        control_sink = ReadVectorSink(fp32);
        request_start = AscendC::GetSystemCycle();
        RequestChain(input_queue, fp32, global_source, row_count);
        request_end = AscendC::GetSystemCycle();
        request_sink = ReadVectorSink(fp32);
    } else {
        request_start = AscendC::GetSystemCycle();
        RequestChain(input_queue, fp32, global_source, row_count);
        request_end = AscendC::GetSystemCycle();
        request_sink = ReadVectorSink(fp32);
        PrepareControl(input_queue);
        control_start = AscendC::GetSystemCycle();
        ControlChain(input_queue, fp32, row_count);
        control_end = AscendC::GetSystemCycle();
        control_sink = ReadVectorSink(fp32);
    }

    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + worker * kOutputValuesPerWorker);
    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, static_cast<int64_t>(control_sink * 2.0f));
    result.SetValue(3, static_cast<int64_t>(request_sink * 2.0f));
    result.SetValue(4, request_count);
    result.SetValue(5, kRequestBytes);
    result.SetValue(6, pair_order);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
