// Ascend 910B first MTE request-to-completion microbenchmark.
//
// Each AIV executes the CANN 8.5.1 GET_TILING_DATA materialization used by
// source-inspected AIV kernels, initializes one local buffer, and then times
// one dynamic-parameter DataCopy followed by one selected hard completion
// edge, paired with a control interval on the same AIV in the same launch.
// The two interval orders are swept independently so the request-minus-control
// result isolates a first body MTE completion without cross-launch subtraction
// or replaying any production operator. Directions 4 and 5 retain the same
// transfer but wrap it in the generic TQue producer/consumer lifecycle used by
// source-defined CANN Vector pipelines. Directions 6 and 7 replace the raw
// copy with the DataCopyPad form used by the CANN 8.5.1 RMSNorm source.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;
constexpr uint32_t kTilingBytes = 28;

struct GenericTilingData {
    uint32_t field0;
    uint32_t field1;
    uint32_t field2;
    uint32_t field3;
    uint32_t field4;
    uint32_t field5;
    uint32_t field6;
};

static_assert(sizeof(GenericTilingData) == kTilingBytes);

__aicore__ inline void MaterializeTilingData(
        GenericTilingData* destination, __gm__ uint8_t* source) {
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) \
        || defined(__DAV_310R6_CUBE__) || defined(__GET_CODE_CHANNEL__)
    copy_data_align64(
        reinterpret_cast<uint8_t*>(destination), source, kTilingBytes);
#else
    __ubuf__ uint8_t* tiling_data_in_ub =
        reinterpret_cast<__ubuf__ uint8_t*>(get_imm(0));
    constexpr uint32_t burst_count = (kTilingBytes + 31) / 32;
    copy_gm_to_ubuf(
        tiling_data_in_ub, source, 0, 1, burst_count, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64(
        reinterpret_cast<uint8_t*>(destination),
        (__ubuf__ uint8_t*)tiling_data_in_ub, kTilingBytes);
    pipe_barrier(PIPE_ALL);
#endif
}

}  // namespace

extern "C" __global__ __aicore__ void first_mte_completion(
        __gm__ uint8_t* tiling, __gm__ uint8_t* source,
        __gm__ uint8_t* destination, __gm__ int64_t* output,
        int64_t direction, int64_t tile_bytes, int64_t stride_bytes,
        int64_t pair_order) {
    const int64_t block_index = AscendC::GetBlockIdx();

    GenericTilingData tiling_data{};
    MaterializeTilingData(&tiling_data, tiling);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> output_queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> rms_input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> rms_output_queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> gather_index_queue;
    pipe.InitBuffer(buffer, tile_bytes);
    if (direction == 4) {
        pipe.InitBuffer(input_queue, 2, tile_bytes);
    } else if (direction == 5) {
        pipe.InitBuffer(output_queue, 2, tile_bytes);
    } else if (direction == 6) {
        pipe.InitBuffer(rms_input_queue, 1, tile_bytes);
    } else if (direction == 7) {
        pipe.InitBuffer(rms_output_queue, 1, tile_bytes);
    } else if (direction == 8) {
        pipe.InitBuffer(gather_index_queue, 1, tile_bytes);
    }
    auto local_bytes = buffer.Get<uint8_t>();
    auto local_half = buffer.Get<half>();
    const AscendC::DataCopyParams block_copy_params = {
        1, static_cast<uint16_t>(tile_bytes / 32), 0, 0};
    // CANN 8.5.1 DataCopyPad uses a byte-valued blockLen on Ascend 910B,
    // unlike the 32-byte block count used by the plain DataCopy overload.
    // Keep the two contracts separate so a nominal payload is never silently
    // reduced by a factor of 32 in the source-form RMSNorm cells.
    const AscendC::DataCopyParams byte_copy_params = {
        1, static_cast<uint16_t>(tile_bytes), 0, 0};
    AscendC::GlobalTensor<uint8_t> global_source;
    global_source.SetGlobalBuffer(source + block_index * stride_bytes);

    AscendC::LocalTensor<half> queued_output;
    if (direction == 1 || direction == 3) {
        AscendC::Duplicate<half>(
            local_half, static_cast<half>(1.0f),
            static_cast<uint32_t>(tile_bytes / sizeof(half)));
        AscendC::PipeBarrier<PIPE_V>();
    } else if (direction == 5 || direction == 7) {
        queued_output = (
            direction == 5
            ? output_queue.AllocTensor<half>()
            : rms_output_queue.AllocTensor<half>());
        AscendC::Duplicate<half>(
            queued_output, static_cast<half>(1.0f),
            static_cast<uint32_t>(tile_bytes / sizeof(half)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::LocalTensor<uint8_t> queued_input;
    AscendC::LocalTensor<int32_t> gather_indices;
    int64_t gather_index_sum = 0;

    int64_t control_start = 0;
    int64_t control_end = 0;
    int64_t request_start = 0;
    int64_t request_end = 0;
    if (pair_order == 0) {
        control_start = AscendC::GetSystemCycle();
        AscendC::PipeBarrier<PIPE_ALL>();
        control_end = AscendC::GetSystemCycle();
        request_start = AscendC::GetSystemCycle();
        if (direction == 0 || direction == 2) {
            AscendC::DataCopy(local_bytes, global_source, block_copy_params);
            if (direction == 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
        } else if (direction == 1 || direction == 3) {
            AscendC::GlobalTensor<half> half_destination;
            half_destination.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(
                    destination + block_index * stride_bytes));
            AscendC::DataCopy(
                half_destination, local_half, block_copy_params);
            if (direction == 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            }
        } else if (direction == 4) {
            auto produced = input_queue.AllocTensor<uint8_t>();
            AscendC::DataCopy(produced, global_source, block_copy_params);
            input_queue.EnQue(produced);
            queued_input = input_queue.DeQue<uint8_t>();
        } else if (direction == 5) {
            AscendC::GlobalTensor<half> half_destination;
            half_destination.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(
                    destination + block_index * stride_bytes));
            output_queue.EnQue(queued_output);
            auto consumed = output_queue.DeQue<half>();
            AscendC::DataCopy(
                half_destination, consumed, block_copy_params);
            output_queue.FreeTensor(consumed);
        } else if (direction == 6) {
            auto produced = rms_input_queue.AllocTensor<uint8_t>();
            AscendC::DataCopyPadParams pad_params;
            AscendC::DataCopyPad(
                produced, global_source, byte_copy_params, pad_params);
            rms_input_queue.EnQue(produced);
            queued_input = rms_input_queue.DeQue<uint8_t>();
        } else if (direction == 7) {
            AscendC::GlobalTensor<half> half_destination;
            half_destination.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(
                    destination + block_index * stride_bytes));
            rms_output_queue.EnQue(queued_output);
            auto consumed = rms_output_queue.DeQue<half>();
            AscendC::DataCopyPad(
                half_destination, consumed, byte_copy_params);
            rms_output_queue.FreeTensor(consumed);
        } else if (direction == 8) {
            auto produced = gather_index_queue.AllocTensor<int32_t>();
            AscendC::DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(tile_bytes);
            params.srcStride = 0;
            params.dstStride = 0;
            AscendC::DataCopyPadExtParams<uint8_t> pad_params{
                false, 0, 0, 0};
            AscendC::DataCopyPad(
                produced.template ReinterpretCast<uint8_t>(),
                global_source, params, pad_params);
            gather_index_queue.EnQue(produced);
            gather_indices = gather_index_queue.DeQue<int32_t>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
            for (int64_t index = 0; index < tile_bytes / 4; ++index) {
                gather_index_sum += gather_indices.GetValue(index);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        request_end = AscendC::GetSystemCycle();
    } else {
        request_start = AscendC::GetSystemCycle();
        if (direction == 0 || direction == 2) {
            AscendC::DataCopy(local_bytes, global_source, block_copy_params);
            if (direction == 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
        } else if (direction == 1 || direction == 3) {
            AscendC::GlobalTensor<half> half_destination;
            half_destination.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(
                    destination + block_index * stride_bytes));
            AscendC::DataCopy(
                half_destination, local_half, block_copy_params);
            if (direction == 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            }
        } else if (direction == 4) {
            auto produced = input_queue.AllocTensor<uint8_t>();
            AscendC::DataCopy(produced, global_source, block_copy_params);
            input_queue.EnQue(produced);
            queued_input = input_queue.DeQue<uint8_t>();
        } else if (direction == 5) {
            AscendC::GlobalTensor<half> half_destination;
            half_destination.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(
                    destination + block_index * stride_bytes));
            output_queue.EnQue(queued_output);
            auto consumed = output_queue.DeQue<half>();
            AscendC::DataCopy(
                half_destination, consumed, block_copy_params);
            output_queue.FreeTensor(consumed);
        } else if (direction == 6) {
            auto produced = rms_input_queue.AllocTensor<uint8_t>();
            AscendC::DataCopyPadParams pad_params;
            AscendC::DataCopyPad(
                produced, global_source, byte_copy_params, pad_params);
            rms_input_queue.EnQue(produced);
            queued_input = rms_input_queue.DeQue<uint8_t>();
        } else if (direction == 7) {
            AscendC::GlobalTensor<half> half_destination;
            half_destination.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(
                    destination + block_index * stride_bytes));
            rms_output_queue.EnQue(queued_output);
            auto consumed = rms_output_queue.DeQue<half>();
            AscendC::DataCopyPad(
                half_destination, consumed, byte_copy_params);
            rms_output_queue.FreeTensor(consumed);
        } else if (direction == 8) {
            auto produced = gather_index_queue.AllocTensor<int32_t>();
            AscendC::DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(tile_bytes);
            params.srcStride = 0;
            params.dstStride = 0;
            AscendC::DataCopyPadExtParams<uint8_t> pad_params{
                false, 0, 0, 0};
            AscendC::DataCopyPad(
                produced.template ReinterpretCast<uint8_t>(),
                global_source, params, pad_params);
            gather_index_queue.EnQue(produced);
            gather_indices = gather_index_queue.DeQue<int32_t>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
            for (int64_t index = 0; index < tile_bytes / 4; ++index) {
                gather_index_sum += gather_indices.GetValue(index);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        request_end = AscendC::GetSystemCycle();
        control_start = AscendC::GetSystemCycle();
        AscendC::PipeBarrier<PIPE_ALL>();
        control_end = AscendC::GetSystemCycle();
    }

    int64_t sink = 0;
    if (direction == 0 || direction == 2) {
        sink = static_cast<int64_t>(local_bytes.GetValue(0));
    } else if (direction == 4 || direction == 6) {
        sink = static_cast<int64_t>(queued_input.GetValue(0));
        if (direction == 4) {
            input_queue.FreeTensor(queued_input);
        } else {
            rms_input_queue.FreeTensor(queued_input);
        }
    } else if (direction == 8) {
        sink = gather_index_sum;
        gather_index_queue.FreeTensor(gather_indices);
    }
    volatile uint32_t* fields =
        reinterpret_cast<volatile uint32_t*>(&tiling_data);
    sink += static_cast<int64_t>(
        fields[0] + fields[1] + fields[2] + fields[3]
        + fields[4] + fields[5] + fields[6]);

    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    result.SetValue(0, control_end - control_start);
    result.SetValue(1, request_end - request_start);
    result.SetValue(2, sink);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
