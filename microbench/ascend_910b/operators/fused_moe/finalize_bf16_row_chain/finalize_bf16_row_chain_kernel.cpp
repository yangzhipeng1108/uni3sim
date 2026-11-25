// Isolate the indexed BF16 row-accumulation chain in CANN 8.5.1
// MoeFinalizeRoutingV2 key 20036.  This is not the production operator: it
// has no routing, skips, bias, host tiling, expert dispatch, or production H.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kRowsPerOutput = 2;
constexpr int64_t kIndexAlignment = 8;
constexpr int64_t kResultStrideElements = 8;

__aicore__ inline bool IsSupportedRows(int64_t row_elements) {
    return row_elements == 6080 || row_elements == 6208;
}

}  // namespace

extern "C" __global__ __aicore__ void finalize_bf16_row_chain(
        __gm__ half* gm_source, __gm__ int32_t* gm_indices,
        __gm__ half* gm_output, __gm__ int64_t* gm_results,
        int64_t row_elements, int64_t source_groups,
        int64_t iterations, int64_t source_mode) {
    const int64_t worker = AscendC::GetBlockIdx();
    if (!IsSupportedRows(row_elements) || source_groups <= 0
            || iterations <= 0 || (source_mode != 0 && source_mode != 1)) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> index_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> first_bf16_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> second_bf16_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> first_fp32_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> second_fp32_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> output_bf16_buffer;
    pipe.InitBuffer(index_buffer, kIndexAlignment * sizeof(int32_t));
    pipe.InitBuffer(first_bf16_buffer, row_elements * sizeof(half));
    pipe.InitBuffer(second_bf16_buffer, row_elements * sizeof(half));
    pipe.InitBuffer(first_fp32_buffer, row_elements * sizeof(float));
    pipe.InitBuffer(second_fp32_buffer, row_elements * sizeof(float));
    pipe.InitBuffer(output_bf16_buffer, row_elements * sizeof(half));

    auto indices = index_buffer.Get<int32_t>();
    auto first_bf16 = first_bf16_buffer.Get<half>();
    auto second_bf16 = second_bf16_buffer.Get<half>();
    auto first_fp32 = first_fp32_buffer.Get<float>();
    auto second_fp32 = second_fp32_buffer.Get<float>();
    auto output_bf16 = output_bf16_buffer.Get<half>();

    const AscendC::DataCopyParams row_copy = {
        1, static_cast<uint16_t>(row_elements * sizeof(half)), 0, 0};
    const AscendC::DataCopyPadParams row_pad = {false, 0, 0, 0};
    const AscendC::DataCopyExtParams index_copy = {
        1, static_cast<uint32_t>(kRowsPerOutput * sizeof(int32_t)), 0, 0, 0};
    const AscendC::DataCopyPadExtParams<int32_t> index_pad = {
        true, 0, static_cast<uint8_t>(kIndexAlignment - kRowsPerOutput), 0};

    AscendC::GlobalTensor<half> source;
    AscendC::GlobalTensor<int32_t> index_source;
    AscendC::GlobalTensor<half> destination;
    AscendC::GlobalTensor<int64_t> results;
    results.SetGlobalBuffer(gm_results + worker * kResultStrideElements);

    const int64_t cycle_start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const int64_t group = iteration % source_groups;
        const int64_t group_offset = (
            (worker * source_groups + group) * kRowsPerOutput * row_elements);
        int64_t first_index = 0;
        int64_t second_index = 1;
        if (source_mode == 1) {
            index_source.SetGlobalBuffer(
                gm_indices + (worker * source_groups + group) * kRowsPerOutput);
            AscendC::DataCopyPad(indices, index_source, index_copy, index_pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            first_index = static_cast<int64_t>(indices.GetValue(0));
            second_index = static_cast<int64_t>(indices.GetValue(1));
        }
        source.SetGlobalBuffer(gm_source + group_offset);
        AscendC::DataCopyPad(
            first_bf16, source[first_index * row_elements], row_copy, row_pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::DataCopyPad(
            second_bf16, source[second_index * row_elements], row_copy, row_pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(first_fp32, first_bf16, AscendC::RoundMode::CAST_NONE,
                      row_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(first_fp32, first_fp32, 1.0f, row_elements);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Cast(second_fp32, second_bf16, AscendC::RoundMode::CAST_NONE,
                      row_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(second_fp32, second_fp32, 1.0f, row_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(first_fp32, first_fp32, second_fp32, row_elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(output_bf16, first_fp32, AscendC::RoundMode::CAST_RINT,
                      row_elements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        destination.SetGlobalBuffer(
            gm_output + (worker * iterations + iteration) * row_elements);
        AscendC::DataCopyPad(destination, output_bf16, row_copy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    const int64_t cycle_end = AscendC::GetSystemCycle();
    results.SetValue(0, cycle_end - cycle_start);
    results.SetValue(1, iterations);
    results.SetValue(2, source_mode);
}
