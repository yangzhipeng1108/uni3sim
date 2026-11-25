// Isolate the key-1003 ProcessAfter workspace reduction from CANN
// MoeComputeExpertTokens.  The probe retains its DataCopyPad -> Cast ->
// WholeReduceMax -> Cast -> DataCopyPad source order, without the production
// search body or any production/evaluation duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kPhysicalAivCores = 48;
constexpr int64_t kInt32Alignment = 8;

__aicore__ inline int64_t AlignInt32(int64_t elements) {
    return (elements + kInt32Alignment - 1) / kInt32Alignment
        * kInt32Alignment;
}

}  // namespace

extern "C" __global__ __aicore__ void compute_expert_tokens_reduce(
        __gm__ int32_t* gm_workspace, __gm__ int32_t* gm_output,
        __gm__ int64_t* gm_cycles, int64_t experts,
        int64_t used_before) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t normal_rows =
        (experts + kPhysicalAivCores - 1) / kPhysicalAivCores;
    const int64_t used_after =
        (experts + normal_rows - 1) / normal_rows;
    if (block >= used_after) {
        gm_cycles[block] = 0;
        return;
    }
    const int64_t rows = block + 1 == used_after
        ? experts - (used_after - 1) * normal_rows
        : normal_rows;
    const int64_t aligned_width = AlignInt32(used_before);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> workspace_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> input_cast_buffer;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> output_cast_buffer;
    pipe.InitBuffer(
        workspace_queue, 1,
        static_cast<size_t>(normal_rows * aligned_width * sizeof(int32_t)));
    pipe.InitBuffer(
        output_queue, 1,
        static_cast<size_t>(normal_rows * sizeof(int32_t)));
    pipe.InitBuffer(
        input_cast_buffer,
        static_cast<size_t>(normal_rows * aligned_width * sizeof(float)));
    pipe.InitBuffer(
        output_cast_buffer,
        static_cast<size_t>(normal_rows * sizeof(float)));

    AscendC::GlobalTensor<int32_t> workspace;
    AscendC::GlobalTensor<int32_t> output;
    workspace.SetGlobalBuffer(
        gm_workspace + block * normal_rows * used_before,
        rows * aligned_width);
    output.SetGlobalBuffer(gm_output + block * normal_rows, rows);

    const int64_t begin = AscendC::GetSystemCycle();
    auto input_local = workspace_queue.AllocTensor<int32_t>();
    const bool padded = used_before % kInt32Alignment != 0;
    AscendC::DataCopyParams input_copy{
        static_cast<uint16_t>(rows),
        static_cast<uint16_t>(used_before * sizeof(int32_t)),
        static_cast<uint16_t>(0), static_cast<uint16_t>(0)};
    AscendC::DataCopyPadParams input_pad{
        padded, static_cast<uint8_t>(0),
        static_cast<uint8_t>(padded ? kInt32Alignment - used_before
                                    % kInt32Alignment : 0),
        static_cast<uint8_t>(0)};
    AscendC::DataCopyPad(input_local, workspace, input_copy, input_pad);
    workspace_queue.EnQue(input_local);

    input_local = workspace_queue.DeQue<int32_t>();
    auto output_local = output_queue.AllocTensor<int32_t>();
    auto input_float = input_cast_buffer.Get<float>();
    auto output_float = output_cast_buffer.Get<float>();
    AscendC::Cast(
        input_float, input_local, AscendC::RoundMode::CAST_NONE,
        rows * aligned_width);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceMax<float>(
        output_float, input_float, static_cast<uint64_t>(used_before),
        static_cast<int32_t>(rows), 1, 1,
        static_cast<int32_t>(aligned_width * sizeof(float) / 32),
        AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(
        output_local, output_float, AscendC::RoundMode::CAST_ROUND,
        AlignInt32(rows));
    output_queue.EnQue(output_local);
    workspace_queue.FreeTensor(input_local);

    output_local = output_queue.DeQue<int32_t>();
    AscendC::DataCopyParams output_copy{
        static_cast<uint16_t>(1),
        static_cast<uint16_t>(rows * sizeof(int32_t)),
        static_cast<uint16_t>(0), static_cast<uint16_t>(0)};
    AscendC::DataCopyPad(output, output_local, output_copy);
    output_queue.FreeTensor(output_local);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    const int64_t end = AscendC::GetSystemCycle();
    gm_cycles[block] = end - begin;
}
