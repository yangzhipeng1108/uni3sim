// Independent CANN 8.5.1 measurement of the FIA CalcExp Brcb dependency.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

extern "C" __global__ __aicore__ void fia_brcb_service(
        __gm__ int64_t* gm_cycles, __gm__ uint32_t* gm_validation,
        uint32_t row_chunk_round, uint32_t iterations) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(gm_cycles);
    if (row_chunk_round < 8 || row_chunk_round > 64
            || row_chunk_round % 8 != 0 || iterations == 0) {
        cycles.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> source_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> destination_buffer;
    pipe.InitBuffer(source_buffer, 64 * sizeof(float));
    pipe.InitBuffer(destination_buffer, 512 * sizeof(uint32_t));
    auto source = source_buffer.Get<float>();
    auto destination = destination_buffer.Get<uint32_t>();
    AscendC::Duplicate<float>(source, 1.0f, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(
        destination, source.ReinterpretCast<uint32_t>(),
        row_chunk_round / 8, AscendC::BrcbRepeatParams(1, 8));
    AscendC::PipeBarrier<PIPE_V>();

    const int64_t begin = AscendC::GetSystemCycle();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::Brcb(
            destination, source.ReinterpretCast<uint32_t>(),
            row_chunk_round / 8, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }
    const int64_t end = AscendC::GetSystemCycle();
    cycles.SetValue(0, end - begin);

    AscendC::GlobalTensor<uint32_t> validation;
    validation.SetGlobalBuffer(gm_validation);
    validation.SetValue(0, destination.GetValue(0));
}
