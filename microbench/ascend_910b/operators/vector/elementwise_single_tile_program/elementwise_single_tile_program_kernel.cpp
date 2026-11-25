// Isolate the finite one-tile memory/control program shared by the CANN
// 8.5.1 BF16 Add and Mul BroadcastOneDimSch path.  The probe retains the
// source TPipe/TBuf allocation, two GM-to-UB copies, the three cast phases,
// one UB-to-GM copy, and terminal fence.  It deliberately omits the Add/Mul
// arithmetic node, so it is not a production elementwise op.
#include "kernel_operator.h"

#include "elementwise_single_tile_program_contract.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

using namespace elementwise_single_tile_program_contract;

extern "C" __global__ __aicore__ void elementwise_single_tile_program(
        __gm__ bfloat16_t* source_a, __gm__ bfloat16_t* source_b,
        __gm__ bfloat16_t* destination, __gm__ int64_t* worker_cycles) {
    const int64_t worker = AscendC::GetBlockIdx();
    if (worker < 0 || worker >= kAivWorkers) {
        return;
    }
    const uint32_t elements = static_cast<uint32_t>(
        worker == kAivWorkers - 1
        ? kTailTileElements : kFullTileElements);
    const int64_t offset = worker * kFullTileElements;

    AscendC::GlobalTensor<bfloat16_t> global_a;
    AscendC::GlobalTensor<bfloat16_t> global_b;
    AscendC::GlobalTensor<bfloat16_t> global_destination;
    global_a.SetGlobalBuffer(source_a + offset);
    global_b.SetGlobalBuffer(source_b + offset);
    global_destination.SetGlobalBuffer(destination + offset);

    const int64_t cycle_start = AscendC::GetSystemCycle();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kTbufBytes);
    auto pool = buffer.Get<uint8_t>();
    constexpr int64_t slot_bytes =
        kFullTileElements * kMaximumDtypeBytes;
    auto local_a = pool[0 * slot_bytes].ReinterpretCast<bfloat16_t>();
    auto local_b = pool[1 * slot_bytes].ReinterpretCast<bfloat16_t>();
    auto fp32_a = pool[2 * slot_bytes].ReinterpretCast<float>();
    auto fp32_b = pool[3 * slot_bytes].ReinterpretCast<float>();
    auto fp32_result = pool[4 * slot_bytes].ReinterpretCast<float>();
    local_a.SetBufferLen(kFullTileElements);
    local_b.SetBufferLen(kFullTileElements);
    fp32_a.SetBufferLen(kFullTileElements);
    fp32_b.SetBufferLen(kFullTileElements);
    fp32_result.SetBufferLen(kFullTileElements);

    AscendC::DataCopy(local_a, global_a, elements);
    AscendC::DataCopy(local_b, global_b, elements);
    AscendC::PipeBarrier<PIPE_MTE2>();

    AscendC::Cast(
        fp32_a, local_a, AscendC::RoundMode::CAST_NONE, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(
        fp32_b, local_b, AscendC::RoundMode::CAST_NONE, elements);
    AscendC::PipeBarrier<PIPE_V>();

    // Replace the binary node with a constant write; no Add/Mul payload is
    // executed by this mechanism probe.
    AscendC::Duplicate(fp32_result, 1.0f, elements);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Cast(
        local_a, fp32_result, AscendC::RoundMode::CAST_RINT, elements);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::DataCopy(global_destination, local_a, elements);
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t cycle_end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(worker_cycles);
    const int64_t result_offset = worker * kResultValuesPerWorker;
    cycles.SetValue(result_offset, cycle_start);
    cycles.SetValue(result_offset + 1, cycle_end);
    cycles.SetValue(result_offset + 2, cycle_end - cycle_start);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(
            cycles[result_offset]);
}
