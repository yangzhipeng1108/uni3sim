// Independent Ascend 910B CANN NBuffer33 L1-to-L0 pair benchmark.
//
// This is deliberately not a GEMM.  It isolates the exact CANN 8.5.1
// deterministic-split-K source primitive for one 128x128x128 BF16 base tile:
// non-transposed A1->A2 and B1->B2 LoadData3D V2Pro calls, their MTE1->Cube
// completion edge, and the source's three-A/two-B cyclic buffer ownership.
// GM->L1 population occurs before timing.  The timed interval contains no
// Cube, FixPipe, workspace, output, or production-operator duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kBaseM = 128;
constexpr int64_t kBaseN = 128;
constexpr int64_t kBaseK = 128;
constexpr int64_t kA1SlotCount = 3;
constexpr int64_t kB1SlotCount = 2;
constexpr int64_t kL0SlotCount = 2;
constexpr int64_t kATileElements = kBaseM * kBaseK;
constexpr int64_t kBTileElements = kBaseK * kBaseN;
constexpr uint8_t kPadList[] = {0, 0, 0, 0};
constexpr uint64_t kMPosBit = 48;
constexpr uint64_t kKPosBit = 32;
constexpr uint64_t kMStepBit = 16;

__aicore__ inline int64_t Align16(int64_t value) {
    return (value + 15) / 16 * 16;
}

__aicore__ inline void CopyNd2Nz(
        AscendC::LocalTensor<bfloat16_t>& destination,
        AscendC::GlobalTensor<bfloat16_t> source, int64_t rows,
        int64_t columns) {
    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = static_cast<uint16_t>(rows);
    parameters.dValue = static_cast<uint16_t>(columns);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(columns);
    parameters.dstNzC0Stride = static_cast<uint16_t>(Align16(rows));
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

__aicore__ inline void PrepareNBuffer33LoadData() {
    // These are the non-transposed CFG_MDL 128x128x128 source preparations:
    // LoadToA2::Prepare then LoadToB2::Prepare.  Both have 128-wide source
    // axes; B uses fMatrixCtrl in the following V2Pro LoadData instruction.
    AscendC::Load3DSetFMatrixCal(1, kBaseM, kPadList);
    AscendC::Load3DSetFMatrixBCal(1, kBaseK, kPadList);
}

__aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    AscendC::LoadData3DParamsV2Pro parameters;
    parameters.channelSize = kBaseK;
    parameters.extConfig = (static_cast<uint64_t>(kBaseM) << kMStepBit)
        | static_cast<uint64_t>(kBaseK);
    AscendC::LoadData(destination, source, parameters);
}

__aicore__ inline void LoadB(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    AscendC::LoadData3DParamsV2Pro parameters;
    parameters.channelSize = kBaseN;
    parameters.extConfig = (static_cast<uint64_t>(kBaseK) << kMStepBit)
        | static_cast<uint64_t>(kBaseN);
    parameters.fMatrixCtrl = true;
    AscendC::LoadData(destination, source, parameters);
}

__aicore__ inline void RunPair(
        AscendC::LocalTensor<bfloat16_t>& l0a,
        AscendC::LocalTensor<bfloat16_t>& l0b,
        const AscendC::LocalTensor<bfloat16_t>& l1a,
        const AscendC::LocalTensor<bfloat16_t>& l1b,
        int64_t l0_slot) {
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0_slot);
    LoadA(l0a, l1a);
    LoadB(l0b, l1b);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0_slot);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0_slot);
    // In the full source program, this is the MTE1 completion at which the
    // associated A1/B1 producer slot becomes refillable.  No MTE2 producer
    // exists in this isolated interval, so emitting MTE1_MTE2 without its
    // matching source consumer would fill the hardware event queue and alter
    // the service being measured.
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0_slot);
}

}  // namespace

extern "C" __global__ __aicore__ void aic_n_buffer33_mte1_pair(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_cycles,
        int64_t iterations) {
    const int64_t block = AscendC::GetBlockIdx();
    if (iterations <= 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a1_buffers[kA1SlotCount];
    AscendC::TBuf<AscendC::TPosition::B1> b1_buffers[kB1SlotCount];
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buffers[kL0SlotCount];
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buffers[kL0SlotCount];
    for (int64_t slot = 0; slot < kA1SlotCount; ++slot) {
        pipe.InitBuffer(a1_buffers[slot], kATileElements * sizeof(bfloat16_t));
    }
    for (int64_t slot = 0; slot < kB1SlotCount; ++slot) {
        pipe.InitBuffer(b1_buffers[slot], kBTileElements * sizeof(bfloat16_t));
    }
    for (int64_t slot = 0; slot < kL0SlotCount; ++slot) {
        pipe.InitBuffer(l0a_buffers[slot], kATileElements * sizeof(bfloat16_t));
        pipe.InitBuffer(l0b_buffers[slot], kBTileElements * sizeof(bfloat16_t));
    }

    AscendC::LocalTensor<bfloat16_t> a1[kA1SlotCount];
    AscendC::LocalTensor<bfloat16_t> b1[kB1SlotCount];
    AscendC::LocalTensor<bfloat16_t> l0a[kL0SlotCount];
    AscendC::LocalTensor<bfloat16_t> l0b[kL0SlotCount];
    for (int64_t slot = 0; slot < kA1SlotCount; ++slot) {
        a1[slot] = a1_buffers[slot].Get<bfloat16_t>();
    }
    for (int64_t slot = 0; slot < kB1SlotCount; ++slot) {
        b1[slot] = b1_buffers[slot].Get<bfloat16_t>();
    }
    for (int64_t slot = 0; slot < kL0SlotCount; ++slot) {
        l0a[slot] = l0a_buffers[slot].Get<bfloat16_t>();
        l0b[slot] = l0b_buffers[slot].Get<bfloat16_t>();
    }

    AscendC::GlobalTensor<bfloat16_t> source;
    source.SetGlobalBuffer(gm_source);
    for (int64_t slot = 0; slot < kA1SlotCount; ++slot) {
        CopyNd2Nz(a1[slot], source[slot * kATileElements], kBaseM, kBaseK);
    }
    const int64_t b_source_offset = kA1SlotCount * kATileElements;
    for (int64_t slot = 0; slot < kB1SlotCount; ++slot) {
        CopyNd2Nz(
            b1[slot], source[b_source_offset + slot * kBTileElements],
            kBaseK, kBaseN);
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    PrepareNBuffer33LoadData();
    for (int64_t slot = 0; slot < kL0SlotCount; ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(slot);
    }

    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const int64_t a_slot = iteration % kA1SlotCount;
        const int64_t b_slot = iteration % kB1SlotCount;
        const int64_t l0_slot = iteration % kL0SlotCount;
        RunPair(l0a[l0_slot], l0b[l0_slot], a1[a_slot], b1[b_slot], l0_slot);
    }
    const int64_t end = AscendC::GetSystemCycle();
    for (int64_t slot = 0; slot < kL0SlotCount; ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(slot);
    }

    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(
        gm_cycles + block * kCycleSlotBytes / sizeof(int64_t));
    cycles.SetValue(0, end - begin);
    cycles.SetValue(1, 1);
}
