// Independent CANN 8.5.1 probe: drained ND2NZ descriptor service as a function
// of AIC concurrency under two source-address layouts.
//
// The AIC descriptor program, its slot lifetime and its timing boundary are
// character-identical to the published `fia_gqa_drain_count_law` probe; the
// only added degree of freedom is where each worker's descriptors are read
// from.  `address_mode = 0` keeps that probe's worker-disjoint, set-skewed
// layout and must reproduce its published service.  `address_mode = 1` packs
// every worker's descriptors into one shared, densely strided region, which is
// how a KV tensor is laid out when the launch splits it across cores.
//
// It omits Q, P, MMAD, Fixpipe, softmax, AIV work, output composition and
// cross-domain handoff, so it cannot reproduce a production Attention
// duration, and it measures no production or evaluation duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kMaximumWorkers = 24;
constexpr int64_t kMaximumRows = 256;
constexpr int64_t kMaximumStrideElements = 1024;
constexpr int64_t kMaximumCount = 128;
// Disjoint layout: one descriptor walks rows * source row stride elements, and
// a non power of two skew keeps consecutive descriptors out of the same L2
// sets, so the service measured there is DRAM service rather than
// associativity.  This is the published probe's layout.
constexpr int64_t kSkewElements = 8192;
constexpr int64_t kDescriptorSpanElements =
    kMaximumRows * kMaximumStrideElements + kSkewElements;
constexpr int64_t kPasses = 2;
constexpr int64_t kElementsPerWorker =
    kPasses * kMaximumCount * kDescriptorSpanElements;
// Packed layout: descriptors tile one shared region back to back, with no skew
// and no per-worker partition, and consecutive workers take consecutive tiles.
constexpr int64_t kPackedTileElements = kMaximumRows * kMaximumStrideElements;
constexpr int64_t kCycleSlotElements = 64 / sizeof(int64_t);
constexpr int64_t kL1Slots = 2;

__aicore__ inline void StoreResult(
        AscendC::GlobalTensor<int64_t>& destination,
        int64_t cycles, int64_t valid) {
    destination.SetValue(0, cycles);
    destination.SetValue(1, valid);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(destination);
}

}  // namespace

extern "C" __global__ __aicore__ void fia_mte2_concurrency_locality(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_cycles,
        int64_t rows, int64_t columns, int64_t stride_elements,
        int64_t count, int64_t pass_index, int64_t address_mode,
        int64_t launch_offset) {
    const int64_t worker = AscendC::GetBlockIdx();
    const int64_t workers = AscendC::GetBlockNum();
    AscendC::GlobalTensor<int64_t> cycle_slot;
    cycle_slot.SetGlobalBuffer(gm_cycles + worker * kCycleSlotElements);
    if (rows <= 0 || rows > kMaximumRows || columns <= 0
            || columns > kMaximumStrideElements
            || stride_elements < columns
            || stride_elements > kMaximumStrideElements
            || count <= 0 || count > kMaximumCount
            || pass_index < 0 || pass_index >= kPasses
            || address_mode < 0 || address_mode > 1
            || launch_offset < 0
            || (launch_offset + 1) * count > kMaximumCount
            || workers > kMaximumWorkers) {
        StoreResult(cycle_slot, -1, 0);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::B1> l1_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buffer;
    pipe.InitBuffer(
        l1_buffer,
        kL1Slots * kMaximumRows * 128 * sizeof(bfloat16_t));
    pipe.InitBuffer(l0b_buffer, 16 * 16 * sizeof(bfloat16_t));
    auto l1 = l1_buffer.Get<bfloat16_t>();
    auto l0b = l0b_buffer.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> source;
    source.SetGlobalBuffer(gm_source);

    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = static_cast<uint16_t>(rows);
    parameters.dValue = static_cast<uint16_t>(columns);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(stride_elements);
    parameters.dstNzC0Stride = static_cast<uint16_t>(rows);
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;

    for (int64_t slot = 0; slot < kL1Slots; ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(slot));
    }

    // Launch l reads the descriptors at [l * count, (l + 1) * count) of the
    // pass, so repeated timed launches in one process never revisit an address
    // and each stays DRAM-cold in the same sense as the first.
    const int64_t first = launch_offset * count;
    const int64_t disjoint_base =
        worker * kElementsPerWorker + pass_index * kMaximumCount
                                          * kDescriptorSpanElements;
    // Packed: tile index (descriptor * workers + worker) puts the concurrently
    // issued descriptors of one step next to each other, which is what a KV
    // tensor split across cores presents to the memory system.
    const int64_t packed_base =
        pass_index * kMaximumCount * kMaximumWorkers * kPackedTileElements;

    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t index = 0; index < count; ++index) {
        const int64_t slot = index % kL1Slots;
        const int64_t offset = address_mode == 0
            ? disjoint_base + (first + index) * kDescriptorSpanElements
            : packed_base
                  + ((first + index) * workers + worker) * kPackedTileElements;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(slot));
        AscendC::DataCopy(
            l1[slot * kMaximumRows * 128], source[offset], parameters);
        // The source's per-descriptor completion edge: signalled, waited and
        // consumed by MTE1 before the slot is released, so no descriptor stays
        // in flight.  The MTE1 consumer is what makes the edge real; WaitFlag
        // alone stalls only the MTE1 pipe.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            static_cast<uint32_t>(slot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
            static_cast<uint32_t>(slot));
        AscendC::LoadData2DParams consume;
        consume.startIndex = 0;
        consume.repeatTimes = 1;
        consume.srcStride = 1;
        consume.dstGap = 0;
        consume.ifTranspose = false;
        AscendC::LoadData(l0b, l1[slot * kMaximumRows * 128], consume);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(slot));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();
    for (int64_t index = 0; index < kL1Slots; ++index) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(index));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    StoreResult(cycle_slot, end - begin, 1);
}
