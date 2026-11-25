// Independent CANN 8.5.1 regular-FIA GQA drained ND2NZ descriptor probe.
//
// Scope: only the K/V GM-to-L1 ingress edge that the FIA source executes when
// `qkvDSameFlag == true`, where `CopyKToL1`/`CopyVToL1` is followed at once by
// its `WaitFlag<HardEvent::MTE2_MTE1>` and the descriptor program is therefore
// drained after every descriptor.  It omits Q, P, MMAD, Fixpipe, softmax, AIV
// work, output composition and cross-domain handoff, so it cannot reproduce a
// production Attention duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kMaximumWorkers = 24;
constexpr int64_t kMaximumRows = 256;
constexpr int64_t kMaximumStrideElements = 1024;
constexpr int64_t kMaximumCount = 128;
// Address span one descriptor walks: rows * source row stride.
// One descriptor walks rows * source row stride elements.  A non power of
// two skew is added so consecutive descriptors do not collide in the same L2
// sets; a 512 KiB power-of-two step thrashes a fixed set and measures
// associativity, not DRAM service.
constexpr int64_t kSkewElements = 8192;
constexpr int64_t kDescriptorSpanElements =
    kMaximumRows * kMaximumStrideElements + kSkewElements;
constexpr int64_t kPasses = 2;
constexpr int64_t kElementsPerWorker =
    kPasses * kMaximumCount * kDescriptorSpanElements;
constexpr int64_t kCycleSlotElements = 64 / sizeof(int64_t);
constexpr int64_t kL1Slots = 4;
constexpr int64_t kMaximumStreams = 2;

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

extern "C" __global__ __aicore__ void fia_gqa_drain_count_law(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_cycles,
        int64_t rows, int64_t columns, int64_t stride_elements,
        int64_t count, int64_t pass_index, int64_t streams) {
    const int64_t worker = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> cycle_slot;
    cycle_slot.SetGlobalBuffer(gm_cycles + worker * kCycleSlotElements);
    if (rows <= 0 || rows > kMaximumRows || columns <= 0
            || columns > kMaximumStrideElements
            || stride_elements < columns
            || stride_elements > kMaximumStrideElements
            || count <= 0 || count > kMaximumCount
            || pass_index < 0 || pass_index >= kPasses
            || streams < 1 || streams > kMaximumStreams
            || count % streams != 0
            || AscendC::GetBlockNum() > kMaximumWorkers) {
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
    source.SetGlobalBuffer(gm_source + worker * kElementsPerWorker);

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

    // Stream s owns slot pair {2s, 2s+1}.  Alternating streams lets MTE2 hold
    // one stream's descriptor in flight while MTE1 drains the other's, which
    // is the MM1/MM2 structure under test.  Stream 1 reads a disjoint region.
    int64_t slot = 0;
    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t index = 0; index < count; ++index) {
        const int64_t stream = index % streams;
        const int64_t step = index / streams;
        slot = 2 * stream + (step % 2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(slot));
        AscendC::DataCopy(
            l1[slot * kMaximumRows * 128],
            source[((pass_index * kMaximumCount)
                    + (stream * (kMaximumCount / kMaximumStreams)) + step)
                   * kDescriptorSpanElements],
            parameters);
        // The source's per-descriptor completion edge: signalled, waited and
        // consumed by MTE1 before the slot is released, so no descriptor
        // stays in flight.  The MTE1 consumer is what makes the edge real;
        // WaitFlag alone stalls only the MTE1 pipe.
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
