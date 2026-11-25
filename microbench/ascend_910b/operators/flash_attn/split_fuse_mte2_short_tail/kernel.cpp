// Independent BF16 paged-KV MTE2-to-MTE1 finite stream, CANN 8.5.1 geometry.
// It has no MMAD, AIV, softmax or Attention output program.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {
constexpr int64_t kHeads = 8;
constexpr int64_t kRows = 128;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kStride = kHeads * kHeadDim;
constexpr int64_t kPageElements = kRows * kStride;
constexpr int64_t kL1Slots = 2;
constexpr int64_t kCycleSlotElements = 64 / sizeof(int64_t);

__aicore__ inline void StoreResult(
        AscendC::GlobalTensor<int64_t>& destination,
        int64_t cycles, int64_t valid) {
    destination.SetValue(0, cycles);
    destination.SetValue(1, valid);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(destination);
}

__aicore__ inline void Consume(
        AscendC::LocalTensor<bfloat16_t>& l1,
        AscendC::LocalTensor<bfloat16_t>& l0b, int64_t slot) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
        static_cast<uint32_t>(slot));
    AscendC::LoadData2DParams parameters;
    parameters.startIndex = 0;
    parameters.repeatTimes = 1;
    parameters.srcStride = 1;
    parameters.dstGap = 0;
    parameters.ifTranspose = false;
    AscendC::LoadData(
        l0b, l1[slot * kRows * kHeadDim], parameters);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
        static_cast<uint32_t>(slot));
}
}  // namespace

extern "C" __global__ __aicore__ void split_fuse_mte2_interleaved(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_cycles,
        int64_t active_groups, int64_t descriptor_count, int64_t issue_mode,
        int64_t base_page) {
    const int64_t worker = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> cycle_slot;
    cycle_slot.SetGlobalBuffer(gm_cycles + worker * kCycleSlotElements);
    if (AscendC::GetBlockNum() != active_groups
            || (active_groups != 24)
            || (descriptor_count != 3 && descriptor_count != 4
                && descriptor_count != 5 && descriptor_count != 7)
            || (issue_mode != 0 && issue_mode != 1)
            || base_page < 0) {
        StoreResult(cycle_slot, -1, 0);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::B1> l1_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buffer;
    pipe.InitBuffer(
        l1_buffer,
        kL1Slots * kRows * kHeadDim * sizeof(bfloat16_t));
    pipe.InitBuffer(l0b_buffer, 16 * 16 * sizeof(bfloat16_t));
    auto l1 = l1_buffer.Get<bfloat16_t>();
    auto l0b = l0b_buffer.Get<bfloat16_t>();
    AscendC::GlobalTensor<bfloat16_t> source;
    source.SetGlobalBuffer(gm_source);

    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = static_cast<uint16_t>(kRows);
    parameters.dValue = static_cast<uint16_t>(kHeadDim);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(kStride);
    parameters.dstNzC0Stride = static_cast<uint16_t>(kRows);
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;

    for (int64_t slot = 0; slot < kL1Slots; ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(slot));
    }
    const int64_t request = worker / kHeads;
    const int64_t head = worker % kHeads;
    const int64_t requests = active_groups / kHeads;
    const int64_t begin = AscendC::GetSystemCycle();
    if (issue_mode == 0) {
        for (int64_t ordinal = 0; ordinal < descriptor_count; ++ordinal) {
            const int64_t slot = ordinal % kL1Slots;
            const int64_t page = ordinal * requests + request;
            const int64_t source_offset = (base_page + page) * kPageElements
                + head * kHeadDim;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                static_cast<uint32_t>(slot));
            AscendC::DataCopy(
                l1[slot * kRows * kHeadDim], source[source_offset],
                parameters);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
                static_cast<uint32_t>(slot));
            Consume(l1, l0b, slot);
        }
    } else {
        for (int64_t ordinal = 0; ordinal < descriptor_count; ++ordinal) {
            const int64_t slot = ordinal % kL1Slots;
            const int64_t page = ordinal * requests + request;
            const int64_t source_offset = (base_page + page) * kPageElements
                + head * kHeadDim;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                static_cast<uint32_t>(slot));
            AscendC::DataCopy(
                l1[slot * kRows * kHeadDim], source[source_offset],
                parameters);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
                static_cast<uint32_t>(slot));
            if (ordinal > 0) {
                Consume(l1, l0b, (ordinal - 1) % kL1Slots);
            }
        }
        Consume(l1, l0b, (descriptor_count - 1) % kL1Slots);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();
    for (int64_t slot = 0; slot < kL1Slots; ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(slot));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    StoreResult(cycle_slot, end - begin, 1);
}
