// Independent CANN 8.5.1 FIA AIC scalar control-floor benchmark.
//
// Preserves the 1-AIC:2-AIV actor topology and the source's per-descriptor L1
// slot lifetime, with the descriptor itself removed.  It contains no DataCopy,
// LoadData, Mmad, Fixpipe, Vector arithmetic or production memory traffic, so
// it cannot reproduce a production Attention duration and is not a partial FIA
// reimplementation.  It measures the control skeleton alone.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr int64_t kCycleSlotElements = 64 / sizeof(int64_t);
constexpr int64_t kMaximumGroups = 24;
constexpr int64_t kMaximumIters = 256;
constexpr int64_t kSlots = 2;
constexpr int64_t kMaximumGmReads = 512;
constexpr int64_t kMaximumHandoffs = 128;
constexpr uint16_t kFlagC2V = 6;
constexpr uint16_t kFlagV2C = 7;
constexpr uint8_t kFiaSyncMode2 = 2;
// Worker-disjoint probe region, strided past the 192 MiB shared L2 so a
// dependent read is not served from a warm line.
constexpr int64_t kProbeStrideElements = 4096;
// Shared-line mode walks one small cache-resident region, the order of the
// production tiling struct, that every AIC core reads.
constexpr int64_t kSharedElements = 512;
constexpr int64_t kProbeElementsPerWorker =
    kMaximumGmReads * kProbeStrideElements;

}  // namespace

extern "C" __global__ __aicore__ void aic_scalar_control_floor(
        __gm__ int64_t* gm_cycles, __gm__ int64_t* gm_probe, int64_t iters,
        int64_t active_aic_groups, int64_t gm_reads, int64_t handoffs,
        int64_t shared) {
    if (iters < 0 || iters > kMaximumIters || active_aic_groups <= 0
            || active_aic_groups > kMaximumGroups
            || gm_reads < 0 || gm_reads > kMaximumGmReads
            || handoffs < 0 || handoffs > kMaximumHandoffs
            || shared < 0 || shared > 1) {
        return;
    }
    const int64_t block = AscendC::GetBlockIdx();
    int64_t slot_index = block;
    if ASCEND_IS_AIC {
        slot_index = block;
    } else {
        slot_index = active_aic_groups + block;
    }
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(gm_cycles + slot_index * kCycleSlotElements);

    if ASCEND_IS_AIV {
        // The FIA-mandatory AIV prelude, then the mirror side of each
        // AIC-to-AIV round trip under the source's own sync mode.
        const int64_t aiv_begin = AscendC::GetSystemCycle();
        AscendC::SyncAll();
        for (int64_t round = 0; round < handoffs; ++round) {
            AscendC::CrossCoreWaitFlag(kFlagC2V);
            AscendC::CrossCoreSetFlag<kFiaSyncMode2, PIPE_MTE3>(kFlagV2C);
        }
        const int64_t aiv_end = AscendC::GetSystemCycle();
        cycles.SetValue(0, aiv_end - aiv_begin);
        cycles.SetValue(1, 1);
        AscendC::DataCacheCleanAndInvalid<
            int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(cycles);
        return;
    }

    for (int64_t slot = 0; slot < kSlots; ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(slot));
    }
    // Address arithmetic kept live through a checksum so the scalar work
    // cannot be eliminated.
    int64_t checksum = 0;
    int64_t slot = 0;
    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t index = 0; index < iters; ++index) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(slot));
        // One descriptor's source and destination address arithmetic.
        const int64_t source_offset = index * 256 * 1024 + block * 128;
        const int64_t destination_offset = slot * 256 * 128;
        checksum += source_offset - destination_offset;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            static_cast<uint32_t>(slot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
            static_cast<uint32_t>(slot));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(slot));
        slot = (slot + 1) % kSlots;
    }
    // Dependent serialized scalar GM reads: each index derives from the
    // previous returned value, so the chain cannot be reordered or elided.
    if (gm_reads > 0) {
        AscendC::GlobalTensor<int64_t> probe;
        if (shared == 1) {
            // Every core walks the same region: the production regime, where
            // all AIC cores read one tiling struct.
            probe.SetGlobalBuffer(gm_probe);
        } else {
            probe.SetGlobalBuffer(gm_probe + block * kProbeElementsPerWorker);
        }
        int64_t cursor = 0;
        for (int64_t index = 0; index < gm_reads; ++index) {
            cursor = probe.GetValue(cursor);
            checksum += cursor;
            if (shared == 1) {
                cursor = (cursor + 1) % kSharedElements;
            } else {
                cursor = (cursor + kProbeStrideElements)
                    % kProbeElementsPerWorker;
            }
        }
    }
    for (int64_t round = 0; round < handoffs; ++round) {
        AscendC::CrossCoreSetFlag<kFiaSyncMode2, PIPE_FIX>(kFlagC2V);
        AscendC::CrossCoreWaitFlag(kFlagV2C);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();
    for (int64_t index = 0; index < kSlots; ++index) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            static_cast<uint32_t>(index));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    cycles.SetValue(0, end - begin);
    cycles.SetValue(1, 2);
    // Recorded, not gated: writing the accumulator to GM is what keeps the
    // address arithmetic from being eliminated.
    cycles.SetValue(2, checksum);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(cycles);
}
