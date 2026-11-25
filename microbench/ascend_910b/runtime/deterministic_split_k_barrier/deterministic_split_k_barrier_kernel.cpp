// Independent CANN 8.5.1 deterministic Split-K control-lifecycle probe.
//
// This is deliberately not a GEMM: it contains no operand, Cube, FixPipe,
// workspace, vector reduction, or output traffic.  It only reproduces the
// L2-split source's alternating AIC -> paired-AIV and paired-AIV -> AIC flag
// order, its AIV and AIC SyncAll calls, and its PIPE_ALL boundaries.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr int64_t kAicGroups = 24;
constexpr int64_t kAivPerGroup = 2;
constexpr int64_t kActorsPerGroup = 3;
constexpr int64_t kSlotElements = 8;
constexpr uint16_t kAicToAivFlagBase = 200;
constexpr uint16_t kAivToAicFlagBase = 202;
constexpr uint8_t kInterCoreMode = 0x2;

__aicore__ inline void Store(
        AscendC::GlobalTensor<int64_t>& result, const int64_t values[8]) {
    for (int64_t index = 0; index < kSlotElements; ++index) {
        result.SetValue(index, values[index]);
    }
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}

}  // namespace

extern "C" __global__ __aicore__ void deterministic_split_k_barrier(
        __gm__ int64_t* gm_result, int64_t slab_count, int64_t mode) {
    const int64_t physical = AscendC::GetBlockIdx();
    int64_t group = 0;
    int64_t actor = 0;
    if ASCEND_IS_AIC {
        group = physical;
    } else {
        group = physical / kAivPerGroup;
        actor = 1 + physical % kAivPerGroup;
    }
    const int64_t slot_index = group * kActorsPerGroup + actor;
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(gm_result + slot_index * kSlotElements);
    if (group < 0 || group >= kAicGroups || slab_count < 1 || slab_count > 2
            || (mode != 0 && mode != 1)) {
        const int64_t invalid[8] = {-1, 0, 0, 0, 0, 0, 0, 0};
        Store(result, invalid);
        return;
    }

    // mode=0 is the source-matched collective lifecycle.  mode=1 retains
    // the same CrossCore flag ordering and pipe boundaries but excludes only
    // SyncAll, isolating the all-participant collective contribution.
    const bool source_collective = mode == 0;
    int64_t aic_reverse_wait_cycles = 0;
    int64_t aic_sync_cycles = 0;
    int64_t aiv_first_sync_cycles = 0;
    int64_t aiv_second_sync_cycles = 0;
    const int64_t start = AscendC::GetSystemCycle();
    for (int64_t slab = 0; slab < slab_count; ++slab) {
        const uint16_t pingpong = static_cast<uint16_t>(slab & 1);
        if ASCEND_IS_AIC {
            if (slab > 0) {
                const int64_t wait_start = AscendC::GetSystemCycle();
                AscendC::CrossCoreWaitFlag(kAivToAicFlagBase + pingpong);
                const int64_t wait_end = AscendC::GetSystemCycle();
                aic_reverse_wait_cycles += wait_end - wait_start;
                if (source_collective) {
                    const int64_t sync_start = AscendC::GetSystemCycle();
                    AscendC::SyncAll();
                    aic_sync_cycles += AscendC::GetSystemCycle() - sync_start;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_FIX>(
                kAicToAivFlagBase + pingpong);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            AscendC::CrossCoreWaitFlag(kAicToAivFlagBase + pingpong);
            if (source_collective) {
                const int64_t sync_start = AscendC::GetSystemCycle();
                AscendC::SyncAll();
                const int64_t sync_cycles = AscendC::GetSystemCycle() - sync_start;
                if (slab == 0) {
                    aiv_first_sync_cycles = sync_cycles;
                } else {
                    aiv_second_sync_cycles = sync_cycles;
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if (slab + 1 < slab_count) {
                AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_MTE3>(
                    kAivToAicFlagBase + static_cast<uint16_t>((slab + 1) & 1));
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }
    const int64_t values[8] = {
        AscendC::GetSystemCycle() - start,
        aic_reverse_wait_cycles,
        aic_sync_cycles,
        aiv_first_sync_cycles,
        aiv_second_sync_cycles,
        slab_count,
        mode,
        1,
    };
    Store(result, values);
}
