// Independent Ascend 910B AIC<->AIV synchronization-pipeline benchmark.
//
// The timed region contains no production arithmetic and no tensor traffic.
// It reproduces only the three CrossCore handoffs and three-task ring order
// used by the fixed CANN 8.5.1 FIA source:
//
//   AIC: MM1-ready -> wait Vec1-ready -> MM2-ready
//   AIV: wait MM1-ready -> Vec1-ready -> wait MM2-ready
//
// One logical AIC group has two physical AIV actors, matching the production
// KERNEL_TYPE_MIX_AIC_1_2 contract.  Device-cycle results are written to
// separate cache lines after the timed region so they cannot create a timed
// GM service or cross-actor false sharing.
#include "kernel_operator.h"

#define CROSS_DOMAIN_CONTRACT_AICORE
#include "cross_domain_contract.h"
#undef CROSS_DOMAIN_CONTRACT_AICORE

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr uint16_t kMm1Ready = 10;
constexpr uint16_t kVec1Ready = 11;
constexpr uint16_t kMm2Ready = 12;
constexpr uint8_t kInterCoreMode = 2;

__aicore__ inline void StoreCycle(
        AscendC::GlobalTensor<int64_t>& cycle_slot, int64_t value) {
    cycle_slot.SetValue(0, value);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(cycle_slot);
}

}  // namespace

extern "C" __global__ __aicore__ void cross_domain_rt(
        __gm__ int64_t* gm_cycles, __gm__ int64_t* gm_task_counts) {
    const int64_t physical_block_index = AscendC::GetBlockIdx();
    int64_t aic_group = 0;
    int64_t actor_in_group = 0;
    if ASCEND_IS_AIC {
        aic_group = physical_block_index;
        actor_in_group = 0;
    } else {
        aic_group = physical_block_index / cross_domain_contract::kAivPerAic;
        actor_in_group = 1
            + physical_block_index % cross_domain_contract::kAivPerAic;
    }
    const int64_t actor_index =
        aic_group * cross_domain_contract::kActorsPerAicGroup
        + actor_in_group;
    AscendC::GlobalTensor<int64_t> cycle_slot;
    cycle_slot.SetGlobalBuffer(
        gm_cycles
        + actor_index * cross_domain_contract::kCycleSlotElements);
    if (aic_group < 0
            || aic_group >= cross_domain_contract::kMaximumAicBlocks) {
        StoreCycle(cycle_slot, -11);
        return;
    }
    AscendC::GlobalTensor<int64_t> task_counts;
    task_counts.SetGlobalBuffer(gm_task_counts);
    const int64_t task_count = task_counts.GetValue(aic_group);
    if (!cross_domain_contract::IsFrozenTaskCount(task_count)) {
        StoreCycle(cycle_slot, -12);
        return;
    }

    const int64_t cycle_start = AscendC::GetSystemCycle();
    // This is the exact empty three-stage ring around FIA's three-entry task
    // cache.  At tick t, AIC starts task t then advances task t-1 to MM2;
    // each AIV advances task t-1 through Vec1 and task t-2 through Vec2.
    #pragma unroll 1
    for (int64_t tick = 0; tick < task_count + 2; ++tick) {
        if ASCEND_IS_AIC {
            if (tick < task_count) {
                AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_FIX>(
                    kMm1Ready);
            }
            if (tick >= 1 && tick <= task_count) {
                AscendC::CrossCoreWaitFlag(kVec1Ready);
                AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_FIX>(
                    kMm2Ready);
            }
        } else {
            if (tick >= 1 && tick <= task_count) {
                AscendC::CrossCoreWaitFlag(kMm1Ready);
                AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_MTE3>(
                    kVec1Ready);
            }
            if (tick >= 2 && tick <= task_count + 1) {
                AscendC::CrossCoreWaitFlag(kMm2Ready);
            }
        }
    }
    const int64_t cycle_end = AscendC::GetSystemCycle();
    StoreCycle(cycle_slot, cycle_end - cycle_start);
}
