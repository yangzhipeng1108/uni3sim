// Independent CANN 8.5.1 FIA AIC/AIV handoff-edge benchmark.
//
// The three source-ordered CrossCore edges are executed without production
// arithmetic or tensor traffic.  GetSystemCycle is a device-global 50 MHz
// counter, so sums of receiver-minus-producer timestamps retain each edge's
// completion latency without timed GM communication between actors.
#include "kernel_operator.h"

#define FIA_EDGE_HANDOFF_CONTRACT_AICORE
#include "fia_edge_handoff_contract.h"
#undef FIA_EDGE_HANDOFF_CONTRACT_AICORE

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr uint16_t kMm1Ready = 10;
constexpr uint16_t kVec1Ready = 11;
constexpr uint16_t kMm2Ready = 12;
constexpr uint8_t kInterCoreMode = 2;

__aicore__ inline void StoreResults(
        AscendC::GlobalTensor<int64_t>& slot,
        AscendC::GlobalTensor<int64_t>& tail, int64_t elapsed,
        int64_t first, int64_t second, int64_t third,
        const int64_t (&prefixes)[9]) {
    slot.SetValue(0, elapsed);
    slot.SetValue(1, first);
    slot.SetValue(2, second);
    slot.SetValue(3, third);
    for (int64_t index = 0; index < 4; ++index) {
        slot.SetValue(4 + index, prefixes[index]);
    }
    for (int64_t index = 4; index < 9; ++index) {
        tail.SetValue(index - 4, prefixes[index]);
    }
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(tail);
}

}  // namespace

extern "C" __global__ __aicore__ void fia_edge_handoff_rt(
        __gm__ int64_t* gm_results, int64_t task_count) {
    const int64_t physical_block_index = AscendC::GetBlockIdx();
    int64_t aic_group = 0;
    int64_t actor_in_group = 0;
    if ASCEND_IS_AIC {
        aic_group = physical_block_index;
    } else {
        aic_group = physical_block_index
            / fia_edge_handoff_contract::kAivPerAic;
        actor_in_group = 1 + physical_block_index
            % fia_edge_handoff_contract::kAivPerAic;
    }
    const int64_t actor_index =
        aic_group * fia_edge_handoff_contract::kActorsPerAicGroup
        + actor_in_group;
    AscendC::GlobalTensor<int64_t> result_slot;
    result_slot.SetGlobalBuffer(
        gm_results
        + actor_index * fia_edge_handoff_contract::kResultSlotElements);
    AscendC::GlobalTensor<int64_t> result_tail;
    result_tail.SetGlobalBuffer(
        gm_results
        + actor_index * fia_edge_handoff_contract::kResultSlotElements + 8);
    if (aic_group < 0
            || aic_group >= fia_edge_handoff_contract::kMaximumAicGroups) {
        const int64_t empty[9] = {};
        StoreResults(result_slot, result_tail, -11, 0, 0, 0, empty);
        return;
    }
    if (!fia_edge_handoff_contract::IsFrozenTaskCount(task_count)) {
        const int64_t empty[9] = {};
        StoreResults(result_slot, result_tail, -12, 0, 0, 0, empty);
        return;
    }

    int64_t first_sum = 0;
    int64_t second_sum = 0;
    int64_t third_sum = 0;
    int64_t prefixes[9] = {};
    const int64_t cycle_start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t task = 0; task < task_count; ++task) {
        if ASCEND_IS_AIC {
            first_sum += AscendC::GetSystemCycle();
            AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_FIX>(kMm1Ready);
            AscendC::CrossCoreWaitFlag(kVec1Ready);
            second_sum += AscendC::GetSystemCycle();
            third_sum += AscendC::GetSystemCycle();
            AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_FIX>(kMm2Ready);
        } else {
            AscendC::CrossCoreWaitFlag(kMm1Ready);
            first_sum += AscendC::GetSystemCycle();
            second_sum += AscendC::GetSystemCycle();
            AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_MTE3>(kVec1Ready);
            AscendC::CrossCoreWaitFlag(kMm2Ready);
            third_sum += AscendC::GetSystemCycle();
        }
        if (task < 3) {
            const int64_t base = 3 * task;
            prefixes[base] = first_sum;
            prefixes[base + 1] = second_sum;
            prefixes[base + 2] = third_sum;
        }
    }
    const int64_t cycle_end = AscendC::GetSystemCycle();
    StoreResults(
        result_slot, result_tail, cycle_end - cycle_start,
        first_sum, second_sum, third_sum, prefixes);
}
