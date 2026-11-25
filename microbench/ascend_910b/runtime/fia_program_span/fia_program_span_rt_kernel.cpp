// CANN 8.5.1 FIA nonempty controller/program-span mechanism benchmark.
//
// The timed boundary is the source-required main-program boundary: before the
// three-entry RunInfo task cache is initialized through completion of the
// final valid Vec2 body.  It reproduces the exact CreateTask/ShouldExecute/
// ExecuteTask cache rotation and MM1->Vec1->MM2->Vec2 CrossCore order from
// fia_kernel_nonquant.h.  The bodies are deliberately tiny actor-local scalar
// work: they make every source branch nonempty without copying a production
// attention data body or consuming any production duration.
#include "kernel_operator.h"

#define FIA_PROGRAM_SPAN_CONTRACT_AICORE
#include "fia_program_span_contract.h"
#undef FIA_PROGRAM_SPAN_CONTRACT_AICORE

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr uint16_t kMm1Ready = 10;
constexpr uint16_t kVec1Ready = 11;
constexpr uint16_t kMm2Ready = 12;
constexpr uint8_t kInterCoreMode = 2;
constexpr int64_t kTaskCacheSize = 3;

struct RunInfo {
    uint64_t token;
    bool is_valid;
};

__aicore__ inline uint64_t NonemptyActorBody(
        uint64_t accumulator, const RunInfo& info, uint64_t stage) {
    // The result is stored after timing, preventing dead-code elimination.
    // These scalar operations are not a proxy for any production arithmetic.
    accumulator ^= info.token + stage * 0x9e3779b97f4a7c15ULL;
    accumulator = accumulator * 0xbf58476d1ce4e5b9ULL
        + 0x94d049bb133111ebULL;
    return accumulator ^ (accumulator >> 29);
}

__aicore__ inline bool ShouldExecuteTask(const RunInfo (&cache)[3]) {
    for (int64_t index = 0; index < kTaskCacheSize; ++index) {
        if (cache[index].is_valid) {
            return true;
        }
    }
    return false;
}

__aicore__ inline void StoreResult(
        AscendC::GlobalTensor<int64_t>& result_slot,
        int64_t cycles, uint64_t checksum) {
    result_slot.SetValue(0, cycles);
    result_slot.SetValue(1, static_cast<int64_t>(checksum));
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result_slot);
}

}  // namespace

extern "C" __global__ __aicore__ void fia_program_span_rt(
        __gm__ int64_t* gm_results, __gm__ int64_t* gm_task_counts) {
    const int64_t physical_block_index = AscendC::GetBlockIdx();
    int64_t aic_group = 0;
    int64_t actor_in_group = 0;
    if ASCEND_IS_AIC {
        aic_group = physical_block_index;
    } else {
        aic_group = physical_block_index
            / fia_program_span_contract::kAivPerAic;
        actor_in_group = 1 + physical_block_index
            % fia_program_span_contract::kAivPerAic;
    }
    const int64_t actor_index =
        aic_group * fia_program_span_contract::kActorsPerAicGroup
        + actor_in_group;
    AscendC::GlobalTensor<int64_t> result_slot;
    result_slot.SetGlobalBuffer(
        gm_results
        + actor_index * fia_program_span_contract::kCycleSlotElements);
    if (aic_group < 0
            || aic_group >= fia_program_span_contract::kMaximumAicGroups) {
        StoreResult(result_slot, -11, 0);
        return;
    }
    AscendC::GlobalTensor<int64_t> task_counts;
    task_counts.SetGlobalBuffer(gm_task_counts);
    const int64_t task_count = task_counts.GetValue(aic_group);
    if (!fia_program_span_contract::IsFrozenTaskCount(task_count)) {
        StoreResult(result_slot, -12, 0);
        return;
    }

    uint64_t accumulator = static_cast<uint64_t>(actor_index + 1);
    const int64_t cycle_start = AscendC::GetSystemCycle();
    for (int64_t repetition = 0;
            repetition < fia_program_span_contract::kProgramRepetitions;
            ++repetition) {
        RunInfo cache[3] = {};
        uint64_t created_task_count = 0;
        uint64_t executed_task_count = 0;
        bool should_dispatch_task = true;
        bool should_execute_task = false;
        while (should_dispatch_task || should_execute_task) {
            should_dispatch_task = created_task_count
                < static_cast<uint64_t>(task_count);
            if (should_dispatch_task) {
                RunInfo& current = cache[created_task_count % kTaskCacheSize];
                current.token = (
                    static_cast<uint64_t>(aic_group + 1) << 32)
                    ^ created_task_count ^ static_cast<uint64_t>(repetition);
                current.is_valid = true;
                ++created_task_count;
            }
            should_execute_task = ShouldExecuteTask(cache);
            if (should_execute_task) {
                RunInfo& current = cache[executed_task_count % kTaskCacheSize];
                RunInfo& previous = cache[
                    (executed_task_count + 2) % kTaskCacheSize];
                RunInfo& oldest = cache[
                    (executed_task_count + 1) % kTaskCacheSize];
                if (current.is_valid) {
                    if ASCEND_IS_AIC {
                        accumulator = NonemptyActorBody(
                            accumulator, current, 1);
                        AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_FIX>(
                            kMm1Ready);
                    }
                }
                if (previous.is_valid) {
                    if ASCEND_IS_AIV {
                        AscendC::CrossCoreWaitFlag(kMm1Ready);
                        accumulator = NonemptyActorBody(
                            accumulator, previous, 2);
                        AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_MTE3>(
                            kVec1Ready);
                    }
                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreWaitFlag(kVec1Ready);
                        accumulator = NonemptyActorBody(
                            accumulator, previous, 3);
                        AscendC::CrossCoreSetFlag<kInterCoreMode, PIPE_FIX>(
                            kMm2Ready);
                    }
                }
                if (oldest.is_valid) {
                    if ASCEND_IS_AIV {
                        AscendC::CrossCoreWaitFlag(kMm2Ready);
                        accumulator = NonemptyActorBody(
                            accumulator, oldest, 4);
                    }
                    oldest.is_valid = false;
                }
                ++executed_task_count;
            }
        }
    }
    const int64_t cycle_end = AscendC::GetSystemCycle();
    StoreResult(
        result_slot,
        (cycle_end - cycle_start)
            / fia_program_span_contract::kProgramRepetitions,
        accumulator);
}
