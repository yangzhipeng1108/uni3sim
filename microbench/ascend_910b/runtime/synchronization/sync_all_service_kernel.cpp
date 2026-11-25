// Independent Ascend 910B AIV SyncAll completion benchmark.
//
// The timed region contains exactly one device-wide AIV synchronization and
// no production arithmetic or memory traffic.  Each participant records its
// own elapsed system-counter cycles after the barrier into a separate cache
// line; the report uses the slowest participant as the complete barrier span.
// The 1 AIC : 2 AIV task type deliberately matches production FIA.  AIC
// actors return without entering the timed region; SyncAll covers only the
// two AIV actors belonging to every launched logical AIC group.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr int64_t kCycleSlotElements = 64 / sizeof(int64_t);

}  // namespace

extern "C" __global__ __aicore__ void sync_all_service(
        __gm__ int64_t* gm_cycles) {
    if ASCEND_IS_AIC {
        return;
    } else {
        const int64_t block_index = AscendC::GetBlockIdx();
        AscendC::GlobalTensor<int64_t> cycles;
        cycles.SetGlobalBuffer(gm_cycles + block_index * kCycleSlotElements);

        const int64_t start = AscendC::GetSystemCycle();
        AscendC::SyncAll();
        const int64_t end = AscendC::GetSystemCycle();

        cycles.SetValue(0, end - start);
        AscendC::DataCacheCleanAndInvalid<
            int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(cycles);
    }
}
