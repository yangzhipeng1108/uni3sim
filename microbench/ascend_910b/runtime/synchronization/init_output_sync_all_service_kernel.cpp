// Independent Ascend 910B AIV output-initialization phase benchmark.
//
// This isolates the source-level sequence used by CANN FIA before its main
// task loop: each AIV initializes one disjoint BF16 output slice, completes
// the MTE3 operation, and enters one device-wide AIV SyncAll.  It contains no
// attention arithmetic, Q/K/V traffic, or production workload duration.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr int64_t kCycleSlotElements = 64 / sizeof(int64_t);

}  // namespace

extern "C" __global__ __aicore__ void init_output_sync_all_service(
        __gm__ bfloat16_t* gm_output, __gm__ int64_t* gm_cycles,
        int64_t elements_per_aiv) {
    AscendC::TPipe pipe;
    if ASCEND_IS_AIC {
        return;
    } else {
        const int64_t block_index = AscendC::GetBlockIdx();
        AscendC::GlobalTensor<bfloat16_t> output;
        output.SetGlobalBuffer(gm_output + block_index * elements_per_aiv);
        AscendC::GlobalTensor<int64_t> cycles;
        cycles.SetGlobalBuffer(gm_cycles + block_index * kCycleSlotElements);

        const int64_t start = AscendC::GetSystemCycle();
        constexpr uint32_t kOutputEventId = 0U;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kOutputEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kOutputEventId);
        matmul::InitOutput<bfloat16_t>(
            output, static_cast<uint64_t>(elements_per_aiv), 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kOutputEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kOutputEventId);
        AscendC::SyncAll();
        const int64_t end = AscendC::GetSystemCycle();

        cycles.SetValue(0, end - start);
        AscendC::DataCacheCleanAndInvalid<
            int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(cycles);
    }
}
