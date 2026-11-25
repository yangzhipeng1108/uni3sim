// Minimal non-empty 0 AIC : 1 AIV mixed task.  It isolates the CANN
// KERNEL_TYPE_MIX_AIV_1_0 task topology without implementing MoeInitRouting
// or consuming any production operator duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

namespace {

constexpr int64_t kValuesPerCacheLine = 8;

}  // namespace

extern "C" __global__ __aicore__ void nonempty_mix_aiv(
        __gm__ int64_t* validation) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> slot;
    slot.SetGlobalBuffer(validation + block_index * kValuesPerCacheLine);
    slot.SetValue(0, block_index + 1);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot);
}
