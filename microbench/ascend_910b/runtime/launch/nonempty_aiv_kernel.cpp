// Minimal non-empty AIV task for measuring the CANN device-task entry/exit
// boundary independently of any production operator.  Each physical AIV
// performs one scalar integer add and one cache-line-isolated validation store.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kValuesPerCacheLine = 8;

}  // namespace

extern "C" __global__ __aicore__ void nonempty_aiv(
        __gm__ int64_t* validation) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> slot;
    slot.SetGlobalBuffer(validation + block_index * kValuesPerCacheLine);
    slot.SetValue(0, block_index + 1);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot);
}
