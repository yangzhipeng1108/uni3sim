// Record the global SYS_CNT value observed at entry by every physical AIV.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kValuesPerCacheLine = 8;

}  // namespace

extern "C" __global__ __aicore__ void aiv_worker_start(
        __gm__ int64_t* timestamps) {
    const int64_t block_index = AscendC::GetBlockIdx();
    const int64_t start = AscendC::GetSystemCycle();
    const int64_t end = AscendC::GetSystemCycle();
    AscendC::GlobalTensor<int64_t> slot;
    slot.SetGlobalBuffer(timestamps + block_index * kValuesPerCacheLine);
    slot.SetValue(0, start);
    slot.SetValue(1, end);
    slot.SetValue(2, block_index + 1);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot);
}
