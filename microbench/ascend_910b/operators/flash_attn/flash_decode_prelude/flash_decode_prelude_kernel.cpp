// Independent CANN FIA FlashDecode terminal-prelude benchmark.
//
// The production source executes these operations after its main attention
// program and before FlashDecode arithmetic.  This benchmark preserves the
// 1-AIC:2-AIV actor topology but contains no attention payload, tensor work,
// Vector arithmetic, or production memory traffic.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr int64_t kCycleSlotElements = 64 / sizeof(int64_t);
constexpr int64_t kSyncAllOnly = 0;
constexpr int64_t kIcacheAndSyncAll = 1;
constexpr int64_t kCompletePrelude = 2;

__aicore__ inline void InitializeFlashDecodeBuffers() {
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        AscendC::TBuf<> sum_buffer_0;
        AscendC::TBuf<> sum_buffer_1;
        AscendC::TBuf<> max_buffer_0;
        AscendC::TBuf<> max_buffer_1;
        AscendC::TBuf<> exp_buffer;
        AscendC::TBuf<> mm2_buffer_0;
        AscendC::TBuf<> mm2_buffer_1;
        AscendC::TBuf<> reduce_buffer;
        AscendC::TBuf<> output_buffer;
        AscendC::TBuf<> max_ub_buffer_0;
        AscendC::TBuf<> sum_ub_buffer_0;
        AscendC::TBuf<> max_ub_buffer_1;
        AscendC::TBuf<> sum_ub_buffer_1;
        AscendC::TBuf<> lse_ub_buffer;
        pipe.Reset();
        pipe.InitBuffer(sum_buffer_0, 6 * 1024);
        pipe.InitBuffer(sum_buffer_1, 6 * 1024);
        pipe.InitBuffer(max_buffer_0, 6 * 1024);
        pipe.InitBuffer(max_buffer_1, 6 * 1024);
        pipe.InitBuffer(exp_buffer, 6 * 1024);
        pipe.InitBuffer(mm2_buffer_0, 16 * 1024);
        pipe.InitBuffer(mm2_buffer_1, 16 * 1024);
        pipe.InitBuffer(reduce_buffer, 16 * 1024);
        pipe.InitBuffer(output_buffer, 16 * 1024);
        pipe.InitBuffer(max_ub_buffer_0, 256);
        pipe.InitBuffer(sum_ub_buffer_0, 256);
        pipe.InitBuffer(max_ub_buffer_1, 256);
        pipe.InitBuffer(sum_ub_buffer_1, 256);
        pipe.InitBuffer(lse_ub_buffer, 256);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void flash_decode_prelude(
        __gm__ int64_t* gm_cycles, int64_t mode,
        int64_t active_aic_groups) {
    if (mode < kSyncAllOnly || mode > kCompletePrelude
            || active_aic_groups <= 0 || active_aic_groups > 24) {
        return;
    }
    const int64_t block = AscendC::GetBlockIdx();
    int64_t slot = 0;
    if ASCEND_IS_AIC {
        slot = block;
    } else {
        slot = active_aic_groups + block;
    }
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(gm_cycles + slot * kCycleSlotElements);

    const int64_t begin = AscendC::GetSystemCycle();
    if (mode == kCompletePrelude) {
        InitializeFlashDecodeBuffers();
    }
    if (mode != kSyncAllOnly) {
        AscendC::ICachePreLoad(2);
    }
    AscendC::SyncAll();
    const int64_t end = AscendC::GetSystemCycle();

    cycles.SetValue(0, end - begin);
    cycles.SetValue(1, mode + 1);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(cycles);
}
