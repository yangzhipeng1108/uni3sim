// Isolate the two CANN AIV TQue initializations used by GatherV3 key 10330.
//
// The loop repeats only the source-identical TPipe lifetime so that the
// 20 ns SYS_CNT timer can resolve it.  It contains no queue access, data
// transfer, index decode, GatherV3 computation, or production operator.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;
constexpr uint32_t kBufferDepth = 2;
constexpr uint32_t kDataBufferBytes = 512;
constexpr uint32_t kIndexBufferBytes = 8;

}  // namespace

extern "C" __global__ __aicore__ void single_tbuf_lifecycle(
        __gm__ int64_t* output, int64_t payload_bytes, int64_t iterations) {
    const int64_t block_index = AscendC::GetBlockIdx();
    if (payload_bytes <= 0 || iterations <= 0) {
        return;
    }
    const int64_t cycle_start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::TPipe pipe;
        AscendC::TQueBind<
            AscendC::QuePosition::VECIN,
            AscendC::QuePosition::VECOUT,
            1> data_queue;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> index_queue;
        pipe.InitBuffer(data_queue, kBufferDepth, kDataBufferBytes);
        pipe.InitBuffer(index_queue, kBufferDepth, kIndexBufferBytes);
    }
    const int64_t cycle_end = AscendC::GetSystemCycle();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    result.SetValue(0, cycle_end - cycle_start);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
