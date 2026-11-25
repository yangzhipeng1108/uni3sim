// Isolate one CANN AIV TPipe scope containing one VECCALC TBuf.
//
// The loop repeats the same workload-independent lifecycle only to resolve
// it against the 20 ns SYS_CNT tick. It contains no slot lookup, tensor copy,
// token loop, cache layout, or production operator computation.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;

}  // namespace

extern "C" __global__ __aicore__ void single_tbuf_lifecycle(
        __gm__ int64_t* output, int64_t payload_bytes, int64_t iterations) {
    const int64_t block_index = AscendC::GetBlockIdx();
    const int64_t cycle_start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> token_buffer;
        pipe.InitBuffer(token_buffer, static_cast<uint32_t>(payload_bytes));
    }
    const int64_t cycle_end = AscendC::GetSystemCycle();
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(output + block_index * kOutputValuesPerBlock);
    result.SetValue(0, cycle_end - cycle_start);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
