// Ascend 910B concurrent first-request GlobalTensor<int32_t>::GetValue
// microbenchmark.
//
// Unlike dram_latency_kernel.cpp, this kernel deliberately performs no
// GetValue warmup.  It does execute the source-defined ReshapeAndCache tiling
// materialization before the timer so that the measured first request has the
// same immediately preceding mechanism as production keys 200 and 202.  All
// AIVs read adjacent int32 values from one shared slot array, matching the
// source's contiguous AllocateTask partition without executing the operator.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kOutputValuesPerBlock = 8;
constexpr uint32_t kTilingBytes = 28;

struct ReshapeAndCacheTilingData {
    uint32_t numTokens;
    uint32_t numHeads;
    uint32_t headSizeK;
    uint32_t headSizeV;
    uint32_t numBlocks;
    uint32_t blockDim;
    uint32_t numBatchs;
};

static_assert(sizeof(ReshapeAndCacheTilingData) == kTilingBytes);

__aicore__ inline void MaterializeTilingData(
        ReshapeAndCacheTilingData* destination,
        __gm__ uint8_t* source) {
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) \
        || defined(__DAV_310R6_CUBE__) || defined(__GET_CODE_CHANNEL__)
    copy_data_align64(
        reinterpret_cast<uint8_t*>(destination), source, kTilingBytes);
#else
    __ubuf__ uint8_t* tiling_data_in_ub =
        reinterpret_cast<__ubuf__ uint8_t*>(get_imm(0));
    constexpr uint32_t burst_count = (kTilingBytes + 31) / 32;
    copy_gm_to_ubuf(
        tiling_data_in_ub, source, 0, 1, burst_count, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64(
        reinterpret_cast<uint8_t*>(destination),
        (__ubuf__ uint8_t*)tiling_data_in_ub, kTilingBytes);
    pipe_barrier(PIPE_ALL);
#endif
}

}  // namespace

extern "C" __global__ __aicore__ void first_get_value(
        __gm__ uint8_t* tiling, __gm__ int32_t* buffer,
        __gm__ int64_t* output,
        int64_t request_count) {
    const int64_t block_index = AscendC::GetBlockIdx();

    ReshapeAndCacheTilingData tiling_data{};
    MaterializeTilingData(&tiling_data, tiling);

    AscendC::GlobalTensor<int32_t> values;
    values.SetGlobalBuffer(buffer);
    int32_t value = 0;

    const int64_t start = AscendC::GetSystemCycle();
    if (request_count == 1) {
        value = values.GetValue(block_index);
    }
    // Keep the optional scalar global read inside the measured interval.
    // Both the zero-request control and one-request cell execute the same
    // completion barrier, so their paired difference isolates GetValue while
    // preventing the compiler from moving the otherwise independently used
    // load below the ending counter read.
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();

    volatile uint32_t* fields =
        reinterpret_cast<volatile uint32_t*>(&tiling_data);
    const int64_t tiling_sum = static_cast<int64_t>(
        fields[0] + fields[1] + fields[2] + fields[3]
        + fields[4] + fields[5] + fields[6]);
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(
        output + block_index * kOutputValuesPerBlock);
    result.SetValue(0, end - start);
    result.SetValue(1, static_cast<int64_t>(value) + tiling_sum);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
