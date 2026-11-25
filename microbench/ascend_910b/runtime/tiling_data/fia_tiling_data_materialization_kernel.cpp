// Measure the CANN 8.5.1 FIA tiling-data materialization mechanism.
//
// The payload size and 1-AIC:2-AIV task ratio are launch facts captured from
// FusedInferAttentionScore.  The timed region contains only the generic
// GET_TILING_DATA_WITH_STRUCT copy sequence: no attention arithmetic,
// production input traffic, or cross-domain synchronization is present.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr int64_t kAivPerAic = 2;
constexpr int64_t kActorsPerGroup = 3;
constexpr int64_t kCycleSlotElements = 8;
constexpr int64_t kValidationSlotElements = 16;
constexpr uint32_t kPayloadBytes = 1560;
constexpr uint32_t kPayloadTransferBytes = 1568;

struct alignas(8) FiaTilingData {
    uint8_t payload[kPayloadBytes];
};

static_assert(sizeof(FiaTilingData) == kPayloadBytes);

__aicore__ inline void MaterializeTilingData(
        FiaTilingData* destination, __gm__ uint8_t* source) {
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) \
        || defined(__DAV_310R6_CUBE__) || defined(__GET_CODE_CHANNEL__)
    copy_data_align64(
        reinterpret_cast<uint8_t*>(destination), source, kPayloadBytes);
#else
    __ubuf__ uint8_t* tiling_data_in_ub =
        reinterpret_cast<__ubuf__ uint8_t*>(get_imm(0));
    constexpr uint32_t burst_count = kPayloadTransferBytes / 32;
    copy_gm_to_ubuf(
        tiling_data_in_ub, source, 0, 1, burst_count, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64(
        reinterpret_cast<uint8_t*>(destination),
        (__ubuf__ uint8_t*)tiling_data_in_ub, kPayloadBytes);
    pipe_barrier(PIPE_ALL);
#endif
}

__aicore__ inline int64_t ActorIndex() {
    const int64_t physical_block = AscendC::GetBlockIdx();
    if ASCEND_IS_AIC {
        return physical_block * kActorsPerGroup;
    }
    const int64_t group = physical_block / kAivPerAic;
    const int64_t lane = physical_block % kAivPerAic;
    return group * kActorsPerGroup + 1 + lane;
}

}  // namespace

extern "C" __global__ __aicore__ void fia_tiling_data_materialization(
        __gm__ uint8_t* gm_tiling, __gm__ int64_t* gm_cycles,
        __gm__ uint32_t* gm_validation, int64_t iterations) {
    const int64_t actor = ActorIndex();
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(gm_cycles + actor * kCycleSlotElements);
    if (iterations <= 0) {
        cycles.SetValue(0, -1);
        return;
    }

    FiaTilingData tiling_data{};
    for (int64_t warmup = 0; warmup < 16; ++warmup) {
        MaterializeTilingData(&tiling_data, gm_tiling);
    }
    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        MaterializeTilingData(&tiling_data, gm_tiling);
    }
    const int64_t end = AscendC::GetSystemCycle();
    cycles.SetValue(0, end - start);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(cycles);

    volatile uint32_t* fields =
        reinterpret_cast<volatile uint32_t*>(&tiling_data);
    AscendC::GlobalTensor<uint32_t> validation;
    validation.SetGlobalBuffer(
        gm_validation + actor * kValidationSlotElements);
    validation.SetValue(0, fields[0] + fields[194] + fields[389]);
    AscendC::DataCacheCleanAndInvalid<
        uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(validation);
}
