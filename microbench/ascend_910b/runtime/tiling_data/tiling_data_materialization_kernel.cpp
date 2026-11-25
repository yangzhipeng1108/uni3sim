// Measure CANN AIV InitTilingData materialization mechanisms.
//
// The 28-byte ReshapeAndCache, 56-byte BroadcastOneDim, 80-byte SwiGlu, and
// 192-byte GatherV3 payloads use the generic typed-copy sequence. The 48-byte
// Rope payload isolates its distinct source sequence: a 64-byte rounded
// GM->UB copy, PIPE_ALL barrier, twelve uint32_t scalar reads, and a second
// PIPE_ALL barrier. This is not a production operator and contains no
// production arithmetic.
#include "kernel_operator.h"

// GroupedMatmul materializes these members in its cube-core entry path.
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kCycleValuesPerBlock = 8;
constexpr int64_t kValidationValuesPerBlock = 16;
constexpr uint32_t kReshapeAndCacheTilingBytes = 28;
constexpr uint32_t kRopeTilingBytes = 48;
constexpr uint32_t kRopeTransferBytes = 64;
constexpr uint32_t kBroadcastOneDimTilingBytes = 56;
constexpr uint32_t kSwiGluTilingBytes = 80;
constexpr uint32_t kGatherV3TilingBytes = 192;
// CANN 8.5.1 GroupedMatmul's captured GMMTilingData places the complete
// GMMBaseParams at inline offset zero and TCubeTiling at byte offset 1672.
// The production entry materializes these as two separate typed members, so
// the pair path below must retain two distinct copy operations rather than
// coalescing them into one 1872-byte descriptor.
constexpr uint32_t kGroupedMatmulBaseParamsBytes = 1672;
constexpr uint32_t kGroupedMatmulTCubeTilingBytes = 200;
constexpr uint32_t kGroupedMatmulMetadataPairBytes =
    kGroupedMatmulBaseParamsBytes + kGroupedMatmulTCubeTilingBytes;

struct ReshapeAndCacheTilingData {
    uint32_t numTokens;
    uint32_t numHeads;
    uint32_t headSizeK;
    uint32_t headSizeV;
    uint32_t numBlocks;
    uint32_t blockDim;
    uint32_t numBatchs;
};

static_assert(
    sizeof(ReshapeAndCacheTilingData) == kReshapeAndCacheTilingBytes);

struct RopeTilingData {
    uint32_t words[12];
};

static_assert(sizeof(RopeTilingData) == kRopeTilingBytes);

struct BroadcastOneDimTilingData {
    int32_t scalarFlag;
    int32_t ubSplitAxis;
    int32_t ubFormer;
    int32_t ubTail;
    int32_t blockNum;
    int64_t blockFormer;
    int64_t blockTail;
    char scalarData[16];
};

static_assert(
    sizeof(BroadcastOneDimTilingData) == kBroadcastOneDimTilingBytes);

struct SwiGluTilingData {
    uint32_t is32BAligned;
    uint32_t isDoubleBuffer;
    uint64_t rowLen;
    uint64_t colLen;
    uint32_t baseRowLen;
    uint32_t baseColLen;
    uint32_t activateLeft;
    uint32_t biasIsEmpty;
    uint32_t quantScaleIsEmpty;
    uint32_t activateScaleIsEmpty;
    uint64_t swiColLen;
    uint64_t perRowLen;
    uint64_t modRowLen;
    uint32_t usedCoreNum;
    uint8_t placeholder[4];
};

static_assert(sizeof(SwiGluTilingData) == kSwiGluTilingBytes);

struct GatherV3TilingData {
    int64_t values[24];
};

static_assert(sizeof(GatherV3TilingData) == kGatherV3TilingBytes);

struct GroupedMatmulBaseParams {
    uint32_t words[kGroupedMatmulBaseParamsBytes / sizeof(uint32_t)];
};

static_assert(
    sizeof(GroupedMatmulBaseParams) == kGroupedMatmulBaseParamsBytes);

struct GroupedMatmulTCubeTiling {
    uint32_t words[kGroupedMatmulTCubeTilingBytes / sizeof(uint32_t)];
};

static_assert(
    sizeof(GroupedMatmulTCubeTiling) == kGroupedMatmulTCubeTilingBytes);

template <typename TilingData, uint32_t TilingBytes>
__aicore__ inline void MaterializeTilingData(
        TilingData* destination,
        __gm__ uint8_t* source) {
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) \
        || defined(__DAV_310R6_CUBE__) || defined(__GET_CODE_CHANNEL__)
    copy_data_align64(
        reinterpret_cast<uint8_t*>(destination), source, TilingBytes);
#else
    __ubuf__ uint8_t* tiling_data_in_ub =
        reinterpret_cast<__ubuf__ uint8_t*>(get_imm(0));
    constexpr uint32_t burst_count = (TilingBytes + 31) / 32;
    copy_gm_to_ubuf(
        tiling_data_in_ub, source, 0, 1, burst_count, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64(
        reinterpret_cast<uint8_t*>(destination),
        (__ubuf__ uint8_t*)tiling_data_in_ub, TilingBytes);
    pipe_barrier(PIPE_ALL);
#endif
}

template <typename TilingData, uint32_t TilingBytes>
__aicore__ inline void MeasureMaterialization(
        __gm__ uint8_t* gm_tiling, int64_t iterations,
        int64_t phase_repetitions, int64_t& elapsed_cycles,
        uint32_t& validation_sum) {
    TilingData tiling_data{};
    for (int64_t warmup = 0; warmup < 16; ++warmup) {
        for (int64_t phase = 0; phase < phase_repetitions; ++phase) {
            MaterializeTilingData<TilingData, TilingBytes>(
                &tiling_data, gm_tiling);
        }
    }
    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        #pragma unroll 1
        for (int64_t phase = 0; phase < phase_repetitions; ++phase) {
            MaterializeTilingData<TilingData, TilingBytes>(
                &tiling_data, gm_tiling);
        }
    }
    const int64_t end = AscendC::GetSystemCycle();

    volatile uint32_t* words =
        reinterpret_cast<volatile uint32_t*>(&tiling_data);
    validation_sum = 0;
    for (uint32_t index = 0; index < TilingBytes / sizeof(uint32_t); ++index) {
        validation_sum += words[index];
    }
    elapsed_cycles = end - start;
}

// This preserves the source-visible ordering in GroupedMatmul's entry macro:
// GET_TILING_DATA_MEMBER(gmmBaseParams) then
// GET_TILING_DATA_MEMBER(mmTilingData).  It intentionally has no matrix,
// routing, output, or production-operator work.
__aicore__ inline void MeasureGroupedMatmulMetadataPair(
        __gm__ uint8_t* gm_tiling, int64_t iterations,
        int64_t phase_repetitions, int64_t& elapsed_cycles,
        uint32_t& validation_sum) {
    GroupedMatmulBaseParams base_params{};
    GroupedMatmulTCubeTiling tcube_tiling{};
    for (int64_t warmup = 0; warmup < 16; ++warmup) {
        for (int64_t phase = 0; phase < phase_repetitions; ++phase) {
            MaterializeTilingData<
                GroupedMatmulBaseParams,
                kGroupedMatmulBaseParamsBytes>(&base_params, gm_tiling);
            MaterializeTilingData<
                GroupedMatmulTCubeTiling,
                kGroupedMatmulTCubeTilingBytes>(
                    &tcube_tiling,
                    gm_tiling + kGroupedMatmulBaseParamsBytes);
        }
    }
    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        #pragma unroll 1
        for (int64_t phase = 0; phase < phase_repetitions; ++phase) {
            MaterializeTilingData<
                GroupedMatmulBaseParams,
                kGroupedMatmulBaseParamsBytes>(&base_params, gm_tiling);
            MaterializeTilingData<
                GroupedMatmulTCubeTiling,
                kGroupedMatmulTCubeTilingBytes>(
                    &tcube_tiling,
                    gm_tiling + kGroupedMatmulBaseParamsBytes);
        }
    }
    const int64_t end = AscendC::GetSystemCycle();

    validation_sum = 0;
    for (uint32_t index = 0;
            index < kGroupedMatmulBaseParamsBytes / sizeof(uint32_t);
            ++index) {
        validation_sum += base_params.words[index];
    }
    for (uint32_t index = 0;
            index < kGroupedMatmulTCubeTilingBytes / sizeof(uint32_t);
            ++index) {
        validation_sum += tcube_tiling.words[index];
    }
    elapsed_cycles = end - start;
}

__aicore__ inline void MaterializeRopeTilingData(
        volatile RopeTilingData* destination,
        AscendC::LocalTensor<uint32_t> tiling_data_in_ub,
        AscendC::GlobalTensor<uint32_t> source) {
    AscendC::DataCopy(
        tiling_data_in_ub, source,
        kRopeTransferBytes / sizeof(uint32_t));
    AscendC::PipeBarrier<PIPE_ALL>();
    destination->words[0] = tiling_data_in_ub.GetValue(0);
    destination->words[1] = tiling_data_in_ub.GetValue(1);
    destination->words[2] = tiling_data_in_ub.GetValue(2);
    destination->words[3] = tiling_data_in_ub.GetValue(3);
    destination->words[4] = tiling_data_in_ub.GetValue(4);
    destination->words[5] = tiling_data_in_ub.GetValue(5);
    destination->words[6] = tiling_data_in_ub.GetValue(6);
    destination->words[7] = tiling_data_in_ub.GetValue(7);
    destination->words[8] = tiling_data_in_ub.GetValue(8);
    destination->words[9] = tiling_data_in_ub.GetValue(9);
    destination->words[10] = tiling_data_in_ub.GetValue(10);
    destination->words[11] = tiling_data_in_ub.GetValue(11);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void MeasureRopeMaterialization(
        __gm__ uint8_t* gm_tiling, int64_t iterations,
        int64_t& elapsed_cycles, uint32_t& validation_sum) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tiling_buffer;
    pipe.InitBuffer(tiling_buffer, kRopeTransferBytes);
    const AscendC::LocalTensor<uint32_t> tiling_data_in_ub =
        tiling_buffer.Get<uint32_t>();
    AscendC::GlobalTensor<uint32_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gm_tiling));
    volatile RopeTilingData tiling_data{};
    for (int64_t warmup = 0; warmup < 16; ++warmup) {
        MaterializeRopeTilingData(&tiling_data, tiling_data_in_ub, source);
    }
    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        MaterializeRopeTilingData(&tiling_data, tiling_data_in_ub, source);
    }
    const int64_t end = AscendC::GetSystemCycle();

    validation_sum = 0;
    for (uint32_t index = 0;
            index < kRopeTilingBytes / sizeof(uint32_t); ++index) {
        validation_sum += tiling_data.words[index];
    }
    elapsed_cycles = end - start;
}

}  // namespace

extern "C" __global__ __aicore__ void tiling_data_materialization(
        __gm__ uint8_t* gm_tiling, __gm__ int64_t* gm_cycles,
        __gm__ uint32_t* gm_validation, int64_t payload_bytes,
        int64_t phase_repetitions, int64_t iterations) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(
        gm_cycles + block_index * kCycleValuesPerBlock);
    if (iterations <= 0 || phase_repetitions <= 0) {
        cycles.SetValue(0, -1);
        return;
    }

    int64_t elapsed_cycles = 0;
    uint32_t validation_sum = 0;
    if (payload_bytes == kReshapeAndCacheTilingBytes) {
        MeasureMaterialization<
            ReshapeAndCacheTilingData, kReshapeAndCacheTilingBytes>(
                gm_tiling, iterations, phase_repetitions,
                elapsed_cycles, validation_sum);
    } else if (payload_bytes == kBroadcastOneDimTilingBytes) {
        MeasureMaterialization<
            BroadcastOneDimTilingData, kBroadcastOneDimTilingBytes>(
                gm_tiling, iterations, phase_repetitions,
                elapsed_cycles, validation_sum);
    } else if (payload_bytes == kRopeTilingBytes) {
        MeasureRopeMaterialization(
            gm_tiling, iterations, elapsed_cycles, validation_sum);
    } else if (payload_bytes == kSwiGluTilingBytes) {
        MeasureMaterialization<
            SwiGluTilingData, kSwiGluTilingBytes>(
                gm_tiling, iterations, phase_repetitions,
                elapsed_cycles, validation_sum);
    } else if (payload_bytes == kGatherV3TilingBytes) {
        MeasureMaterialization<
            GatherV3TilingData, kGatherV3TilingBytes>(
                gm_tiling, iterations, phase_repetitions,
                elapsed_cycles, validation_sum);
    } else if (payload_bytes == kGroupedMatmulMetadataPairBytes) {
        MeasureGroupedMatmulMetadataPair(
            gm_tiling, iterations, phase_repetitions,
            elapsed_cycles, validation_sum);
    } else {
        cycles.SetValue(0, -1);
        return;
    }
    cycles.SetValue(0, elapsed_cycles);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(cycles);

    AscendC::GlobalTensor<uint32_t> validation;
    validation.SetGlobalBuffer(
        gm_validation + block_index * kValidationValuesPerBlock);
    validation.SetValue(0, validation_sum);
    AscendC::DataCacheCleanAndInvalid<
        uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(validation);
}
