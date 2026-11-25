// Independent Ascend 910B GroupedMatmul B-ingress address-topology probe.
//
// This is deliberately not a FusedMoE implementation.  It issues only the
// source-visible BF16 CopyND2NZOnTheFly B1 fill, its MTE2->MTE1/MTE1->M
// consumer edges, and four small Cube operations per B1 group.  A is prefetched
// before timing and C is never written.  The two layouts differ only in the
// physical expert offsets used by the real TOK2/E8/TOPK2/H6144/INT16384 W1
// program: adjacent [1,2,3] versus sparse [1,4,6].
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

using namespace AscendC;

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kActiveAic = 24;
constexpr int64_t kExperts = 8;
constexpr int64_t kActiveExperts = 3;
constexpr int64_t kNTilesPerExpert = 128;
constexpr int64_t kTileN = 256;
constexpr int64_t kProjectionK = 6144;
constexpr int64_t kB1K = 256;
constexpr int64_t kB1GroupsPerOutputTile = kProjectionK / kB1K;
constexpr int64_t kOutputTiles = kActiveExperts * kNTilesPerExpert;
constexpr int64_t kB1GroupsPerWorker = (
    kOutputTiles / kActiveAic * kB1GroupsPerOutputTile);
constexpr int64_t kExpertSpan = kNTilesPerExpert * kTileN * kProjectionK;
constexpr int64_t kB1Elements = kB1K * kTileN;
constexpr int64_t kL0AElements = 16 * 64;
constexpr int64_t kL0BElements = 64 * kTileN;
constexpr int64_t kL0CElements = 16 * kTileN;
constexpr int64_t kCycleSlotBytes = 64;

__aicore__ inline int64_t ExpertOffset(int64_t layout, int64_t active_expert) {
    if (layout == 0) {
        constexpr int64_t kAdjacent[kActiveExperts] = {1, 2, 3};
        return kAdjacent[active_expert] * kExpertSpan;
    }
    constexpr int64_t kSparse[kActiveExperts] = {1, 4, 6};
    return kSparse[active_expert] * kExpertSpan;
}

__aicore__ inline void CopyB1OnTheFly(
        LocalTensor<bfloat16_t>& destination,
        GlobalTensor<bfloat16_t>& source) {
    // CANN 8.5.1 BF16 CFG_MDL's B1 issue: 16 C0-width copies, each over
    // 256 logical N rows with the physical K=6144 source-row stride.
    constexpr int64_t kC0 = 16;
    constexpr int64_t kCopyWidths = kTileN / kC0;
    constexpr int64_t kDestinationWidthStride = (
        (kB1K / kC0) * kC0 * kC0);
    constexpr int64_t kSourceGap = kProjectionK * kElementBytes / 32 - 1;
    DataCopyEnhancedParams enhanced {};
    #pragma unroll 1
    for (int64_t width = 0; width < kCopyWidths; ++width) {
        DataCopy(destination[width * kDestinationWidthStride],
                 source[width * kC0],
                 {static_cast<uint16_t>(kB1K), 1,
                  static_cast<uint16_t>(kSourceGap), 0}, enhanced);
    }
}

__aicore__ inline void CopyAForCube(
        LocalTensor<bfloat16_t>& destination,
        GlobalTensor<bfloat16_t>& source) {
    Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = 16;
    parameters.dValue = 64;
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = 64;
    parameters.dstNzC0Stride = 16;
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    DataCopy(destination, source, parameters);
}

__aicore__ inline void LoadA(
        LocalTensor<bfloat16_t>& destination,
        const LocalTensor<bfloat16_t>& source) {
    LoadData2DParams parameters{0, 4, 1, 0, 0, false, 0};
    LoadData(destination, source, parameters);
}

__aicore__ inline void LoadB(
        LocalTensor<bfloat16_t>& destination,
        const LocalTensor<bfloat16_t>& source, int64_t k_index) {
    constexpr int64_t kBlocksN = kTileN / 16;
    constexpr int64_t kBlocksK = 64 / 16;
    LoadData2DParams parameters{
        0, static_cast<uint8_t>(kBlocksN),
        static_cast<uint16_t>(kB1K / 16), 0, 0, true, 0};
    const auto b1_k_tile = source[k_index * kL0BElements];
    #pragma unroll 1
    for (int64_t block_k = 0; block_k < kBlocksK; ++block_k) {
        LoadData(destination[block_k * kBlocksN * 256],
                 b1_k_tile[block_k * kTileN], parameters);
    }
}

__aicore__ inline void RunOneOutputTile(
        LocalTensor<bfloat16_t>& b1_ping,
        LocalTensor<bfloat16_t>& b1_pong,
        LocalTensor<bfloat16_t>& a1,
        LocalTensor<bfloat16_t>& l0a_ping,
        LocalTensor<bfloat16_t>& l0a_pong,
        LocalTensor<bfloat16_t>& l0b_ping,
        LocalTensor<bfloat16_t>& l0b_pong,
        LocalTensor<float>& l0c,
        __gm__ bfloat16_t* source_base,
        int64_t layout, int64_t active_expert, int64_t n_index) {
    #pragma unroll 1
    for (int64_t b1_group = 0; b1_group < kB1GroupsPerOutputTile;
            ++b1_group) {
        const int64_t b1_slot = b1_group & 1;
        auto b1 = b1_slot == 0 ? b1_ping : b1_pong;
        WaitFlag<HardEvent::MTE1_MTE2>(b1_slot);
        GlobalTensor<bfloat16_t> source;
        const int64_t offset = ExpertOffset(layout, active_expert)
            + n_index * kTileN * kProjectionK + b1_group * kB1K;
        source.SetGlobalBuffer(source_base + offset);
        source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        CopyB1OnTheFly(b1, source);
        SetFlag<HardEvent::MTE2_MTE1>(b1_slot);
        WaitFlag<HardEvent::MTE2_MTE1>(b1_slot);
        #pragma unroll 1
        for (int64_t k_index = 0; k_index < 4; ++k_index) {
            const int64_t l0_slot = k_index & 1;
            auto l0a = l0_slot == 0 ? l0a_ping : l0a_pong;
            auto l0b = l0_slot == 0 ? l0b_ping : l0b_pong;
            WaitFlag<HardEvent::M_MTE1>(l0_slot);
            LoadA(l0a, a1);
            LoadB(l0b, b1, k_index);
            SetFlag<HardEvent::MTE1_M>(l0_slot);
            WaitFlag<HardEvent::MTE1_M>(l0_slot);
            MmadParams parameters;
            parameters.m = 16;
            parameters.n = kTileN;
            parameters.k = 64;
            parameters.cmatrixInitVal = k_index == 0;
            parameters.cmatrixSource = false;
            Mmad(l0c, l0a, l0b, parameters);
            SetFlag<HardEvent::M_MTE1>(l0_slot);
        }
        SetFlag<HardEvent::MTE1_MTE2>(b1_slot);
    }
}

__aicore__ inline void RunOneSourcePattern(
        LocalTensor<bfloat16_t>& b1_ping,
        LocalTensor<bfloat16_t>& b1_pong,
        LocalTensor<bfloat16_t>& a1,
        LocalTensor<bfloat16_t>& l0a_ping,
        LocalTensor<bfloat16_t>& l0a_pong,
        LocalTensor<bfloat16_t>& l0b_ping,
        LocalTensor<bfloat16_t>& l0b_pong,
        LocalTensor<float>& l0c,
        __gm__ bfloat16_t* source_base, int64_t layout, int64_t core) {
    // This is the GMMProcess count / cur_block source assignment for three
    // populated one-M-tile experts, each with 128 N tiles and 24 AICs.
    int64_t count = 0;
    #pragma unroll 1
    for (int64_t active_expert = 0; active_expert < kActiveExperts;
            ++active_expert) {
        const int64_t cur_count = count + kNTilesPerExpert;
        int64_t cur_block = core >= count ? core : core + kActiveAic;
        while (cur_block < cur_count) {
            const int64_t n_index = cur_block - count;
            RunOneOutputTile(
                b1_ping, b1_pong, a1, l0a_ping, l0a_pong, l0b_ping,
                l0b_pong, l0c, source_base, layout, active_expert, n_index);
            cur_block += kActiveAic;
        }
        count = cur_count % kActiveAic;
    }
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_sparse_expert_layout(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_output,
        int64_t layout, int64_t repetitions) {
    const int64_t core = GetBlockIdx();
    if ((layout != 0 && layout != 1) || repetitions <= 0
            || core >= kActiveAic) {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::B1> b1_ping_buffer;
    TBuf<TPosition::B1> b1_pong_buffer;
    TBuf<TPosition::A1> a1_buffer;
    TBuf<TPosition::A2> l0a_ping_buffer;
    TBuf<TPosition::A2> l0a_pong_buffer;
    TBuf<TPosition::B2> l0b_ping_buffer;
    TBuf<TPosition::B2> l0b_pong_buffer;
    TBuf<TPosition::CO1> l0c_buffer;
    pipe.InitBuffer(b1_ping_buffer, kB1Elements * kElementBytes);
    pipe.InitBuffer(b1_pong_buffer, kB1Elements * kElementBytes);
    pipe.InitBuffer(a1_buffer, kL0AElements * kElementBytes);
    pipe.InitBuffer(l0a_ping_buffer, kL0AElements * kElementBytes);
    pipe.InitBuffer(l0a_pong_buffer, kL0AElements * kElementBytes);
    pipe.InitBuffer(l0b_ping_buffer, kL0BElements * kElementBytes);
    pipe.InitBuffer(l0b_pong_buffer, kL0BElements * kElementBytes);
    pipe.InitBuffer(l0c_buffer, kL0CElements * sizeof(float));
    auto b1_ping = b1_ping_buffer.Get<bfloat16_t>();
    auto b1_pong = b1_pong_buffer.Get<bfloat16_t>();
    auto a1 = a1_buffer.Get<bfloat16_t>();
    auto l0a_ping = l0a_ping_buffer.Get<bfloat16_t>();
    auto l0a_pong = l0a_pong_buffer.Get<bfloat16_t>();
    auto l0b_ping = l0b_ping_buffer.Get<bfloat16_t>();
    auto l0b_pong = l0b_pong_buffer.Get<bfloat16_t>();
    auto l0c = l0c_buffer.Get<float>();
    GlobalTensor<bfloat16_t> a_source;
    a_source.SetGlobalBuffer(gm_source + core * kL0AElements);
    CopyAForCube(a1, a_source);
    PipeBarrier<PIPE_MTE2>();
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    const int64_t begin = GetSystemCycle();
    #pragma unroll 1
    for (int64_t repeat = 0; repeat < repetitions; ++repeat) {
        RunOneSourcePattern(
            b1_ping, b1_pong, a1, l0a_ping, l0a_pong, l0b_ping,
            l0b_pong, l0c, gm_source, layout, core);
    }
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
    const int64_t end = GetSystemCycle();
    GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + core * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - begin);
    output.SetValue(1, repetitions * kB1GroupsPerWorker);
}
