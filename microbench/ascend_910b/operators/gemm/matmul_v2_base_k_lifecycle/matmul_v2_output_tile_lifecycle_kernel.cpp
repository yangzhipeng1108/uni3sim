// Isolate MatmulImpl output-tile boundary credits for a MatMulV2 source.
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {

constexpr uint32_t kMaximumAicCount = 24;
constexpr uint32_t kBaseM = 128;
constexpr uint32_t kBaseK = 128;
constexpr uint32_t kDescriptorK = 256;
constexpr uint32_t kMaximumBaseN = 128;
constexpr uint32_t kMaximumK = 4096;
constexpr uint32_t kMaximumOutputTiles = 9;
constexpr uint32_t kResultStride = 16;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;

template <uint32_t ProjectionK, uint32_t BaseN, uint32_t OutputTiles>
__aicore__ constexpr MatmulConfig MakeConfig() {
    MatmulConfig result = GetNormalConfig();
    result.basicM = kBaseM;
    result.basicN = BaseN;
    result.basicK = kBaseK;
    result.singleCoreM = kBaseM;
    result.singleCoreN = BaseN;
    result.singleCoreK = ProjectionK;
    result.stepM = 1;
    result.stepN = 1;
    result.baseMN = kBaseM * BaseN;
    result.singleCoreMN = kBaseM * BaseN;
    result.enableSetTail = false;
    return result;
}

template <uint32_t ProjectionK, uint32_t BaseN, uint32_t OutputTiles>
constexpr static MatmulConfig kConfig =
    MakeConfig<ProjectionK, BaseN, OutputTiles>();

template <uint32_t ProjectionK, uint32_t BaseN, uint32_t OutputTiles>
constexpr tiling::TCubeTiling MakeTiling() {
    constexpr uint32_t descriptor_base_k_tiles = kDescriptorK / kBaseK;
    tiling::TCubeTiling result{};
    result.usedCoreNum = kMaximumAicCount;
    result.M = kBaseM;
    result.N = OutputTiles * BaseN;
    result.Ka = ProjectionK;
    result.Kb = ProjectionK;
    result.singleCoreM = kBaseM;
    result.singleCoreN = OutputTiles * BaseN;
    result.singleCoreK = ProjectionK;
    result.baseM = kBaseM;
    result.baseN = BaseN;
    result.baseK = kBaseK;
    // Each double-buffered A1/B1 slot contains one 256-wide descriptor.
    result.depthA1 = 2 * descriptor_base_k_tiles;
    result.depthB1 = 2 * descriptor_base_k_tiles;
    result.stepM = 1;
    result.stepN = 1;
    result.stepKa = descriptor_base_k_tiles;
    result.stepKb = descriptor_base_k_tiles;
    result.dbL0A = 2;
    result.dbL0B = 2;
    result.dbL0C = 1;
    return result;
}

template <uint32_t ProjectionK, uint32_t BaseN, uint32_t OutputTiles>
constexpr static tiling::TCubeTiling kTiling =
    MakeTiling<ProjectionK, BaseN, OutputTiles>();

template <uint32_t ProjectionK, uint32_t BaseN, uint32_t OutputTiles>
__aicore__ inline void RunProgram(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        uint32_t active_aics, bool force_tile_drain) {
    tiling::TCubeTiling tiling = kTiling<ProjectionK, BaseN, OutputTiles>;
    tiling.usedCoreNum = active_aics;
    TPipe pipe;
    MatmulImpl<AType, BType, CType, BiasType,
        kConfig<ProjectionK, BaseN, OutputTiles>> matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(&tiling, &pipe);
    GlobalTensor<bfloat16_t> a;
    GlobalTensor<bfloat16_t> b;
    GlobalTensor<bfloat16_t> c;
    const uint32_t block = GetBlockIdx();
    a.SetGlobalBuffer(a_gm + block * kBaseM * kMaximumK);
    b.SetGlobalBuffer(b_gm);
    c.SetGlobalBuffer(c_gm + block * OutputTiles * kBaseM * BaseN);

    WaitPreTaskEnd();
    GlobalTensor<int64_t> row;
    row.SetGlobalBuffer(result + block * kResultStride);
    const int64_t begin = GetSystemCycle();
    for (uint32_t output_tile = 0; output_tile < OutputTiles; ++output_tile) {
        matmul.SetOrgShape(
            kBaseM, OutputTiles * BaseN, ProjectionK,
            ProjectionK, OutputTiles * BaseN);
        matmul.SetSingleShape(kBaseM, BaseN, ProjectionK);
        matmul.SetTensorA(a);
        matmul.SetTensorB(b[output_tile * BaseN * ProjectionK]);
        matmul.template IterateAll<false>(
            c[output_tile * kBaseM * BaseN], 0);
        if (force_tile_drain) {
            // This counterfactual does not model production.  It reveals
            // whether the continuous source program retains credits across
            // output tiles by removing all such overlap explicitly.
            PipeBarrier<PIPE_ALL>();
        }
    }
    PipeBarrier<PIPE_ALL>();
    const int64_t end = GetSystemCycle();
    SetNextTaskStart();

    row.SetValue(0, end - begin);
    row.SetValue(1, ProjectionK);
    row.SetValue(2, BaseN);
    row.SetValue(3, OutputTiles);
    row.SetValue(4, active_aics);
    row.SetValue(5, tiling.stepKa);
    row.SetValue(6, tiling.stepKb);
    row.SetValue(7, tiling.depthA1);
    row.SetValue(8, tiling.depthB1);
    row.SetValue(9, force_tile_drain ? 1 : 0);
    row.SetValue(10, tiling.baseM);
    row.SetValue(11, tiling.baseK);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(row);
}

}  // namespace

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

extern "C" __global__ __aicore__ void matmul_v2_output_tile_lifecycle(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        int64_t projection_k, int64_t base_n, int64_t output_tiles,
        int64_t active_aics, int64_t force_tile_drain) {
    if (GetBlockIdx() >= active_aics
            || (projection_k != 1024 && projection_k != 2048
                && projection_k != 4096)
            || (base_n != 112 && base_n != 128)
            || (output_tiles != 1 && output_tiles != 2
                && output_tiles != 4 && output_tiles != 6
                && output_tiles != 9)
            || (active_aics != 22 && active_aics != 24)
            || (force_tile_drain != 0 && force_tile_drain != 1)) {
        return;
    }
#define RUN_CELL(K, N, TILES)                                            \
    if (projection_k == K && base_n == N && output_tiles == TILES) {    \
        RunProgram<K, N, TILES>(                                         \
            a_gm, b_gm, c_gm, result, static_cast<uint32_t>(active_aics), \
            force_tile_drain != 0);                                      \
        return;                                                          \
    }
    RUN_CELL(1024, 112, 1)
    RUN_CELL(1024, 112, 2)
    RUN_CELL(1024, 112, 4)
    RUN_CELL(1024, 112, 6)
    RUN_CELL(1024, 112, 9)
    RUN_CELL(1024, 128, 1)
    RUN_CELL(1024, 128, 2)
    RUN_CELL(1024, 128, 4)
    RUN_CELL(1024, 128, 6)
    RUN_CELL(1024, 128, 9)
    RUN_CELL(2048, 112, 1)
    RUN_CELL(2048, 112, 2)
    RUN_CELL(2048, 112, 4)
    RUN_CELL(2048, 112, 6)
    RUN_CELL(2048, 112, 9)
    RUN_CELL(2048, 128, 1)
    RUN_CELL(2048, 128, 2)
    RUN_CELL(2048, 128, 4)
    RUN_CELL(2048, 128, 6)
    RUN_CELL(2048, 128, 9)
    RUN_CELL(4096, 112, 1)
    RUN_CELL(4096, 112, 2)
    RUN_CELL(4096, 112, 4)
    RUN_CELL(4096, 112, 6)
    RUN_CELL(4096, 112, 9)
    RUN_CELL(4096, 128, 1)
    RUN_CELL(4096, 128, 2)
    RUN_CELL(4096, 128, 4)
    RUN_CELL(4096, 128, 6)
    RUN_CELL(4096, 128, 9)
#undef RUN_CELL
}
