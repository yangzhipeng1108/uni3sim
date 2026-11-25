// Isolate MatmulImpl's base-K-dependent finite worker lifecycle.
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {

// Narrow-M stage ladder: the same complete MatmulImpl program as
// matmul_v2_base_k_lifecycle, re-parameterised to the narrow base tiles the
// production small-M launches actually use.  Compared against the MTE2-only
// credit-stream probe at the same descriptor topology, the difference
// attributes the operator time that is not the operand stream.  This is a
// diagnostic decomposition: its absolute span is not a Forward parameter.
constexpr uint32_t kMaximumK = 4096;

constexpr uint32_t kResultStride = 16;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;

template <uint32_t AicCount, uint32_t BaseM, uint32_t BaseN,
          uint32_t ProjectionK, uint32_t BaseK, uint32_t OutputTiles,
          uint32_t FullA>
__aicore__ constexpr MatmulConfig MakeConfig() {
    constexpr uint32_t kAicCount = AicCount;
    constexpr uint32_t kBaseM = BaseM;
    constexpr uint32_t kBaseN = BaseN;
    MatmulConfig result = GetNormalConfig();
    result.basicM = kBaseM;
    result.basicN = kBaseN;
    result.basicK = BaseK;
    result.singleCoreM = kBaseM;
    result.singleCoreN = kBaseN;
    result.singleCoreK = ProjectionK;
    result.stepM = 1;
    result.stepN = 1;
    result.baseMN = kBaseM * kBaseN;
    result.singleCoreMN = kBaseM * kBaseN;
    result.enableSetTail = false;
    return result;
}

template <uint32_t AicCount, uint32_t BaseM, uint32_t BaseN,
          uint32_t ProjectionK, uint32_t BaseK, uint32_t OutputTiles,
          uint32_t FullA>
constexpr static MatmulConfig kConfig =
    MakeConfig<AicCount, BaseM, BaseN, ProjectionK, BaseK,
               OutputTiles, FullA>();

template <uint32_t AicCount, uint32_t BaseM, uint32_t BaseN,
          uint32_t ProjectionK, uint32_t BaseK, uint32_t OutputTiles,
          uint32_t FullA>
constexpr tiling::TCubeTiling MakeTiling() {
    constexpr uint32_t kAicCount = AicCount;
    constexpr uint32_t kBaseM = BaseM;
    constexpr uint32_t kBaseN = BaseN;
    constexpr uint32_t kOutputTiles = OutputTiles;
    // The production narrow program keeps the worker-local M slab resident in
    // one A1 slot for the whole K axis (FullA) and streams B in 256-element
    // K groups through two slots.  The 96x128 control keeps the published
    // 512/256 step policy.
    constexpr uint32_t step_ka =
        FullA ? (ProjectionK / BaseK) : (512 / BaseK);
    constexpr uint32_t step_kb = 256 / BaseK;
    tiling::TCubeTiling result{};
    result.usedCoreNum = kAicCount;
    result.M = kBaseM;
    result.N = kOutputTiles * kBaseN;
    result.Ka = ProjectionK;
    result.Kb = ProjectionK;
    result.singleCoreM = kBaseM;
    result.singleCoreN = kOutputTiles * kBaseN;
    result.singleCoreK = ProjectionK;
    result.baseM = kBaseM;
    result.baseN = kBaseN;
    result.baseK = BaseK;
    result.depthA1 = (FullA ? 1u : 2u) * step_ka;
    result.depthB1 = 2 * step_kb;
    result.stepM = 1;
    result.stepN = 1;
    result.stepKa = step_ka;
    result.stepKb = step_kb;
    result.dbL0A = 2;
    result.dbL0B = 2;
    result.dbL0C = 1;
    return result;
}

template <uint32_t AicCount, uint32_t BaseM, uint32_t BaseN,
          uint32_t ProjectionK, uint32_t BaseK, uint32_t OutputTiles,
          uint32_t FullA>
constexpr static tiling::TCubeTiling kTiling =
    MakeTiling<AicCount, BaseM, BaseN, ProjectionK, BaseK,
               OutputTiles, FullA>();

template <uint32_t AicCount, uint32_t BaseM, uint32_t BaseN,
          uint32_t ProjectionK, uint32_t BaseK, uint32_t OutputTiles,
          uint32_t FullA>
__aicore__ inline void RunProgram(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result) {
    constexpr uint32_t kAicCount = AicCount;
    constexpr uint32_t kBaseM = BaseM;
    constexpr uint32_t kBaseN = BaseN;
    constexpr uint32_t kOutputTiles = OutputTiles;
    const tiling::TCubeTiling* tiling =
        &kTiling<AicCount, BaseM, BaseN, ProjectionK, BaseK,
                 OutputTiles, FullA>;
    TPipe pipe;
    MatmulImpl<AType, BType, CType, BiasType,
        kConfig<AicCount, BaseM, BaseN, ProjectionK, BaseK,
                OutputTiles, FullA>> matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(tiling, &pipe);
    GlobalTensor<bfloat16_t> a;
    GlobalTensor<bfloat16_t> b;
    GlobalTensor<bfloat16_t> c;
    const uint32_t block = GetBlockIdx();
    a.SetGlobalBuffer(a_gm + block * kBaseM * kMaximumK);
    b.SetGlobalBuffer(b_gm);
    c.SetGlobalBuffer(c_gm + block * kOutputTiles * kBaseM * kBaseN);

    WaitPreTaskEnd();
    GlobalTensor<int64_t> row;
    row.SetGlobalBuffer(result + block * kResultStride);
    const int64_t begin = GetSystemCycle();
    for (uint32_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
        matmul.SetOrgShape(
            kBaseM, kOutputTiles * kBaseN, ProjectionK,
            ProjectionK, kOutputTiles * kBaseN);
        matmul.SetSingleShape(kBaseM, kBaseN, ProjectionK);
        matmul.SetTensorA(a);
        matmul.SetTensorB(b[output_tile * kBaseN * ProjectionK]);
        matmul.template IterateAll<false>(
            c[output_tile * kBaseM * kBaseN], 0);
    }
    PipeBarrier<PIPE_ALL>();
    const int64_t end = GetSystemCycle();
    SetNextTaskStart();

    row.SetValue(0, end - begin);
    row.SetValue(1, ProjectionK);
    row.SetValue(2, BaseK);
    row.SetValue(3, tiling->stepKa);
    row.SetValue(4, tiling->stepKb);
    row.SetValue(5, tiling->depthA1);
    row.SetValue(6, tiling->depthB1);
    row.SetValue(7, tiling->baseM);
    row.SetValue(8, tiling->baseN);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(row);
}

}  // namespace

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

extern "C" __global__ __aicore__ void matmul_v2_narrow_stage_ladder(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        int64_t projection_k, int64_t base_m) {
    // The retained cells are the exact narrow base tiles of the failing
    // production launches - (24 AIC, 16x176) at K=4096 and (23 AIC, 16x224)
    // at K=1536 - plus the published 96x128 base_k_lifecycle geometry as a
    // control that must reproduce its known value.
#define RUN_CELL(AIC, BM, BN, K, OT, FA)                                          \
    if (projection_k == (K) && base_m == (BM)) {                          \
        if (GetBlockIdx() >= (AIC)) {                                     \
            return;                                                       \
        }                                                                 \
        RunProgram<AIC, BM, BN, K, 64, OT, FA>(a_gm, b_gm, c_gm, result);         \
        return;                                                           \
    }
    RUN_CELL(24, 16, 176, 4096, 1, 1)
    RUN_CELL(23, 16, 224, 1536, 1, 1)
    RUN_CELL(24, 96, 128, 4096, 2, 0)
#undef RUN_CELL
}
