// Isolate the finite MatmulImpl lifecycle of the small-M source program.
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {

constexpr uint32_t kPhysicalAicCount = 24;
constexpr uint32_t kBaseM = 16;
constexpr uint32_t kBaseK = 64;
constexpr uint32_t kStepK = 8;
constexpr uint32_t kMaximumK = 2048;
constexpr uint32_t kOutputTiles = 2;
constexpr uint32_t kResultStride = 16;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;

template <uint32_t ProjectionK, uint32_t BaseN, bool StaticNoTail>
__aicore__ constexpr MatmulConfig MakeConfig() {
    MatmulConfig result = GetNormalConfig();
    if constexpr (StaticNoTail) {
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
    }
    return result;
}

template <uint32_t ProjectionK, uint32_t BaseN, bool StaticNoTail>
constexpr static MatmulConfig kConfig =
    MakeConfig<ProjectionK, BaseN, StaticNoTail>();

template <uint32_t ProjectionK, uint32_t BaseN, uint32_t ActiveAics>
constexpr tiling::TCubeTiling MakeTiling() {
    tiling::TCubeTiling result{};
    result.usedCoreNum = ActiveAics;
    result.M = kBaseM;
    result.N = kOutputTiles * BaseN;
    result.Ka = ProjectionK;
    result.Kb = ProjectionK;
    result.singleCoreM = kBaseM;
    result.singleCoreN = kOutputTiles * BaseN;
    result.singleCoreK = ProjectionK;
    result.baseM = kBaseM;
    result.baseN = BaseN;
    result.baseK = kBaseK;
    result.depthA1 = 2 * kStepK;
    result.depthB1 = 2 * kStepK;
    result.stepM = 1;
    result.stepN = 1;
    result.stepKa = kStepK;
    result.stepKb = kStepK;
    result.dbL0A = 2;
    result.dbL0B = 2;
    result.dbL0C = 1;
    return result;
}

template <uint32_t ProjectionK, uint32_t BaseN, uint32_t ActiveAics>
constexpr static tiling::TCubeTiling kTiling =
    MakeTiling<ProjectionK, BaseN, ActiveAics>();

template <uint32_t ProjectionK, uint32_t BaseN, uint32_t ActiveAics,
          bool StaticNoTail>
__aicore__ inline void RunProgram(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        uint32_t logical_m) {
    const tiling::TCubeTiling* tiling =
        &kTiling<ProjectionK, BaseN, ActiveAics>;
    TPipe pipe;
    MatmulImpl<AType, BType, CType, BiasType,
        kConfig<ProjectionK, BaseN, StaticNoTail>> matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(tiling, &pipe);
    GlobalTensor<bfloat16_t> a;
    GlobalTensor<bfloat16_t> b;
    GlobalTensor<bfloat16_t> c;
    const uint32_t block = GetBlockIdx();
    a.SetGlobalBuffer(a_gm + block * kBaseM * kMaximumK);
    b.SetGlobalBuffer(b_gm);
    c.SetGlobalBuffer(c_gm + block * kOutputTiles * kBaseM * BaseN);

    WaitPreTaskEnd();
    GlobalTensor<int64_t> row;
    row.SetGlobalBuffer(result + block * kResultStride);
    const int64_t begin = GetSystemCycle();
    for (uint32_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
        matmul.SetOrgShape(
            logical_m, kOutputTiles * BaseN, ProjectionK,
            ProjectionK, kOutputTiles * BaseN);
        matmul.SetSingleShape(logical_m, BaseN, ProjectionK);
        matmul.SetTensorA(a);
        matmul.SetTensorB(b[output_tile * BaseN * ProjectionK]);
        matmul.template IterateAll<false>(
            c[output_tile * kBaseM * BaseN], 0);
    }
    PipeBarrier<PIPE_ALL>();
    const int64_t end = GetSystemCycle();
    SetNextTaskStart();

    row.SetValue(0, end - begin);
    row.SetValue(1, logical_m);
    row.SetValue(2, ProjectionK);
    row.SetValue(3, BaseN);
    row.SetValue(4, ActiveAics);
    row.SetValue(5, tiling->stepKa);
    row.SetValue(6, tiling->stepKb);
    row.SetValue(7, tiling->depthA1);
    row.SetValue(8, tiling->depthB1);
    row.SetValue(9, StaticNoTail ? 1 : 0);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(row);
}

}  // namespace

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

extern "C" __global__ __aicore__ void matmul_v2_small_m_lifecycle(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        int64_t base_n, int64_t logical_m, int64_t projection_k,
        int64_t static_no_tail) {
    if (GetBlockIdx() >= kPhysicalAicCount
            || (base_n != 96 && base_n != 112)
            || (logical_m != 1 && logical_m != 16)
            || (projection_k != 1024 && projection_k != 2048)
            || (static_no_tail != 0 && static_no_tail != 1)
            || (static_no_tail == 1 && logical_m != 16)
            || (static_no_tail == 0 && logical_m != 1)) {
        return;
    }
#define RUN_CELL(K, N, WORKERS)                                          \
    if (projection_k == K && base_n == N) {                              \
        if (static_no_tail) {                                             \
            RunProgram<K, N, WORKERS, true>(                              \
                a_gm, b_gm, c_gm, result, logical_m);                     \
        } else {                                                          \
            RunProgram<K, N, WORKERS, false>(                             \
                a_gm, b_gm, c_gm, result, logical_m);                     \
        }                                                                 \
        return;                                                           \
    }
    RUN_CELL(1024, 96, 22)
    RUN_CELL(2048, 96, 22)
    RUN_CELL(1024, 112, 23)
    RUN_CELL(2048, 112, 23)
#undef RUN_CELL
}
