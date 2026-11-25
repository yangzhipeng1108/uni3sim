// Isolate the CANN MatmulImpl admission branch for a partial M basic block.
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {

constexpr uint32_t kLaunchedAicCount = 22;
constexpr uint32_t kPhysicalAicCount = 24;
constexpr uint32_t kBaseM = 48;
constexpr uint32_t kBaseN = 96;
constexpr uint32_t kBaseK = 64;
constexpr uint32_t kMaximumK = 4096;
constexpr uint32_t kOutputTiles = 2;
constexpr uint32_t kResultStride = 16;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;

template <uint32_t ProjectionK, bool StaticNoTail>
__aicore__ constexpr MatmulConfig MakeConfig() {
    MatmulConfig result = GetNormalConfig();
    if constexpr (StaticNoTail) {
        result.basicM = kBaseM;
        result.basicN = kBaseN;
        result.basicK = kBaseK;
        result.singleCoreM = kBaseM;
        result.singleCoreN = kBaseN;
        result.singleCoreK = ProjectionK;
        result.stepM = 1;
        result.stepN = 1;
        result.baseMN = kBaseM * kBaseN;
        result.singleCoreMN = kBaseM * kBaseN;
        result.enableSetTail = false;
    }
    return result;
}

template <uint32_t ProjectionK, bool StaticNoTail>
constexpr static MatmulConfig kConfig =
    MakeConfig<ProjectionK, StaticNoTail>();

constexpr tiling::TCubeTiling MakeTiling(
        uint32_t projection_k, uint32_t step_k) {
    tiling::TCubeTiling result{};
    result.usedCoreNum = kLaunchedAicCount;
    result.M = kBaseM;
    result.N = kOutputTiles * kBaseN;
    result.Ka = projection_k;
    result.Kb = projection_k;
    result.singleCoreM = kBaseM;
    result.singleCoreN = kOutputTiles * kBaseN;
    result.singleCoreK = projection_k;
    result.baseM = kBaseM;
    result.baseN = kBaseN;
    result.baseK = kBaseK;
    result.depthA1 = 2 * step_k;
    result.depthB1 = 2 * step_k;
    result.stepM = 1;
    result.stepN = 1;
    result.stepKa = step_k;
    result.stepKb = step_k;
    result.dbL0A = 2;
    result.dbL0B = 2;
    result.dbL0C = 1;
    return result;
}

template <uint32_t ProjectionK, uint32_t StepK>
constexpr static tiling::TCubeTiling kTiling =
    MakeTiling(ProjectionK, StepK);

template <uint32_t ProjectionK, uint32_t StepK, bool StaticNoTail>
__aicore__ inline void RunProgram(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        int64_t logical_m) {
    const tiling::TCubeTiling* tiling = &kTiling<ProjectionK, StepK>;
    TPipe pipe;
    MatmulImpl<AType, BType, CType, BiasType,
        kConfig<ProjectionK, StaticNoTail>> matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(tiling, &pipe);
    GlobalTensor<bfloat16_t> a;
    GlobalTensor<bfloat16_t> b;
    GlobalTensor<bfloat16_t> c;
    const uint32_t block = GetBlockIdx();
    a.SetGlobalBuffer(a_gm + block * kBaseM * kMaximumK);
    b.SetGlobalBuffer(b_gm);
    c.SetGlobalBuffer(
        c_gm + block * kOutputTiles * kBaseM * kBaseN);

    WaitPreTaskEnd();
    GlobalTensor<int64_t> row;
    row.SetGlobalBuffer(result + block * kResultStride);
    const int64_t program_begin = GetSystemCycle();
    for (uint32_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
        const int64_t begin = GetSystemCycle();
        matmul.SetOrgShape(
            logical_m, kOutputTiles * kBaseN, ProjectionK,
            ProjectionK, kOutputTiles * kBaseN);
        matmul.SetSingleShape(logical_m, kBaseN, ProjectionK);
        matmul.SetTensorA(a);
        matmul.SetTensorB(b[output_tile * kBaseN * ProjectionK]);
        matmul.template IterateAll<false>(
            c[output_tile * kBaseM * kBaseN], 0);
        const int64_t end = GetSystemCycle();
        row.SetValue(output_tile, end - begin);
    }
    PipeBarrier<PIPE_ALL>();
    const int64_t program_end = GetSystemCycle();
    SetNextTaskStart();

    row.SetValue(2, logical_m);
    row.SetValue(3, ProjectionK);
    row.SetValue(4, StepK);
    row.SetValue(5, tiling->depthA1);
    row.SetValue(6, tiling->depthB1);
    row.SetValue(7, tiling->baseM);
    row.SetValue(8, tiling->baseN);
    row.SetValue(9, tiling->baseK);
    row.SetValue(10, StaticNoTail ? 1 : 0);
    row.SetValue(11, program_end - program_begin);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(row);
}

}  // namespace

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

extern "C" __global__ __aicore__ void matmul_v2_m_tail_admission(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        int64_t logical_m, int64_t projection_k, int64_t step_k,
        int64_t static_no_tail) {
    const uint32_t block = GetBlockIdx();
    if (block >= kPhysicalAicCount || (logical_m != 40 && logical_m != 48)
            || (projection_k != 512 && projection_k != 1024
                && projection_k != 2048 && projection_k != 4096)
            || (step_k != 4 && step_k != 8)
            || (static_no_tail != 0 && static_no_tail != 1)
            || (static_no_tail == 1 && logical_m != 48)) {
        return;
    }
#define RUN_CELL(K, STEP)                                                 \
    if (projection_k == K && step_k == STEP) {                            \
        if (static_no_tail) {                                             \
            RunProgram<K, STEP, true>(                                    \
                a_gm, b_gm, c_gm, result, logical_m);                     \
        } else {                                                          \
            RunProgram<K, STEP, false>(                                   \
                a_gm, b_gm, c_gm, result, logical_m);                     \
        }                                                                 \
        return;                                                           \
    }
    RUN_CELL(512, 4)
    RUN_CELL(512, 8)
    RUN_CELL(1024, 4)
    RUN_CELL(1024, 8)
    RUN_CELL(2048, 4)
    RUN_CELL(2048, 8)
    RUN_CELL(4096, 4)
    RUN_CELL(4096, 8)
#undef RUN_CELL
}
