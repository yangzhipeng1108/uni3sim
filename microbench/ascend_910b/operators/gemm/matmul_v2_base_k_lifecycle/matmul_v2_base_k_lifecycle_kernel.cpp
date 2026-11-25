// Isolate MatmulImpl's base-K-dependent finite worker lifecycle.
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {

constexpr uint32_t kAicCount = 24;
constexpr uint32_t kBaseM = 96;
constexpr uint32_t kBaseN = 128;
constexpr uint32_t kMaximumK = 4096;
constexpr uint32_t kOutputTiles = 2;
constexpr uint32_t kResultStride = 16;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;

template <uint32_t ProjectionK, uint32_t BaseK>
__aicore__ constexpr MatmulConfig MakeConfig() {
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

template <uint32_t ProjectionK, uint32_t BaseK>
constexpr static MatmulConfig kConfig = MakeConfig<ProjectionK, BaseK>();

template <uint32_t ProjectionK, uint32_t BaseK>
constexpr tiling::TCubeTiling MakeTiling() {
    constexpr uint32_t step_ka = 512 / BaseK;
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
    result.depthA1 = 2 * step_ka;
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

template <uint32_t ProjectionK, uint32_t BaseK>
constexpr static tiling::TCubeTiling kTiling = MakeTiling<ProjectionK, BaseK>();

template <uint32_t ProjectionK, uint32_t BaseK>
__aicore__ inline void RunProgram(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result) {
    const tiling::TCubeTiling* tiling = &kTiling<ProjectionK, BaseK>;
    TPipe pipe;
    MatmulImpl<AType, BType, CType, BiasType,
        kConfig<ProjectionK, BaseK>> matmul;
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

extern "C" __global__ __aicore__ void matmul_v2_base_k_lifecycle(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        int64_t projection_k, int64_t base_k) {
    if (GetBlockIdx() >= kAicCount
            || (projection_k != 1024 && projection_k != 2048
                && projection_k != 4096)
            || (base_k != 32 && base_k != 64 && base_k != 128)) {
        return;
    }
#define RUN_CELL(K, BK)                                                   \
    if (projection_k == K && base_k == BK) {                             \
        RunProgram<K, BK>(a_gm, b_gm, c_gm, result);                     \
        return;                                                           \
    }
    RUN_CELL(1024, 32)
    RUN_CELL(1024, 64)
    RUN_CELL(1024, 128)
    RUN_CELL(2048, 32)
    RUN_CELL(2048, 64)
    RUN_CELL(2048, 128)
    RUN_CELL(4096, 32)
    RUN_CELL(4096, 64)
    RUN_CELL(4096, 128)
#undef RUN_CELL
}
