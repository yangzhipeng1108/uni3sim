// Independent full-A MatMulV2 Scalar-control probe.
//
// This retains the 24-AIC, 16x176x64, A1=1/B1=2 descriptor topology of the
// K=4096 narrow-M source program. It compares its dynamic M=1 branch with a
// static M=16 control at two full-A K extents. It does not reproduce a
// production operator: whole-program cycles include MTE2, MTE1, Cube and
// FixPipe and are never a Forward parameter.
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {

constexpr uint32_t kAicWorkers = 24;
constexpr uint32_t kBaseM = 16;
constexpr uint32_t kBaseN = 176;
constexpr uint32_t kBaseK = 64;
constexpr uint32_t kMaximumK = 4096;
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
constexpr static MatmulConfig kConfig = MakeConfig<ProjectionK, StaticNoTail>();

template <uint32_t ProjectionK>
constexpr tiling::TCubeTiling MakeTiling() {
    tiling::TCubeTiling result{};
    result.usedCoreNum = kAicWorkers;
    result.M = kBaseM;
    result.N = kBaseN;
    result.Ka = ProjectionK;
    result.Kb = ProjectionK;
    result.singleCoreM = kBaseM;
    result.singleCoreN = kBaseN;
    result.singleCoreK = ProjectionK;
    result.baseM = kBaseM;
    result.baseN = kBaseN;
    result.baseK = kBaseK;
    result.depthA1 = 1;
    result.depthB1 = 2;
    result.stepM = 1;
    result.stepN = 1;
    result.stepKa = ProjectionK / kBaseK;
    result.stepKb = 4;
    result.dbL0A = 2;
    result.dbL0B = 2;
    result.dbL0C = 1;
    return result;
}

template <uint32_t ProjectionK>
constexpr static tiling::TCubeTiling kTiling = MakeTiling<ProjectionK>();

template <uint32_t ProjectionK, bool StaticNoTail>
__aicore__ inline void RunProgram(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        uint32_t logical_m) {
    TPipe pipe;
    MatmulImpl<AType, BType, CType, BiasType,
        kConfig<ProjectionK, StaticNoTail>> matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(&kTiling<ProjectionK>, &pipe);
    const uint32_t block = GetBlockIdx();
    GlobalTensor<bfloat16_t> a;
    GlobalTensor<bfloat16_t> b;
    GlobalTensor<bfloat16_t> c;
    a.SetGlobalBuffer(a_gm + block * kBaseM * kMaximumK);
    b.SetGlobalBuffer(b_gm + block * kBaseN * kMaximumK);
    c.SetGlobalBuffer(c_gm + block * kBaseM * kBaseN);

    WaitPreTaskEnd();
    const int64_t begin = GetSystemCycle();
    matmul.SetOrgShape(
        logical_m, kBaseN, ProjectionK, ProjectionK, kBaseN);
    matmul.SetSingleShape(logical_m, kBaseN, ProjectionK);
    matmul.SetTensorA(a);
    matmul.SetTensorB(b);
    matmul.template IterateAll<false>(c, 0);
    PipeBarrier<PIPE_ALL>();
    const int64_t elapsed = GetSystemCycle() - begin;
    SetNextTaskStart();

    GlobalTensor<int64_t> row;
    row.SetGlobalBuffer(result + block * kResultStride);
    row.SetValue(0, elapsed);
    row.SetValue(1, logical_m);
    row.SetValue(2, ProjectionK);
    row.SetValue(3, StaticNoTail ? 1 : 0);
    row.SetValue(4, kAicWorkers);
    row.SetValue(5, kTiling<ProjectionK>.depthA1);
    row.SetValue(6, kTiling<ProjectionK>.depthB1);
    row.SetValue(7, kTiling<ProjectionK>.stepKa);
    row.SetValue(8, kTiling<ProjectionK>.stepKb);
    row.SetValue(9, kBaseN);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(row);
}

}  // namespace

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

extern "C" __global__ __aicore__ void matmul_v2_full_a_scalar_control(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        int64_t logical_m, int64_t projection_k, int64_t static_no_tail) {
    if (GetBlockIdx() >= kAicWorkers
            || (logical_m != 1 && logical_m != kBaseM)
            || (projection_k != 2048 && projection_k != 4096)
            || (static_no_tail != 0 && static_no_tail != 1)
            || (static_no_tail == 1 && logical_m != kBaseM)
            || (static_no_tail == 0 && logical_m != 1)) {
        return;
    }
#define RUN_CELL(K)                                                       \
    if (projection_k == K) {                                              \
        if (static_no_tail) {                                             \
            RunProgram<K, true>(a_gm, b_gm, c_gm, result, logical_m);    \
        } else {                                                          \
            RunProgram<K, false>(a_gm, b_gm, c_gm, result, logical_m);   \
        }                                                                 \
        return;                                                           \
    }
    RUN_CELL(2048)
    RUN_CELL(4096)
#undef RUN_CELL
}
