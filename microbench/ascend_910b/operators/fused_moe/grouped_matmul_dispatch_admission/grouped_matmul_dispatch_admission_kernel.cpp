// Isolate CANN GMMCompute's asynchronous IterateAll admission for the exact
// BF16 base-M16/base-M64 source programs. The timer excludes MTE, Cube, FixPipe, and
// completion; a single final PIPE_ALL merely makes the probe safe to return.
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace {

constexpr uint32_t kAicCount = 24;
constexpr uint32_t kMaximumIssues = 32;
constexpr uint32_t kResultStrideElements = 48;
constexpr uint32_t kProjectionW1 = 1;
constexpr uint32_t kProjectionW2 = 2;
constexpr uint32_t kProgramM16 = 1;
constexpr uint32_t kProgramM64 = 2;
constexpr uint32_t kMaximumM = 64;
constexpr uint32_t kN = 256;
constexpr uint32_t kMaximumK = 2048;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::NZ, bfloat16_t, false>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;

// Exact CANN 8.5.1 grouped_matmul `matmulCFGUnitFlag` used for the no-quant
// cube-only BF16 path. It is intentionally not a generic Matmul config.
constexpr MatmulConfig kGmmCubeConfig{
    false, false, true, 0, 0, 0, false, false, false, false, false, 0, 0,
    0, 0, 0, 0, 0, true};

constexpr tiling::TCubeTiling MakeGmmTiling(
        uint32_t base_m, uint32_t depth_a1, uint32_t step_ka,
        uint32_t projection_k) {
    tiling::TCubeTiling result{};
    result.usedCoreNum = 1;
    result.M = base_m;
    result.N = kN;
    result.Ka = projection_k;
    result.Kb = projection_k;
    result.singleCoreM = base_m;
    result.singleCoreN = kN;
    result.singleCoreK = projection_k;
    result.baseM = base_m;
    result.baseN = 256;
    result.baseK = 64;
    result.depthA1 = depth_a1;
    result.depthB1 = 8;
    result.stepM = 1;
    result.stepN = 1;
    result.stepKa = step_ka;
    result.stepKb = 4;
    result.dbL0A = 2;
    result.dbL0B = 2;
    result.dbL0C = 1;
    return result;
}

constexpr tiling::TCubeTiling kM16W1Tiling = MakeGmmTiling(16, 120, 60, 2048);
constexpr tiling::TCubeTiling kM16W2Tiling = MakeGmmTiling(16, 120, 60, 512);
constexpr tiling::TCubeTiling kM64W1Tiling = MakeGmmTiling(64, 24, 12, 2048);
constexpr tiling::TCubeTiling kM64W2Tiling = MakeGmmTiling(64, 24, 12, 512);

}  // namespace

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

extern "C" __global__ __aicore__ void grouped_matmul_dispatch_admission(
        __gm__ bfloat16_t* a_gm, __gm__ bfloat16_t* b_gm,
        __gm__ bfloat16_t* c_gm, __gm__ int64_t* result,
        int64_t issue_count, int64_t projection, int64_t program) {
    const uint32_t block = GetBlockIdx();
    if (block >= kAicCount || issue_count <= 0
            || issue_count > static_cast<int64_t>(kMaximumIssues)
            || (projection != kProjectionW1 && projection != kProjectionW2)
            || (program != kProgramM16 && program != kProgramM64)) {
        return;
    }

    const uint32_t projection_k = projection == kProjectionW1 ? 2048U : 512U;
    const uint32_t base_m = program == kProgramM16 ? 16U : 64U;
    const tiling::TCubeTiling* tiling = program == kProgramM16
        ? (projection == kProjectionW1 ? &kM16W1Tiling : &kM16W2Tiling)
        : (projection == kProjectionW1 ? &kM64W1Tiling : &kM64W2Tiling);
    TPipe pipe;
    MatmulImpl<AType, BType, CType, BiasType, kGmmCubeConfig> matmul;
    // GMM_CUBE_IMP executes this client object on the AIC sub-block only.
    matmul.SetSubBlockIdx(0);
    matmul.Init(tiling, &pipe);
    GlobalTensor<bfloat16_t> a;
    GlobalTensor<bfloat16_t> b;
    GlobalTensor<bfloat16_t> c;
    a.SetGlobalBuffer(a_gm + block * kMaximumM * kMaximumK);
    b.SetGlobalBuffer(b_gm);
    c.SetGlobalBuffer(c_gm + block * kMaximumIssues * kMaximumM * kN);

    // Source has exactly one pre-task wait before its group loop. There is no
    // per-tile wait: a timestamp grows only when client admission itself does.
    WaitPreTaskEnd();
    for (int64_t issue = 0; issue < issue_count; ++issue) {
        const int64_t begin = GetSystemCycle();
        matmul.SetOrgShape(base_m, kN, projection_k, projection_k, kN);
        matmul.SetSingleShape(base_m, kN, projection_k);
        matmul.SetTensorA(a);
        matmul.SetTensorB(b);
        matmul.template IterateAll<false>(c[issue * kMaximumM * kN], 0);
        const int64_t end = GetSystemCycle();
        GlobalTensor<int64_t> row;
        row.SetGlobalBuffer(result + block * kResultStrideElements);
        row.SetValue(issue, end - begin);
    }
    // Never include the drain in a post/admission sample.
    PipeBarrier<PIPE_ALL>();
    SetNextTaskStart();

    GlobalTensor<int64_t> row;
    row.SetGlobalBuffer(result + block * kResultStrideElements);
    row.SetValue(32, static_cast<int64_t>(tiling->baseM));
    row.SetValue(33, static_cast<int64_t>(tiling->baseN));
    row.SetValue(34, static_cast<int64_t>(tiling->baseK));
    row.SetValue(35, static_cast<int64_t>(tiling->depthA1));
    row.SetValue(36, static_cast<int64_t>(tiling->depthB1));
    row.SetValue(37, static_cast<int64_t>(tiling->stepKa));
    row.SetValue(38, static_cast<int64_t>(tiling->stepKb));
    row.SetValue(39, static_cast<int64_t>(tiling->dbL0A));
    row.SetValue(40, static_cast<int64_t>(tiling->dbL0B));
    row.SetValue(41, static_cast<int64_t>(tiling->dbL0C));
    row.SetValue(42, projection);
    row.SetValue(43, issue_count);
    row.SetValue(44, program);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(row);
}
