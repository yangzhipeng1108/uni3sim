// AIC cube mini-GEMM microbench (910B) — Matmul client API. Real MMAD consumer
// -> LoadData(B GM->L1) genuinely L2-bound. Mirrors mat_mul_deterministic_splitk
// _kernel.h:505-528 exactly: constexpr static locals in the __aicore__ kernel,
// singleCoreM=0 (GetMatmulApiTiling uses basicM/N/K path). SWEEP n_experts runtime.
#include "kernel_operator.h"
#include "adv_api/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

using A_T = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using B_T = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using C_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BIAS_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;

// BM/BN/BK are build-time constants. The host rejects any runtime shape that
// differs from them, so the reported logical bytes describe the compiled MMAD.
#ifndef BN
#error "BN must be supplied by CMake through BN_VAL"
#endif
#ifndef BM
#error "BM must be supplied by CMake through BM_VAL"
#endif
#ifndef BK
#error "BK must be supplied by CMake through BK_VAL"
#endif

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

extern "C" __global__ __aicore__ void mini_gemm(__gm__ half* aGM, __gm__ half* bGM,
                                                 __gm__ half* cGM, __gm__ int64_t* out,
                                                 int64_t n_experts, int64_t b_stride_elems) {
    constexpr int32_t BM_ = BM, BN_ = BN, BK_ = BK;  // BM/BN/BK are -D macros (literals)
    constexpr static MatmulShapeParams shapeParams = {(uint32_t)BM_, (uint32_t)BN_, (uint32_t)BK_, (uint32_t)BM_, (uint32_t)BN_, (uint32_t)BK_};
    constexpr static MatmulQuantParams quantParams = {false, false};
    constexpr static MatmulBatchParams batchParams = {false, BatchMode::NONE};
    constexpr static MatmulFuncParams funcParams = {false, false, false, false, 0,
        IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true};
    constexpr static MatmulConfig mmCFG =
        GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams, quantParams, batchParams, funcParams);
    constexpr static MatmulApiStaticTiling STATIC_TILING =
        GetMatmulApiTiling<A_T, B_T, C_T, BIAS_T>(mmCFG);
    constexpr tiling::TCubeTiling TILING = []() {
        const auto& s = STATIC_TILING;
        tiling::TCubeTiling t{};
        t.usedCoreNum = 1; t.M = BM; t.N = BN; t.Ka = BK; t.Kb = BK;
        t.singleCoreM = BM; t.singleCoreN = BN; t.singleCoreK = BK;
        t.baseM = s.baseM; t.baseN = s.baseN; t.baseK = s.baseK;
        t.depthA1 = s.depthA1; t.depthB1 = s.depthB1;
        t.stepM = s.stepM; t.stepN = s.stepN;
        t.isBias = s.isBias; t.transLength = s.transLength;
        t.iterateOrder = s.iterateOrder; t.shareMode = s.shareMode;
        t.shareL1Size = s.shareL1Size; t.shareL0CSize = s.shareL0CSize; t.shareUbSize = s.shareUbSize;
        t.stepKa = s.stepKa; t.stepKb = s.stepKb;
        t.depthAL1CacheUB = s.depthAL1CacheUB; t.depthBL1CacheUB = s.depthBL1CacheUB;
        t.dbL0A = s.dbL0A; t.dbL0B = s.dbL0B; t.dbL0C = s.dbL0C;
        t.ALayoutInfoB = s.ALayoutInfoB; t.ALayoutInfoS = s.ALayoutInfoS;
        t.BLayoutInfoB = s.BLayoutInfoB; t.BLayoutInfoS = s.BLayoutInfoS;
        return t;
    }();

    TPipe pipe;
    MatmulImpl<A_T, B_T, C_T, BIAS_T, STATIC_TILING> mm;
    mm.Init(&TILING, &pipe);
    GlobalTensor<half> ga, gb, gc;
    const int64_t block_idx = GetBlockIdx();
    ga.SetGlobalBuffer(aGM);
    gb.SetGlobalBuffer(bGM);
    gc.SetGlobalBuffer(cGM + block_idx * BM * BN);
    mm.SetOrgShape(BM, BN, BK, BK, BN);
    mm.SetTensorA(ga);
    for (int64_t e = 0; e < n_experts; e++) { mm.SetTensorB(gb[e * b_stride_elems]); mm.IterateAll(gc, false); }
    AscendC::PipeBarrier<PIPE_FIX>();
    int64_t c0 = GetSystemCycle();
    for (int64_t e = 0; e < n_experts; e++) { mm.SetTensorB(gb[e * b_stride_elems]); mm.IterateAll(gc, false); }
    AscendC::PipeBarrier<PIPE_FIX>();
    int64_t c1 = GetSystemCycle();
    out[block_idx] = c1 - c0;
}
