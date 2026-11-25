// Full RmsNorm pipeline microbench (910B AIV domain).
// Measures the COMPLETE rmsnorm AIV pipe latency vs hidden:
// DataCopy(GM->UB) + ReduceSum(count) + Muls(scale) + DataCopy(UB->GM)
// This combines mechanisms and is diagnostic only, not a Forward parameter.
// GetSystemCycle = kernel-internal, no host overhead.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

extern "C" __global__ __aicore__ void rmsnorm_full(__gm__ uint8_t* gm_src,
                                                     __gm__ uint8_t* gm_dst,
                                                     __gm__ int64_t* out,
                                                     int64_t n_elems,
                                                     int64_t n_iters) {
    if (AscendC::GetBlockIdx() != 0) return;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> inBuf;
    pipe.InitBuffer(inBuf, (size_t)n_elems * sizeof(float));
    auto in = inBuf.Get<float>();

    AscendC::TBuf<AscendC::TPosition::VECOUT> outBuf;
    pipe.InitBuffer(outBuf, (size_t)n_elems * sizeof(float));
    auto y = outBuf.Get<float>();

    AscendC::TBuf<AscendC::TPosition::VECCALC> resBuf;
    pipe.InitBuffer(resBuf, 64);
    auto res = resBuf.Get<float>();

    AscendC::GlobalTensor<float> g;
    g.SetGlobalBuffer((__gm__ float*)gm_src);
    AscendC::GlobalTensor<float> g_out;
    g_out.SetGlobalBuffer((__gm__ float*)gm_dst);

    // Warmup
    AscendC::DataCopy(in, g, (size_t)n_elems);
    AscendC::PipeBarrier<PIPE_MTE2>();
    // `y` is scratch for ReduceSum and is overwritten by Muls immediately
    // afterward, avoiding a third full-row UB allocation.
    AscendC::ReduceSum<float>(
        res, in, y, static_cast<int32_t>(n_elems));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(y, in, (float)0.5, (uint32_t)n_elems);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopy(g_out, y, (size_t)n_elems);
    AscendC::PipeBarrier<PIPE_MTE3>();

    int64_t c0 = AscendC::GetSystemCycle();
    for (int64_t i = 0; i < n_iters; i++) {
        AscendC::DataCopy(in, g, (size_t)n_elems);
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::ReduceSum<float>(
            res, in, y, static_cast<int32_t>(n_elems));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(y, in, (float)0.5, (uint32_t)n_elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(g_out, y, (size_t)n_elems);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
    int64_t c1 = AscendC::GetSystemCycle();
    out[0] = c1 - c0;
}
