// Diagnostic complete-row reduction pipeline (910B AIV domain).
// Per iteration: GM->UB DataCopy + Level-2 ReduceSum(count) + barriers.
// This combines mechanisms and is not a Forward parameter; use the separate
// reduce_sum_latency benchmark for UB-resident reduction service latency.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

extern "C" __global__ __aicore__ void norm_reduce(__gm__ uint8_t* gm_src,
                                                   __gm__ int64_t* out,
                                                   int64_t n_elems,
                                                   int64_t n_iters) {
    if (AscendC::GetBlockIdx() != 0) return;

    AscendC::TPipe pipe;
    // Input buffer (fp32, n_elems * 4 bytes)
    AscendC::TBuf<AscendC::TPosition::VECIN> inBuf;
    pipe.InitBuffer(inBuf, (size_t)n_elems * sizeof(float));
    auto in = inBuf.Get<float>();

    // Output buffer for the reduce result (1 float = 4 bytes, round up to 64)
    AscendC::TBuf<AscendC::TPosition::VECOUT> outBuf;
    pipe.InitBuffer(outBuf, 64);
    auto res = outBuf.Get<float>();

    AscendC::TBuf<AscendC::TPosition::VECCALC> temporaryBuf;
    pipe.InitBuffer(
        temporaryBuf, static_cast<size_t>(n_elems) * sizeof(float));
    auto temporary = temporaryBuf.Get<float>();

    AscendC::GlobalTensor<float> g;
    g.SetGlobalBuffer((__gm__ float*)gm_src);

    // Warmup
    AscendC::DataCopy(in, g, (size_t)n_elems);
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::ReduceSum<float>(
        res, in, temporary, static_cast<int32_t>(n_elems));
    AscendC::PipeBarrier<PIPE_V>();

    int64_t c0 = AscendC::GetSystemCycle();
    for (int64_t i = 0; i < n_iters; i++) {
        AscendC::DataCopy(in, g, (size_t)n_elems);
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::ReduceSum<float>(
            res, in, temporary, static_cast<int32_t>(n_elems));
        AscendC::PipeBarrier<PIPE_V>();
    }
    int64_t c1 = AscendC::GetSystemCycle();
    out[0] = c1 - c0;  // total cycles
}
