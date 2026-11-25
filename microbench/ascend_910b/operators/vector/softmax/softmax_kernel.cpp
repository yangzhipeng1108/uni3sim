// UB-resident AIV softmax-vector mechanism microbenchmark (910B, MB1).
//
// Each measured row executes a complete FP32 ReduceMax(count), elementwise
// shift surrogate, Exp, complete ReduceSum(count), and normalization surrogate.
// The Level-2 reductions cover all n_cols elements; unlike WholeReduce*(mask),
// their count is not limited to one 64-element FP32 repeat.  Input setup and all
// GM/L2 traffic are outside the timed loop, so this measures vector-pipe service
// rather than a production FlashAttention kernel or its end-to-end duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

extern "C" __global__ __aicore__ void softmax_bench(__gm__ int64_t* out,
                                                    int64_t n_cols,
                                                    int64_t n_rows,
                                                    int64_t n_iters) {
    int64_t bid = AscendC::GetBlockIdx();

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> inBuf;
    pipe.InitBuffer(inBuf, (size_t)n_cols * sizeof(float));
    auto in = inBuf.Get<float>();
    AscendC::TBuf<AscendC::TPosition::VECCALC> temporaryBuf;
    // FP32 ReduceMax emits two values per 64-element first-stage repeat;
    // ReduceSum needs no more.  Keep one 32-byte-aligned slot per pair plus a
    // guard block instead of duplicating the full input tensor, so the
    // mechanism remains measurable through the dataset's 32K KV length.
    const size_t temporaryBytes = (
        (((size_t)n_cols + 31) / 32 * sizeof(float) + 31) / 32 * 32 + 32);
    pipe.InitBuffer(temporaryBuf, temporaryBytes);
    auto temporary = temporaryBuf.Get<float>();
    AscendC::TBuf<AscendC::TPosition::VECOUT> rBuf;
    pipe.InitBuffer(rBuf, 64);
    auto res = rBuf.Get<float>();

    // Initialize once outside timing.  Muls(0) below restores a finite, stable
    // input before each Exp without introducing GM traffic.
    AscendC::Duplicate<float>(in, 0.0f, (uint32_t)n_cols);
    AscendC::PipeBarrier<PIPE_V>();

    int64_t c0 = AscendC::GetSystemCycle();
    for (int64_t it = 0; it < n_iters; it++) {
        for (int64_t r = 0; r < n_rows; r++) {
            AscendC::ReduceMax<float>(
                res, in, temporary, (int32_t)n_cols, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(in, in, 0.0f, (uint32_t)n_cols);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(in, in, (uint32_t)n_cols);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float>(
                res, in, temporary, (int32_t)n_cols);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(in, in, 0.0f, (uint32_t)n_cols);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    int64_t c1 = AscendC::GetSystemCycle();
    if (bid < 64) out[bid] = c1 - c0;  // per-block cycles
}
