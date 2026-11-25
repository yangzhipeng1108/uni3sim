// Isolate the source-form InitOutput followed by SyncAll used by CANN's
// MoeComputeExpertTokens key-1003 program.  It contains no production search,
// reduction, input, or operator duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

extern "C" __global__ __aicore__ void compute_expert_tokens_sync_phase(
        __gm__ int32_t* workspace, __gm__ int64_t* output_cycles,
        int64_t elements_per_worker, int64_t iterations) {
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::TPipe pipe;
    const int64_t worker = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(output_cycles);
    if (elements_per_worker <= 0 || elements_per_worker > 512
            || iterations <= 0) {
        output.SetValue(worker, -1);
        return;
    }
    AscendC::GlobalTensor<int32_t> worker_workspace;
    worker_workspace.SetGlobalBuffer(
        workspace + worker * elements_per_worker, elements_per_worker);

    AscendC::InitOutput<int32_t>(
        worker_workspace, elements_per_worker, static_cast<int32_t>(0));
    AscendC::SyncAll();
    const int64_t begin = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::InitOutput<int32_t>(
            worker_workspace, elements_per_worker, static_cast<int32_t>(0));
        AscendC::SyncAll();
    }
    const int64_t end = AscendC::GetSystemCycle();
    output.SetValue(worker, end - begin);
}
