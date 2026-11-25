// Isolate the key-1001 MoeComputeExpertTokensInt32SS shared InitOutput plus
// SyncAll phase.  The source initializes its one output from AIV worker zero
// and synchronizes all 48 AIVs before it enters CopyIn/Compute/atomic
// CopyOut.  Those remaining production phases are intentionally absent here.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

extern "C" __global__ __aicore__ void compute_expert_tokens_ss_init_sync_phase(
        __gm__ int32_t* output, __gm__ int64_t* output_cycles,
        int64_t experts, int64_t iterations) {
    if ASCEND_IS_AIC {
        return;
    }
    // The source owns a TPipe for this AIV task before entering InitOutput
    // and SyncAll, even though this isolated phase allocates no UB queues.
    AscendC::TPipe pipe;
    const int64_t worker = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(output_cycles);
    if (experts <= 0 || experts > 256 || iterations <= 0) {
        cycles.SetValue(worker, -1);
        return;
    }
    AscendC::GlobalTensor<int32_t> shared_output;
    shared_output.SetGlobalBuffer(output, experts);

    // Establish the same phase state without including it in the clock.
    if (worker == 0) {
        AscendC::InitOutput<int32_t>(
            shared_output, experts, static_cast<int32_t>(0));
    }
    AscendC::SyncAll();

    const int64_t begin = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        if (worker == 0) {
            AscendC::InitOutput<int32_t>(
                shared_output, experts, static_cast<int32_t>(0));
        }
        AscendC::SyncAll();
    }
    const int64_t end = AscendC::GetSystemCycle();
    cycles.SetValue(worker, end - begin);
}
