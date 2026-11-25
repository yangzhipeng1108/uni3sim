// Independent Ascend 910B AIC Fixpipe (L0C-to-GM) write-back residency probe.
//
// Pairs with aiv_gm_write_residency, which covers the AIV UB-to-GM MTE3 path.
// CANN FIA drives its mm1Res/mm2Res workspace through Fixpipe and its
// vec1Res/vec2Res workspace through MTE3, so the two engines need separate
// evidence.  A small BF16 MMAD initializes one L0C tile before the measured
// region; the region then stores that tile into a per-core ring of `slots`
// GM slots, `passes` times, so the distinct extent and the written volume can
// be varied independently.  No production operator is reproduced.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kM = 128;
constexpr int64_t kN = 128;
constexpr int64_t kK = 16;

}  // namespace

extern "C" __global__ __aicore__ void aic_fixpipe_write_residency(
        __gm__ bfloat16_t* gm_a, __gm__ bfloat16_t* gm_b,
        __gm__ float* destination, int64_t slots, int64_t passes) {
    (void)gm_a;
    (void)gm_b;
    const int64_t core = AscendC::GetBlockIdx();
    const int64_t tile_elements = kM * kN;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_buffer;
    pipe.InitBuffer(l0c_buffer, tile_elements * sizeof(float));
    // The stored values are irrelevant to a write-traffic probe, so no
    // MMAD is issued: L0C is stored as allocated.
    auto l0c = l0c_buffer.Get<float>();

    if (passes <= 0 || slots <= 0) {
        // Floor cell: the launch still runs its MMAD and its control path but
        // stores nothing, which isolates any per-launch DRAM write cost.
        return;
    }

    AscendC::GlobalTensor<float> global_destination;
    global_destination.SetGlobalBuffer(
        destination + core * slots * tile_elements);

    AscendC::FixpipeParamsV220 parameters;
    parameters.mSize = static_cast<uint16_t>(kM);
    parameters.nSize = static_cast<uint16_t>(kN);
    parameters.srcStride = static_cast<uint16_t>(kM);
    parameters.dstStride = static_cast<uint16_t>(kN);
    parameters.ndNum = 1;

    #pragma unroll 1
    for (int64_t pass = 0; pass < passes; ++pass) {
        const int64_t slot = pass % slots;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
            global_destination[slot * tile_elements], l0c, parameters);
    }
    AscendC::PipeBarrier<PIPE_FIX>();
}
