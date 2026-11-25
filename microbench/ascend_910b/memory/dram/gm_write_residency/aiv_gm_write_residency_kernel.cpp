// Independent Ascend 910B AIV GM write-back residency probe.
//
// Each AIV writes its own disjoint contiguous GM region through ordinary
// UB-to-GM DataCopy, exactly the path CANN FIA uses for its Vec1/Vec2
// workspace and final output.  Nothing is read unless `readback` asks for
// it, and the stored value is a constant, so this is not a production
// operator: it isolates only how much of a GM write stream reaches DRAM.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kTileElements = 8192;   // 16 KiB per DataCopy

}  // namespace

extern "C" __global__ __aicore__ void aiv_gm_write_residency(
        __gm__ half* destination, __gm__ half* sink,
        int64_t elements_per_worker, int64_t passes, int64_t readback) {
    if (elements_per_worker <= 0 || passes <= 0) {
        return;
    }
    const int64_t worker = AscendC::GetBlockIdx();
    const int64_t worker_offset = worker * elements_per_worker;

    AscendC::GlobalTensor<half> global_destination;
    global_destination.SetGlobalBuffer(destination + worker_offset);
    AscendC::GlobalTensor<half> global_sink;
    global_sink.SetGlobalBuffer(sink + worker * kTileElements);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> output;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> input;
    pipe.InitBuffer(output, 2, kTileElements * sizeof(half));
    pipe.InitBuffer(input, 2, kTileElements * sizeof(half));

    const int64_t tiles = elements_per_worker / kTileElements;
    for (int64_t pass = 0; pass < passes; ++pass) {
        #pragma unroll 1
        for (int64_t tile = 0; tile < tiles; ++tile) {
            auto local = output.AllocTensor<half>();
            AscendC::Duplicate<half>(
                local, static_cast<half>(1.0f), kTileElements);
            output.EnQue(local);
            local = output.DeQue<half>();
            AscendC::DataCopy(
                global_destination[tile * kTileElements], local,
                kTileElements);
            output.FreeTensor(local);
        }
    }
    if (readback != 0) {
        // Re-read what was just written, the way FIA's Vec1 producer and MM2
        // consumer share a workspace inside one launch.
        #pragma unroll 1
        for (int64_t tile = 0; tile < tiles; ++tile) {
            auto local = input.AllocTensor<half>();
            AscendC::DataCopy(
                local, global_destination[tile * kTileElements],
                kTileElements);
            input.EnQue(local);
            local = input.DeQue<half>();
            if (tile == tiles - 1) {
                auto store = output.AllocTensor<half>();
                AscendC::Duplicate<half>(
                    store, static_cast<half>(2.0f), kTileElements);
                output.EnQue(store);
                store = output.DeQue<half>();
                AscendC::DataCopy(global_sink, store, kTileElements);
                output.FreeTensor(store);
            }
            input.FreeTensor(local);
        }
    }
}
