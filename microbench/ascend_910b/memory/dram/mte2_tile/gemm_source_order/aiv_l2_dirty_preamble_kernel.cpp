// Independent AIV write preamble for GEMM shared-L2 arbitration probes.
//
// Each AIV writes a disjoint subset of one retained GM footprint.  The
// following AIC source-order probe measures whether replacing that write
// footprint changes its read completion.  No production GEMM arithmetic or
// duration is consumed by this mechanism benchmark.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kTileBytes = 32 * 1024;

}  // namespace

extern "C" __global__ __aicore__ void aiv_l2_dirty_preamble(
        __gm__ uint8_t* gm_destination, int64_t total_bytes) {
    const int64_t worker = AscendC::GetBlockIdx();
    const int64_t workers = AscendC::GetBlockNum();
    if (total_bytes <= 0 || total_bytes % kTileBytes != 0
            || workers <= 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> source_buffer;
    pipe.InitBuffer(source_buffer, kTileBytes);
    auto source = source_buffer.Get<half>();
    AscendC::Duplicate<half>(
        source, static_cast<half>(1.0f),
        static_cast<uint32_t>(kTileBytes / sizeof(half)));
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::GlobalTensor<half> destination;
    const int64_t tile_count = total_bytes / kTileBytes;
    for (int64_t tile = worker; tile < tile_count; tile += workers) {
        destination.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(gm_destination)
            + tile * kTileBytes / sizeof(half));
        AscendC::DataCopy(
            destination, source,
            static_cast<uint32_t>(kTileBytes / sizeof(half)));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}
