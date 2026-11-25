// MTE2 tile transfer latency + bandwidth microbench (910B AIV domain).
//
// The correct Little's Law granularity for NPU is the MTE2 BULK TILE transfer,
// not single-element scalar access. This kernel issues N DataCopy(GM->UB) of a
// fixed tile:
//   mode 0 (hard_event_completion): every DataCopy is followed by the exact
//                    MTE2_MTE3 SetFlag/WaitFlag completion edge used by CANN
//                    8.5.1 ReshapeAndCache key 200.
//   mode 1 (pipelined): no barrier, ping-pong UB buffers → MTE2 queue fills →
//                    steady-state = MTE2 tile BANDWIDTH (max outstanding).
//   mode 2 (dynamic_params_completion): the same completion edge as mode 0,
//                    but DataCopy is issued through the runtime DataCopyParams
//                    overload used by CANN ReshapeAndCache.  The paired
//                    mode-2 minus mode-0 interval isolates descriptor setup
//                    without replaying a production operator.
// per_block_BW (Little's Law) = tile_bytes / latency; saturated BW from mode 1.
// Timing: GetSystemCycle (SYS_CNT @ 50MHz, host-calibrated).
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

extern "C" __global__ __aicore__ void mte2_tile(__gm__ uint8_t* gm_src,
                                                __gm__ int64_t* out,
                                                int64_t tile_bytes,
                                                int64_t n_tiles,
                                                int64_t n_iters,
                                                int64_t mode) {
    if (AscendC::GetBlockIdx() != 0) return;

    AscendC::TPipe pipe;
    // Two UB tiles for ping-pong (pipelined mode). tile_bytes <= 96KB to fit.
    AscendC::TBuf<AscendC::TPosition::VECIN> buf0, buf1;
    pipe.InitBuffer(buf0, tile_bytes);
    pipe.InitBuffer(buf1, tile_bytes);
    auto t0 = buf0.Get<uint8_t>();
    auto t1 = buf1.Get<uint8_t>();

    AscendC::GlobalTensor<uint8_t> g;
    g.SetGlobalBuffer(gm_src);
    int64_t n_elems = tile_bytes;  // uint8 → 1 byte/elem

    // Warmup
    AscendC::DataCopy(t0, g, n_elems);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);

    int64_t c0 = AscendC::GetSystemCycle();
    if (mode == 0) {
        // Serial: 1 outstanding → latency per tile
        #pragma unroll 1
        for (int64_t i = 0; i < n_iters; i++) {
            // Walk the complete host-validated footprint (>2x L2).
            g.SetGlobalBuffer(gm_src + (i % n_tiles) * tile_bytes);
            AscendC::DataCopy(t0, g, n_elems);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        }
    } else if (mode == 2) {
        const AscendC::DataCopyParams copy_params = {
            1, static_cast<uint16_t>(tile_bytes / 32), 0, 0};
        #pragma unroll 1
        for (int64_t i = 0; i < n_iters; i++) {
            g.SetGlobalBuffer(gm_src + (i % n_tiles) * tile_bytes);
            AscendC::DataCopy(t0, g, copy_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        }
    } else if (mode == 1) {
        // Pipelined: ping-pong, MTE2 queue fills → bandwidth
        #pragma unroll 1
        for (int64_t i = 0; i < n_iters; i++) {
            auto& t = (i & 1) ? t1 : t0;
            g.SetGlobalBuffer(gm_src + (i % n_tiles) * tile_bytes);
            AscendC::DataCopy(t, g, n_elems);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
    } else {
        AscendC::GlobalTensor<int64_t> error_out;
        error_out.SetGlobalBuffer(out);
        error_out.SetValue(0, -1);
        return;
    }
    int64_t c1 = AscendC::GetSystemCycle();

    uint8_t validation_value;
    if (mode == 0 || mode == 2 || ((n_iters - 1) & 1) == 0) {
        validation_value = t0.GetValue(0);
    } else {
        validation_value = t1.GetValue(0);
    }

    AscendC::GlobalTensor<int64_t> go;
    go.SetGlobalBuffer(out);
    go.SetValue(0, c1 - c0);
    go.SetValue(1, static_cast<int64_t>(validation_value));
}
