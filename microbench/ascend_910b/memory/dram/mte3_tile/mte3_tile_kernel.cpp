// Independent Ascend 910B AIV MTE3 (UB -> GM) mechanism benchmark.
//
// One immutable FP16 UB source tile is initialized before timing.  The timed
// interval writes a large GM footprint in one of three mechanism modes:
//
//   hard_event_completion: every DataCopy is followed by the same
//       MTE3_MTE2 SetFlag/WaitFlag completion edge used by CANN 8.5.1
//       ReshapeAndCache key 200;
//   pipeline: all stores are issued before one terminal MTE3 completion edge.
//   dynamic_params_completion: the same completion edge as
//       hard_event_completion, but DataCopy uses the runtime DataCopyParams
//       overload used by CANN ReshapeAndCache.  Its paired difference from
//       hard_event_completion isolates descriptor setup.
//
// This isolates the byte transfer plus completion edge.  It does not execute
// a production operator or include its MTE2, scalar, or address work.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

template <bool CompleteEveryStore, bool DynamicParams>
__aicore__ inline void RunStores(
        __gm__ half* gm_destination,
        const AscendC::LocalTensor<half>& source0,
        int64_t tile_bytes, int64_t destination_tiles, int64_t iterations) {
    int64_t destination_index = 0;
    const int64_t tile_elements = tile_bytes / sizeof(half);
    const AscendC::DataCopyParams copy_params = {
        1, static_cast<uint16_t>(tile_bytes / 32), 0, 0};
    AscendC::GlobalTensor<half> destination;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        destination.SetGlobalBuffer(
            gm_destination + destination_index * tile_elements);
        if constexpr (DynamicParams) {
            AscendC::DataCopy(destination, source0, copy_params);
        } else {
            AscendC::DataCopy(destination, source0, tile_elements);
        }
        ++destination_index;
        if (destination_index == destination_tiles) {
            destination_index = 0;
        }
        if constexpr (CompleteEveryStore) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
    if constexpr (!CompleteEveryStore) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

extern "C" __global__ __aicore__ void mte3_tile(
        __gm__ uint8_t* gm_destination, __gm__ int64_t* out, int64_t tile_bytes,
        int64_t destination_tiles, int64_t iterations,
        int64_t mode, int64_t source_buffers, int64_t barrier_window) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(out);
    const bool valid_completion_event = mode == 0 && barrier_window == 1;
    const bool valid_pipeline_window = mode == 1 && barrier_window == 0;
    const bool valid_dynamic_completion = (
        mode == 2 && barrier_window == 1);
    if (tile_bytes <= 0 || tile_bytes > 32 * 1024
            || tile_bytes % 32 != 0 || destination_tiles <= 0
            || iterations <= 0 || (mode != 0 && mode != 1 && mode != 2)
            || source_buffers != 1
            || (!valid_completion_event && !valid_pipeline_window
                && !valid_dynamic_completion)
            || iterations % source_buffers != 0
            || (mode == 0 && iterations % barrier_window != 0)) {
        output.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buffer0;
    pipe.InitBuffer(buffer0, tile_bytes);
    auto source0 = buffer0.Get<half>();
    const uint32_t tile_elements = static_cast<uint32_t>(
        tile_bytes / sizeof(half));
    AscendC::Duplicate<half>(source0, static_cast<half>(1.0f), tile_elements);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::GlobalTensor<half> destination;
    destination.SetGlobalBuffer((__gm__ half*)gm_destination);
    AscendC::DataCopy(destination, source0, tile_elements);
    AscendC::PipeBarrier<PIPE_MTE3>();

    const int64_t cycle_start = AscendC::GetSystemCycle();
    if (mode == 1) {
        RunStores<false, false>((__gm__ half*)gm_destination, source0,
                     tile_bytes, destination_tiles, iterations);
    } else if (mode == 0 && barrier_window == 1) {
        RunStores<true, false>((__gm__ half*)gm_destination, source0,
                     tile_bytes, destination_tiles, iterations);
    } else if (mode == 2 && barrier_window == 1) {
        RunStores<true, true>((__gm__ half*)gm_destination, source0,
                     tile_bytes, destination_tiles, iterations);
    } else {
        output.SetValue(0, -3);
        return;
    }
    const int64_t cycle_end = AscendC::GetSystemCycle();
    output.SetValue(0, cycle_end - cycle_start);
}
