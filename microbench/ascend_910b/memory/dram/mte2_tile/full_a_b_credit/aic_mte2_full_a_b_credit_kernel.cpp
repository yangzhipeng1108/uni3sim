// Independent MTE2-only full-A/B-credit probe for the CANN 8.5.1 MatMulV2
// small-M decode launch family (persistent full-A, split_k=1).
//
// The trace is derived from the CANN 8.5.1 open mat_mul_v3 AL1-full-load
// template (mat_mul_l1_full_load.h): one A1 slot holds the full-K 16xK A
// slab resident for the whole launch, B streams from GM through a two-slot
// B1 double buffer, and consecutive output tiles (waves) carry no explicit
// cross-wave barrier.  The optional credit mode uses zero-work MTE2-to-MTE1
// handoffs to return B1 credits at the queue-defined points.  It
// intentionally omits MTE1 loads, Cube, FixPipe, workspace, AIV, and any
// production/evaluation duration: it is a descriptor/credit-topology
// mechanism probe, not a GEMM.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kB1Slots = 2;
constexpr int64_t kMaxWaves = 24;
constexpr int64_t kOutputSlotsPerBlock = 32;

__aicore__ inline int64_t Align16(int64_t value) {
    return (value + 15) / 16 * 16;
}

__aicore__ inline void CopyNd2Nz(
        AscendC::LocalTensor<bfloat16_t>& destination,
        AscendC::GlobalTensor<bfloat16_t>& source, int64_t rows,
        int64_t columns, int64_t source_row_stride) {
    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = static_cast<uint16_t>(rows);
    parameters.dValue = static_cast<uint16_t>(columns);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(source_row_stride);
    parameters.dstNzC0Stride = static_cast<uint16_t>(Align16(rows));
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

__aicore__ inline void CopyBSlab(
        __gm__ bfloat16_t* gm_b, int64_t block_b_offset, int64_t slab,
        int64_t first_tile, int64_t b_source_tiles, int64_t b_tile_span,
        int64_t b_rows, int64_t b_columns, int64_t b_source_stride,
        int64_t slot, bool preserve_credits, event_t* b_credit_free,
        event_t* b_data_ready, AscendC::LocalTensor<bfloat16_t>& b0,
        AscendC::LocalTensor<bfloat16_t>& b1) {
    if (preserve_credits) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            b_credit_free[slot]);
    }
    // Cyclic source inside this AIC's private footprint partition so the
    // stream stays L2-cold across launches.
    const int64_t tile = (first_tile + slab) % b_source_tiles;
    AscendC::GlobalTensor<bfloat16_t> source;
    source.SetGlobalBuffer(gm_b + block_b_offset + tile * b_tile_span);
    CopyNd2Nz(slot == 0 ? b0 : b1, source, b_rows, b_columns,
              b_source_stride);
    if (preserve_credits) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            b_data_ready[slot]);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_full_a_b_credit(
        __gm__ bfloat16_t* gm_a, __gm__ bfloat16_t* gm_b,
        __gm__ int64_t* gm_output, int64_t a_rows, int64_t a_columns,
        int64_t b_rows, int64_t b_columns, int64_t b_source_stride,
        int64_t b_tile_span, int64_t b_source_tiles, int64_t first_tile,
        int64_t waves, int64_t n_slabs, int64_t mode) {
    using namespace AscendC;
    const bool credits = mode != 0;
    const int64_t block = AscendC::GetBlockIdx();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer;
    AscendC::TBuf<AscendC::TPosition::A1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> b_buffer1;
    pipe.InitBuffer(a_buffer, Align16(a_rows) * a_columns * kElementBytes);
    pipe.InitBuffer(b_buffer0, Align16(b_rows) * b_columns * kElementBytes);
    pipe.InitBuffer(b_buffer1, Align16(b_rows) * b_columns * kElementBytes);
    auto a0 = a_buffer.Get<bfloat16_t>();
    auto b0 = b_buffer0.Get<bfloat16_t>();
    auto b1 = b_buffer1.Get<bfloat16_t>();

    event_t b_credit_free[kB1Slots];
    event_t b_data_ready[kB1Slots];
    if (credits) {
        for (int64_t slot = 0; slot < kB1Slots; ++slot) {
            b_credit_free[slot] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<
                    AscendC::HardEvent::MTE1_MTE2>());
            b_data_ready[slot] = static_cast<event_t>(
                GetTPipePtr()->AllocEventID<
                    AscendC::HardEvent::MTE2_MTE1>());
            // Both B1 slots start with their credit available.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                b_credit_free[slot]);
        }
    }

    const int64_t block_b_offset = block * b_source_tiles * b_tile_span;
    const int64_t total_slabs = waves * n_slabs;

    // Counter wake-up: the AIC system counter does not tick during the
    // first tens of microseconds after kernel entry, so short timed
    // regions read near-zero.  Stream a fixed dummy slab batch through
    // MTE2 first and discard it; timestamps taken afterwards are real.
    // The dummy batch is offset past the timed tiles so the timed section
    // cannot re-read what the wake-up just streamed through L2.
    {
        AscendC::GlobalTensor<bfloat16_t> warm_source;
        #pragma unroll 1
        for (int64_t warm = 0; warm < 32; ++warm) {
            const int64_t tile = (first_tile + total_slabs + warm)
                % b_source_tiles;
            warm_source.SetGlobalBuffer(
                gm_b + block_b_offset + tile * b_tile_span);
            CopyNd2Nz(b0, warm_source, b_rows, b_columns, b_source_stride);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    }

    // Persistent full-A: one descriptor loads the whole 16xK slab once;
    // the source template keeps it resident in L1 for every wave.
    AscendC::GlobalTensor<bfloat16_t> a_source;
    a_source.SetGlobalBuffer(gm_a + block * a_rows * a_columns);
    const int64_t a_start = AscendC::GetSystemCycle();
    CopyNd2Nz(a0, a_source, a_rows, a_columns, a_columns);
    AscendC::PipeBarrier<PIPE_MTE2>();
    const int64_t a_cycles = AscendC::GetSystemCycle() - a_start;

    int64_t wave_cycles[kMaxWaves];
    const int64_t t_start = AscendC::GetSystemCycle();
    if (credits) {
        // Steady two-slot pipeline: prefetch both slots, then each consumed
        // slab returns its credit through a deliberately zero-work MTE1
        // acknowledgement and the freed slot immediately receives the slab
        // two positions ahead.  Wave boundaries add no barrier, matching the
        // source template's barrier-free Iterate loop.
        CopyBSlab(gm_b, block_b_offset, 0, first_tile, b_source_tiles,
                  b_tile_span, b_rows, b_columns, b_source_stride, 0, credits,
                  b_credit_free, b_data_ready, b0, b1);
        if (total_slabs > 1) {
            CopyBSlab(gm_b, block_b_offset, 1, first_tile, b_source_tiles,
                      b_tile_span, b_rows, b_columns, b_source_stride, 1,
                      credits, b_credit_free, b_data_ready, b0, b1);
        }
        int64_t wave = 0;
        int64_t wave_start = t_start;
        #pragma unroll 1
        for (int64_t slab = 0; slab < total_slabs; ++slab) {
            const int64_t slot = slab & 1;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
                b_data_ready[slot]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                b_credit_free[slot]);
            if (slab + 2 < total_slabs) {
                CopyBSlab(gm_b, block_b_offset, slab + 2, first_tile,
                          b_source_tiles, b_tile_span, b_rows, b_columns,
                          b_source_stride, slot, credits, b_credit_free,
                          b_data_ready, b0, b1);
            }
            if ((slab + 1) % n_slabs == 0) {
                const int64_t now = AscendC::GetSystemCycle();
                wave_cycles[wave] = now - wave_start;
                ++wave;
                wave_start = now;
            }
        }
        // Drain every returned credit before the terminal barrier and event
        // release, mirroring the source queue teardown.
        for (int64_t slot = 0; slot < kB1Slots; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                b_credit_free[slot]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t slot = 0; slot < kB1Slots; ++slot) {
            GetTPipePtr()->ReleaseEventID<
                AscendC::HardEvent::MTE1_MTE2>(b_credit_free[slot]);
            GetTPipePtr()->ReleaseEventID<
                AscendC::HardEvent::MTE2_MTE1>(b_data_ready[slot]);
        }
    } else {
        // Descriptor-only drain: issue the wave's slabs with no credit
        // chain and fully drain MTE2 at each wave boundary.
        int64_t wave_start = t_start;
        #pragma unroll 1
        for (int64_t wave = 0; wave < waves; ++wave) {
            #pragma unroll 1
            for (int64_t s = 0; s < n_slabs; ++s) {
                CopyBSlab(gm_b, block_b_offset, wave * n_slabs + s,
                          first_tile, b_source_tiles, b_tile_span, b_rows,
                          b_columns, b_source_stride, s & 1, credits,
                          b_credit_free, b_data_ready, b0, b1);
            }
            AscendC::PipeBarrier<PIPE_MTE2>();
            const int64_t now = AscendC::GetSystemCycle();
            wave_cycles[wave] = now - wave_start;
            wave_start = now;
        }
    }
    const int64_t total_cycles = AscendC::GetSystemCycle() - t_start;

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(gm_output + block * kOutputSlotsPerBlock);
    output.SetValue(0, total_cycles);
    output.SetValue(1, a_cycles);
    output.SetValue(2, total_slabs);
    output.SetValue(3, mode);
    for (int64_t wave = 0; wave < waves; ++wave) {
        output.SetValue(4 + wave, wave_cycles[wave]);
    }
}
