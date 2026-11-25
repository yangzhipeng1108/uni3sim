// Independent Ascend 910B mix AIC+AIV strided GM-to-L1 ND2NZ probe.
//
// Identical AIC work and addressing to aic_mte2_nd2nz_stride_kernel.cpp; the
// only difference is the task type.  CANN FusedInferAttentionScore runs as a
// mix kernel (22 AIC + 44 AIV), while the AIC-only probe does not, so this
// pairs with it to isolate whether the msprof L2 read hit counter keeps its
// AIC-only semantics when AIV subcores are resident.  The AIV side performs a
// small fixed GM->UB copy so the launch really is mixed; its own traffic is
// negligible next to the AIC stream and is reported separately by msprof.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kVectorElements = 128;

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

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_nd2nz_mix(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_output,
        int64_t rows, int64_t columns, int64_t source_row_stride,
        int64_t source_tile_stride, int64_t source_tiles,
        int64_t iterations, int64_t mode, int64_t worker_group,
        int64_t interleave_columns, int64_t prologue_offset) {
    const int64_t block = AscendC::GetBlockIdx();

    if ASCEND_IS_AIV {
        // Keep the launch genuinely mixed without adding a comparable GM
        // stream: one small fixed-size read from the head of the buffer.
        AscendC::TPipe vector_pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> vector_buffer;
        vector_pipe.InitBuffer(
            vector_buffer, kVectorElements * kElementBytes);
        auto vector_local = vector_buffer.Get<bfloat16_t>();
        AscendC::GlobalTensor<bfloat16_t> vector_source;
        vector_source.SetGlobalBuffer(gm_source);
        AscendC::DataCopy(vector_local, vector_source, kVectorElements);
        return;
    }

    const int64_t local_elements = Align16(rows) * columns;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> buffer;
    AscendC::TBuf<AscendC::TPosition::A1> peer_buffer;
    pipe.InitBuffer(buffer, local_elements * kElementBytes);
    pipe.InitBuffer(peer_buffer, local_elements * kElementBytes);
    auto local = buffer.Get<bfloat16_t>();
    auto peer_local = peer_buffer.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> source;
    // interleave_columns != 0 reproduces the CANN FIA n2 addressing: every
    // AIC reads a different `columns`-wide slice of the SAME rows, adjacent
    // in GM, so no byte is read twice yet neighbouring slices can fall in one
    // L2 line.  Otherwise the workers take disjoint tile partitions.
    const int64_t partition = block / worker_group;
    const int64_t block_source_offset = (
        interleave_columns != 0
        ? block * interleave_columns
        : partition * source_tiles * source_tile_stride);
    // The prologue copy primes the MTE2 path, but it reads the SAME tile the
    // timed loop starts on.  With source_tiles == 1 that silently makes any
    // "single pass" cell a two-pass cell, which reads out as a 50% hit that
    // is real reuse rather than a property of one cold traversal.  In
    // interleaved mode, where source_tiles is pinned to 1, prime the path on
    // a disjoint dummy row instead so the measured tile stays cold.
    if (interleave_columns != 0) {
        AscendC::GlobalTensor<bfloat16_t> prime;
        // Per-worker prime region.  A single shared region would let the
        // workers' own prologues hit each other, which lands in the same
        // counters as the adjacent-slice sharing under test.
        prime.SetGlobalBuffer(
            gm_source + prologue_offset + block * source_tile_stride);
        CopyNd2Nz(local, prime, rows, columns, source_row_stride);
    } else {
        source.SetGlobalBuffer(gm_source + block_source_offset);
        CopyNd2Nz(local, source, rows, columns, source_row_stride);
    }
    AscendC::PipeBarrier<PIPE_MTE2>();

    const int64_t start = AscendC::GetSystemCycle();
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const int64_t tile = iteration % source_tiles;
        source.SetGlobalBuffer(
            gm_source + block_source_offset + tile * source_tile_stride);
        if (mode == 0 || (iteration & 1) == 0) {
            CopyNd2Nz(local, source, rows, columns, source_row_stride);
        } else {
            CopyNd2Nz(peer_local, source, rows, columns, source_row_stride);
        }
        if (mode == 0) {
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    }
    if (mode == 1) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, 1);
}
