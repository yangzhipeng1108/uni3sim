// Independent Ascend 910B AIC GM-to-L1 ND2NZ service benchmark.
//
// This is one memory mechanism, not an attention kernel.  It measures the
// public DataCopy(Nd2NzParams) operation used by CANN FIA CopyQ/K/VToL1.
// The serial mode preserves the source's MTE2->MTE1 completion dependency;
// the stream mode measures the same descriptor without an intervening wait.
// The partitioned-stream mode gives every active AIC a disjoint position in
// one shared source ring, allowing a footprint larger than L2 to measure the
// aggregate cold-memory service without duplicating a production operator.
// The paired-stream modes issue a BF16 GroupedMatmul base-K operand pair
// (32x64 or 128x64 A and 64x256 B) to their distinct A1/B1 destinations.
// Mode 3 drains once after the complete stream; mode 4 drains after each
// source-defined stepK=4 group.  They isolate MTE2 descriptor credit/drain
// behavior without Cube arithmetic, output traffic, expert routing, or
// production duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;

__aicore__ inline int64_t Align16(int64_t value) {
    return (value + 15) / 16 * 16;
}

__aicore__ inline void CopyNd2Nz(
        AscendC::LocalTensor<bfloat16_t>& destination,
        AscendC::GlobalTensor<bfloat16_t>& source, int64_t rows,
        int64_t columns) {
    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = static_cast<uint16_t>(rows);
    parameters.dValue = static_cast<uint16_t>(columns);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(columns);
    parameters.dstNzC0Stride = static_cast<uint16_t>(Align16(rows));
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_nd2nz(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_output,
        int64_t rows, int64_t columns, int64_t source_tiles,
        int64_t iterations, int64_t mode) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t tile_elements = rows * columns;
    const int64_t local_elements = Align16(rows) * columns;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> paired_buffer;
    pipe.InitBuffer(buffer0, local_elements * kElementBytes);
    pipe.InitBuffer(buffer1, local_elements * kElementBytes);
    if (mode == 3 || mode == 4) {
        pipe.InitBuffer(paired_buffer, 64 * 256 * kElementBytes);
    }
    auto local0 = buffer0.Get<bfloat16_t>();
    auto local1 = buffer1.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> source;
    source.SetGlobalBuffer(gm_source);
    CopyNd2Nz(local0, source, rows, columns);
    if (mode == 3 || mode == 4) {
        auto paired_local = paired_buffer.Get<bfloat16_t>();
        source.SetGlobalBuffer(gm_source + tile_elements);
        CopyNd2Nz(paired_local, source, 64, 256);
    }
    AscendC::PipeBarrier<PIPE_MTE2>();

    const int64_t start = AscendC::GetSystemCycle();
    if (mode == 0) {
        for (int64_t iteration = 0; iteration < iterations; ++iteration) {
            const int64_t tile = (iteration + block) % source_tiles;
            source.SetGlobalBuffer(gm_source + tile * tile_elements);
            CopyNd2Nz(local0, source, rows, columns);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    } else if (mode == 1 || mode == 2) {
        for (int64_t iteration = 0; iteration < iterations; ++iteration) {
            const int64_t tile = mode == 1
                ? (iteration + block) % source_tiles
                : (iteration * AscendC::GetBlockNum() + block) % source_tiles;
            source.SetGlobalBuffer(gm_source + tile * tile_elements);
            if ((iteration & 1) == 0) {
                CopyNd2Nz(local0, source, rows, columns);
            } else {
                CopyNd2Nz(local1, source, rows, columns);
            }
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
    } else if (mode == 3 || mode == 4) {
        auto paired_local = paired_buffer.Get<bfloat16_t>();
        const int64_t paired_tile_elements = 64 * 256;
        const int64_t pair_span_tiles = 1 + paired_tile_elements / tile_elements;
        const int64_t pair_slots = source_tiles / pair_span_tiles;
        for (int64_t iteration = 0; iteration < iterations; ++iteration) {
            const int64_t pair = (
                iteration * AscendC::GetBlockNum() + block) % pair_slots;
            source.SetGlobalBuffer(
                gm_source + pair * pair_span_tiles * tile_elements);
            CopyNd2Nz(local0, source, rows, 64);
            source.SetGlobalBuffer(
                gm_source + (pair * pair_span_tiles + 1) * tile_elements);
            CopyNd2Nz(paired_local, source, 64, 256);
            if (mode == 4 && (iteration & 3) == 3) {
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
        }
        if (mode == 3) {
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    } else {
        AscendC::GlobalTensor<int64_t> output;
        output.SetGlobalBuffer(
            gm_output + block * kCycleSlotBytes / sizeof(int64_t));
        output.SetValue(0, -1);
        return;
    }
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, 1);
}
