// Independent source-shaped MatMulV2 A1/B1 cold-pair MTE2 probe.
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

extern "C" __global__ __aicore__ void aic_mte2_legacy_l1_pair(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output, int64_t a_rows, int64_t a_columns,
        int64_t a_source_row_stride, int64_t a_source_tile_stride,
        int64_t a_source_tiles, int64_t b_rows, int64_t b_columns,
        int64_t b_source_row_stride, int64_t b_source_tile_stride,
        int64_t b_source_tiles, int64_t pairs_per_program, int64_t programs,
        int64_t drain_after_each_pair) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t a_local_elements = Align16(a_rows) * a_columns;
    const int64_t b_local_elements = Align16(b_rows) * b_columns;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    pipe.InitBuffer(a_buffer0, a_local_elements * kElementBytes);
    pipe.InitBuffer(a_buffer1, a_local_elements * kElementBytes);
    pipe.InitBuffer(b_buffer0, b_local_elements * kElementBytes);
    pipe.InitBuffer(b_buffer1, b_local_elements * kElementBytes);
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = a_buffer1.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();

    const int64_t a_block_offset = block * a_source_tiles
        * a_source_tile_stride;
    const int64_t b_block_offset = block * b_source_tiles
        * b_source_tile_stride;
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;
    a_source.SetGlobalBuffer(gm_a_source + a_block_offset);
    b_source.SetGlobalBuffer(gm_b_source + b_block_offset);
    CopyNd2Nz(
        a_local0, a_source, a_rows, a_columns, a_source_row_stride);
    CopyNd2Nz(
        b_local0, b_source, b_rows, b_columns, b_source_row_stride);
    AscendC::PipeBarrier<PIPE_MTE2>();

    const int64_t start = AscendC::GetSystemCycle();
    for (int64_t program = 0; program < programs; ++program) {
        for (int64_t pair = 0; pair < pairs_per_program; ++pair) {
            const int64_t sequence = program * pairs_per_program + pair;
            a_source.SetGlobalBuffer(
                gm_a_source + a_block_offset
                + (sequence % a_source_tiles) * a_source_tile_stride);
            b_source.SetGlobalBuffer(
                gm_b_source + b_block_offset
                + (sequence % b_source_tiles) * b_source_tile_stride);
            auto a_local = (pair & 1) == 0 ? a_local0 : a_local1;
            auto b_local = (pair & 1) == 0 ? b_local0 : b_local1;
            CopyNd2Nz(
                a_local, a_source, a_rows, a_columns,
                a_source_row_stride);
            CopyNd2Nz(
                b_local, b_source, b_rows, b_columns,
                b_source_row_stride);
            if (drain_after_each_pair == 1) {
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
        }
        if (drain_after_each_pair == 0) {
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    }
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, 1);
}
