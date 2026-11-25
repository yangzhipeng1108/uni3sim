// Independent MTE2-only replica of one CANN 8.5.1 MatMulV2 ingress program.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
#ifndef GEMM_SOURCE_A_ROWS
#define GEMM_SOURCE_A_ROWS 128
#endif
#ifndef GEMM_SOURCE_A_COLUMNS
#define GEMM_SOURCE_A_COLUMNS 512
#endif
#ifndef GEMM_SOURCE_B_COLUMNS
#define GEMM_SOURCE_B_COLUMNS 256
#endif
#ifndef GEMM_SOURCE_MAXIMUM_B_ROWS
#define GEMM_SOURCE_MAXIMUM_B_ROWS 256
#endif
constexpr int64_t kARows = GEMM_SOURCE_A_ROWS;
constexpr int64_t kAColumns = GEMM_SOURCE_A_COLUMNS;
constexpr int64_t kMaximumBRows = GEMM_SOURCE_MAXIMUM_B_ROWS;
constexpr int64_t kBColumns = GEMM_SOURCE_B_COLUMNS;
constexpr int64_t kBDescriptorsPerA = kAColumns / kBColumns;

static_assert(kAColumns % kBColumns == 0);
static_assert(2 * (kARows * kAColumns + kMaximumBRows * kBColumns)
              * kElementBytes <= 512 * 1024);

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

__aicore__ inline void RunOutputTile(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        AscendC::LocalTensor<bfloat16_t>& a_local0,
        AscendC::LocalTensor<bfloat16_t>& a_local1,
        AscendC::LocalTensor<bfloat16_t>& b_local0,
        AscendC::LocalTensor<bfloat16_t>& b_local1,
        int64_t m_tile, int64_t n_tile, int64_t k_slabs, int64_t b_rows,
        int64_t source_row_stride) {
    const int64_t a_row_offset = m_tile * kARows * source_row_stride;
    const int64_t b_row_offset = n_tile * b_rows * source_row_stride;
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;
    #pragma unroll 1
    for (int64_t slab = 0; slab < k_slabs; ++slab) {
        const int64_t k_offset = slab * kAColumns;
        const int64_t remaining_columns = source_row_stride - k_offset;
        const int64_t a_columns = (
            remaining_columns < kAColumns ? remaining_columns : kAColumns);
        a_source.SetGlobalBuffer(gm_a_source + a_row_offset + k_offset);
        if ((slab & 1) == 0) {
            CopyNd2Nz(
                a_local0, a_source, kARows, a_columns,
                source_row_stride);
        } else {
            CopyNd2Nz(
                a_local1, a_source, kARows, a_columns,
                source_row_stride);
        }
        #pragma unroll 1
        for (int64_t b_descriptor = 0;
                b_descriptor * kBColumns < a_columns; ++b_descriptor) {
            const int64_t remaining_b_columns = (
                a_columns - b_descriptor * kBColumns);
            const int64_t b_columns = remaining_b_columns < kBColumns
                ? remaining_b_columns : kBColumns;
            b_source.SetGlobalBuffer(
                gm_b_source + b_row_offset + k_offset
                + b_descriptor * kBColumns);
            if ((b_descriptor & 1) == 0) {
                CopyNd2Nz(
                    b_local0, b_source, b_rows, b_columns,
                    source_row_stride);
            } else {
                CopyNd2Nz(
                    b_local1, b_source, b_rows, b_columns,
                    source_row_stride);
            }
        }
    }
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_gemm_source_order(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output, int64_t m_groups,
        int64_t worker_m_tiles, int64_t worker_n_tiles,
        int64_t logical_n_tiles, int64_t output_tiles_per_worker,
        int64_t k_slabs, int64_t b_rows, int64_t source_row_stride) {
    const int64_t entry = AscendC::GetSystemCycle();
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t m_group = block % m_groups;
    const int64_t n_group = block / m_groups;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    pipe.InitBuffer(a_buffer0, kARows * kAColumns * kElementBytes);
    pipe.InitBuffer(a_buffer1, kARows * kAColumns * kElementBytes);
    pipe.InitBuffer(b_buffer0, b_rows * kBColumns * kElementBytes);
    pipe.InitBuffer(b_buffer1, b_rows * kBColumns * kElementBytes);
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = a_buffer1.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();

    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t output_tile = 0;
            output_tile < output_tiles_per_worker; ++output_tile) {
        const int64_t local_m = output_tile % worker_m_tiles;
        const int64_t local_n = output_tile / worker_m_tiles;
        const int64_t m_tile = m_group * worker_m_tiles + local_m;
        const int64_t candidate_n_start = n_group * worker_n_tiles;
        const int64_t maximum_n_start = logical_n_tiles - worker_n_tiles;
        const int64_t n_group_start = (
            candidate_n_start < maximum_n_start
            ? candidate_n_start : maximum_n_start);
        const int64_t n_tile = n_group_start + local_n;
        RunOutputTile(
            gm_a_source, gm_b_source, a_local0, a_local1, b_local0,
            b_local1, m_tile, n_tile, k_slabs, b_rows,
            source_row_stride);
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, output_tiles_per_worker * k_slabs);
    output.SetValue(2, entry);
    output.SetValue(3, end);
}
