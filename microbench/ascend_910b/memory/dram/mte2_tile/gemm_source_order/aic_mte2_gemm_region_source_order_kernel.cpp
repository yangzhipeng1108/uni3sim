// Exact MTE2-only ingress order for the 4096x8192x8192 MatMulV2 program.
#include "kernel_operator.h"
#include "aic_mte2_gemm_region_schedule.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kARows = 128;
constexpr int64_t kAColumns = 512;
constexpr int64_t kBRows = 256;
constexpr int64_t kBColumns = 256;

static_assert(2 * (kARows * kAColumns + kBRows * kBColumns)
              * kElementBytes == 512 * 1024);

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
        int64_t m_tile, int64_t n_tile, int64_t k_slabs) {
    const int64_t source_row_stride = k_slabs * kAColumns;
    const int64_t a_row_offset = m_tile * kARows * source_row_stride;
    const int64_t b_row_offset = n_tile * kBRows * source_row_stride;
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;
    #pragma unroll 1
    for (int64_t slab = 0; slab < k_slabs; ++slab) {
        const int64_t k_offset = slab * kAColumns;
        a_source.SetGlobalBuffer(gm_a_source + a_row_offset + k_offset);
        if ((slab & 1) == 0) {
            CopyNd2Nz(
                a_local0, a_source, kARows, kAColumns,
                source_row_stride);
        } else {
            CopyNd2Nz(
                a_local1, a_source, kARows, kAColumns,
                source_row_stride);
        }
        b_source.SetGlobalBuffer(
            gm_b_source + b_row_offset + k_offset);
        CopyNd2Nz(
            b_local0, b_source, kBRows, kBColumns,
            source_row_stride);
        b_source.SetGlobalBuffer(
            gm_b_source + b_row_offset + k_offset + kBColumns);
        CopyNd2Nz(
            b_local1, b_source, kBRows, kBColumns,
            source_row_stride);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_gemm_region_source_order(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output, int64_t k_slabs,
        int64_t m_regions, int64_t n_regions,
        int64_t m_block, int64_t n_block,
        int64_t m_tiles, int64_t n_tiles) {
    const int64_t block = AscendC::GetBlockIdx();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    pipe.InitBuffer(a_buffer0, kARows * kAColumns * kElementBytes);
    pipe.InitBuffer(a_buffer1, kARows * kAColumns * kElementBytes);
    pipe.InitBuffer(b_buffer0, kBRows * kBColumns * kElementBytes);
    pipe.InitBuffer(b_buffer1, kBRows * kBColumns * kElementBytes);
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = a_buffer1.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();

    int64_t output_tiles = 0;
    int64_t tile_checksum = 0;
    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t region = 0;
            region < m_regions * n_regions; ++region) {
        const int64_t rounds =
            GemmRegionSchedule::WorkerRounds(
                block, region, m_block, n_block);
        #pragma unroll 1
        for (int64_t round = 0; round < rounds; ++round) {
            const auto tile =
                GemmRegionSchedule::WorkerTile(
                    block, region, round, n_regions, m_block, n_block);
            if (tile.m >= m_tiles || tile.n >= n_tiles) {
                continue;
            }
            RunOutputTile(
                gm_a_source, gm_b_source, a_local0, a_local1,
                b_local0, b_local1, tile.m, tile.n, k_slabs);
            ++output_tiles;
            tile_checksum += tile.m * n_tiles + tile.n + 1;
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, output_tiles * k_slabs);
    output.SetValue(2, output_tiles);
    output.SetValue(3, tile_checksum);
}
