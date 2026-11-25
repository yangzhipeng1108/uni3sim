// Independent A4/B4 MTE2-to-MTE1 completion probe for a MatMulV3 region.
// It consumes four 64-K pairs per 256-K A1/B1 descriptor and has no Cube or
// output path, isolating source credit release from production GEMM duration.
#include "kernel_operator.h"
#include "aic_mte2_gemm_region_schedule.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kARows = 128;
constexpr int64_t kAColumns = 256;
constexpr int64_t kBRows = 256;
constexpr int64_t kBColumns = 256;
constexpr int64_t kBaseK = 64;

static_assert(2 * (kARows * kAColumns + kBRows * kBColumns)
              * kElementBytes <= 512 * 1024);
static_assert(2 * (kARows * kBaseK + kBaseK * kBColumns)
              * kElementBytes == 96 * 1024);

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

__aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    constexpr int64_t block_use_m = kARows / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_k),
        static_cast<uint16_t>(block_use_m), 0, 0, false, 0};
    for (int64_t index = 0; index < block_use_m; ++index) {
        AscendC::LoadData(
            destination[index * block_use_k * 256], source[index * 256],
            parameters);
    }
}

__aicore__ inline void LoadB(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source,
        int64_t base_pair) {
    constexpr int64_t block_use_n = kBColumns / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_n),
        static_cast<uint16_t>(kBRows / 16), 0, 0, true, 0};
    for (int64_t index = 0; index < block_use_k; ++index) {
        AscendC::LoadData(
            destination[index * block_use_n * 256],
            source[base_pair * kBaseK * kBColumns + index * 256],
            parameters);
    }
}

__aicore__ inline void ConsumeBaseKPair(
        AscendC::TQue<AscendC::TPosition::A2, 2>& l0a_queue,
        AscendC::TQue<AscendC::TPosition::B2, 2>& l0b_queue,
        const AscendC::LocalTensor<bfloat16_t>& a_local,
        const AscendC::LocalTensor<bfloat16_t>& b_local,
        int64_t base_pair) {
    auto l0a = l0a_queue.AllocTensor<bfloat16_t>();
    auto l0b = l0b_queue.AllocTensor<bfloat16_t>();
    LoadA(l0a, a_local);
    LoadB(l0b, b_local, base_pair);
    l0a_queue.EnQue(l0a);
    l0b_queue.EnQue(l0b);
    auto completed_l0a = l0a_queue.DeQue<bfloat16_t>();
    auto completed_l0b = l0b_queue.DeQue<bfloat16_t>();
    l0a_queue.FreeTensor(completed_l0a);
    l0b_queue.FreeTensor(completed_l0b);
}

__aicore__ inline void RunOutputTile(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        AscendC::TQue<AscendC::TPosition::A1, 2>& a_queue,
        AscendC::TQue<AscendC::TPosition::B1, 2>& b_queue,
        AscendC::TQue<AscendC::TPosition::A2, 2>& l0a_queue,
        AscendC::TQue<AscendC::TPosition::B2, 2>& l0b_queue,
        int64_t m_tile, int64_t n_tile, int64_t k_groups) {
    const int64_t source_row_stride = k_groups * kAColumns;
    const int64_t a_row_offset = m_tile * kARows * source_row_stride;
    const int64_t b_row_offset = n_tile * kBRows * source_row_stride;
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;
    #pragma unroll 1
    for (int64_t group = 0; group < k_groups; ++group) {
        const int64_t k_offset = group * kAColumns;
        a_source.SetGlobalBuffer(gm_a_source + a_row_offset + k_offset);
        auto a_producer = a_queue.AllocTensor<bfloat16_t>();
        CopyNd2Nz(
            a_producer, a_source, kARows, kAColumns, source_row_stride);
        a_queue.EnQue(a_producer);
        auto a_consumer = a_queue.DeQue<bfloat16_t>();
        b_source.SetGlobalBuffer(gm_b_source + b_row_offset + k_offset);
        auto b_producer = b_queue.AllocTensor<bfloat16_t>();
        CopyNd2Nz(
            b_producer, b_source, kBRows, kBColumns, source_row_stride);
        b_queue.EnQue(b_producer);
        auto b_consumer = b_queue.DeQue<bfloat16_t>();
        #pragma unroll 1
        for (int64_t base_pair = 0; base_pair < kAColumns / kBaseK;
                ++base_pair) {
            ConsumeBaseKPair(
                l0a_queue, l0b_queue, a_consumer, b_consumer, base_pair);
        }
        b_queue.FreeTensor(b_consumer);
        a_queue.FreeTensor(a_consumer);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void
aic_mte2_gemm_region_a4b4_ingress_completion(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output, int64_t k_groups,
        int64_t m_regions, int64_t n_regions,
        int64_t m_block, int64_t n_block,
        int64_t m_tiles, int64_t n_tiles) {
    const int64_t block = AscendC::GetBlockIdx();
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 2> a_queue;
    AscendC::TQue<AscendC::TPosition::B1, 2> b_queue;
    AscendC::TQue<AscendC::TPosition::A2, 2> l0a_queue;
    AscendC::TQue<AscendC::TPosition::B2, 2> l0b_queue;
    pipe.InitBuffer(a_queue, 2, kARows * kAColumns * kElementBytes);
    pipe.InitBuffer(b_queue, 2, kBRows * kBColumns * kElementBytes);
    pipe.InitBuffer(l0a_queue, 2, kARows * kBaseK * kElementBytes);
    pipe.InitBuffer(l0b_queue, 2, kBaseK * kBColumns * kElementBytes);

    int64_t output_tiles = 0;
    int64_t tile_checksum = 0;
    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t region = 0; region < m_regions * n_regions; ++region) {
        const int64_t rounds = GemmRegionSchedule::WorkerRounds(
            block, region, m_block, n_block);
        #pragma unroll 1
        for (int64_t round = 0; round < rounds; ++round) {
            const auto tile = GemmRegionSchedule::WorkerTile(
                block, region, round, n_regions, m_block, n_block);
            if (tile.m >= m_tiles || tile.n >= n_tiles) {
                continue;
            }
            RunOutputTile(
                gm_a_source, gm_b_source, a_queue, b_queue,
                l0a_queue, l0b_queue, tile.m, tile.n, k_groups);
            ++output_tiles;
            tile_checksum += tile.m * n_tiles + tile.n + 1;
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, output_tiles * k_groups * (kAColumns / kBaseK));
    output.SetValue(2, output_tiles);
    output.SetValue(3, tile_checksum);
}
