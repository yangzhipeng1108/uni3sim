// Independent MTE2-to-MTE1 ingress replica of a CANN 8.5.1 MatMulV2
// source program.  The default program executes no Cube or output work; a
// separately named build may retain the dependent Cube consumer while still
// omitting FixPipe and the complete production operator.
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
constexpr int64_t kBaseK = 64;
constexpr int64_t kBDescriptorsPerA = kAColumns / kBColumns;

#ifndef GEMM_INGRESS_KERNEL_NAME
#define GEMM_INGRESS_KERNEL_NAME aic_mte2_gemm_ingress_completion
#endif

static_assert(kAColumns % kBColumns == 0);
static_assert(kBColumns % kBaseK == 0);
static_assert(2 * (kARows * kAColumns + kMaximumBRows * kBColumns)
              * kElementBytes <= 512 * 1024);
static_assert(2 * (kARows * kBaseK + kBaseK * kMaximumBRows)
              * kElementBytes <= 96 * 1024);

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
        const AscendC::LocalTensor<bfloat16_t>& source, int64_t b_rows) {
    const int64_t block_use_n = b_rows / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_n), 1, 0, 0, false, 0};
    for (int64_t index = 0; index < block_use_k; ++index) {
        AscendC::LoadData(
            destination[index * block_use_n * 256],
            source[index * b_rows * 16], parameters);
    }
}

__aicore__ inline void ConsumeBaseKPair(
        AscendC::TQue<AscendC::TPosition::A2, 2>& l0a_queue,
        AscendC::TQue<AscendC::TPosition::B2, 2>& l0b_queue,
        const AscendC::LocalTensor<bfloat16_t>& a_local,
        const AscendC::LocalTensor<bfloat16_t>& b_local, int64_t b_rows
#ifdef GEMM_ENABLE_CUBE_CONSUMER
        , AscendC::LocalTensor<float>& l0c, bool initialize_l0c
#endif
        ) {
    auto l0a = l0a_queue.AllocTensor<bfloat16_t>();
    auto l0b = l0b_queue.AllocTensor<bfloat16_t>();
    LoadA(l0a, a_local);
    LoadB(l0b, b_local, b_rows);
    l0a_queue.EnQue(l0a);
    l0b_queue.EnQue(l0b);
    auto completed_l0a = l0a_queue.DeQue<bfloat16_t>();
    auto completed_l0b = l0b_queue.DeQue<bfloat16_t>();
#ifdef GEMM_ENABLE_CUBE_CONSUMER
    AscendC::MmadParams parameters;
    parameters.m = static_cast<uint16_t>(kARows);
    parameters.n = static_cast<uint16_t>(b_rows);
    parameters.k = static_cast<uint16_t>(kBaseK);
    parameters.cmatrixInitVal = initialize_l0c;
    parameters.cmatrixSource = false;
    AscendC::Mmad(l0c, completed_l0a, completed_l0b, parameters);
#endif
    l0a_queue.FreeTensor(completed_l0a);
    l0b_queue.FreeTensor(completed_l0b);
}

__aicore__ inline void RunOutputTile(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        AscendC::TQue<AscendC::TPosition::A1, 2>& a_queue,
        AscendC::TQue<AscendC::TPosition::B1, 2>& b_queue,
        AscendC::TQue<AscendC::TPosition::A2, 2>& l0a_queue,
        AscendC::TQue<AscendC::TPosition::B2, 2>& l0b_queue,
#ifdef GEMM_ENABLE_CUBE_CONSUMER
        AscendC::LocalTensor<float>& l0c,
#endif
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
        auto a_producer = a_queue.AllocTensor<bfloat16_t>();
        CopyNd2Nz(
            a_producer, a_source, kARows, a_columns,
            source_row_stride);
        a_queue.EnQue(a_producer);
        auto a_consumer = a_queue.DeQue<bfloat16_t>();
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
            auto b_producer = b_queue.AllocTensor<bfloat16_t>();
            CopyNd2Nz(
                b_producer, b_source, b_rows, b_columns,
                source_row_stride);
            b_queue.EnQue(b_producer);
            auto b_consumer = b_queue.DeQue<bfloat16_t>();
            #pragma unroll 1
            for (int64_t base_pair = 0;
                    base_pair < b_columns / kBaseK; ++base_pair) {
                ConsumeBaseKPair(
                    l0a_queue, l0b_queue, a_consumer, b_consumer, b_rows
#ifdef GEMM_ENABLE_CUBE_CONSUMER
                    , l0c, slab == 0 && b_descriptor == 0 && base_pair == 0
#endif
                    );
            }
            b_queue.FreeTensor(b_consumer);
        }
        a_queue.FreeTensor(a_consumer);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void GEMM_INGRESS_KERNEL_NAME(
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
    AscendC::TQue<AscendC::TPosition::A1, 2> a_queue;
    AscendC::TQue<AscendC::TPosition::B1, 2> b_queue;
    AscendC::TQue<AscendC::TPosition::A2, 2> l0a_queue;
    AscendC::TQue<AscendC::TPosition::B2, 2> l0b_queue;
#ifdef GEMM_ENABLE_CUBE_CONSUMER
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_buffer;
#endif
    pipe.InitBuffer(
        a_queue, 2, kARows * kAColumns * kElementBytes);
    pipe.InitBuffer(
        b_queue, 2, b_rows * kBColumns * kElementBytes);
    pipe.InitBuffer(
        l0a_queue, 2, kARows * kBaseK * kElementBytes);
    pipe.InitBuffer(
        l0b_queue, 2, kBaseK * b_rows * kElementBytes);
#ifdef GEMM_ENABLE_CUBE_CONSUMER
    pipe.InitBuffer(l0c_buffer, kARows * b_rows * sizeof(float));
    auto l0c = l0c_buffer.Get<float>();
#endif

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
            gm_a_source, gm_b_source, a_queue, b_queue, l0a_queue,
            l0b_queue,
#ifdef GEMM_ENABLE_CUBE_CONSUMER
            l0c,
#endif
            m_tile, n_tile, k_slabs, b_rows,
            source_row_stride);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(
        1, output_tiles_per_worker * source_row_stride / kBaseK);
    output.SetValue(2, entry);
    output.SetValue(3, end);
}
