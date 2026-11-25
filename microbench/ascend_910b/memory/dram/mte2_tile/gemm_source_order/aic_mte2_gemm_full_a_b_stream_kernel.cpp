// Independent MTE2-only probe for a full-A / streamed-B MatMulV2 program.
//
// This is deliberately not a GEMM: it performs neither MTE1, Cube nor output
// work.  It isolates the descriptor topology used by the normalized
// 16x1280x8192 persistent program: one 16x8192 A1 descriptor per output tile
// and thirty-two 256x256 B1 descriptors, with two B1 slots.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kActiveAics = 23;
constexpr int64_t kLogicalNTiles = 112;
constexpr int64_t kWorkerNTiles = 5;
constexpr int64_t kOutputTilesPerWorker = 5;
constexpr int64_t kARows = 16;
constexpr int64_t kAColumns = 8192;
constexpr int64_t kBRows = 256;
constexpr int64_t kBColumns = 256;
constexpr int64_t kBDescriptorsPerA = kAColumns / kBColumns;

static_assert(kAColumns % kBColumns == 0);
static_assert(kActiveAics * kWorkerNTiles >= kLogicalNTiles);
static_assert(kOutputTilesPerWorker == kWorkerNTiles);
static_assert((kARows * kAColumns + 2 * kBRows * kBColumns)
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

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_gemm_full_a_b_stream(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t candidate_n_start = block * kWorkerNTiles;
    const int64_t maximum_n_start = kLogicalNTiles - kWorkerNTiles;
    const int64_t n_start = candidate_n_start < maximum_n_start
        ? candidate_n_start : maximum_n_start;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    pipe.InitBuffer(a_buffer, kARows * kAColumns * kElementBytes);
    pipe.InitBuffer(b_buffer0, kBRows * kBColumns * kElementBytes);
    pipe.InitBuffer(b_buffer1, kBRows * kBColumns * kElementBytes);
    auto a_local = a_buffer.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;

    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t output_tile = 0;
            output_tile < kOutputTilesPerWorker; ++output_tile) {
        // The normalized program owns one M tile; every N tile starts one
        // full-K A descriptor and streams its own B descriptors afterwards.
        a_source.SetGlobalBuffer(gm_a_source);
        CopyNd2Nz(a_local, a_source, kARows, kAColumns, kAColumns);
        const int64_t n_tile = n_start + output_tile;
        const int64_t b_row_offset = n_tile * kBRows * kAColumns;
        #pragma unroll 1
        for (int64_t descriptor = 0;
                descriptor < kBDescriptorsPerA; ++descriptor) {
            b_source.SetGlobalBuffer(
                gm_b_source + b_row_offset + descriptor * kBColumns);
            if ((descriptor & 1) == 0) {
                CopyNd2Nz(
                    b_local0, b_source, kBRows, kBColumns, kAColumns);
            } else {
                CopyNd2Nz(
                    b_local1, b_source, kBRows, kBColumns, kAColumns);
            }
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(
        1, kOutputTilesPerWorker * (1 + kBDescriptorsPerA));
    output.SetValue(2, start);
    output.SetValue(3, end);
}
