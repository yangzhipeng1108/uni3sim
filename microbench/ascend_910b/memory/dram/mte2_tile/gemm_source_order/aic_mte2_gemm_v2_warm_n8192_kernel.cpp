// Independent warm-L2 MTE2 issue probe for one decoded MatMulV2 program.
//
// It contains no Cube, MTE1 or FixPipe work.  The descriptor topology is the
// CANN 8.5.1 rank-three B1/M1/K8192/N8192 payload: 22 AICs, base 16x128x128,
// A1/B1 depth two, sixteen 512-K slabs and a 20x3 plus 2x2 N tail.  An
// untimed identical launch warms the immutable 128-MiB B allocation; each
// timed launch then measures only the source-order MTE2 completion.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kActiveAics = 22;
constexpr int64_t kFullOutputWorkers = 20;
constexpr int64_t kFullOutputTiles = 3;
constexpr int64_t kTailOutputTiles = 2;
constexpr int64_t kLogicalNTiles = (
    kFullOutputWorkers * kFullOutputTiles
    + (kActiveAics - kFullOutputWorkers) * kTailOutputTiles);
constexpr int64_t kMRows = 16;
constexpr int64_t kNRows = 128;
constexpr int64_t kKColumns = 512;
constexpr int64_t kSourceRowStride = 8192;
constexpr int64_t kKSlabs = kSourceRowStride / kKColumns;

static_assert(kLogicalNTiles == 64);
static_assert(kKSlabs == 16);
static_assert(
    (2 * kMRows * kKColumns + 2 * kNRows * kKColumns)
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

__aicore__ inline int64_t OutputTilesForWorker(int64_t block) {
    return block < kFullOutputWorkers ? kFullOutputTiles : kTailOutputTiles;
}

__aicore__ inline int64_t FirstNTileForWorker(int64_t block) {
    if (block < kFullOutputWorkers) {
        return block * kFullOutputTiles;
    }
    return kFullOutputWorkers * kFullOutputTiles
        + (block - kFullOutputWorkers) * kTailOutputTiles;
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_gemm_v2_warm_n8192(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t output_tiles = OutputTilesForWorker(block);
    const int64_t first_n_tile = FirstNTileForWorker(block);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    pipe.InitBuffer(a_buffer0, kMRows * kKColumns * kElementBytes);
    pipe.InitBuffer(a_buffer1, kMRows * kKColumns * kElementBytes);
    pipe.InitBuffer(b_buffer0, kNRows * kKColumns * kElementBytes);
    pipe.InitBuffer(b_buffer1, kNRows * kKColumns * kElementBytes);
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = a_buffer1.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;

    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t output_tile = 0; output_tile < output_tiles; ++output_tile) {
        const int64_t n_tile = first_n_tile + output_tile;
        #pragma unroll 1
        for (int64_t slab = 0; slab < kKSlabs; ++slab) {
            const int64_t k_offset = slab * kKColumns;
            a_source.SetGlobalBuffer(gm_a_source + k_offset);
            b_source.SetGlobalBuffer(
                gm_b_source + n_tile * kNRows * kSourceRowStride
                + k_offset);
            if (((output_tile * kKSlabs + slab) & 1) == 0) {
                CopyNd2Nz(
                    a_local0, a_source, kMRows, kKColumns,
                    kSourceRowStride);
                CopyNd2Nz(
                    b_local0, b_source, kNRows, kKColumns,
                    kSourceRowStride);
            } else {
                CopyNd2Nz(
                    a_local1, a_source, kMRows, kKColumns,
                    kSourceRowStride);
                CopyNd2Nz(
                    b_local1, b_source, kNRows, kKColumns,
                    kSourceRowStride);
            }
        }
    }
    // The counter read is issued by the scalar pipe.  A MTE2-local barrier
    // only orders commands within that pipe; PIPE_ALL publishes its terminal
    // completion to the scalar SystemCycle measurement without introducing a
    // production consumer stage.
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, output_tiles * kKSlabs * 2);
    output.SetValue(2, output_tiles);
}
