// Independent MTE2-only probe for one full-A MatMulV2 source queue.
//
// The CANN launch geometry is 23 AICs, a 16x96x128 base tile, one 16x4096
// A1 descriptor, and five worker-local N tiles.  Each N tile streams eight
// 96x512 B1 descriptors through two B1 buffers.  No MTE1, Cube, FixPipe, or
// output operation is present: this measures only the source-visible MTE2
// queue completion and cannot be a production GEMM duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kActiveAics = 23;
constexpr int64_t kLogicalNTiles = 115;
constexpr int64_t kWorkerNTiles = 5;
constexpr int64_t kOutputTilesPerWorker = 5;
constexpr int64_t kARows = 16;
constexpr int64_t kAColumns = 4096;
constexpr int64_t kBRows = 96;
constexpr int64_t kBColumns = 512;
constexpr int64_t kBDescriptorsPerOutputTile = kAColumns / kBColumns;
constexpr int64_t kBDescriptorBytes = kBRows * kBColumns * kElementBytes;
constexpr int64_t kCleanEvictionDescriptorsPerWorker = 128;
constexpr int64_t kCleanEvictionBytesPerWorker = (
    kCleanEvictionDescriptorsPerWorker * kBDescriptorBytes);

static_assert(kActiveAics * kWorkerNTiles == kLogicalNTiles);
static_assert(kAColumns % kBColumns == 0);
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

extern "C" __global__ __aicore__ void
aic_mte2_matmul_v2_full_a_k4096_n96(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ bfloat16_t* gm_clean_eviction, __gm__ int64_t* gm_output) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t n_start = block * kWorkerNTiles;

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
    AscendC::GlobalTensor<bfloat16_t> eviction_source;

    // Every source descriptor in the production-program trace is a first
    // pass over its own matrix line.  The canonical L2 replay model similarly
    // classifies this 90 MiB B stream as cold.  A disjoint 276 MiB clean MTE2
    // prelude exceeds the published 192 MiB L2 capacity before each timed
    // program, while remaining outside the measured MTE2-only interval.
    const int64_t eviction_offset = (
        block * kCleanEvictionDescriptorsPerWorker * kBRows * kBColumns);
    #pragma unroll 1
    for (int64_t descriptor = 0;
            descriptor < kCleanEvictionDescriptorsPerWorker; ++descriptor) {
        eviction_source.SetGlobalBuffer(
            gm_clean_eviction + eviction_offset
            + descriptor * kBRows * kBColumns);
        if ((descriptor & 1) == 0) {
            CopyNd2Nz(
                b_local0, eviction_source, kBRows, kBColumns, kBColumns);
        } else {
            CopyNd2Nz(
                b_local1, eviction_source, kBRows, kBColumns, kBColumns);
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    const int64_t start = AscendC::GetSystemCycle();
    a_source.SetGlobalBuffer(gm_a_source);
    CopyNd2Nz(a_local, a_source, kARows, kAColumns, kAColumns);
    #pragma unroll 1
    for (int64_t output_tile = 0;
            output_tile < kOutputTilesPerWorker; ++output_tile) {
        const int64_t n_tile = n_start + output_tile;
        const int64_t b_row_offset = n_tile * kBRows * kAColumns;
        #pragma unroll 1
        for (int64_t descriptor = 0;
                descriptor < kBDescriptorsPerOutputTile; ++descriptor) {
            b_source.SetGlobalBuffer(
                gm_b_source + b_row_offset + descriptor * kBColumns);
            if ((descriptor & 1) == 0) {
                CopyNd2Nz(
                    b_local0, b_source, kBRows, kBColumns, kAColumns);
            } else {
                CopyNd2Nz(
                    b_local1, b_source, kBRows, kBColumns, kAColumns);
                // MatMulV2's MTE1 consumer returns the two B1 slots before
                // the next descriptor pair.  This MTE2-only program has no
                // MTE1 work, so retain that queue credit lifecycle with the
                // same zero-work handoff as the A1/B20 source holdout.  The
                // handoff is a producer-credit substitute, not a production
                // GEMM timing term.
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
                AscendC::PipeBarrier<PIPE_ALL>();
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
        1, 1 + kOutputTilesPerWorker * kBDescriptorsPerOutputTile);
    output.SetValue(2, start);
    output.SetValue(3, end);
}
