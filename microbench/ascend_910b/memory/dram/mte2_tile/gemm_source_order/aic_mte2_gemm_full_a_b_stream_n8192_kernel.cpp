// Independent MTE2-only probe for the 16x256x64 full-A / streamed-B program.
//
// This is not a GEMM: it excludes MTE1, Cube, and output work.  It retains
// exactly the source descriptor topology of the normalized M=1, K=N=8192
// program: 16 AICs, two output tiles per AIC, one 16x8192 A1 descriptor per
// output tile, and thirty-two 256x256 B1 descriptors with two B1 slots.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kActiveAics = 16;
constexpr int64_t kLogicalNTiles = 32;
constexpr int64_t kWorkerNTiles = 2;
constexpr int64_t kOutputTilesPerWorker = 2;
constexpr int64_t kARows = 16;
constexpr int64_t kAColumns = 8192;
constexpr int64_t kBRows = 256;
constexpr int64_t kBColumns = 256;
constexpr int64_t kBDescriptorsPerA = kAColumns / kBColumns;
constexpr int64_t kCleanEvictionBytesPerWorker = 16 * 1024 * 1024;
constexpr int64_t kBDescriptorBytes = (
    kBRows * kBColumns * kElementBytes);
constexpr int64_t kCleanEvictionDescriptorsPerWorker = (
    kCleanEvictionBytesPerWorker / kBDescriptorBytes);

static_assert(kAColumns % kBColumns == 0);
static_assert(kCleanEvictionBytesPerWorker % kBDescriptorBytes == 0);
static_assert(kActiveAics * kWorkerNTiles == kLogicalNTiles);
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
aic_mte2_gemm_full_a_b_stream_n8192(
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

    // The target B matrix is 128 MiB and can otherwise remain resident in
    // the shared 192 MiB L2 after host initialization.  The source counter
    // records DRAM ingress, so evict it with a disjoint, clean 256 MiB MTE2
    // stream before each timed source program.  This prelude is outside the
    // timed interval and has no MTE1, Cube, or output work.
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
    #pragma unroll 1
    for (int64_t output_tile = 0;
            output_tile < kOutputTilesPerWorker; ++output_tile) {
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
    output.SetValue(1, kOutputTilesPerWorker * (1 + kBDescriptorsPerA));
    output.SetValue(2, start);
    output.SetValue(3, end);
}
