// Independent MTE2 and MTE2-to-MTE1 replicas of the CANN 8.5.1
// MatMulV2 M64/N5120/K1536 source program.  No Cube or output work runs.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kARows = 64;
constexpr int64_t kAColumns = 1536;
constexpr int64_t kBRows = 224;
constexpr int64_t kBColumns = 256;
constexpr int64_t kBaseK = 64;
constexpr int64_t kBSlabs = kAColumns / kBColumns;

static_assert(kAColumns % kBColumns == 0);
static_assert(kBColumns % kBaseK == 0);
static_assert(
    (kARows * kAColumns + 2 * kBRows * kBColumns) * kElementBytes
    <= 512 * 1024);
static_assert(2 * kARows * kBaseK * kElementBytes <= 64 * 1024);
static_assert(2 * kBaseK * kBRows * kElementBytes <= 64 * 1024);

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
    parameters.srcDValue = static_cast<uint16_t>(kAColumns);
    parameters.dstNzC0Stride = static_cast<uint16_t>(Align16(rows));
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

__aicore__ inline void StoreResult(
        __gm__ int64_t* gm_output, int64_t start, int64_t end,
        int64_t count) {
    const int64_t block = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, count);
    output.SetValue(2, start);
    output.SetValue(3, end);
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_gemm_m64_source_order(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output) {
    const int64_t block = AscendC::GetBlockIdx();
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
    a_source.SetGlobalBuffer(gm_a_source);
    const int64_t b_row_offset = block * kBRows * kAColumns;

    const int64_t start = AscendC::GetSystemCycle();
    CopyNd2Nz(a_local, a_source, kARows, kAColumns);
    #pragma unroll 1
    for (int64_t slab = 0; slab < kBSlabs; ++slab) {
        b_source.SetGlobalBuffer(
            gm_b_source + b_row_offset + slab * kBColumns);
        if ((slab & 1) == 0) {
            CopyNd2Nz(b_local0, b_source, kBRows, kBColumns);
        } else {
            CopyNd2Nz(b_local1, b_source, kBRows, kBColumns);
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    const int64_t end = AscendC::GetSystemCycle();
    StoreResult(gm_output, start, end, 1 + kBSlabs);
}
