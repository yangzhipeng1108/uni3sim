// Independent source-shaped MatMulV2 key-98513 A1/B1 MTE2 program probe.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kARows = 16;
constexpr int64_t kBRows = 64;
constexpr int64_t kKColumns = 512;
constexpr int64_t kASourceRowStride = 5120;
constexpr int64_t kBSourceRowStride = 1536;
constexpr int64_t kKSlabs = 10;

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

extern "C" __global__ __aicore__ void aic_mte2_small_m_program(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output) {
    const int64_t block = AscendC::GetBlockIdx();

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    pipe.InitBuffer(a_buffer0, kARows * kKColumns * kElementBytes);
    pipe.InitBuffer(a_buffer1, kARows * kKColumns * kElementBytes);
    pipe.InitBuffer(b_buffer0, kBRows * kKColumns * kElementBytes);
    pipe.InitBuffer(b_buffer1, kBRows * kKColumns * kElementBytes);
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = a_buffer1.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;
    const int64_t b_block_offset = block * kBRows * kBSourceRowStride;
    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t slab = 0; slab < kKSlabs; ++slab) {
        a_source.SetGlobalBuffer(gm_a_source + slab * kKColumns);
        b_source.SetGlobalBuffer(
            gm_b_source + b_block_offset + slab * kKColumns);
        if ((slab & 1) == 0) {
            CopyNd2Nz(
                a_local0, a_source, kARows, kKColumns, kASourceRowStride);
            CopyNd2Nz(
                b_local0, b_source, kBRows, kKColumns, kBSourceRowStride);
        } else {
            CopyNd2Nz(
                a_local1, a_source, kARows, kKColumns, kASourceRowStride);
            CopyNd2Nz(
                b_local1, b_source, kBRows, kKColumns, kBSourceRowStride);
        }
        if ((slab & 1) == 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, kKSlabs);
}
