// Independent finite AIC GM-to-L1 ND2NZ program with a completion edge after
// every copy.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;

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

extern "C" __global__ __aicore__ void aic_mte2_nd2nz_finite_drain(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_output,
        int64_t rows, int64_t columns, int64_t source_row_stride,
        int64_t source_tile_span, int64_t base_tile,
        int64_t descriptor_count) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t local_elements = Align16(rows) * columns;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> buffer;
    pipe.InitBuffer(buffer, local_elements * kElementBytes);
    auto local = buffer.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> source;
    const int64_t block_base = (
        base_tile + block * descriptor_count * source_tile_span);
    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t descriptor = 0;
            descriptor < descriptor_count; ++descriptor) {
        source.SetGlobalBuffer(
            gm_source + block_base + descriptor * source_tile_span);
        CopyNd2Nz(local, source, rows, columns, source_row_stride);
        // This is the MTE2-to-MTE1 completion edge used by the CANN FIA cube
        // path before L1 is consumed.  PipeBarrier<PIPE_MTE2> alone only
        // orders MTE2 issue and does not make GetSystemCycle observe transfer
        // completion.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, descriptor_count);
}
