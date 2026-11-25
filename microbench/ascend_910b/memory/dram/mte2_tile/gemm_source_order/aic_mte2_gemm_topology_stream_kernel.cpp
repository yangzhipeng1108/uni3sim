// Independent MTE2-only probe for a parameterised MatMulV2 A1/B1 source
// program.
//
// This is deliberately not a GEMM: it performs no MTE1, Cube, or output work.
// It isolates one source descriptor topology - a_descriptors A1 descriptors of
// a_rows x a_columns followed by b_descriptors B1 descriptors of
// b_rows x b_columns, at the physical source row stride, with a1_depth and
// b1_depth L1 slots - on active_aics AI cores.  The topology arrives through
// GM and is read before the timed region so the measurement contains only
// descriptor service.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kMaximumDepth = 2;

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

extern "C" __global__ __aicore__ void aic_mte2_gemm_topology_stream(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_topology, __gm__ int64_t* gm_output) {
    const int64_t block = AscendC::GetBlockIdx();

    AscendC::GlobalTensor<int64_t> topology;
    topology.SetGlobalBuffer(gm_topology);
    const int64_t a_rows = topology.GetValue(0);
    const int64_t a_columns = topology.GetValue(1);
    const int64_t b_rows = topology.GetValue(2);
    const int64_t b_columns = topology.GetValue(3);
    const int64_t source_row_stride = topology.GetValue(4);
    const int64_t a1_depth = topology.GetValue(5);
    const int64_t b1_depth = topology.GetValue(6);
    const int64_t a_descriptors = topology.GetValue(7);
    const int64_t b_descriptors = topology.GetValue(8);
    const int64_t a_tile_rows = topology.GetValue(9);
    const int64_t b_tile_rows = topology.GetValue(10);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    const int64_t a_bytes = a_tile_rows * a_columns * kElementBytes;
    const int64_t b_bytes = b_tile_rows * b_columns * kElementBytes;
    pipe.InitBuffer(a_buffer0, a_bytes);
    if (a1_depth == kMaximumDepth) {
        pipe.InitBuffer(a_buffer1, a_bytes);
    }
    pipe.InitBuffer(b_buffer0, b_bytes);
    if (b1_depth == kMaximumDepth) {
        pipe.InitBuffer(b_buffer1, b_bytes);
    }
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = (a1_depth == kMaximumDepth)
        ? a_buffer1.Get<bfloat16_t>() : a_local0;
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = (b1_depth == kMaximumDepth)
        ? b_buffer1.Get<bfloat16_t>() : b_local0;
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;

    const int64_t a_row_offset = block * a_rows * source_row_stride;
    const int64_t b_row_offset = block * b_rows * source_row_stride;

    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t descriptor = 0; descriptor < a_descriptors; ++descriptor) {
        a_source.SetGlobalBuffer(
            gm_a_source + a_row_offset
            + (descriptor % 2) * a_columns);
        if ((descriptor & 1) == 0 || a1_depth != kMaximumDepth) {
            CopyNd2Nz(a_local0, a_source, a_rows, a_columns,
                      source_row_stride);
        } else {
            CopyNd2Nz(a_local1, a_source, a_rows, a_columns,
                      source_row_stride);
        }
        // Production releases each A1 slot after its MTE1 consumer.  This
        // MTE2-only probe substitutes a zero-work consumer handoff so it
        // measures producer service without overflowing the physical A1 queue
        // or importing MTE1/Cube duration.
        if (a1_depth != kMaximumDepth || (descriptor & 1) == 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    #pragma unroll 1
    for (int64_t descriptor = 0; descriptor < b_descriptors; ++descriptor) {
        b_source.SetGlobalBuffer(
            gm_b_source + b_row_offset
            + (descriptor % (source_row_stride / b_columns)) * b_columns);
        if ((descriptor & 1) == 0 || b1_depth != kMaximumDepth) {
            CopyNd2Nz(b_local0, b_source, b_rows, b_columns,
                      source_row_stride);
        } else {
            CopyNd2Nz(b_local1, b_source, b_rows, b_columns,
                      source_row_stride);
        }
        if (b1_depth != kMaximumDepth || (descriptor & 1) == 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, a_descriptors + b_descriptors);
    output.SetValue(2, start);
    output.SetValue(3, end);
}
