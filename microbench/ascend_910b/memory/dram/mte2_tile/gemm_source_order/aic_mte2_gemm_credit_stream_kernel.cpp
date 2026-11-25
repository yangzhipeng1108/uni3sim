// Independent MTE2-only probe for concurrent A1/B1 descriptor admission under
// the production credit discipline.
//
// This is deliberately not a GEMM: it performs no MTE1, Cube, or output work.
// It differs from aic_mte2_gemm_topology_stream in exactly one respect, which
// is the point of the probe: instead of draining the pipeline with a
// PipeBarrier<PIPE_ALL> after every operand pair, each buffer slot carries its
// own MTE2->MTE1 event and a descriptor only waits for the slot it is about to
// reuse.  A and B use disjoint event identifiers, so the two streams contend
// for one MTE2 exactly as the source program's depth-limited outstanding
// window does, rather than being serialised by a global barrier.
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

extern "C" __global__ __aicore__ void aic_mte2_gemm_credit_stream(
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
    // The source interleaves one A group with the B groups that consume it.
    const int64_t b_per_a = (b_descriptors + a_descriptors - 1) / a_descriptors;

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
    const int64_t b_tiles_per_row = source_row_stride / b_columns;

    int64_t b_issued = 0;
    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t a_index = 0; a_index < a_descriptors; ++a_index) {
        const int64_t a_slot = a_index % a1_depth;
        // Credit return: this A slot may be refilled only once its previous
        // occupant has been consumed.  EVENT_ID0/1 are the A slots.
        if (a_index >= a1_depth) {
            if (a_slot == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
            }
        }
        a_source.SetGlobalBuffer(
            gm_a_source + a_row_offset + (a_index % 2) * a_columns);
        CopyNd2Nz(a_slot == 0 ? a_local0 : a_local1, a_source, a_rows,
                  a_columns, source_row_stride);
        if (a_slot == 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        }
        #pragma unroll 1
        for (int64_t j = 0; j < b_per_a && b_issued < b_descriptors; ++j) {
            const int64_t b_slot = b_issued % b1_depth;
            // EVENT_ID2/3 are the B slots, disjoint from A so the two
            // streams are not cross-serialised.
            if (b_issued >= b1_depth) {
                if (b_slot == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
                }
            }
            b_source.SetGlobalBuffer(
                gm_b_source + b_row_offset
                + (b_issued % b_tiles_per_row) * b_columns);
            CopyNd2Nz(b_slot == 0 ? b_local0 : b_local1, b_source, b_rows,
                      b_columns, source_row_stride);
            if (b_slot == 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
            }
            ++b_issued;
        }
    }
    // Drain the outstanding credits exactly once, outside the steady loop.
    for (int64_t slot = 0; slot < a1_depth && slot < a_descriptors; ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
            slot == 0 ? EVENT_ID0 : EVENT_ID1);
    }
    for (int64_t slot = 0; slot < b1_depth && slot < b_descriptors; ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
            slot == 0 ? EVENT_ID2 : EVENT_ID3);
    }
    // One barrier, after the loop, so the scalar timestamp observes the
    // drained MTE2 stream.  The steady loop itself stays barrier-free; that
    // is the whole difference from aic_mte2_gemm_topology_stream.
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, a_descriptors + b_issued);
    output.SetValue(2, start);
    output.SetValue(3, end);
}
