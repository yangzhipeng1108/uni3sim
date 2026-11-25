// Independent Ascend 910B FP32 L0C-to-GM direct Fixpipe mechanism benchmark.
//
// A small BF16 MMAD initializes one L0C tile before GetSystemCycle.  The timed
// region contains only public Fixpipe calls and explicit PIPE_FIX completion.
// Public Fixpipe on dav_c220 starts with its own PIPE_FIX barrier, so the
// "stream" mode below is deliberately named and reported as serialized
// steady-state service.  It is not evidence of an asynchronous queue.
#include "kernel_operator.h"
#define FIXPIPE_SERVICE_AICORE
#include "fixpipe_service_contract.h"
#undef FIXPIPE_SERVICE_AICORE

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kRowMajor = 0;
constexpr int64_t kSerial = 0;
constexpr int64_t kMaximumM = 128;
constexpr int64_t kMaximumN = 128;
constexpr int64_t kK = 16;

__aicore__ inline void StoreCycle(
        AscendC::GlobalTensor<int64_t>& cycle_slot, int64_t value) {
    cycle_slot.SetValue(0, value);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(cycle_slot);
}

template <bool RowMajor>
__aicore__ inline void IssueFixpipe(
        AscendC::GlobalTensor<float>& destination,
        const AscendC::LocalTensor<float>& source, int64_t m, int64_t n,
        int64_t src_stride, int64_t dst_stride) {
    AscendC::FixpipeParamsV220 parameters;
    parameters.mSize = static_cast<uint16_t>(m);
    parameters.nSize = static_cast<uint16_t>(n);
    parameters.srcStride = static_cast<uint16_t>(src_stride);
    parameters.dstStride = static_cast<uint16_t>(dst_stride);
    parameters.ndNum = 1;
    if constexpr (RowMajor) {
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
            destination, source, parameters);
    } else {
        AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(
            destination, source, parameters);
    }
}

__aicore__ inline void InitializeL0C(
        AscendC::TPipe& pipe, __gm__ bfloat16_t* gm_a,
        __gm__ bfloat16_t* gm_b, int64_t m, int64_t n,
        AscendC::LocalTensor<float>& l0c) {
    const int64_t init_m = (m + 15) / 16 * 16;
    const int64_t init_n = (n + 15) / 16 * 16;
    AscendC::TBuf<AscendC::TPosition::A1> l1_a_buffer;
    AscendC::TBuf<AscendC::TPosition::B1> l1_b_buffer;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buffer;
    pipe.InitBuffer(l1_a_buffer, kMaximumM * kK * sizeof(bfloat16_t));
    pipe.InitBuffer(l1_b_buffer, kMaximumN * kK * sizeof(bfloat16_t));
    pipe.InitBuffer(l0a_buffer, kMaximumM * kK * sizeof(bfloat16_t));
    pipe.InitBuffer(l0b_buffer, kMaximumN * kK * sizeof(bfloat16_t));

    AscendC::GlobalTensor<bfloat16_t> a_global;
    AscendC::GlobalTensor<bfloat16_t> b_global;
    a_global.SetGlobalBuffer(gm_a);
    b_global.SetGlobalBuffer(gm_b);
    auto a_l1 = l1_a_buffer.Get<bfloat16_t>();
    auto b_l1 = l1_b_buffer.Get<bfloat16_t>();
    auto a_l0 = l0a_buffer.Get<bfloat16_t>();
    auto b_l0 = l0b_buffer.Get<bfloat16_t>();
    AscendC::Nd2NzParams a_copy;
    a_copy.ndNum = 1;
    a_copy.nValue = static_cast<uint16_t>(init_m);
    a_copy.dValue = static_cast<uint16_t>(kK);
    a_copy.srcNdMatrixStride = 0;
    a_copy.srcDValue = static_cast<uint16_t>(kK);
    a_copy.dstNzC0Stride = static_cast<uint16_t>(init_m);
    a_copy.dstNzNStride = 1;
    a_copy.dstNzMatrixStride = 0;
    AscendC::DataCopy(a_l1, a_global, a_copy);
    AscendC::Nd2NzParams b_copy;
    b_copy.ndNum = 1;
    b_copy.nValue = static_cast<uint16_t>(init_n);
    b_copy.dValue = static_cast<uint16_t>(kK);
    b_copy.srcNdMatrixStride = 0;
    b_copy.srcDValue = static_cast<uint16_t>(kK);
    b_copy.dstNzC0Stride = static_cast<uint16_t>(init_n);
    b_copy.dstNzNStride = 1;
    b_copy.dstNzMatrixStride = 0;
    AscendC::DataCopy(b_l1, b_global, b_copy);
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);

    // dav_c220 L1->L0A/B supports the legacy 2D load contract.  Nd2Nz above
    // forms m-by-K A and n-by-K B tiles; B is transposed into K-by-n for MMAD.
    AscendC::LoadData2DParams a_parameters;
    a_parameters.startIndex = 0;
    a_parameters.repeatTimes = static_cast<uint8_t>(init_m / 16);
    a_parameters.srcStride = 1;
    a_parameters.dstGap = 0;
    a_parameters.ifTranspose = false;
    AscendC::LoadData(a_l0, a_l1, a_parameters);

    AscendC::LoadData2DParams b_parameters;
    b_parameters.startIndex = 0;
    b_parameters.repeatTimes = static_cast<uint8_t>(init_n / 16);
    b_parameters.srcStride = 1;
    b_parameters.dstGap = 0;
    b_parameters.ifTranspose = true;
    AscendC::LoadData(b_l0, b_l1, b_parameters);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID1);

    AscendC::MmadParams mm_parameters;
    mm_parameters.m = static_cast<uint16_t>(init_m);
    mm_parameters.n = static_cast<uint16_t>(n);
    mm_parameters.k = static_cast<uint16_t>(kK);
    mm_parameters.cmatrixInitVal = true;
    mm_parameters.cmatrixSource = false;
    AscendC::Mmad(l0c, a_l0, b_l0, mm_parameters);
    AscendC::PipeBarrier<PIPE_M>();
}

}  // namespace

extern "C" __global__ __aicore__ void fixpipe_service(
        __gm__ bfloat16_t* gm_a, __gm__ bfloat16_t* gm_b,
        __gm__ float* gm_destination, __gm__ int64_t* gm_cycles,
        int64_t m, int64_t n, int64_t src_stride, int64_t dst_stride,
        int64_t layout, int64_t mode, int64_t stream_depth,
        int64_t destination_slots, int64_t invocations) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> cycle_slot;
    cycle_slot.SetGlobalBuffer(
        gm_cycles
        + block_index * fixpipe_service_contract::kCycleSlotElements);
    const bool valid_shape = m >= 1 && m <= kMaximumM
        && n >= 1 && n <= kMaximumN;
    const bool valid_depth = stream_depth == 1;
    if (!valid_shape) {
        StoreCycle(cycle_slot, -11);
        return;
    }
    if (layout != kRowMajor) {
        StoreCycle(cycle_slot, -13);
        return;
    }
    if (mode != kSerial) {
        StoreCycle(cycle_slot, -14);
        return;
    }
    if (!valid_depth) {
        StoreCycle(cycle_slot, -15);
        return;
    }
    if (stream_depth != 1) {
        StoreCycle(cycle_slot, -16);
        return;
    }
    if (invocations <= 0) {
        StoreCycle(cycle_slot, -17);
        return;
    }
    const int64_t expected_src_stride = (m + 15) / 16 * 16;
    if (!fixpipe_service_contract::is_frozen_invocation_count(
                m, n, stream_depth, invocations)
            || src_stride != expected_src_stride
            || dst_stride < n || dst_stride % 32 != 0
            || dst_stride > 8192 || destination_slots <= 0
            || destination_slots > invocations) {
        StoreCycle(cycle_slot, -2);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_buffer;
    pipe.InitBuffer(l0c_buffer, kMaximumM * kMaximumN * sizeof(float));
    auto l0c = l0c_buffer.Get<float>();
    InitializeL0C(pipe, gm_a, gm_b, m, n, l0c);
    // Establish the architectural MMAD-producer -> Fixpipe-consumer edge.
    // This handshake, like MMAD initialization itself, is outside timing.
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

    int64_t tile_span_elements = 0;
    fixpipe_service_contract::DestinationExtent destination_extent{
        0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (!fixpipe_service_contract::checked_row_major_span(
                m, n, dst_stride, &tile_span_elements)
            || !fixpipe_service_contract::checked_destination_extent(
                block_index, invocations, destination_slots,
                tile_span_elements,
                &destination_extent)) {
        StoreCycle(cycle_slot, -3);
        return;
    }
    AscendC::GlobalTensor<float> warm_destination;
    warm_destination.SetGlobalBuffer(
        gm_destination + destination_extent.warmup_destination_offset);
    IssueFixpipe<true>(
        warm_destination, l0c, m, n, src_stride, dst_stride);
    AscendC::PipeBarrier<PIPE_FIX>();

    const int64_t cycle_start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t invocation = 0; invocation < invocations; ++invocation) {
        const int64_t destination_offset =
            destination_extent.block_offset
            + invocation % destination_extent.slot_count
                * destination_extent.slot_pitch_elements
            + fixpipe_service_contract::kDestinationBaseOffsetElements;
        AscendC::GlobalTensor<float> destination;
        destination.SetGlobalBuffer(gm_destination + destination_offset);
        IssueFixpipe<true>(
            destination, l0c, m, n, src_stride, dst_stride);
        AscendC::PipeBarrier<PIPE_FIX>();
    }
    // Explicitly complete the final direct L0C->GM transfer before sampling.
    AscendC::PipeBarrier<PIPE_FIX>();
    const int64_t cycle_end = AscendC::GetSystemCycle();
    StoreCycle(cycle_slot, cycle_end - cycle_start);
}
