// Independent CANN MatMulV3-65536 BF16 L0C-to-GM store mechanism probe.
#include "kernel_operator.h"
#define GEMM_V3_STORE_AICORE
#include "gemm_v3_store_contract.h"
#undef GEMM_V3_STORE_AICORE

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

__aicore__ inline void StoreCycle(
        AscendC::GlobalTensor<int64_t>& slot, int64_t value) {
    slot.SetValue(0, value);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot);
}

template <typename DstT>
__aicore__ inline void StoreTile(
        AscendC::GlobalTensor<DstT>& destination,
        const AscendC::LocalTensor<float>& l0c,
        int64_t m, int64_t n) {
    AscendC::FixpipeParamsV220 parameters;
    parameters.mSize = static_cast<uint16_t>(m);
    parameters.nSize = static_cast<uint16_t>(n);
    parameters.srcStride = static_cast<uint16_t>((m + 15) / 16 * 16);
    parameters.dstStride = static_cast<uint16_t>(n);
    parameters.ndNum = 1;
    if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value) {
        parameters.quantPre = QuantMode_t::F322BF16;
    }
    AscendC::Fixpipe<DstT, float, AscendC::CFG_ROW_MAJOR>(
        destination, l0c, parameters);
}

__aicore__ inline void InitializeL0C(
        AscendC::TPipe& pipe, __gm__ bfloat16_t* gm_a,
        __gm__ bfloat16_t* gm_b, int64_t m, int64_t n,
        AscendC::LocalTensor<float>& l0c) {
    AscendC::TBuf<AscendC::TPosition::A1> l1_a_buffer;
    AscendC::TBuf<AscendC::TPosition::B1> l1_b_buffer;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buffer;
    pipe.InitBuffer(
        l1_a_buffer, gemm_v3_store_contract::kMaximumM
            * gemm_v3_store_contract::kK * sizeof(bfloat16_t));
    pipe.InitBuffer(
        l1_b_buffer, gemm_v3_store_contract::kMaximumN
            * gemm_v3_store_contract::kK * sizeof(bfloat16_t));
    pipe.InitBuffer(
        l0a_buffer, gemm_v3_store_contract::kMaximumM
            * gemm_v3_store_contract::kK * sizeof(bfloat16_t));
    pipe.InitBuffer(
        l0b_buffer, gemm_v3_store_contract::kMaximumN
            * gemm_v3_store_contract::kK * sizeof(bfloat16_t));

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
    a_copy.nValue = static_cast<uint16_t>(m);
    a_copy.dValue = static_cast<uint16_t>(gemm_v3_store_contract::kK);
    a_copy.srcDValue = static_cast<uint16_t>(gemm_v3_store_contract::kK);
    a_copy.dstNzC0Stride = static_cast<uint16_t>(m);
    a_copy.dstNzNStride = 1;
    AscendC::DataCopy(a_l1, a_global, a_copy);
    AscendC::Nd2NzParams b_copy;
    b_copy.ndNum = 1;
    b_copy.nValue = static_cast<uint16_t>(n);
    b_copy.dValue = static_cast<uint16_t>(gemm_v3_store_contract::kK);
    b_copy.srcDValue = static_cast<uint16_t>(gemm_v3_store_contract::kK);
    b_copy.dstNzC0Stride = static_cast<uint16_t>(n);
    b_copy.dstNzNStride = 1;
    AscendC::DataCopy(b_l1, b_global, b_copy);
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);

    AscendC::LoadData2DParams a_parameters;
    a_parameters.repeatTimes = static_cast<uint8_t>(m / 16);
    a_parameters.srcStride = 1;
    AscendC::LoadData(a_l0, a_l1, a_parameters);
    AscendC::LoadData2DParams b_parameters;
    b_parameters.repeatTimes = static_cast<uint8_t>(n / 16);
    b_parameters.srcStride = 1;
    b_parameters.ifTranspose = true;
    AscendC::LoadData(b_l0, b_l1, b_parameters);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID1);

    AscendC::MmadParams mm_parameters;
    mm_parameters.m = static_cast<uint16_t>(m);
    mm_parameters.n = static_cast<uint16_t>(n);
    mm_parameters.k = static_cast<uint16_t>(gemm_v3_store_contract::kK);
    mm_parameters.cmatrixInitVal = true;
    AscendC::Mmad(l0c, a_l0, b_l0, mm_parameters);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
}

template <typename DstT>
__aicore__ inline int64_t Measure(
        __gm__ uint8_t* gm_destination,
        const AscendC::LocalTensor<float>& l0c,
        int64_t m, int64_t n, int64_t invocations) {
    const int64_t tile_elements = m * n;
    const int64_t block_elements = tile_elements * (invocations + 1);
    const int64_t block_offset = AscendC::GetBlockIdx() * block_elements;
    AscendC::GlobalTensor<DstT> warm_destination;
    warm_destination.SetGlobalBuffer(
        reinterpret_cast<__gm__ DstT*>(gm_destination)
        + block_offset + invocations * tile_elements);
    StoreTile(warm_destination, l0c, m, n);
    AscendC::PipeBarrier<PIPE_FIX>();

    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t invocation = 0; invocation < invocations; ++invocation) {
        AscendC::GlobalTensor<DstT> destination;
        destination.SetGlobalBuffer(
            reinterpret_cast<__gm__ DstT*>(gm_destination)
            + block_offset + invocation * tile_elements);
        StoreTile(destination, l0c, m, n);
    }
    // The public CANN call contains the pre-issue PIPE_FIX barrier.  One final
    // barrier is required so the last store completes before cycle sampling.
    AscendC::PipeBarrier<PIPE_FIX>();
    return AscendC::GetSystemCycle() - start;
}

}  // namespace

extern "C" __global__ __aicore__ void gemm_v3_store(
        __gm__ bfloat16_t* gm_a, __gm__ bfloat16_t* gm_b,
        __gm__ uint8_t* gm_destination, __gm__ int64_t* gm_cycles,
        int64_t dtype, int64_t m, int64_t n, int64_t invocations) {
    const int64_t block_index = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> cycle_slot;
    cycle_slot.SetGlobalBuffer(
        gm_cycles + block_index * gemm_v3_store_contract::kCycleSlotElements);
    const bool steady_cell = gemm_v3_store_contract::is_frozen_cell(
        dtype, m, n, AscendC::GetBlockNum());
    const bool finite_cell = gemm_v3_store_contract::is_frozen_finite_cell(
        dtype, m, n, AscendC::GetBlockNum());
    const bool long_stream_cell =
        gemm_v3_store_contract::is_frozen_long_stream_cell(
            dtype, m, n, AscendC::GetBlockNum());
    const bool steady_count =
        invocations == gemm_v3_store_contract::kShortInvocations
        || invocations == gemm_v3_store_contract::kLongInvocations;
    const bool finite_count =
        invocations == gemm_v3_store_contract::kFiniteShortInvocations
        || invocations == gemm_v3_store_contract::kFiniteLongInvocations;
    const bool long_stream_count =
        invocations == gemm_v3_store_contract::kLongStreamShortInvocations
        || invocations == gemm_v3_store_contract::kLongStreamLongInvocations;
    if ((!steady_cell || !steady_count)
            && (!finite_cell || !finite_count)
            && (!long_stream_cell || !long_stream_count)) {
        StoreCycle(cycle_slot, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::CO1> l0c_buffer;
    pipe.InitBuffer(
        l0c_buffer, gemm_v3_store_contract::kMaximumM
            * gemm_v3_store_contract::kMaximumN * sizeof(float));
    auto l0c = l0c_buffer.Get<float>();
    InitializeL0C(pipe, gm_a, gm_b, m, n, l0c);
    const int64_t cycles = dtype == gemm_v3_store_contract::kBfloat16
        ? Measure<bfloat16_t>(gm_destination, l0c, m, n, invocations)
        : Measure<float>(gm_destination, l0c, m, n, invocations);
    StoreCycle(cycle_slot, cycles);
}
