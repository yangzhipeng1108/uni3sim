// Independent Ascend 910B B1 queue-cadence probe.
//
// This isolates B1 queue topology: a 4x64x256 ND2NZ B1 fill (physical K
// stride supplied by the host), CANN's MTE2->MTE1 edge, and the source-shaped
// four-K MTE1 B load. Mode 0 uses one Nd2NzParams fill per group as a topology
// control; mode 1 uses CANN's CopyND2NZOnTheFly source issue sequence. Modes
// 2 and 3 add the same respective B1 fills to a source-visible B1->L0B->Cube
// ping-pong primitive; A is prefetched before timing and C is never stored.
// No mode contains routing, output, or production FusedMoE code.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kBaseK = 64;
constexpr int64_t kBaseN = 256;
constexpr int64_t kStepKb = 4;
constexpr int64_t kB1Rows = kBaseK * kStepKb;
constexpr int64_t kB1Elements = kB1Rows * kBaseN;
constexpr int64_t kL0BElements = kBaseK * kBaseN;
constexpr int64_t kL0AElements = 16 * kBaseK;
constexpr int64_t kL0CElements = 16 * kBaseN;

__aicore__ inline void CopyB1(
        AscendC::LocalTensor<bfloat16_t>& destination,
        AscendC::GlobalTensor<bfloat16_t>& source,
        int64_t source_row_stride) {
    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = static_cast<uint16_t>(kB1Rows);
    parameters.dValue = static_cast<uint16_t>(kBaseN);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(source_row_stride);
    parameters.dstNzC0Stride = static_cast<uint16_t>(kB1Rows);
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

__aicore__ inline void CopyB1OnTheFly(
        AscendC::LocalTensor<bfloat16_t>& destination,
        AscendC::GlobalTensor<bfloat16_t>& source,
        int64_t source_row_stride) {
    // Exact CANN 8.5.1 CFG_MDL CopyND2NZOnTheFly branch for the target's
    // BF16 B tile: 16 width-C0 copies, each with a 256-row source stride.
    constexpr int64_t c0_size = 16;
    constexpr int64_t copy_widths = kBaseN / c0_size;
    constexpr int64_t destination_width_stride = (
        (kB1Rows / c0_size) * c0_size * c0_size);
    const int64_t source_gap = (
        source_row_stride * kElementBytes / 32 - 1);
    if (source_gap < 0 || source_gap >= UINT16_MAX) {
        return;
    }
    AscendC::DataCopyEnhancedParams parameters {};
    for (int64_t width_index = 0;
            width_index < copy_widths; ++width_index) {
        AscendC::DataCopy(
            destination[width_index * destination_width_stride],
            source[width_index * c0_size],
            {static_cast<uint16_t>(kB1Rows), 1,
             static_cast<uint16_t>(source_gap), 0},
            parameters);
    }
}

__aicore__ inline void LoadB(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    constexpr int64_t block_use_n = kBaseN / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_n),
        static_cast<uint16_t>(kB1Rows / 16), 0, 0, true, 0};
    for (int64_t index = 0; index < block_use_k; ++index) {
        AscendC::LoadData(
            destination[index * block_use_n * 256],
            source[index * kBaseN], parameters);
    }
}

__aicore__ inline void LoadAForCube(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(kBaseK / 16), 1, 0, 0, false, 0};
    AscendC::LoadData(destination, source, parameters);
}

__aicore__ inline void LoadBForCube(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source,
        int64_t k_index) {
    constexpr int64_t block_use_n = kBaseN / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_n),
        static_cast<uint16_t>(kB1Rows / 16), 0, 0, true, 0};
    const auto b1_k_tile = source[k_index * kL0BElements];
    #pragma unroll 1
    for (int64_t block_k = 0; block_k < block_use_k; ++block_k) {
        AscendC::LoadData(
            destination[block_k * block_use_n * 256],
            b1_k_tile[block_k * kBaseN], parameters);
    }
}

__aicore__ inline void CopyAForCube(
        AscendC::LocalTensor<bfloat16_t>& destination,
        AscendC::GlobalTensor<bfloat16_t>& source) {
    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = 16;
    parameters.dValue = static_cast<uint16_t>(kBaseK);
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = static_cast<uint16_t>(kBaseK);
    parameters.dstNzC0Stride = 16;
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

__aicore__ inline void RunPanelCubeCadence(
        AscendC::LocalTensor<bfloat16_t>& b1_ping,
        AscendC::LocalTensor<bfloat16_t>& b1_pong,
        AscendC::LocalTensor<bfloat16_t>& a1,
        AscendC::LocalTensor<bfloat16_t>& l0a_ping,
        AscendC::LocalTensor<bfloat16_t>& l0a_pong,
        AscendC::LocalTensor<bfloat16_t>& l0b_ping,
        AscendC::LocalTensor<bfloat16_t>& l0b_pong,
        AscendC::LocalTensor<float>& l0c,
        __gm__ bfloat16_t* gm_source,
        int64_t source_row_stride, int64_t source_group_span,
        int64_t source_groups, int64_t iterations, int64_t block_source_base,
        int64_t source_mode) {
    // This follows the source-visible MTE1_MTE2 (B1 reuse), MTE2_MTE1
    // (B1 ready), and M_MTE1 (L0 ping-pong) edges.  It is deliberately a
    // B ingress primitive: A is prefetched once, C is never stored, and no
    // routing, group-list, or production FusedMoE code is present.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::GlobalTensor<bfloat16_t> source;
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const int64_t b1_slot = iteration & 1;
        const int64_t group = iteration % source_groups;
        auto b1 = b1_slot == 0 ? b1_ping : b1_pong;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(b1_slot);
        source.SetGlobalBuffer(
            gm_source + block_source_base + group * source_group_span);
        if (source_mode == 0) {
            CopyB1(b1, source, source_row_stride);
        } else {
            CopyB1OnTheFly(b1, source, source_row_stride);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(b1_slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(b1_slot);
        #pragma unroll 1
        for (int64_t k_index = 0; k_index < kStepKb; ++k_index) {
            const int64_t l0_slot = k_index & 1;
            auto l0a = l0_slot == 0 ? l0a_ping : l0a_pong;
            auto l0b = l0_slot == 0 ? l0b_ping : l0b_pong;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0_slot);
            LoadAForCube(l0a, a1);
            LoadBForCube(l0b, b1, k_index);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0_slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0_slot);
            AscendC::MmadParams parameters;
            parameters.m = 16;
            parameters.n = static_cast<uint16_t>(kBaseN);
            parameters.k = static_cast<uint16_t>(kBaseK);
            parameters.cmatrixInitVal = k_index == 0;
            parameters.cmatrixSource = false;
            AscendC::Mmad(l0c, l0a, l0b, parameters);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0_slot);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(b1_slot);
    }
    // Match the source program's event drain before leaving the primitive.
    // This also makes the timed interval include the final Cube consumer
    // rather than returning while a ping-pong slot is still in flight.
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_b1_queue_cadence(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_output,
        int64_t source_row_stride, int64_t source_group_span,
        int64_t source_groups, int64_t iterations, int64_t source_mode) {
    const int64_t block = AscendC::GetBlockIdx();
    if (source_row_stride < kBaseN || source_groups <= 0 || iterations <= 0
            || source_mode < 0 || source_mode > 3) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::B1> b1_ping_buffer;
    AscendC::TBuf<AscendC::TPosition::B1> b1_pong_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_ping_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_pong_buffer;
    pipe.InitBuffer(b1_ping_buffer, kB1Elements * kElementBytes);
    pipe.InitBuffer(b1_pong_buffer, kB1Elements * kElementBytes);
    pipe.InitBuffer(l0b_ping_buffer, kL0BElements * kElementBytes);
    pipe.InitBuffer(l0b_pong_buffer, kL0BElements * kElementBytes);
    auto b1_ping = b1_ping_buffer.Get<bfloat16_t>();
    auto b1_pong = b1_pong_buffer.Get<bfloat16_t>();
    auto l0b_ping = l0b_ping_buffer.Get<bfloat16_t>();
    auto l0b_pong = l0b_pong_buffer.Get<bfloat16_t>();
    const int64_t block_source_base = (
        block * source_groups * source_group_span);

    if (source_mode == 2 || source_mode == 3) {
        AscendC::TBuf<AscendC::TPosition::A1> a1_buffer;
        AscendC::TBuf<AscendC::TPosition::A2> l0a_ping_buffer;
        AscendC::TBuf<AscendC::TPosition::A2> l0a_pong_buffer;
        AscendC::TBuf<AscendC::TPosition::CO1> l0c_buffer;
        pipe.InitBuffer(a1_buffer, kL0AElements * kElementBytes);
        pipe.InitBuffer(l0a_ping_buffer, kL0AElements * kElementBytes);
        pipe.InitBuffer(l0a_pong_buffer, kL0AElements * kElementBytes);
        pipe.InitBuffer(l0c_buffer, kL0CElements * sizeof(float));
        auto a1 = a1_buffer.Get<bfloat16_t>();
        auto l0a_ping = l0a_ping_buffer.Get<bfloat16_t>();
        auto l0a_pong = l0a_pong_buffer.Get<bfloat16_t>();
        auto l0c = l0c_buffer.Get<float>();
        AscendC::GlobalTensor<bfloat16_t> a_source;
        a_source.SetGlobalBuffer(gm_source + block * kL0AElements);
        CopyAForCube(a1, a_source);
        AscendC::PipeBarrier<PIPE_MTE2>();
        const int64_t begin = AscendC::GetSystemCycle();
        RunPanelCubeCadence(
            b1_ping, b1_pong, a1, l0a_ping, l0a_pong, l0b_ping, l0b_pong,
            l0c, gm_source, source_row_stride, source_group_span,
            source_groups, iterations, block_source_base,
            source_mode == 2 ? 0 : 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        const int64_t end = AscendC::GetSystemCycle();
        AscendC::GlobalTensor<int64_t> output;
        output.SetGlobalBuffer(
            gm_output + block * kCycleSlotBytes / sizeof(int64_t));
        output.SetValue(0, end - begin);
        output.SetValue(1, iterations);
        return;
    }

    // Both L0B slots are initially free for MTE1.  The empty M consumer below
    // only releases an MTE1-loaded slot; it adds no production computation.
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::GlobalTensor<bfloat16_t> source;

    const int64_t begin = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const int64_t slot = iteration & 1;
        const int64_t group = iteration % source_groups;
        auto b1 = slot == 0 ? b1_ping : b1_pong;
        auto l0b = slot == 0 ? l0b_ping : l0b_pong;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(slot);
        source.SetGlobalBuffer(
            gm_source + block_source_base + group * source_group_span);
        if (source_mode == 0) {
            CopyB1(b1, source, source_row_stride);
        } else {
            CopyB1OnTheFly(b1, source, source_row_stride);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
        LoadB(l0b, b1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(slot);
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - begin);
    output.SetValue(1, iterations);
}
