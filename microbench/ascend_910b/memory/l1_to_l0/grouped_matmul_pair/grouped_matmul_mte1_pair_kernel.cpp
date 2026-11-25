// Independent Ascend 910B GroupedMatmul L1-to-L0 pair benchmark.
//
// The timed region reproduces one BF16 CFG_MDL SplitA/SplitB base-K pair:
// LoadData2D A for BMx64, LoadData2D B for 64x256, and the MTE1-to-M
// completion event. GM-to-L1 population occurs before timing. There is no
// Cube work, output traffic, or complete production operator in the probe.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kMaximumARows = 128;
constexpr int64_t kBaseK = 64;
constexpr int64_t kMaximumBaseN = 256;
constexpr int64_t kMaximumBStepK = 256;
constexpr int64_t kMaximumAElements = kMaximumARows * kBaseK;
constexpr int64_t kMaximumBElements = kMaximumBStepK * kMaximumBaseN;

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
    parameters.srcDValue = static_cast<uint16_t>(columns);
    parameters.dstNzC0Stride = static_cast<uint16_t>(Align16(rows));
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

template <int64_t BaseM>
__aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    constexpr int64_t block_use_m = BaseM / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_k),
        static_cast<uint16_t>(block_use_m), 0, 0, false, 0};
    for (int64_t index = 0; index < block_use_m; ++index) {
        AscendC::LoadData(
            destination[index * block_use_k * 256], source[index * 256],
            parameters);
    }
}

__aicore__ inline void LoadB(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source, int64_t base_n,
        int64_t b_step_k) {
    const int64_t block_use_n = base_n / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_n),
        static_cast<uint16_t>(b_step_k / 16), 0, 0, true, 0};
    for (int64_t index = 0; index < block_use_k; ++index) {
        AscendC::LoadData(
            destination[index * block_use_n * 256], source[index * base_n],
            parameters);
    }
}

template <int64_t BaseM>
__aicore__ inline void RunPair(
        AscendC::LocalTensor<bfloat16_t>& l0a,
        AscendC::LocalTensor<bfloat16_t>& l0b,
        const AscendC::LocalTensor<bfloat16_t>& l1a,
        const AscendC::LocalTensor<bfloat16_t>& l1b, int64_t base_n,
        int64_t b_step_k) {
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    LoadA<BaseM>(l0a, l1a);
    LoadB(l0b, l1b, base_n, b_step_k);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
}

}  // namespace

extern "C" __global__ __aicore__ void grouped_matmul_mte1_pair(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_cycles,
        int64_t base_m, int64_t base_n, int64_t b_step_k,
        int64_t iterations) {
    const int64_t block = AscendC::GetBlockIdx();
    if (base_m < 16 || base_m > 128 || base_m % 16 != 0
            || base_n < 16 || base_n > kMaximumBaseN || base_n % 16 != 0
            || b_step_k < kBaseK || b_step_k > kMaximumBStepK
            || b_step_k % kBaseK != 0
            || iterations <= 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> l1a_buffer;
    AscendC::TBuf<AscendC::TPosition::B1> l1b_buffer;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buffer;
    pipe.InitBuffer(
        l1a_buffer, kMaximumAElements * sizeof(bfloat16_t));
    pipe.InitBuffer(l1b_buffer, kMaximumBElements * sizeof(bfloat16_t));
    pipe.InitBuffer(
        l0a_buffer, kMaximumAElements * sizeof(bfloat16_t));
    pipe.InitBuffer(
        l0b_buffer, kBaseK * kMaximumBaseN * sizeof(bfloat16_t));
    auto l1a = l1a_buffer.Get<bfloat16_t>();
    auto l1b = l1b_buffer.Get<bfloat16_t>();
    auto l0a = l0a_buffer.Get<bfloat16_t>();
    auto l0b = l0b_buffer.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> source_a;
    AscendC::GlobalTensor<bfloat16_t> source_b;
    source_a.SetGlobalBuffer(gm_source);
    source_b.SetGlobalBuffer(gm_source + kMaximumAElements);
    CopyNd2Nz(l1a, source_a, base_m, kBaseK);
    CopyNd2Nz(l1b, source_b, b_step_k, base_n);
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);

    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        if (base_m == 16) {
            RunPair<16>(l0a, l0b, l1a, l1b, base_n, b_step_k);
        } else if (base_m == 32) {
            RunPair<32>(l0a, l0b, l1a, l1b, base_n, b_step_k);
        } else if (base_m == 48) {
            RunPair<48>(l0a, l0b, l1a, l1b, base_n, b_step_k);
        } else if (base_m == 64) {
            RunPair<64>(l0a, l0b, l1a, l1b, base_n, b_step_k);
        } else if (base_m == 80) {
            RunPair<80>(l0a, l0b, l1a, l1b, base_n, b_step_k);
        } else if (base_m == 96) {
            RunPair<96>(l0a, l0b, l1a, l1b, base_n, b_step_k);
        } else if (base_m == 112) {
            RunPair<112>(l0a, l0b, l1a, l1b, base_n, b_step_k);
        } else {
            RunPair<128>(l0a, l0b, l1a, l1b, base_n, b_step_k);
        }
    }
    const int64_t end = AscendC::GetSystemCycle();
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);

    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(
        gm_cycles + block * kCycleSlotBytes / sizeof(int64_t));
    cycles.SetValue(0, end - begin);
    cycles.SetValue(1, 1);
}
