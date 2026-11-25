// Independent finite GroupedMatmul GM-to-L1 completion-span probe.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

#ifndef GROUPED_PAIR_BASE_M
#define GROUPED_PAIR_BASE_M 16
#endif
#ifndef GROUPED_PAIR_ACTIVE_M
#define GROUPED_PAIR_ACTIVE_M GROUPED_PAIR_BASE_M
#endif
#ifndef GROUPED_PAIR_B_ROW_STRIDE
#define GROUPED_PAIR_B_ROW_STRIDE 256
#endif

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kBaseM = GROUPED_PAIR_BASE_M;
constexpr int64_t kActiveM = GROUPED_PAIR_ACTIVE_M;
constexpr int64_t kBaseN = 256;
constexpr int64_t kBaseK = 64;
constexpr int64_t kBSourceRowStride = GROUPED_PAIR_B_ROW_STRIDE;
constexpr int64_t kATileElements = kBaseM * kBaseK;
constexpr int64_t kBTileElements = kBaseK * kBaseN;
constexpr int64_t kBSourceTileElements = (
    (kBaseK - 1) * kBSourceRowStride + kBaseN);
static_assert(kBaseM == 16 || kBaseM == 64 || kBaseM == 128,
              "grouped pair probe supports only M16, M64, and M128 source tiles");
static_assert(kActiveM == 16 || kActiveM == 32 || kActiveM == 48
              || kActiveM == kBaseM,
              "grouped pair probe supports M16/M32/M48 tails or the full tile");
static_assert(kActiveM <= kBaseM,
              "grouped pair active rows cannot exceed the base-M allocation");
static_assert(kBSourceRowStride >= kBaseN && kBSourceRowStride <= 65535,
              "grouped pair B source-row stride is outside ND2NZ bounds");

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
    parameters.dstNzC0Stride = static_cast<uint16_t>(rows);
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

__aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    constexpr int64_t block_use_m = kActiveM / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_k),
        static_cast<uint16_t>(block_use_m), 0, 0, false, 0};
    AscendC::LoadData(destination, source, parameters);
}

__aicore__ inline void LoadB(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    constexpr int64_t block_use_n = kBaseN / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_n),
        static_cast<uint16_t>(kBaseN / 16), 0, 0, true, 0};
    for (int64_t index = 0; index < block_use_k; ++index) {
        AscendC::LoadData(
            destination[index * block_use_n * 256], source[index * 256],
            parameters);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void grouped_pair_completion(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output, int64_t pairs,
        int64_t b_source_tiles_per_worker) {
    const int64_t block = AscendC::GetBlockIdx();
    if (pairs <= 0 || pairs % 4 != 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buffer;
    pipe.InitBuffer(a_buffer0, kATileElements * kElementBytes);
    pipe.InitBuffer(a_buffer1, kATileElements * kElementBytes);
    pipe.InitBuffer(b_buffer0, kBTileElements * kElementBytes);
    pipe.InitBuffer(b_buffer1, kBTileElements * kElementBytes);
    pipe.InitBuffer(l0a_buffer, kATileElements * kElementBytes);
    pipe.InitBuffer(l0b_buffer, kBTileElements * kElementBytes);
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = a_buffer1.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();
    auto l0a = l0a_buffer.Get<bfloat16_t>();
    auto l0b = l0b_buffer.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;
    a_source.SetGlobalBuffer(gm_a_source + block * kATileElements);
    CopyNd2Nz(a_local0, a_source, kActiveM, kBaseK, kBaseK);
    CopyNd2Nz(a_local1, a_source, kActiveM, kBaseK, kBaseK);
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);

    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t pair = 0; pair < pairs; ++pair) {
        const int64_t sequence = (
            (pair % b_source_tiles_per_worker) * AscendC::GetBlockNum()
            + block);
        auto a_local = (pair & 1) == 0 ? a_local0 : a_local1;
        auto b_local = (pair & 1) == 0 ? b_local0 : b_local1;
        a_source.SetGlobalBuffer(gm_a_source + block * kATileElements);
        b_source.SetGlobalBuffer(
            gm_b_source + sequence * kBSourceTileElements);
        CopyNd2Nz(a_local, a_source, kActiveM, kBaseK, kBaseK);
        CopyNd2Nz(
            b_local, b_source, kBaseK, kBaseN, kBSourceRowStride);
        if ((pair & 3) == 3) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            LoadA(l0a, a_local);
            LoadB(l0b, b_local);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - begin);
    output.SetValue(1, 1);
}
