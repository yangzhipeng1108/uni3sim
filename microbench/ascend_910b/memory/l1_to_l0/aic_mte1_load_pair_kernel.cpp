// Independent Ascend 910B L1-to-L0 source-primitive benchmark.
//
// The timed region is one repeated FIA-shaped L0A/L0B load pair: dynamic
// LoadData3D for A, the source-required fmatrix reset, one or more LoadData2D
// calls for B, and the MTE1->M completion event.  GM->L1 population happens
// before timing.  There is no attention arithmetic, Vector work, or output
// traffic in the timed interval.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kMaximumL1Elements = 256 * 128;
constexpr int64_t kMaximumL0Elements = 128 * 128;

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

template <int64_t MatrixRows>
__aicore__ inline void ResetLoad3dConfig() {
    constexpr uint8_t padding[] = {0, 0, 0, 255};
    AscendC::SetFmatrix(
        MatrixRows / 16, 16, padding, AscendC::FmatrixMode::FMATRIX_LEFT);
    AscendC::SetLoadDataPaddingValue<bfloat16_t>(0);
}

template <int64_t MatrixRows>
__aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source,
        int64_t k_extension) {
    AscendC::LoadData3DParamsV2<bfloat16_t> parameters;
    parameters.l1H = 1;
    parameters.l1W = 16;
    parameters.padList[0] = 0;
    parameters.padList[1] = 0;
    parameters.padList[2] = 0;
    parameters.padList[3] = 255;
    parameters.mStartPt = 0;
    parameters.kStartPt = 0;
    parameters.mExtension = 16;
    parameters.kExtension = static_cast<uint16_t>(k_extension);
    parameters.channelSize = static_cast<uint16_t>(k_extension);
    parameters.strideW = 1;
    parameters.strideH = 1;
    parameters.filterW = 1;
    parameters.filterSizeW = 0;
    parameters.filterH = 1;
    parameters.filterSizeH = 0;
    parameters.dilationFilterW = 1;
    parameters.dilationFilterH = 1;
    parameters.enTranspose = 0;
    parameters.fMatrixCtrl = 0;
    AscendC::LoadData(destination, source, parameters);
    ResetLoad3dConfig<MatrixRows>();
}

__aicore__ inline void LoadMm1B(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source,
        int64_t head_dim) {
    AscendC::LoadData2DParams parameters;
    parameters.startIndex = 0;
    parameters.repeatTimes = 8;
    parameters.srcStride = 1;
    parameters.dstGap = 0;
    parameters.ifTranspose = false;
    const int64_t k_loops = head_dim / 16;
    for (int64_t index = 0; index < k_loops; ++index) {
        AscendC::LoadData(
            destination[128 * index * 16], source[256 * index * 16],
            parameters);
    }
}

__aicore__ inline void LoadMm2B(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source,
        int64_t head_dim) {
    AscendC::LoadData2DParams parameters;
    parameters.startIndex = 0;
    parameters.repeatTimes = static_cast<uint8_t>(head_dim / 16);
    parameters.srcStride = 16;
    parameters.dstGap = 0;
    parameters.ifTranspose = true;
    for (int64_t index = 0; index < 8; ++index) {
        AscendC::LoadData(
            destination[index * 16 * head_dim], source[index * 16 * 16],
            parameters);
    }
}

template <int64_t MatrixRows>
__aicore__ inline void RunPair(
        AscendC::LocalTensor<bfloat16_t>& l0a,
        AscendC::LocalTensor<bfloat16_t>& l0b,
        const AscendC::LocalTensor<bfloat16_t>& l1a,
        const AscendC::LocalTensor<bfloat16_t>& l1b,
        int64_t head_dim, bool mm2) {
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    LoadA<MatrixRows>(l0a, l1a, mm2 ? 128 : head_dim);
    if (mm2) {
        LoadMm2B(l0b, l1b, head_dim);
    } else {
        LoadMm1B(l0b, l1b, head_dim);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte1_load_pair(
        __gm__ bfloat16_t* gm_source, __gm__ int64_t* gm_cycles,
        int64_t mode, int64_t iterations) {
    const int64_t block = AscendC::GetBlockIdx();
    const bool mm2 = mode >= 2;
    const int64_t head_dim = (mode == 0 || mode == 2) ? 64 : 128;
    if (mode < 0 || mode > 3 || iterations <= 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> l1a_buffer;
    AscendC::TBuf<AscendC::TPosition::B1> l1b_buffer;
    AscendC::TBuf<AscendC::TPosition::A2> l0a_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buffer;
    pipe.InitBuffer(
        l1a_buffer, kMaximumL1Elements * sizeof(bfloat16_t));
    pipe.InitBuffer(
        l1b_buffer, kMaximumL1Elements * sizeof(bfloat16_t));
    pipe.InitBuffer(
        l0a_buffer, kMaximumL0Elements * sizeof(bfloat16_t));
    pipe.InitBuffer(
        l0b_buffer, kMaximumL0Elements * sizeof(bfloat16_t));
    auto l1a = l1a_buffer.Get<bfloat16_t>();
    auto l1b = l1b_buffer.Get<bfloat16_t>();
    auto l0a = l0a_buffer.Get<bfloat16_t>();
    auto l0b = l0b_buffer.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> source_a;
    AscendC::GlobalTensor<bfloat16_t> source_b;
    source_a.SetGlobalBuffer(gm_source);
    source_b.SetGlobalBuffer(gm_source + kMaximumL1Elements);
    CopyNd2Nz(l1a, source_a, 16, mm2 ? 256 : head_dim);
    CopyNd2Nz(l1b, source_b, 256, head_dim);
    AscendC::PipeBarrier<PIPE_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    if (mm2) {
        ResetLoad3dConfig<128>();
    } else {
        ResetLoad3dConfig<256>();
    }
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);

    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        if (mm2) {
            RunPair<128>(l0a, l0b, l1a, l1b, head_dim, true);
        } else {
            RunPair<256>(l0a, l0b, l1a, l1b, head_dim, false);
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
