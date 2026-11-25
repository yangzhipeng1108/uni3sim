// Independent source-shaped MatMulV2 A1/B1 MTE2 descriptor program probe.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kAHeight = 128;
constexpr int64_t kAWidth = 512;
constexpr int64_t kBHeight = 256;
constexpr int64_t kBWidth = 240;
constexpr int64_t kADescriptorsPerProgram = 112;
constexpr int64_t kBDescriptorsPerProgram = 168;
constexpr int64_t kSourceTiles = 24;

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

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_l1_program(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output, int64_t programs) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t a_tile_elements = kAHeight * kAWidth;
    const int64_t b_tile_elements = kBHeight * kBWidth;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    pipe.InitBuffer(a_buffer0, a_tile_elements * kElementBytes);
    pipe.InitBuffer(a_buffer1, a_tile_elements * kElementBytes);
    pipe.InitBuffer(b_buffer0, b_tile_elements * kElementBytes);
    pipe.InitBuffer(b_buffer1, b_tile_elements * kElementBytes);
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = a_buffer1.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;
    a_source.SetGlobalBuffer(
        gm_a_source + (block % kSourceTiles) * a_tile_elements);
    b_source.SetGlobalBuffer(
        gm_b_source + (block % kSourceTiles) * b_tile_elements);
    CopyNd2Nz(a_local0, a_source, kAHeight, kAWidth);
    CopyNd2Nz(b_local0, b_source, kBHeight, kBWidth);
    AscendC::PipeBarrier<PIPE_MTE2>();

    const int64_t start = AscendC::GetSystemCycle();
    for (int64_t program = 0; program < programs; ++program) {
        const int64_t source_tile = (program + block) % kSourceTiles;
        a_source.SetGlobalBuffer(gm_a_source + source_tile * a_tile_elements);
        for (int64_t descriptor = 0;
                descriptor < kADescriptorsPerProgram; ++descriptor) {
            auto destination = (descriptor & 1) == 0 ? a_local0 : a_local1;
            CopyNd2Nz(destination, a_source, kAHeight, kAWidth);
        }
        b_source.SetGlobalBuffer(gm_b_source + source_tile * b_tile_elements);
        for (int64_t descriptor = 0;
                descriptor < kBDescriptorsPerProgram; ++descriptor) {
            auto destination = (descriptor & 1) == 0 ? b_local0 : b_local1;
            CopyNd2Nz(destination, b_source, kBHeight, kBWidth);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, 1);
}
