// Independent grouped-MatMul A/B GM-to-L1 residency and stepK drain probe.
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

extern "C" __global__ __aicore__ void aic_mte2_grouped_pair(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output, int64_t a_rows, int64_t b_columns,
        int64_t a_source_tiles, int64_t b_source_tiles, int64_t iterations,
        int64_t drain_interval) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t a_tile_elements = a_rows * 64;
    const int64_t b_tile_elements = 64 * b_columns;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    const int64_t a_buffer_bytes = Align16(a_rows) * 64 * kElementBytes;
    const int64_t b_buffer_bytes = b_tile_elements * kElementBytes;
    pipe.InitBuffer(a_buffer0, a_buffer_bytes);
    pipe.InitBuffer(a_buffer1, a_buffer_bytes);
    pipe.InitBuffer(b_buffer0, b_buffer_bytes);
    pipe.InitBuffer(b_buffer1, b_buffer_bytes);
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = a_buffer1.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();

    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;
    a_source.SetGlobalBuffer(
        gm_a_source + (block % a_source_tiles) * a_tile_elements);
    b_source.SetGlobalBuffer(
        gm_b_source + (block % b_source_tiles) * b_tile_elements);
    CopyNd2Nz(a_local0, a_source, a_rows, 64);
    CopyNd2Nz(b_local0, b_source, 64, b_columns);
    AscendC::PipeBarrier<PIPE_MTE2>();

    const int64_t start = AscendC::GetSystemCycle();
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const int64_t sequence = iteration * AscendC::GetBlockNum() + block;
        a_source.SetGlobalBuffer(
            gm_a_source + (sequence % a_source_tiles) * a_tile_elements);
        auto a_local = (iteration & 1) == 0 ? a_local0 : a_local1;
        CopyNd2Nz(a_local, a_source, a_rows, 64);
        b_source.SetGlobalBuffer(
            gm_b_source + (sequence % b_source_tiles) * b_tile_elements);
        auto b_local = (iteration & 1) == 0 ? b_local0 : b_local1;
        CopyNd2Nz(b_local, b_source, 64, b_columns);
        if (drain_interval == 4 && (iteration & 3) == 3) {
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    }
    if (drain_interval == 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, 1);
}
