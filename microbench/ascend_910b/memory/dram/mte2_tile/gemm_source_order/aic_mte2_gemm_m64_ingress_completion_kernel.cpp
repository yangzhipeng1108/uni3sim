// MTE2-to-MTE1 completion replica of the CANN 8.5.1 MatMulV2
// M64/N5120/K1536 source program.  No Cube or output work runs.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kARows = 64;
constexpr int64_t kAColumns = 1536;
constexpr int64_t kBRows = 224;
constexpr int64_t kBColumns = 256;
constexpr int64_t kBaseK = 64;
constexpr int64_t kBSlabs = kAColumns / kBColumns;

static_assert(kAColumns % kBColumns == 0);
static_assert(kBColumns % kBaseK == 0);
static_assert(
    (kARows * kAColumns + 2 * kBRows * kBColumns) * kElementBytes
    <= 512 * 1024);
static_assert(2 * kARows * kBaseK * kElementBytes <= 64 * 1024);
static_assert(2 * kBaseK * kBRows * kElementBytes <= 64 * 1024);

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
    parameters.srcDValue = static_cast<uint16_t>(kAColumns);
    parameters.dstNzC0Stride = static_cast<uint16_t>(Align16(rows));
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::DataCopy(destination, source, parameters);
}

__aicore__ inline void LoadA(
        AscendC::LocalTensor<bfloat16_t>& destination,
        const AscendC::LocalTensor<bfloat16_t>& source) {
    constexpr int64_t block_use_m = kARows / 16;
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
        const AscendC::LocalTensor<bfloat16_t>& source) {
    constexpr int64_t block_use_n = kBRows / 16;
    constexpr int64_t block_use_k = kBaseK / 16;
    AscendC::LoadData2DParams parameters{
        0, static_cast<uint8_t>(block_use_n),
        static_cast<uint16_t>(kBColumns / 16), 0, 0, true, 0};
    for (int64_t index = 0; index < block_use_k; ++index) {
        AscendC::LoadData(
            destination[index * block_use_n * 256], source[index * 256],
            parameters);
    }
}

__aicore__ inline void ConsumeBaseKPair(
        AscendC::TQue<AscendC::TPosition::A2, 2>& l0a_queue,
        AscendC::TQue<AscendC::TPosition::B2, 2>& l0b_queue,
        const AscendC::LocalTensor<bfloat16_t>& a_local,
        const AscendC::LocalTensor<bfloat16_t>& b_local) {
    auto l0a = l0a_queue.AllocTensor<bfloat16_t>();
    auto l0b = l0b_queue.AllocTensor<bfloat16_t>();
    LoadA(l0a, a_local);
    LoadB(l0b, b_local);
    l0a_queue.EnQue(l0a);
    l0b_queue.EnQue(l0b);
    auto completed_l0a = l0a_queue.DeQue<bfloat16_t>();
    auto completed_l0b = l0b_queue.DeQue<bfloat16_t>();
    l0a_queue.FreeTensor(completed_l0a);
    l0b_queue.FreeTensor(completed_l0b);
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_gemm_m64_ingress_completion(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output) {
    const int64_t block = AscendC::GetBlockIdx();
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> a_queue;
    AscendC::TQue<AscendC::TPosition::B1, 2> b_queue;
    AscendC::TQue<AscendC::TPosition::A2, 2> l0a_queue;
    AscendC::TQue<AscendC::TPosition::B2, 2> l0b_queue;
    pipe.InitBuffer(a_queue, 1, kARows * kAColumns * kElementBytes);
    pipe.InitBuffer(b_queue, 2, kBRows * kBColumns * kElementBytes);
    pipe.InitBuffer(l0a_queue, 2, kARows * kBaseK * kElementBytes);
    pipe.InitBuffer(l0b_queue, 2, kBaseK * kBRows * kElementBytes);
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;
    a_source.SetGlobalBuffer(gm_a_source);
    const int64_t b_row_offset = block * kBRows * kAColumns;

    const int64_t start = AscendC::GetSystemCycle();
    auto a_producer = a_queue.AllocTensor<bfloat16_t>();
    CopyNd2Nz(a_producer, a_source, kARows, kAColumns);
    a_queue.EnQue(a_producer);
    auto a_consumer = a_queue.DeQue<bfloat16_t>();
    #pragma unroll 1
    for (int64_t slab = 0; slab < kBSlabs; ++slab) {
        b_source.SetGlobalBuffer(
            gm_b_source + b_row_offset + slab * kBColumns);
        auto b_producer = b_queue.AllocTensor<bfloat16_t>();
        CopyNd2Nz(b_producer, b_source, kBRows, kBColumns);
        b_queue.EnQue(b_producer);
        auto b_consumer = b_queue.DeQue<bfloat16_t>();
        #pragma unroll 1
        for (int64_t base_pair = 0;
                base_pair < kBColumns / kBaseK; ++base_pair) {
            ConsumeBaseKPair(
                l0a_queue, l0b_queue, a_consumer, b_consumer);
        }
        b_queue.FreeTensor(b_consumer);
    }
    a_queue.FreeTensor(a_consumer);
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, kAColumns / kBaseK);
    output.SetValue(2, start);
    output.SetValue(3, end);
}
