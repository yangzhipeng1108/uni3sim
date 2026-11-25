// Independent MTE2-only replica of the CANN 8.5.1 MatMulV2
// M2220/N4096/K12288 persistent-worker operand program. No Cube or output
// work runs.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

namespace {

constexpr int64_t kElementBytes = sizeof(bfloat16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kM = 2220;
constexpr int64_t kAlignedM = 2224;
constexpr int64_t kN = 4096;
constexpr int64_t kK = 12288;
constexpr int64_t kWorkerGridM = 3;
constexpr int64_t kWorkerGridN = 8;
constexpr int64_t kSingleCoreM = 768;
constexpr int64_t kSingleCoreN = 512;
constexpr int64_t kBaseM = 256;
constexpr int64_t kBaseN = 128;
constexpr int64_t kBaseK = 64;
constexpr int64_t kAColumns = 256;
constexpr int64_t kBRows = kBaseN;
constexpr int64_t kBColumns = 256;
constexpr int64_t kKSlabs = kK / kAColumns;

static_assert(kM <= kAlignedM && kAlignedM % 16 == 0);
static_assert(kWorkerGridM * kSingleCoreM >= kAlignedM);
static_assert(kWorkerGridN * kSingleCoreN == kN);
static_assert(kAColumns == kBColumns && kAColumns % kBaseK == 0);
static_assert(2 * (kBaseM * kAColumns + kBRows * kBColumns)
              * kElementBytes <= 512 * 1024);

__aicore__ inline int64_t Align16(int64_t value) {
    return (value + 15) / 16 * 16;
}

__aicore__ inline int64_t Minimum(int64_t left, int64_t right) {
    return left < right ? left : right;
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

__aicore__ inline void StoreResult(
        __gm__ int64_t* gm_output, int64_t start, int64_t end,
        int64_t count) {
    const int64_t block = AscendC::GetBlockIdx();
    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(
        gm_output + block * kCycleSlotBytes / sizeof(int64_t));
    output.SetValue(0, end - start);
    output.SetValue(1, count);
}

}  // namespace

extern "C" __global__ __aicore__ void aic_mte2_gemm_m2220_source_order(
        __gm__ bfloat16_t* gm_a_source, __gm__ bfloat16_t* gm_b_source,
        __gm__ int64_t* gm_output) {
    const int64_t block = AscendC::GetBlockIdx();
    const int64_t m_worker = block % kWorkerGridM;
    const int64_t n_worker = block / kWorkerGridM;
    const int64_t m_worker_start = Minimum(
        m_worker * kSingleCoreM, kAlignedM - kSingleCoreM);
    const int64_t n_worker_start = n_worker * kSingleCoreN;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer0;
    AscendC::TBuf<AscendC::TPosition::A1> a_buffer1;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer0;
    AscendC::TBuf<AscendC::TPosition::B1> b_buffer1;
    pipe.InitBuffer(a_buffer0, kBaseM * kAColumns * kElementBytes);
    pipe.InitBuffer(a_buffer1, kBaseM * kAColumns * kElementBytes);
    pipe.InitBuffer(b_buffer0, kBRows * kBColumns * kElementBytes);
    pipe.InitBuffer(b_buffer1, kBRows * kBColumns * kElementBytes);
    auto a_local0 = a_buffer0.Get<bfloat16_t>();
    auto a_local1 = a_buffer1.Get<bfloat16_t>();
    auto b_local0 = b_buffer0.Get<bfloat16_t>();
    auto b_local1 = b_buffer1.Get<bfloat16_t>();
    AscendC::GlobalTensor<bfloat16_t> a_source;
    AscendC::GlobalTensor<bfloat16_t> b_source;

    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t local_n = 0; local_n < kSingleCoreN; local_n += kBaseN) {
        #pragma unroll 1
        for (int64_t local_m = 0; local_m < kSingleCoreM;
                local_m += kBaseM) {
            const int64_t m_start = m_worker_start + local_m;
            const int64_t n_start = n_worker_start + local_n;
            #pragma unroll 1
            for (int64_t slab = 0; slab < kKSlabs; ++slab) {
                const int64_t k_start = slab * kAColumns;
                a_source.SetGlobalBuffer(
                    gm_a_source + m_start * kK + k_start);
                b_source.SetGlobalBuffer(
                    gm_b_source + n_start * kK + k_start);
                if ((slab & 1) == 0) {
                    CopyNd2Nz(
                        a_local0, a_source, kBaseM, kAColumns, kK);
                    CopyNd2Nz(
                        b_local0, b_source, kBRows, kBColumns, kK);
                } else {
                    CopyNd2Nz(
                        a_local1, a_source, kBaseM, kAColumns, kK);
                    CopyNd2Nz(
                        b_local1, b_source, kBRows, kBColumns, kK);
                }
            }
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    const int64_t end = AscendC::GetSystemCycle();
    StoreResult(
        gm_output, start, end,
        (kSingleCoreM / kBaseM) * (kSingleCoreN / kBaseN) * kKSlabs);
}
