// Finite memory/control program for the CANN 8.5.1 BF16 RMSNorm key-0
// source.  The probe preserves gamma/input/output/rstd queue order and the
// four-queue/three-buffer lifetime, but replaces all normalization arithmetic
// with a constant Duplicate.  It therefore measures no production operator.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kResultValuesPerWorker = 8;
constexpr uint32_t kMaximumHidden = 8192;
constexpr uint32_t kRowGroup = 64;

}  // namespace

extern "C" __global__ __aicore__ void rmsnorm_key0_finite_program(
        __gm__ half* source_x, __gm__ half* source_gamma,
        __gm__ half* destination_y, __gm__ float* destination_rstd,
        __gm__ int64_t* worker_cycles, uint32_t rows, uint32_t hidden) {
    const uint32_t worker = AscendC::GetBlockIdx();
    const uint32_t worker_count = AscendC::GetBlockNum();
    if (rows == 0 || hidden == 0 || hidden > kMaximumHidden
            || hidden % 16 != 0 || worker_count == 0
            || worker_count > rows) {
        return;
    }
    const uint32_t block_factor =
        (rows + worker_count - 1) / worker_count;
    const uint32_t row_start = worker * block_factor;
    const uint32_t row_work = (
        worker < worker_count - 1
        ? block_factor : rows - (worker_count - 1) * block_factor);
    const uint32_t row_bytes = hidden * sizeof(half);

    AscendC::GlobalTensor<half> global_x;
    AscendC::GlobalTensor<half> global_gamma;
    AscendC::GlobalTensor<half> global_y;
    AscendC::GlobalTensor<float> global_rstd;
    global_x.SetGlobalBuffer(source_x + row_start * hidden);
    global_gamma.SetGlobalBuffer(source_gamma);
    global_y.SetGlobalBuffer(destination_y + row_start * hidden);
    global_rstd.SetGlobalBuffer(destination_rstd + worker * 8);

    const int64_t cycle_start = AscendC::GetSystemCycle();
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_x;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_gamma;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_y;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_rstd;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_fp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> squared_x;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_fp32;
    pipe.InitBuffer(input_x, 1, row_bytes);
    pipe.InitBuffer(input_gamma, 1, row_bytes);
    pipe.InitBuffer(output_y, 1, row_bytes);
    pipe.InitBuffer(output_rstd, 1, kRowGroup * sizeof(float));
    pipe.InitBuffer(x_fp32, kMaximumHidden * sizeof(float));
    pipe.InitBuffer(squared_x, kMaximumHidden * sizeof(float));
    pipe.InitBuffer(reduce_fp32, kRowGroup * sizeof(float));

    auto gamma = input_gamma.AllocTensor<half>();
    AscendC::DataCopy(gamma, global_gamma, hidden);
    input_gamma.EnQue(gamma);
    gamma = input_gamma.DeQue<half>();

    uint32_t completed_rows = 0;
    const uint32_t groups = (row_work + kRowGroup - 1) / kRowGroup;
    #pragma unroll 1
    for (uint32_t group = 0; group < groups; ++group) {
        const uint32_t remaining = row_work - completed_rows;
        const uint32_t group_rows =
            remaining < kRowGroup ? remaining : kRowGroup;
        auto rstd = output_rstd.AllocTensor<float>();
        #pragma unroll 1
        for (uint32_t row = 0; row < group_rows; ++row) {
            const uint32_t offset = completed_rows * hidden;
            auto x = input_x.AllocTensor<half>();
            AscendC::DataCopy(x, global_x[offset], hidden);
            input_x.EnQue(x);
            x = input_x.DeQue<half>();

            auto y = output_y.AllocTensor<half>();
            AscendC::Duplicate<half>(
                y, static_cast<half>(1.0f), hidden);
            AscendC::PipeBarrier<PIPE_V>();
            output_y.EnQue(y);
            input_x.FreeTensor(x);
            y = output_y.DeQue<half>();
            AscendC::DataCopy(global_y[offset], y, hidden);
            output_y.FreeTensor(y);
            ++completed_rows;
        }
        AscendC::Duplicate<float>(rstd, 1.0f, 8);
        AscendC::PipeBarrier<PIPE_V>();
        output_rstd.EnQue(rstd);
        rstd = output_rstd.DeQue<float>();
        AscendC::DataCopy(global_rstd[group * 8], rstd, 8);
        output_rstd.FreeTensor(rstd);
    }
    input_gamma.FreeTensor(gamma);
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t cycle_end = AscendC::GetSystemCycle();

    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(worker_cycles);
    const int64_t result_offset =
        static_cast<int64_t>(worker) * kResultValuesPerWorker;
    cycles.SetValue(result_offset, cycle_start);
    cycles.SetValue(result_offset + 1, cycle_end);
    cycles.SetValue(result_offset + 2, cycle_end - cycle_start);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(cycles[result_offset]);
}
