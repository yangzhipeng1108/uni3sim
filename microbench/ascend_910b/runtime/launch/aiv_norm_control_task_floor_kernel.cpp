// Independent AIV task floor for a normalization-style control footprint.
// The kernel retains source-derived row distribution, four queue lifecycles,
// and three temporary-buffer declarations, but deliberately performs no
// production normalization arithmetic or tensor data movement.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

using namespace AscendC;

constexpr uint32_t kTensorElements = 256;
constexpr uint32_t kRowFactor = 64;
constexpr uint32_t kReduceElements = 64;
constexpr int64_t kValuesPerCacheLine = 8;

class NormControlTask {
public:
    __aicore__ inline void Init(
            __gm__ int64_t* validation, uint32_t num_rows) {
        block_index_ = AscendC::GetBlockIdx();
        const uint32_t block_count = AscendC::GetBlockNum();
        block_factor_ = (num_rows + block_count - 1) / block_count;
        if (block_index_ < block_count - 1) {
            row_work_ = block_factor_;
        } else {
            row_work_ = num_rows - (block_count - 1) * block_factor_;
        }
        validation_.SetGlobalBuffer(
            validation + block_index_ * kValuesPerCacheLine);

        constexpr uint32_t kBf16Bytes =
            kTensorElements * sizeof(bfloat16_t);
        constexpr uint32_t kFp32Bytes =
            kTensorElements * sizeof(float);
        pipe_.InitBuffer(input_x_, 1, kBf16Bytes);
        pipe_.InitBuffer(input_gamma_, 1, kBf16Bytes);
        pipe_.InitBuffer(output_y_, 1, kBf16Bytes);
        pipe_.InitBuffer(output_rstd_, 1, kRowFactor * sizeof(float));
        pipe_.InitBuffer(x_fp32_, kFp32Bytes);
        pipe_.InitBuffer(squared_x_, kFp32Bytes);
        pipe_.InitBuffer(reduce_fp32_, kReduceElements * sizeof(float));
    }

    __aicore__ inline void Process() {
        auto gamma = input_gamma_.AllocTensor<bfloat16_t>();
        input_gamma_.EnQue(gamma);
        gamma = input_gamma_.DeQue<bfloat16_t>();

        const uint32_t outer_groups =
            (row_work_ + kRowFactor - 1) / kRowFactor;
        uint32_t completed_rows = 0;
        #pragma unroll 1
        for (uint32_t outer = 0; outer < outer_groups; ++outer) {
            const uint32_t remaining = row_work_ - completed_rows;
            const uint32_t rows =
                remaining < kRowFactor ? remaining : kRowFactor;
            auto rstd = output_rstd_.AllocTensor<float>();
            #pragma unroll 1
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t logical_offset =
                    (outer * kRowFactor + row) * kTensorElements;
                auto x = input_x_.AllocTensor<bfloat16_t>();
                input_x_.EnQue(x);
                x = input_x_.DeQue<bfloat16_t>();
                input_x_.FreeTensor(x);
                auto y = output_y_.AllocTensor<bfloat16_t>();
                output_y_.EnQue(y);
                y = output_y_.DeQue<bfloat16_t>();
                output_y_.FreeTensor(y);
                completed_rows += (
                    logical_offset == static_cast<uint32_t>(-1)) ? 0 : 1;
            }
            output_rstd_.EnQue(rstd);
            rstd = output_rstd_.DeQue<float>();
            output_rstd_.FreeTensor(rstd);
        }
        input_gamma_.FreeTensor(gamma);

        validation_.SetValue(
            0, static_cast<int64_t>(block_index_ + completed_rows));
        AscendC::DataCacheCleanAndInvalid<
            int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(validation_);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_x_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_gamma_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_y_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_rstd_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_fp32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> squared_x_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_fp32_;
    AscendC::GlobalTensor<int64_t> validation_;
    uint32_t block_index_ = 0;
    uint32_t block_factor_ = 0;
    uint32_t row_work_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void aiv_norm_control_task_floor(
        __gm__ int64_t* validation, uint32_t num_rows) {
    NormControlTask task;
    task.Init(validation, num_rows);
    task.Process();
}
