// Measure the two UB-only compute segments on either side of the B-input
// completion boundary in the CANN BF16 elementwise Mul source program.  This
// probe performs no GM data transfer and is not a production operator.
#include "kernel_operator.h"

#include "elementwise_mul_compute_segments_contract.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

using namespace elementwise_mul_compute_segments_contract;

__aicore__ inline void WaitForVectorCompletion() {
    const event_t event_v_s = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(event_v_s);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(event_v_s);
}

extern "C" __global__ __aicore__ void elementwise_mul_compute_segments(
        __gm__ int64_t* worker_results, int64_t segment, int64_t elements,
        int64_t iterations) {
    const int64_t worker = AscendC::GetBlockIdx();
    if (worker < 0 || worker >= kMaximumWorkers
            || (segment != kPrefixCastA
                && segment != kSuffixCastBMulCastOut)
            || elements <= 0 || elements > kMaximumElements
            || iterations <= 0) {
        return;
    }

    const uint32_t count = static_cast<uint32_t>(elements);
    const size_t fp32_bytes = static_cast<size_t>(elements) * sizeof(float);
    const size_t bf16_bytes =
        static_cast<size_t>(elements) * sizeof(bfloat16_t);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_a_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_b_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> product_buffer;
    AscendC::TBuf<AscendC::TPosition::VECIN> source_buffer;
    AscendC::TBuf<AscendC::TPosition::VECOUT> output_buffer;
    pipe.InitBuffer(input_a_buffer, fp32_bytes);
    pipe.InitBuffer(input_b_buffer, fp32_bytes);
    pipe.InitBuffer(product_buffer, fp32_bytes);
    pipe.InitBuffer(source_buffer, bf16_bytes);
    pipe.InitBuffer(output_buffer, bf16_bytes);
    auto input_a = input_a_buffer.Get<float>();
    auto input_b = input_b_buffer.Get<float>();
    auto product = product_buffer.Get<float>();
    auto source = source_buffer.Get<bfloat16_t>();
    auto output = output_buffer.Get<bfloat16_t>();

    AscendC::Duplicate<float>(input_a, 1.5f, count);
    AscendC::Cast(
        source, input_a, AscendC::RoundMode::CAST_RINT, count);
    AscendC::PipeBarrier<PIPE_V>();

    const int64_t cycle_start = AscendC::GetSystemCycle();
    if (segment == kPrefixCastA) {
        #pragma unroll 1
        for (int64_t iteration = 0; iteration < iterations; ++iteration) {
            AscendC::Cast(
                input_a, source, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else {
        #pragma unroll 1
        for (int64_t iteration = 0; iteration < iterations; ++iteration) {
            AscendC::Cast(
                input_b, source, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, input_a, input_b, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                output, product, AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    WaitForVectorCompletion();
    const int64_t cycle_end = AscendC::GetSystemCycle();

    float sink = 0.0f;
    if (segment == kPrefixCastA) {
        sink = input_a.GetValue(0);
    } else {
        AscendC::Cast(
            input_a, output, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        WaitForVectorCompletion();
        sink = input_a.GetValue(0);
    }
    AscendC::GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(
        worker_results + worker * kResultValuesPerWorker);
    result.SetValue(0, cycle_end - cycle_start);
    result.SetValue(1, static_cast<int64_t>(sink * 1000.0f));
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
