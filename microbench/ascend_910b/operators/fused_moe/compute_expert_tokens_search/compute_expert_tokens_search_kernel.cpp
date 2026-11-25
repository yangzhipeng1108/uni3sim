// Isolate one source-shaped LocalTensor binary-search/output-control request
// from CANN MoeComputeExpertTokens key 1003.  GM input setup, task launch,
// SyncAll, workspace traffic, and the rest of the production operator are
// outside the timed interval.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kMaximumInputElements = 320;
constexpr int64_t kOutputStrideElements = 8;
constexpr int64_t kOutputSlots = 256;

}  // namespace

extern "C" __global__ __aicore__ void compute_expert_tokens_search(
        __gm__ int32_t* gm_input, __gm__ int64_t* gm_cycles,
        __gm__ int32_t* gm_validation, int64_t input_elements,
        int64_t target, int64_t iterations) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> output_buffer;
    pipe.InitBuffer(
        input_buffer,
        static_cast<size_t>(kMaximumInputElements) * sizeof(int32_t));
    pipe.InitBuffer(
        output_buffer,
        static_cast<size_t>(kOutputSlots * kOutputStrideElements)
            * sizeof(int32_t));
    auto input = input_buffer.Get<int32_t>();
    auto output = output_buffer.Get<int32_t>();

    AscendC::GlobalTensor<int32_t> source;
    source.SetGlobalBuffer(gm_input);
    AscendC::DataCopy(
        input, source,
        static_cast<uint32_t>(kMaximumInputElements));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    int32_t final_location = 0;
    int32_t final_value = 0;
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    const int64_t begin = AscendC::GetSystemCycle();
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        int32_t low = 0;
        int32_t high = static_cast<int32_t>(input_elements - 1);
        int32_t target_location = 0;
        while (low <= high) {
            const int32_t middle = (low + high) / 2;
            if (input.GetValue(middle) > target) {
                high = middle - 1;
            } else {
                low = middle + 1;
                target_location = middle;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const bool found = input.GetValue(target_location) == target;
        final_location = target_location;
        final_value = found ? target_location + 1 : target_location;
        const int64_t slot = iteration % kOutputSlots;
        AscendC::Duplicate(
            output[slot * kOutputStrideElements], final_value, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    const int64_t end = AscendC::GetSystemCycle();

    gm_cycles[0] = end - begin;
    gm_validation[0] = final_location;
    gm_validation[1] = final_value;
    gm_validation[2] = output[
        ((iterations - 1) % kOutputSlots) * kOutputStrideElements].GetValue(0);
}
