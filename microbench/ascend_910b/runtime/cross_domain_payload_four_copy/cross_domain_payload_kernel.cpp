// Independent loaded 1:2 AIC/AIV three-handoff ring, without FIA arithmetic.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

namespace {
constexpr uint16_t kMm1Ready = 10;
constexpr uint16_t kVec1Ready = 11;
constexpr uint16_t kMm2Ready = 12;
constexpr uint8_t kMode = 2;
constexpr int64_t kGroups = 24;
constexpr int64_t kHeads = 8;
constexpr int64_t kRows = 128;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageElements = kRows * kHeads * kHeadDim;
constexpr int64_t kCopiesPerTask = 4;
constexpr int64_t kAivElements = 256;
constexpr int64_t kCycleSlotElements = 64 / sizeof(int64_t);

__aicore__ inline void StoreCycle(
        AscendC::GlobalTensor<int64_t>& slot, int64_t value) {
    slot.SetValue(0, value);
    AscendC::DataCacheCleanAndInvalid<
        int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot);
}
}  // namespace

extern "C" __global__ __aicore__ void cross_domain_payload(
        __gm__ bfloat16_t* gm_kv, __gm__ half* gm_aiv,
        __gm__ half* gm_output, __gm__ int64_t* gm_cycles,
        int64_t task_count, int64_t payload_mode) {
    const int64_t physical = AscendC::GetBlockIdx();
    int64_t group = 0;
    int64_t lane = 0;
    if ASCEND_IS_AIC {
        group = physical;
    } else {
        group = physical / 2;
        lane = physical % 2;
    }
    int64_t actor = group * 3;
    if ASCEND_IS_AIC {
    } else {
        actor += 1 + lane;
    }
    AscendC::GlobalTensor<int64_t> cycle_slot;
    cycle_slot.SetGlobalBuffer(gm_cycles + actor * kCycleSlotElements);
    if (group >= kGroups || (task_count != 4 && task_count != 12)
            || (payload_mode != 0 && payload_mode != 1)) {
        StoreCycle(cycle_slot, -1);
        return;
    }

#ifdef __DAV_C220_CUBE__
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::B1> l1_buffer;
    AscendC::TBuf<AscendC::TPosition::B2> l0b_buffer;
    pipe.InitBuffer(l1_buffer, kRows * kHeadDim * sizeof(bfloat16_t));
    pipe.InitBuffer(l0b_buffer, 16 * 16 * sizeof(bfloat16_t));
    auto l1 = l1_buffer.Get<bfloat16_t>();
    auto l0b = l0b_buffer.Get<bfloat16_t>();
    AscendC::GlobalTensor<bfloat16_t> kv;
    kv.SetGlobalBuffer(gm_kv);
    AscendC::Nd2NzParams parameters;
    parameters.ndNum = 1;
    parameters.nValue = kRows;
    parameters.dValue = kHeadDim;
    parameters.srcNdMatrixStride = 0;
    parameters.srcDValue = kHeads * kHeadDim;
    parameters.dstNzC0Stride = kRows;
    parameters.dstNzNStride = 1;
    parameters.dstNzMatrixStride = 0;
    AscendC::LoadData2DParams l0_parameters;
    l0_parameters.startIndex = 0;
    l0_parameters.repeatTimes = 1;
    l0_parameters.srcStride = 1;
    l0_parameters.dstGap = 0;
    l0_parameters.ifTranspose = false;
#endif
#ifdef __DAV_C220_VEC__
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
    pipe.InitBuffer(input_queue, 1, kAivElements * sizeof(half));
    pipe.InitBuffer(output_queue, 1, kAivElements * sizeof(half));
    AscendC::GlobalTensor<half> aiv_source;
    AscendC::GlobalTensor<half> aiv_output;
    aiv_source.SetGlobalBuffer(gm_aiv);
    aiv_output.SetGlobalBuffer(gm_output);
#endif

    const int64_t start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t tick = 0; tick < task_count + 2; ++tick) {
#ifdef __DAV_C220_CUBE__
        if (tick < task_count) {
            if (payload_mode == 1) {
                for (int64_t ordinal = 0; ordinal < kCopiesPerTask;
                     ++ordinal) {
                    const int64_t page =
                        (tick * 3 + group / kHeads) * kCopiesPerTask
                        + ordinal;
                    const int64_t offset = page * kPageElements
                        + (group % kHeads) * kHeadDim;
                    AscendC::DataCopy(l1, kv[offset], parameters);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                    AscendC::LoadData(l0b, l1, l0_parameters);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
                }
            }
            AscendC::CrossCoreSetFlag<kMode, PIPE_FIX>(kMm1Ready);
        }
        if (tick >= 1 && tick <= task_count) {
            AscendC::CrossCoreWaitFlag(kVec1Ready);
            AscendC::CrossCoreSetFlag<kMode, PIPE_FIX>(kMm2Ready);
        }
#endif
#ifdef __DAV_C220_VEC__
        if (tick >= 1 && tick <= task_count) {
            AscendC::CrossCoreWaitFlag(kMm1Ready);
            if (payload_mode == 1) {
                const int64_t offset = ((tick - 1) * 48 + group * 2 + lane)
                    * kAivElements;
                auto input = input_queue.AllocTensor<half>();
                AscendC::DataCopy(input, aiv_source[offset], kAivElements);
                input_queue.EnQue(input);
                input = input_queue.DeQue<half>();
                auto output = output_queue.AllocTensor<half>();
                AscendC::Adds(output, input, static_cast<half>(1),
                              kAivElements);
                input_queue.FreeTensor(input);
                output_queue.EnQue(output);
                output = output_queue.DeQue<half>();
                AscendC::DataCopy(aiv_output[offset], output, kAivElements);
                output_queue.FreeTensor(output);
            }
            AscendC::CrossCoreSetFlag<kMode, PIPE_MTE3>(kVec1Ready);
        }
        if (tick >= 2 && tick <= task_count + 1) {
            AscendC::CrossCoreWaitFlag(kMm2Ready);
        }
#endif
    }
    const int64_t end = AscendC::GetSystemCycle();
    StoreCycle(cycle_slot, end - start);
}
