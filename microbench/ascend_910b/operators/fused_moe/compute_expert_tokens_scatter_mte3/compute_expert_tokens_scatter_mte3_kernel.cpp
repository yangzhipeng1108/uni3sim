// Isolate one source-form CANN DataCopyPad UB-to-GM scatter request and its
// terminal MTE3 completion edge.  This probe contains no production operator
// arithmetic, input data, or duration.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

extern "C" __global__ __aicore__ void compute_expert_tokens_scatter_mte3(
        __gm__ int32_t* destination, __gm__ int64_t* output_cycles,
        int64_t block_count, int64_t iterations) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    AscendC::GlobalTensor<int64_t> output;
    output.SetGlobalBuffer(output_cycles);
    if (block_count <= 0 || block_count > 256 || iterations <= 0) {
        output.SetValue(0, -1);
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> source_buffer;
    pipe.InitBuffer(source_buffer, 8192);
    auto source = source_buffer.Get<int32_t>();
    AscendC::Duplicate<int32_t>(source, 1, 2048);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::GlobalTensor<int32_t> target;
    target.SetGlobalBuffer(destination);
    AscendC::DataCopyParams parameters{
        static_cast<uint16_t>(block_count), static_cast<uint16_t>(4),
        static_cast<uint16_t>(0), static_cast<uint16_t>(0)};
    AscendC::DataCopyPad(target, source, parameters);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

    const int64_t begin = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        AscendC::DataCopyPad(target, source, parameters);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
    const int64_t end = AscendC::GetSystemCycle();
    output.SetValue(0, end - begin);
}
