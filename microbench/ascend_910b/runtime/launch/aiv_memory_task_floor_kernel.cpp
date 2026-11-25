// Minimum source-dependent AIV memory task used to measure pipeline
// fill/drain independently of any production operator.  Every AIV executes
// one 64-byte FP16 GM->UB copy, one UB-resident vector multiply, and one
// 64-byte UB->GM copy.  Worker cache lines are contiguous for the requested
// round count so the benchmark does not impose an artificial inter-worker
// address stride.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {

constexpr int64_t kElementsPerAiv = 32;
constexpr int64_t kBytesPerAiv =
    kElementsPerAiv * static_cast<int64_t>(sizeof(half));
constexpr int64_t kMaximumRounds = 64;

}  // namespace

extern "C" __global__ __aicore__ void aiv_memory_task_floor(
        __gm__ half* source, __gm__ half* destination, int64_t rounds) {
    if (rounds <= 0 || rounds > kMaximumRounds) {
        return;
    }
    const int64_t worker_offset = (
        static_cast<int64_t>(AscendC::GetBlockIdx())
        * rounds * kElementsPerAiv);
    AscendC::GlobalTensor<half> global_source;
    AscendC::GlobalTensor<half> global_destination;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
    pipe.InitBuffer(input_queue, 1, kBytesPerAiv);
    pipe.InitBuffer(output_queue, 1, kBytesPerAiv);

    #pragma unroll 1
    for (int64_t round = 0; round < rounds; ++round) {
        const int64_t offset = worker_offset + round * kElementsPerAiv;
        global_source.SetGlobalBuffer(source + offset);
        global_destination.SetGlobalBuffer(destination + offset);

        AscendC::LocalTensor<half> input = input_queue.AllocTensor<half>();
        AscendC::DataCopy(input, global_source, kElementsPerAiv);
        input_queue.EnQue(input);

        input = input_queue.DeQue<half>();
        AscendC::LocalTensor<half> output = output_queue.AllocTensor<half>();
        AscendC::Mul(output, input, input, kElementsPerAiv);
        output_queue.EnQue(output);
        input_queue.FreeTensor(input);

        output = output_queue.DeQue<half>();
        AscendC::DataCopy(global_destination, output, kElementsPerAiv);
        output_queue.FreeTensor(output);
    }
}
