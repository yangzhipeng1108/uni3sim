// Exact finite 48-AIV q4/q1 source program for the five-tile worker classes.
//
// The probe preserves the production tiler's full/tail ownership and 2R1W
// queue order. Duplicate replaces Mul arithmetic so the measured interval is
// an independent memory/control mechanism rather than a production operator.
#include "kernel_operator.h"

#include "aiv_mul_q4q1_five_tile_program_contract.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

extern "C" __global__ __aicore__ void aiv_mul_q4q1_five_tile_program(
        __gm__ half* source_a, __gm__ half* source_b,
        __gm__ half* destination, __gm__ int64_t* worker_cycles) {
    using namespace aiv_mul_q4q1_five_tile_program_contract;

    const int64_t worker = AscendC::GetBlockIdx();
    const int64_t worker_offset = worker * kBlockFactorElements;
    const int64_t tail_elements = (
        worker < kFullWorkers
        ? kFullWorkerTailElements : kLastWorkerTailElements);

    AscendC::GlobalTensor<half> global_a;
    AscendC::GlobalTensor<half> global_b;
    AscendC::GlobalTensor<half> global_destination;
    global_a.SetGlobalBuffer(source_a + worker_offset);
    global_b.SetGlobalBuffer(source_b + worker_offset);
    global_destination.SetGlobalBuffer(destination + worker_offset);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 4> input_a;
    AscendC::TQue<AscendC::QuePosition::VECIN, 4> input_b;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output;
    pipe.InitBuffer(input_a, 4, kFullTileElements * sizeof(half));
    pipe.InitBuffer(input_b, 4, kFullTileElements * sizeof(half));
    pipe.InitBuffer(output, 1, kFullTileElements * sizeof(half));

    const int64_t cycle_start = AscendC::GetSystemCycle();
    #pragma unroll 1
    for (int64_t tile = 0; tile <= kFullTilesPerWorker; ++tile) {
        const int64_t tile_elements = (
            tile < kFullTilesPerWorker ? kFullTileElements : tail_elements);
        const int64_t offset = tile * kFullTileElements;
        auto local_a = input_a.AllocTensor<half>();
        AscendC::DataCopy(local_a, global_a[offset], tile_elements);
        input_a.EnQue(local_a);
        auto local_b = input_b.AllocTensor<half>();
        AscendC::DataCopy(local_b, global_b[offset], tile_elements);
        input_b.EnQue(local_b);

        local_a = input_a.DeQue<half>();
        local_b = input_b.DeQue<half>();
        auto local_output = output.AllocTensor<half>();
        AscendC::Duplicate<half>(
            local_output, static_cast<half>(1.0f), tile_elements);
        output.EnQue(local_output);
        input_a.FreeTensor(local_a);
        input_b.FreeTensor(local_b);

        local_output = output.DeQue<half>();
        AscendC::DataCopy(
            global_destination[offset], local_output, tile_elements);
        output.FreeTensor(local_output);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t cycle_end = AscendC::GetSystemCycle();
    AscendC::GlobalTensor<int64_t> cycles;
    cycles.SetGlobalBuffer(worker_cycles);
    cycles.SetValue(worker, cycle_end - cycle_start);
}
