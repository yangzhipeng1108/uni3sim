// Second AIV instruction-fetch kernel, distinct symbol, same body as
// aiv_text_footprint_kernel.cpp.
//
// A separate symbol is the whole point: linking both kernels into one
// process lets a single stream alternate between two different kernel
// texts, which is the eviction control the same-binary interleave in
// cold_launch.sh could not provide.  If alternating restores the cold
// penalty the other kernel's text evicts this one, so a production step
// that runs many distinct kernels fetches every one of them; if it stays
// warm both texts are resident together and only first use is cold.
//
// The body issues ALT_TEXT_BODY straight-line vector instructions on one
// UB-resident tile.  The tile, the element count and the memory traffic are
// the same for every ALT_TEXT_BODY, so two variants differ only in how many bytes
// of kernel text the AI vector core has to fetch before it can retire the
// same amount of work.  Pairing a process's first launch (cold instruction
// cache) against later launches of the same binary (warm) therefore isolates
// the instruction fetch, and sweeping ALT_TEXT_BODY turns that isolated cost into
// a rate with a unit.
//
// Alternating Adds with opposite signed immediates keeps every instruction
// distinct (no common-subexpression collapse) and keeps the accumulated value
// bounded, so the sweep cannot be confounded by denormal or infinite operands.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

#ifndef ALT_TEXT_BODY
#define ALT_TEXT_BODY 1
#endif

namespace {

constexpr int64_t kElementsPerAiv = 64;
constexpr int64_t kBytesPerAiv =
    kElementsPerAiv * static_cast<int64_t>(sizeof(half));

template <int Remaining>
struct Body {
    static __aicore__ inline void Apply(
            const AscendC::LocalTensor<half>& tile) {
        AscendC::Adds(
            tile, tile,
            static_cast<half>((Remaining % 2 == 0) ? Remaining : -Remaining),
            kElementsPerAiv);
        Body<Remaining - 1>::Apply(tile);
    }
};

template <>
struct Body<0> {
    static __aicore__ inline void Apply(const AscendC::LocalTensor<half>&) {}
};

}  // namespace

extern "C" __global__ __aicore__ void aiv_text_footprint_alt(
        __gm__ half* source, __gm__ half* destination) {
    const int64_t worker_offset = (
        static_cast<int64_t>(AscendC::GetBlockIdx()) * kElementsPerAiv);
    AscendC::GlobalTensor<half> global_source;
    AscendC::GlobalTensor<half> global_destination;
    global_source.SetGlobalBuffer(source + worker_offset);
    global_destination.SetGlobalBuffer(destination + worker_offset);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue;
    pipe.InitBuffer(input_queue, 1, kBytesPerAiv);
    pipe.InitBuffer(output_queue, 1, kBytesPerAiv);

    AscendC::LocalTensor<half> input = input_queue.AllocTensor<half>();
    AscendC::DataCopy(input, global_source, kElementsPerAiv);
    input_queue.EnQue(input);

    input = input_queue.DeQue<half>();
    AscendC::LocalTensor<half> output = output_queue.AllocTensor<half>();
    AscendC::Adds(output, input, static_cast<half>(0), kElementsPerAiv);
    input_queue.FreeTensor(input);
    Body<ALT_TEXT_BODY>::Apply(output);
    output_queue.EnQue(output);

    output = output_queue.DeQue<half>();
    AscendC::DataCopy(global_destination, output, kElementsPerAiv);
    output_queue.FreeTensor(output);
}
