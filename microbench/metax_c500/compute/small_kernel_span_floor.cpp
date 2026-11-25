/* MetaX C500 profiler-reported duration floor of a very small launch.
 *
 * The C500 counterpart of microbench/nvidia_a800/compute/small_kernel_span_floor.cu.
 * The evaluation's timing target is the mcTracer kernel duration, so the floor
 * that matters for a launch with almost no work is the span mcTracer reports,
 * not a host launch latency and not a MACA-event interval (which on this
 * deployment also contains host submit time).  This driver therefore does
 * nothing but launch, and the duration is read from an mcTracer capture.
 *
 * PROBE_BODY selects the body: 0 is a truly empty CTA, 1 writes one 4-byte
 * word per CTA.  The pair separates the empty-task floor from the floor of a
 * launch that touches memory at all.  The two launch patterns matter more here
 * than on A800: the deployed decode step replays a graph of roughly 1,100
 * back-to-back launches per rank, so the span a launch occupies inside a dense
 * stream is the one the Serving timeline needs, and it need not equal the span
 * of an isolated launch.  Both are emitted.
 *
 * Unlike the A800 probe this one also sweeps the block size, because the C500
 * Serving glue launches span 16 to 1,024 threads per CTA and the A800 result
 * that block size does not move the floor cannot be assumed to carry over.
 *
 * It is a launch and synchronisation mechanism: no operand stream, no operator,
 * no matrix instruction, no production shape and no production duration.
 *
 * Compile (MetaX C500 / MACA):
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o small_kernel_span_floor \
 *       small_kernel_span_floor.cpp
 *
 * Run under mcTracer:
 *   mcTracer --odname <case> ./small_kernel_span_floor <device> <grid> \
 *       <threads> <body> <pattern> <launches>
 *   body: 0 empty, 1 one store per CTA
 *   pattern: 0 isolated (synchronise after each), 1 back-to-back
 */
#include <mc_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void truly_empty_cta() {}

__global__ void one_store_cta(unsigned int* __restrict__ sink) {
    if (threadIdx.x == 0) sink[blockIdx.x] = blockIdx.x + 1u;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::printf("usage: %s device grid threads body pattern launches\n",
                    argv[0]);
        return 2;
    }
    const int device = std::atoi(argv[1]);
    const int grid = std::atoi(argv[2]);
    const int threads = std::atoi(argv[3]);
    const int body = std::atoi(argv[4]);
    const int pattern = std::atoi(argv[5]);
    const int launches = std::atoi(argv[6]);
    if (grid <= 0 || threads <= 0 || launches <= 0) return 2;
    if (body != 0 && body != 1) return 2;
    if (pattern != 0 && pattern != 1) return 2;

    if (mcSetDevice(device) != mcSuccess) return 3;
    unsigned int* sink = nullptr;
    if (mcMalloc(&sink, static_cast<size_t>(grid) * sizeof(*sink)) != mcSuccess)
        return 3;

    for (int i = 0; i < 20; ++i) {
        if (body) one_store_cta<<<grid, threads>>>(sink);
        else truly_empty_cta<<<grid, threads>>>();
    }
    if (mcDeviceSynchronize() != mcSuccess) return 4;

    for (int i = 0; i < launches; ++i) {
        if (body) one_store_cta<<<grid, threads>>>(sink);
        else truly_empty_cta<<<grid, threads>>>();
        if (pattern == 0) {
            if (mcDeviceSynchronize() != mcSuccess) return 4;
        }
    }
    if (mcDeviceSynchronize() != mcSuccess) return 4;
    mcFree(sink);
    std::printf("device=%d grid=%d threads=%d body=%d pattern=%d launches=%d\n",
                device, grid, threads, body, pattern, launches);
    return 0;
}
