/* H200 profiler-reported duration floor of a very small kernel.
 *
 * Owner copy of microbench/nvidia_a800/compute/small_kernel_span_floor.cu with
 * the same bodies; PROBE_SYNC (1 = isolated, 0 = back-to-back) implements the
 * companion launch pattern the A800 reference describes.  The A800 results
 * quoted below are that card's and are kept for provenance only.
 *
 * The evaluation's timing target is the nsys kernel duration, so the floor that
 * matters for a kernel with almost no work is the span nsys reports, not a host
 * launch latency and not a CUDA-event interval (which also contains host submit
 * time and, on this device, quantises to 1024 ns).  This driver therefore does
 * nothing but launch, and the duration is read from an nsys capture.
 *
 * PROBE_BODY selects the body: 0 is a truly empty CTA, 1 writes one 4-byte word
 * per CTA.  The pair separates the empty-task floor from the floor of a kernel
 * that touches memory at all.  PROBE_SYNC in the companion sweep selects
 * isolated launches or back-to-back ones, which distinguishes a floor that only
 * an isolated launch pays from one a pipelined stream pays too.
 *
 * It is a launch and synchronisation mechanism: no operand stream, no operator,
 * no matrix instruction, no production shape and no production duration.
 *
 * Measured with nsys 2025.6.1 on an idle A800, 30 launches, median ns:
 *
 *   grid          48     108     216
 *   empty       1760    1888    1984
 *   one store   2240    2336    2464
 *
 * and back-to-back launches of the one-store body read 2112 at grid 48 and 2208
 * at grid 108, so the floor is within six percent of the isolated one and is not
 * an artefact of synchronising between launches.
 *
 * Build:
 *   nvcc -O3 -arch=sm_90 -DPROBE_THREADS=512 -DPROBE_BODY=1 \
 *        -o small_kernel_span_floor small_kernel_span_floor.cu
 *   nsys profile -t cuda -o span ./small_kernel_span_floor 0 48 30
 */
#include <cuda_runtime.h>
#include <cstdlib>
#ifndef PROBE_THREADS
#define PROBE_THREADS 512
#endif
#ifndef PROBE_SYNC
#define PROBE_SYNC 1
#endif
#ifndef PROBE_BODY
#define PROBE_BODY 0
#endif
constexpr int kThreads = PROBE_THREADS;
__global__ __launch_bounds__(kThreads) void truly_empty_cta() {}
__global__ __launch_bounds__(kThreads)
void one_store_cta(unsigned int* __restrict__ sink) {
    if (threadIdx.x == 0) sink[blockIdx.x] = blockIdx.x + 1u;
}
int main(int argc, char** argv) {
    const int device = std::atoi(argv[1]);
    const int grid = std::atoi(argv[2]);
    const int launches = std::atoi(argv[3]);
    cudaSetDevice(device);
    unsigned int* sink = nullptr;
    cudaMalloc(&sink, grid * sizeof(*sink));
    for (int i = 0; i < 5; ++i) {
#if PROBE_BODY
        one_store_cta<<<grid, kThreads>>>(sink);
#else
        truly_empty_cta<<<grid, kThreads>>>();
#endif
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < launches; ++i) {
#if PROBE_BODY
        one_store_cta<<<grid, kThreads>>>(sink);
#else
        truly_empty_cta<<<grid, kThreads>>>();
#endif
#if PROBE_SYNC
        cudaDeviceSynchronize();
#endif
    }
    cudaDeviceSynchronize();
    cudaFree(sink);
    return 0;
}
