/* A800 L2 read rate against outstanding loads per thread.
 *
 * An L2-resident read stream does not reach the L2 service rate unless enough
 * requests per thread are in flight to cover the L2 latency.  This probe
 * measures that directly: achieved aggregate bytes per second against the
 * number of independent 16-byte loads each thread issues before any of them is
 * consumed, at a fixed grid and block.  It is a bandwidth/latency mechanism.
 * It contains no operator, no reduction, no normalisation, no production shape
 * and no production duration, and reports only bytes per second at a request
 * depth.
 *
 * The buffer stays far inside the 40 MiB L2 and is warmed before timing, so
 * every measured pass is served by L2.  The consuming XOR after each batch is
 * what forces the batch to complete: the compiler cannot sink it, so the batch
 * is exactly the request depth in flight.
 *
 * Forward consumes nothing from this probe, and the reason is the result it
 * produced.  It was written to test whether the plain-RMSNorm under-prediction
 * comes from its reduce pass having only one 16-byte load per thread in flight
 * before the block reduction consumes it.  At the production grid the measured
 * curve is flat -- 2467 GB/s at depth one and 2047 to 2491 GB/s across depths
 * two to sixteen for 1024-thread CTAs, 2083 to 2310 GB/s for 256-thread ones --
 * so request depth per thread does not set the rate once the launch fills the
 * machine, and the hypothesis is excluded.  Kept so the same hypothesis is not
 * retried: an L2-resident stream shaped exactly like that reduce pass reaches
 * 3.4 times the rate the production kernel achieves, which places the residual
 * outside the operand stream.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -DPROBE_DEPTH=4 -DPROBE_THREADS=256 \
 *        -o l2_outstanding_load_depth l2_outstanding_load_depth.cu
 */
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call) do {                                                \
    const cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

#ifndef PROBE_DEPTH
#define PROBE_DEPTH 4
#endif
#ifndef PROBE_THREADS
#define PROBE_THREADS 256
#endif
constexpr int kDepth = PROBE_DEPTH;
constexpr int kThreads = PROBE_THREADS;
// 12 MiB stays far inside the 40 MiB L2 at every measured grid, so the warmed
// pass is L2-resident rather than a DRAM stream.
constexpr long long kBufferBytes = 12LL * 1024 * 1024;
constexpr long long kTrafficBudgetBytes = 3LL * 1024 * 1024 * 1024;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

__device__ __forceinline__ uint4 load_cg(const uint4* address) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address));
    return value;
}

/* Each thread issues kDepth independent 16-byte loads, then consumes all of
 * them.  Consecutive batches are independent, so the in-flight request count
 * per thread is kDepth and no more. */
__global__ __launch_bounds__(kThreads)
void outstanding_depth_stream(const uint4* __restrict__ input,
                              long long vectors,
                              long long batches_per_thread,
                              std::uint32_t* __restrict__ sink) {
    const long long lane =
        static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    const long long lanes =
        static_cast<long long>(gridDim.x) * kThreads;
    unsigned int checksum = 0;
    long long cursor = lane;
    for (long long batch = 0; batch < batches_per_thread; ++batch) {
        uint4 staged[kDepth];
#pragma unroll
        for (int slot = 0; slot < kDepth; ++slot) {
            // Each slot strides by the whole lane population, so every
            // individual load instruction stays exactly as coalesced as it is
            // at depth one and the sweep varies only the request depth.
            long long index = cursor + static_cast<long long>(slot) * lanes;
            index -= (index / vectors) * vectors;
            staged[slot] = load_cg(input + index);
        }
#pragma unroll
        for (int slot = 0; slot < kDepth; ++slot) {
            checksum ^= staged[slot].x ^ staged[slot].y
                        ^ staged[slot].z ^ staged[slot].w;
        }
        cursor += lanes * kDepth;
        cursor -= (cursor / vectors) * vectors;
    }
    if (checksum == 0xffffffffu) sink[blockIdx.x] = checksum;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS GRID\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::atoi(argv[1]);
    const int trials = std::atoi(argv[2]);
    const int grid = std::atoi(argv[3]);
    if (device < 0 || trials < 3 || trials > 15 || grid <= 0) {
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::strcmp(properties.name, kExpectedDevice) != 0
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108
            || properties.l2CacheSize != 41943040) {
        std::fprintf(stderr, "unexpected A800 device contract\n");
        return EXIT_FAILURE;
    }

    const long long vectors = kBufferBytes / 16;
    uint4* input = nullptr;
    std::uint32_t* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&input, kBufferBytes));
    CHECK_CUDA(cudaMalloc(&sink, grid * sizeof(*sink)));
    CHECK_CUDA(cudaMemset(input, 0x5a, kBufferBytes));
    CHECK_CUDA(cudaMemset(sink, 0, grid * sizeof(*sink)));

    const long long lanes = static_cast<long long>(grid) * kThreads;
    const long long bytes_per_batch = lanes * kDepth * 16;
    long long batches = kTrafficBudgetBytes / bytes_per_batch;
    if (batches < 1) batches = 1;
    const long long logical_bytes = batches * bytes_per_batch;

    for (int warmup = 0; warmup < 3; ++warmup) {
        outstanding_depth_stream<<<grid, kThreads>>>(
            input, vectors, batches, sink);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::printf("grid,threads,depth,in_flight_bytes_per_thread,buffer_bytes,"
                "trial,duration_ms,logical_bytes\n");
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        outstanding_depth_stream<<<grid, kThreads>>>(
            input, vectors, batches, sink);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        std::printf("%d,%d,%d,%d,%lld,%d,%.9f,%lld\n",
                    grid, kThreads, kDepth, kDepth * 16, kBufferBytes,
                    trial, milliseconds, logical_bytes);
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
    }
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
