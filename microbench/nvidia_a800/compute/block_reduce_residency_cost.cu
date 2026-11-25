/* A800 block-wide reduction cost against resident CTAs per multiprocessor.
 *
 * A dependent chain of block-wide reductions is a synchronisation mechanism:
 * each round's input depends on the previous round's broadcast result, so the
 * round cost is one full 1024-thread reduction tree plus its two barriers and
 * cannot be overlapped inside the CTA.  This probe measures that round cost
 * directly, at a controlled number of resident CTAs per multiprocessor, and
 * reports only nanoseconds per round.
 *
 * It touches no global memory beyond a single sink write per CTA, contains no
 * operator, no normalisation, no operand stream and no production shape, and
 * consumes no production duration.  Residency is set by a dynamic
 * shared-memory block the reduction never reads, and the probe verifies the
 * residency it asked for rather than assuming it.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -DPROBE_RESIDENT=2 \
 *        -o block_reduce_residency_cost block_reduce_residency_cost.cu
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

#ifndef PROBE_RESIDENT
#define PROBE_RESIDENT 2
#endif
constexpr int kThreads = 1024;
constexpr int kResident = PROBE_RESIDENT;
constexpr int kWarps = kThreads / 32;
// GA100 offers 164 KiB of shared memory per SM and 2048 threads, so a
// 1024-thread CTA can reach two residents.  Requesting more than half the
// shared memory admits exactly one.
constexpr int kResidencyShmemBytes = (kResident == 1) ? (100 * 1024) : 0;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

extern __shared__ std::uint8_t residency_block[];

/* One round: warp-shuffle reduce, publish per warp, barrier, reduce the 32
 * warp results, broadcast, barrier.  The next round's input depends on the
 * broadcast, so rounds are strictly serial inside the CTA. */
__global__ __launch_bounds__(kThreads, kResident)
void dependent_block_reduce(int rounds, float* __restrict__ sink) {
    __shared__ float warp_partials[kWarps];
    __shared__ float broadcast;
    (void)residency_block;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float value = static_cast<float>(threadIdx.x) * 1.0e-6f + 1.0f;
    for (int round = 0; round < rounds; ++round) {
        float partial = value;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
        }
        if (lane == 0) warp_partials[warp] = partial;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
#pragma unroll
            for (int index = 0; index < kWarps; ++index) {
                total += warp_partials[index];
            }
            broadcast = total;
        }
        __syncthreads();
        // The next round's input depends on this round's broadcast, so the
        // compiler cannot hoist or merge rounds.
        value = value * 1.0000001f + broadcast * 1.0e-9f;
        __syncthreads();
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = value;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS GRID ROUNDS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::atoi(argv[1]);
    const int trials = std::atoi(argv[2]);
    const int grid = std::atoi(argv[3]);
    const int rounds = std::atoi(argv[4]);
    if (device < 0 || trials < 3 || trials > 15 || grid <= 0 || rounds < 1) {
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::strcmp(properties.name, kExpectedDevice) != 0
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108) {
        std::fprintf(stderr, "unexpected A800 device contract\n");
        return EXIT_FAILURE;
    }

    if (kResidencyShmemBytes > 0) {
        CHECK_CUDA(cudaFuncSetAttribute(
            dependent_block_reduce,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kResidencyShmemBytes));
    }
    int resident = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, dependent_block_reduce, kThreads, kResidencyShmemBytes));
    if (resident != kResident) {
        std::fprintf(stderr,
                     "expected %d resident CTAs per SM, got %d\n",
                     kResident, resident);
        return EXIT_FAILURE;
    }

    float* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&sink, grid * sizeof(*sink)));
    CHECK_CUDA(cudaMemset(sink, 0, grid * sizeof(*sink)));
    for (int warmup = 0; warmup < 3; ++warmup) {
        dependent_block_reduce<<<grid, kThreads, kResidencyShmemBytes>>>(
            rounds, sink);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::printf("grid,threads,resident_ctas_per_sm,rounds,waves,trial,"
                "duration_ms,total_rounds\n");
    const int waves = (grid + 108 * kResident - 1) / (108 * kResident);
    const long long total_rounds = static_cast<long long>(grid) * rounds;
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        dependent_block_reduce<<<grid, kThreads, kResidencyShmemBytes>>>(
            rounds, sink);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        std::printf("%d,%d,%d,%d,%d,%d,%.9f,%lld\n",
                    grid, kThreads, kResident, rounds, waves, trial,
                    milliseconds, total_rounds);
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
    }
    CHECK_CUDA(cudaFree(sink));
    return EXIT_SUCCESS;
}
