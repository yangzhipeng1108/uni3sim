/*
 * A800 DRAM-miss unloaded-latency microbenchmark.
 *
 * The chain touches 64 MiB of distinct 32-byte sectors, exceeding the 40 MiB
 * L2.  Nodes are 128 bytes apart to prevent adjacent-sector reuse.  A single
 * dependent ``ld.global.cg`` chain exposes unloaded memory latency and
 * ``%globaltimer`` reports nanoseconds independently of dynamic SM clocks.
 * This is a memory-mechanism benchmark, not a production operator.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o dram_latency dram_latency.cu
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                           \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, cudaGetErrorString(error));       \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

constexpr int kTransactionBytes = 32;
// Six times the physical L2 capacity keeps the steady random chain firmly in
// the DRAM-miss regime even under non-LRU replacement and cache compression.
constexpr std::size_t kTouchedBytes = 256ULL * 1024 * 1024;
constexpr int kStrideBytes = 128;
constexpr int kStrideInts = kStrideBytes / sizeof(int);
constexpr int kNodes = static_cast<int>(kTouchedBytes / kTransactionBytes);
constexpr std::size_t kAllocationBytes =
    static_cast<std::size_t>(kNodes) * kStrideBytes;

__device__ __forceinline__ int load_l2_only(const int* address) {
    int value;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(value) : "l"(address));
    return value;
}

__device__ __forceinline__ unsigned long long read_globaltimer() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void chase_dram(const int* buffer, int steps, int* sink,
                           unsigned long long* elapsed_ns) {
    int index = 0;
    for (int iteration = 0; iteration < 128; ++iteration) {
        index = load_l2_only(buffer + index);
    }
    __threadfence();
    unsigned long long start = read_globaltimer();
    #pragma unroll 1
    for (int iteration = 0; iteration < steps; ++iteration) {
        index = load_l2_only(buffer + index);
    }
    unsigned long long end = read_globaltimer();
    *sink = index;
    *elapsed_ns = end - start;
}

int main() {
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (properties.l2CacheSize >= static_cast<int>(kTouchedBytes)) {
        std::fprintf(stderr,
                     "touched bytes %zu must exceed L2 capacity %d\n",
                     kTouchedBytes, properties.l2CacheSize);
        return EXIT_FAILURE;
    }

    std::vector<int> permutation(kNodes);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937 generator(42);
    std::shuffle(permutation.begin(), permutation.end(), generator);
    // Fill every transferred sector with incompressible data.  A zero-filled
    // sparse pointer structure can fit in L2 through Ampere Compute Data
    // Compression even when its logical sector footprint exceeds 40 MiB.
    std::vector<int> host_buffer(kAllocationBytes / sizeof(int));
    std::mt19937 data_generator(7);
    for (int& value : host_buffer) {
        value = static_cast<int>(data_generator());
    }
    for (int position = 0; position < kNodes; ++position) {
        int current = permutation[position];
        int next = permutation[(position + 1) % kNodes];
        host_buffer[static_cast<std::size_t>(current) * kStrideInts] =
            next * kStrideInts;
    }

    int* device_buffer = nullptr;
    int* device_sink = nullptr;
    unsigned long long* device_elapsed_ns = nullptr;
    CUDA_CHECK(cudaMalloc(&device_buffer, kAllocationBytes));
    CUDA_CHECK(cudaMalloc(&device_sink, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_elapsed_ns, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(device_buffer, host_buffer.data(), kAllocationBytes,
                          cudaMemcpyHostToDevice));

    std::printf("GPU=%s L2=%d bytes transaction=%d bytes touched=%zu bytes "
                "allocation=%zu bytes\n",
                properties.name, properties.l2CacheSize,
                kTransactionBytes, kTouchedBytes, kAllocationBytes);
    for (int trial = 0; trial < 3; ++trial) {
        chase_dram<<<1, 1>>>(device_buffer, kNodes, device_sink,
                             device_elapsed_ns);
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long elapsed_ns = 0;
        CUDA_CHECK(cudaMemcpy(&elapsed_ns, device_elapsed_ns,
                              sizeof(elapsed_ns), cudaMemcpyDeviceToHost));
        if (elapsed_ns == 0) {
            std::fprintf(stderr, "globaltimer returned zero elapsed time\n");
            return EXIT_FAILURE;
        }
        std::printf("trial=%d steps=%d elapsed_ns=%llu latency_ns=%.3f\n",
                    trial, kNodes, elapsed_ns,
                    static_cast<double>(elapsed_ns) / kNodes);
    }

    CUDA_CHECK(cudaFree(device_buffer));
    CUDA_CHECK(cudaFree(device_sink));
    CUDA_CHECK(cudaFree(device_elapsed_ns));
    return EXIT_SUCCESS;
}
