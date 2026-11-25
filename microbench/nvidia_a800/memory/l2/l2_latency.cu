/*
 * A800 L2-hit unloaded-latency microbenchmark.
 *
 * The chain contains 16 MiB of distinct 32-byte L2 sectors: larger than the
 * per-SM L1 and smaller than the 40 MiB device L2.  ``ld.global.cg`` bypasses
 * L1, a complete untimed walk primes L2, and the timed walk reads
 * ``%globaltimer`` directly so dynamic SM clocks cannot corrupt nanoseconds.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o l2_latency l2_latency.cu
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
constexpr std::size_t kTouchedBytes = 16ULL * 1024 * 1024;
constexpr int kStrideBytes = 32;
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

__global__ void warm_l2(const int* buffer, int steps, int* sink) {
    int index = 0;
    for (int iteration = 0; iteration < steps; ++iteration) {
        index = load_l2_only(buffer + index);
    }
    *sink = index;
}

__global__ void chase_l2(const int* buffer, int steps, int* sink,
                         unsigned long long* elapsed_ns) {
    int index = 0;
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
    if (properties.l2CacheSize <= static_cast<int>(kTouchedBytes)) {
        std::fprintf(stderr,
                     "L2 capacity %d must exceed touched bytes %zu\n",
                     properties.l2CacheSize, kTouchedBytes);
        return EXIT_FAILURE;
    }

    std::vector<int> permutation(kNodes);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937 generator(42);
    std::shuffle(permutation.begin(), permutation.end(), generator);
    std::vector<int> host_buffer(kAllocationBytes / sizeof(int), 0);
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

    std::printf("GPU=%s L2=%d bytes transaction=%d bytes touched=%zu bytes\n",
                properties.name, properties.l2CacheSize,
                kTransactionBytes, kTouchedBytes);
    for (int trial = 0; trial < 5; ++trial) {
        warm_l2<<<1, 1>>>(device_buffer, kNodes, device_sink);
        CUDA_CHECK(cudaDeviceSynchronize());
        chase_l2<<<1, 1>>>(device_buffer, kNodes, device_sink,
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
