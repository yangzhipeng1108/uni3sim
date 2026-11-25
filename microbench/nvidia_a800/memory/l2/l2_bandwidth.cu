/*
 * NVIDIA A800 L2 sustained read-bandwidth mechanism benchmark.
 *
 * A hand-written ld.global.cg.v4.u32 keeps the access policy fixed: loads
 * bypass L1 and allocate in L2.  The footprint sweep remains below the 40 MiB
 * GA100 L2 capacity while exceeding per-SM L1; each timed pass reads every
 * byte exactly once.  Warmup passes make the timed working set L2-resident.
 *
 * This measures an effective sustained L2 read envelope for one coalesced
 * sector stream.  It does not identify partitions, request credits, write
 * bandwidth, duplex behavior, or a production-operator correction.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o l2_bandwidth l2_bandwidth.cu
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) do {                                              \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, cudaGetErrorString(error));        \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

__device__ __forceinline__ uint4 load_l2_only(const uint4* address) {
    uint4 value;
    asm volatile(
        "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(address));
    return value;
}

__device__ __forceinline__ unsigned int fold(uint4 value) {
    return value.x ^ value.y ^ value.z ^ value.w;
}

__global__ void l2_read_bandwidth_kernel(
        const uint4* data, unsigned int* sink,
        long long elements, int repetitions) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    unsigned int checksum0 = static_cast<unsigned int>(thread);
    unsigned int checksum1 = checksum0 + 1;
    unsigned int checksum2 = checksum0 + 2;
    unsigned int checksum3 = checksum0 + 3;
    unsigned int checksum4 = checksum0 + 4;
    unsigned int checksum5 = checksum0 + 5;
    unsigned int checksum6 = checksum0 + 6;
    unsigned int checksum7 = checksum0 + 7;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        for (long long index = thread;
             index < elements;
            index += 8 * stride) {
            if (index < elements) {
                checksum0 ^= fold(load_l2_only(data + index));
            }
            if (index + stride < elements) {
                checksum1 ^= fold(load_l2_only(data + index + stride));
            }
            if (index + 2 * stride < elements) {
                checksum2 ^= fold(load_l2_only(data + index + 2 * stride));
            }
            if (index + 3 * stride < elements) {
                checksum3 ^= fold(load_l2_only(data + index + 3 * stride));
            }
            if (index + 4 * stride < elements) {
                checksum4 ^= fold(load_l2_only(data + index + 4 * stride));
            }
            if (index + 5 * stride < elements) {
                checksum5 ^= fold(load_l2_only(data + index + 5 * stride));
            }
            if (index + 6 * stride < elements) {
                checksum6 ^= fold(load_l2_only(data + index + 6 * stride));
            }
            if (index + 7 * stride < elements) {
                checksum7 ^= fold(load_l2_only(data + index + 7 * stride));
            }
        }
    }
    sink[thread] = (
        checksum0 ^ checksum1 ^ checksum2 ^ checksum3
        ^ checksum4 ^ checksum5 ^ checksum6 ^ checksum7);
}

struct Result {
    double median_ms;
    double minimum_ms;
    double maximum_ms;
    double bandwidth_gbs;
};

static Result measure(
        const uint4* data, unsigned int* sink, long long elements,
        int grid, int block, int repetitions) {
    int warmup_repetitions = std::max(2, repetitions / 16);
    for (int warmup = 0; warmup < 4; ++warmup) {
        l2_read_bandwidth_kernel<<<grid, block>>>(
            data, sink, elements, warmup_repetitions);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));
    constexpr int TRIALS = 9;
    std::vector<double> milliseconds;
    milliseconds.reserve(TRIALS);
    for (int trial = 0; trial < TRIALS; ++trial) {
        CHECK_CUDA(cudaEventRecord(start));
        l2_read_bandwidth_kernel<<<grid, block>>>(
            data, sink, elements, repetitions);
        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaDeviceSynchronize());
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, end));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
            std::fprintf(stderr, "invalid elapsed time: %.9g ms\n", elapsed_ms);
            std::exit(EXIT_FAILURE);
        }
        milliseconds.push_back(static_cast<double>(elapsed_ms));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));
    std::sort(milliseconds.begin(), milliseconds.end());
    double median_ms = milliseconds[TRIALS / 2];
    double read_bytes = (
        static_cast<double>(elements) * sizeof(uint4) * repetitions);
    return Result{
        median_ms,
        milliseconds.front(),
        milliseconds.back(),
        read_bytes / (median_ms * 1.0e6),
    };
}

int main() {
    cudaDeviceProp properties;
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    if (properties.multiProcessorCount <= 0 || properties.l2CacheSize <= 0) {
        std::fprintf(stderr, "device must report positive SM count and L2 size\n");
        return EXIT_FAILURE;
    }

    constexpr int BLOCK = 256;
    const int footprints_mib[] = {8, 16, 24, 32};
    const int grid_multipliers[] = {1, 2, 4, 8};
    constexpr double TARGET_READ_BYTES = 32.0 * 1024 * 1024 * 1024;
    constexpr int MAX_FOOTPRINT_MIB = 32;
    size_t allocation_bytes = (
        static_cast<size_t>(MAX_FOOTPRINT_MIB) * 1024 * 1024);
    if (allocation_bytes >= static_cast<size_t>(properties.l2CacheSize)) {
        std::fprintf(
            stderr, "maximum footprint %zu must remain below L2 size %d\n",
            allocation_bytes, properties.l2CacheSize);
        return EXIT_FAILURE;
    }

    uint4* data = nullptr;
    unsigned int* sink = nullptr;
    int maximum_grid = (
        properties.multiProcessorCount
        * grid_multipliers[sizeof(grid_multipliers) / sizeof(int) - 1]);
    CHECK_CUDA(cudaMalloc(&data, allocation_bytes));
    CHECK_CUDA(cudaMalloc(
        &sink, static_cast<size_t>(maximum_grid) * BLOCK * sizeof(*sink)));
    CHECK_CUDA(cudaMemset(data, 0x3f, allocation_bytes));
    CHECK_CUDA(cudaMemset(
        sink, 0,
        static_cast<size_t>(maximum_grid) * BLOCK * sizeof(*sink)));

    std::printf("device,%s\n", properties.name);
    std::printf("sm_count,%d\n", properties.multiProcessorCount);
    std::printf("reported_l2_bytes,%d\n", properties.l2CacheSize);
    std::printf("reported_clock_khz,%d\n", properties.clockRate);
    std::printf(
        "footprint_mib,grid,block,repetitions,median_ms,min_ms,max_ms,"
        "logical_read_gbs\n");
    for (int footprint_mib : footprints_mib) {
        size_t footprint_bytes = (
            static_cast<size_t>(footprint_mib) * 1024 * 1024);
        long long elements = static_cast<long long>(
            footprint_bytes / sizeof(uint4));
        int repetitions = std::max(
            16,
            static_cast<int>(std::ceil(
                TARGET_READ_BYTES / static_cast<double>(footprint_bytes))));
        for (int multiplier : grid_multipliers) {
            int grid = properties.multiProcessorCount * multiplier;
            Result result = measure(
                data, sink, elements, grid, BLOCK, repetitions);
            std::printf(
                "%d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f\n",
                footprint_mib, grid, BLOCK, repetitions,
                result.median_ms, result.minimum_ms, result.maximum_ms,
                result.bandwidth_gbs);
            std::fflush(stdout);
        }
    }

    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(data));
    return EXIT_SUCCESS;
}
