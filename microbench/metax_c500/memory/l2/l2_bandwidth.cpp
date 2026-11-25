/*
 * MetaX C500 L2 read-bandwidth mechanism benchmark.
 *
 * The working-set sweep crosses the documented 8 MiB L2 capacity while the
 * access instruction, producer geometry and transaction pattern stay fixed.
 * Every thread stores its checksum.  Vector loads keep the same coalesced
 * 16-byte/thread request form used by the independently audited DRAM sweep.
 * Footprints below L2 are warmed before timing; footprints above L2 provide
 * the DRAM control.
 *
 * This benchmark measures an effective sustained read capacity.  It does not
 * identify L2 partitions, request credits, read/write duplex behavior or a
 * production-operator correction.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o l2_bandwidth l2_bandwidth.cpp
 */

#include <mc_runtime.h>

#define gpuDeviceProp          mcDeviceProp_t
#define gpuDeviceSynchronize   mcDeviceSynchronize
#define gpuError_t             mcError_t
#define gpuEventCreate         mcEventCreate
#define gpuEventDestroy        mcEventDestroy
#define gpuEventElapsedTime    mcEventElapsedTime
#define gpuEventRecord         mcEventRecord
#define gpuEvent_t             mcEvent_t
#define gpuFree                mcFree
#define gpuGetDeviceProperties mcGetDeviceProperties
#define gpuGetErrorString      mcGetErrorString
#define gpuMalloc              mcMalloc
#define gpuMemset              mcMemset
#define gpuSuccess             mcSuccess

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    gpuError_t error = call;                                               \
    if (error != gpuSuccess) {                                             \
        std::fprintf(stderr, "GPU error at %s:%d: %s\n",                \
                     __FILE__, __LINE__, gpuGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

__device__ __forceinline__ std::uint32_t mix_bits(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__device__ __forceinline__ float finite_float(std::uint32_t bits) {
    union {
        std::uint32_t bits;
        float value;
    } converted;
    converted.bits = 0x3f000000U | (bits & 0x007fffffU);
    return converted.value;
}

__global__ void initialize_incompressible_data(
        float4* data, long long elements) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    for (long long index = thread; index < elements; index += stride) {
        std::uint32_t seed = mix_bits(
            static_cast<std::uint32_t>(index)
            ^ static_cast<std::uint32_t>(index >> 32));
        data[index] = make_float4(
            finite_float(mix_bits(seed ^ 0x243f6a88U)),
            finite_float(mix_bits(seed ^ 0x85a308d3U)),
            finite_float(mix_bits(seed ^ 0x13198a2eU)),
            finite_float(mix_bits(seed ^ 0x03707344U)));
    }
}

__global__ void l2_read_bandwidth_kernel(
        const float4* data, float* sink,
        long long elements, int repetitions) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    float4 checksum = make_float4(
        static_cast<float>(thread & 31), 0.0f, 0.0f, 0.0f);
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        for (long long index = thread; index < elements; index += stride) {
            // The empty target-side memory clobber prevents loop-invariant
            // load motion across repetitions without changing the vector
            // instruction or access geometry being measured.
            __asm__ __volatile__("" ::: "memory");
            float4 value = data[index];
            checksum.x += value.x;
            checksum.y += value.y;
            checksum.z += value.z;
            checksum.w += value.w;
        }
    }
    sink[thread] = (
        checksum.x + checksum.y + checksum.z + checksum.w);
}

struct Result {
    double median_ms;
    double minimum_ms;
    double maximum_ms;
    double bandwidth_gbs;
};

static Result measure(
        const float4* data, float* sink, long long elements,
        int grid, int block, int repetitions) {
    int warmup_repetitions = std::max(1, repetitions / 8);
    for (int warmup = 0; warmup < 4; ++warmup) {
        l2_read_bandwidth_kernel<<<grid, block>>>(
            data, sink, elements, warmup_repetitions);
    }
    CHECK(gpuDeviceSynchronize());

    gpuEvent_t start;
    gpuEvent_t end;
    CHECK(gpuEventCreate(&start));
    CHECK(gpuEventCreate(&end));
    std::vector<double> milliseconds;
    constexpr int TRIALS = 7;
    milliseconds.reserve(TRIALS);
    for (int trial = 0; trial < TRIALS; ++trial) {
        CHECK(gpuEventRecord(start));
        l2_read_bandwidth_kernel<<<grid, block>>>(
            data, sink, elements, repetitions);
        CHECK(gpuEventRecord(end));
        CHECK(gpuDeviceSynchronize());
        float elapsed_ms = 0.0f;
        CHECK(gpuEventElapsedTime(&elapsed_ms, start, end));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
            std::fprintf(stderr, "invalid elapsed time: %.9g ms\n", elapsed_ms);
            std::exit(EXIT_FAILURE);
        }
        milliseconds.push_back(static_cast<double>(elapsed_ms));
    }
    CHECK(gpuEventDestroy(start));
    CHECK(gpuEventDestroy(end));
    std::sort(milliseconds.begin(), milliseconds.end());
    double median_ms = milliseconds[TRIALS / 2];
    double read_bytes = (
        static_cast<double>(elements) * sizeof(float4) * repetitions);
    return Result{
        median_ms,
        milliseconds.front(),
        milliseconds.back(),
        read_bytes / (median_ms * 1.0e6),
    };
}

int main() {
    gpuDeviceProp properties;
    CHECK(gpuGetDeviceProperties(&properties, 0));
    if (properties.multiProcessorCount <= 0 || properties.l2CacheSize <= 0) {
        std::fprintf(stderr, "device must report positive AP count and L2 size\n");
        return EXIT_FAILURE;
    }

    constexpr int BLOCK = 256;
    const int footprints_mib[] = {4, 6, 7, 12, 64, 256};
    const int grid_multipliers[] = {1, 2, 4, 8, 16};
    constexpr double TARGET_READ_BYTES = 32.0 * 1024 * 1024 * 1024;
    constexpr int MAX_FOOTPRINT_MIB = 256;
    size_t allocation_bytes = (
        static_cast<size_t>(MAX_FOOTPRINT_MIB) * 1024 * 1024);
    float4* data = nullptr;
    float* sink = nullptr;
    CHECK(gpuMalloc(&data, allocation_bytes));
    CHECK(gpuMalloc(
        &sink,
        static_cast<size_t>(properties.multiProcessorCount)
            * 16 * BLOCK * sizeof(float)));
    CHECK(gpuMemset(
        sink, 0,
        static_cast<size_t>(properties.multiProcessorCount)
            * 16 * BLOCK * sizeof(float)));
    constexpr int INITIALIZE_GRID_MULTIPLIER = 8;
    initialize_incompressible_data<<<
        properties.multiProcessorCount * INITIALIZE_GRID_MULTIPLIER, BLOCK>>>(
            data, allocation_bytes / sizeof(float4));
    CHECK(gpuDeviceSynchronize());

    std::printf("device,%s\n", properties.name);
    std::printf("ap_count,%d\n", properties.multiProcessorCount);
    std::printf("reported_l2_bytes,%zu\n",
                static_cast<size_t>(properties.l2CacheSize));
    std::printf("reported_clock_khz,%d\n", properties.clockRate);
    std::printf(
        "footprint_mib,grid,block,repetitions,median_ms,min_ms,max_ms,"
        "logical_read_gbs\n");
    for (int footprint_mib : footprints_mib) {
        size_t footprint_bytes = (
            static_cast<size_t>(footprint_mib) * 1024 * 1024);
        long long elements = static_cast<long long>(
            footprint_bytes / sizeof(float4));
        int repetitions = std::max(
            8,
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
        }
    }

    CHECK(gpuFree(sink));
    CHECK(gpuFree(data));
    return EXIT_SUCCESS;
}
