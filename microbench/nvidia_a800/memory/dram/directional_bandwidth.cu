/*
 * NVIDIA A800 directional DRAM streaming-capacity microbenchmark.
 *
 * Static kernels issue fixed read/write mixtures over
 * disjoint 2-GiB source and destination arrays.  Every logical operation is a
 * 16-byte, L1-bypassing global load or write-back global store.  The working
 * set in each non-zero direction exceeds the 40-MiB L2, and four complete
 * passes make native event timing long enough for a producer/grid sweep.
 *
 * Device-event duration measures only this independent streaming mechanism.
 * NCU may be used separately for non-duration traffic and instruction
 * evidence; profiler duration is never a parameter source.  Output capacities
 * exclude the small per-thread read-checksum sink, which is reported as a
 * known validation overhead and must be checked with counters before
 * promotion.
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo \
 *       -o directional_bandwidth directional_bandwidth.cu
 *
 * Run the complete native sweep or one mode/grid for counter capture:
 *   CUDA_VISIBLE_DEVICES=3 ./directional_bandwidth
 *   CUDA_VISIBLE_DEVICES=3 ./directional_bandwidth \
 *       --single read50_write50 1728
 */

#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                     __FILE__, __LINE__, cudaGetErrorString(error));        \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

#ifndef DIRECTIONAL_BUFFER_BYTES
#define DIRECTIONAL_BUFFER_BYTES (2ULL * 1024 * 1024 * 1024 - 2048ULL)
#endif
#ifndef DIRECTIONAL_ITERATIONS
#define DIRECTIONAL_ITERATIONS 4
#endif
#ifndef DIRECTIONAL_TRIALS
#define DIRECTIONAL_TRIALS 7
#endif
#ifndef DIRECTIONAL_ENABLE_READ60_WRITE40
#define DIRECTIONAL_ENABLE_READ60_WRITE40 0
#endif

constexpr std::size_t kBufferBytes = DIRECTIONAL_BUFFER_BYTES;
constexpr int kVectorBytes = 16;
constexpr int kOpsPerChunk = 4;
constexpr int kThreeTwoOpsPerChunk = 5;
constexpr int kBlockSize = 256;
constexpr int kIterations = DIRECTIONAL_ITERATIONS;
constexpr int kTrials = DIRECTIONAL_TRIALS;
constexpr int kExpectedSms = 108;
constexpr int kExpectedWarpSize = 32;
constexpr int kExpectedL2Bytes = 40 * 1024 * 1024;
constexpr std::uint32_t kSentinelWord = 0xa5a5a5a5U;
constexpr std::uint32_t kSourceSalt = 0x31415926U;
constexpr std::uint32_t kWriteSalt = 0x9e3779b9U;
static_assert(sizeof(uint4) == kVectorBytes, "uint4 must be 16 bytes");
static_assert(
    kBufferBytes
        % (kExpectedWarpSize * kThreeTwoOpsPerChunk * kVectorBytes) == 0,
    "buffer must tile the five-slot directional warp group");

enum class Mode {
    kRead,
    kRead75Write25,
    kRead60Write40,
    kRead50Write50,
    kRead25Write75,
    kWrite,
};

struct ModeSpec {
    Mode mode;
    const char* name;
    int read_ops;
    int write_ops;
};

constexpr ModeSpec kModes[] = {
    {Mode::kRead, "read", 4, 0},
    {Mode::kRead75Write25, "read75_write25", 3, 1},
#if DIRECTIONAL_ENABLE_READ60_WRITE40
    {Mode::kRead60Write40, "read60_write40", 3, 2},
#endif
    {Mode::kRead50Write50, "read50_write50", 2, 2},
    {Mode::kRead25Write75, "read25_write75", 1, 3},
    {Mode::kWrite, "write", 0, 4},
};

static int parse_positive_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value <= 0 || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a positive integer, got %s\n",
                     name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

static const ModeSpec& parse_mode(const char* text) {
    for (const ModeSpec& spec : kModes) {
        if (std::strcmp(text, spec.name) == 0) {
            return spec;
        }
    }
    std::fprintf(stderr, "unsupported mode %s\n", text);
    std::exit(EXIT_FAILURE);
}

__host__ __device__ __forceinline__ std::uint32_t mix_bits(
        std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__host__ __device__ __forceinline__ uint4 make_value(
        std::uint32_t index, std::uint32_t salt) {
    std::uint32_t x = mix_bits(index ^ salt);
    std::uint32_t y = mix_bits(x ^ 0x68bc21ebU);
    std::uint32_t z = mix_bits(y ^ 0x02e5be93U);
    std::uint32_t w = mix_bits(z ^ 0x967a889bU);
    return make_uint4(x, y, z, w);
}

__global__ void initialize_source(uint4* source, std::size_t element_count) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t stride = (
        static_cast<std::size_t>(blockDim.x) * gridDim.x);
    for (std::size_t index = thread; index < element_count; index += stride) {
        source[index] = make_value(
            static_cast<std::uint32_t>(index), kSourceSalt);
    }
}

__global__ void validate_source(
        const uint4* source, std::size_t element_count,
        unsigned long long* errors) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t stride = (
        static_cast<std::size_t>(blockDim.x) * gridDim.x);
    unsigned long long local_errors = 0;
    for (std::size_t index = thread; index < element_count; index += stride) {
        uint4 expected = make_value(
            static_cast<std::uint32_t>(index), kSourceSalt);
        uint4 observed = source[index];
        local_errors += (
            observed.x != expected.x || observed.y != expected.y
            || observed.z != expected.z || observed.w != expected.w);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

__device__ __forceinline__ uint4 load_global_cg(const uint4* address) {
    uint4 value;
    asm volatile(
        "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(address)
        : "memory");
    return value;
}

__device__ __forceinline__ void store_global_wb(
        uint4* address, const uint4& value) {
    asm volatile(
        "st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
        :: "l"(address), "r"(value.x), "r"(value.y),
           "r"(value.z), "r"(value.w)
        : "memory");
}

template <bool IsRead>
__device__ __forceinline__ void execute_directional_slot(
        const uint4* source, uint4* destination,
        std::size_t index, int iteration,
        std::uint32_t* checksum) {
    if constexpr (IsRead) {
        uint4 value = load_global_cg(source + index);
        *checksum ^= value.x ^ value.y ^ value.z ^ value.w;
    } else {
        std::uint32_t salt = (
            kWriteSalt
            ^ static_cast<std::uint32_t>(iteration) * 0x85ebca6bU);
        uint4 value = make_value(static_cast<std::uint32_t>(index), salt);
        store_global_wb(destination + index, value);
        *checksum ^= value.x;
    }
}

template <int SlotCount, bool Read0, bool Read1, bool Read2, bool Read3,
          bool Read4 = false>
__device__ __forceinline__ void directional_body(
        const uint4* source, uint4* destination,
        std::size_t warp_group_count, int iterations,
        std::uint32_t* read_sink) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t lane = threadIdx.x % kExpectedWarpSize;
    std::size_t warp = thread / kExpectedWarpSize;
    std::size_t warp_stride = (
        static_cast<std::size_t>(blockDim.x) * gridDim.x
        / kExpectedWarpSize);
    std::uint32_t checksum = mix_bits(
        static_cast<std::uint32_t>(thread) ^ 0xd1b54a35U);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t group = warp;
             group < warp_group_count; group += warp_stride) {
            std::size_t base = (
                group * kExpectedWarpSize * SlotCount + lane);
            execute_directional_slot<Read0>(
                source, destination, base, iteration, &checksum);
            execute_directional_slot<Read1>(
                source, destination,
                base + kExpectedWarpSize, iteration, &checksum);
            execute_directional_slot<Read2>(
                source, destination,
                base + 2 * kExpectedWarpSize, iteration, &checksum);
            execute_directional_slot<Read3>(
                source, destination,
                base + 3 * kExpectedWarpSize, iteration, &checksum);
            if constexpr (SlotCount == kThreeTwoOpsPerChunk) {
                execute_directional_slot<Read4>(
                    source, destination,
                    base + 4 * kExpectedWarpSize, iteration, &checksum);
            }
        }
    }
    if constexpr (Read0 || Read1 || Read2 || Read3 || Read4) {
        read_sink[thread] = checksum;
    }
}

extern "C" __global__ void a800_dram_read(
        const uint4* source, uint4* destination,
        std::size_t warp_group_count, int iterations,
        std::uint32_t* read_sink) {
    directional_body<kOpsPerChunk, true, true, true, true>(
        source, destination, warp_group_count, iterations, read_sink);
}

extern "C" __global__ void a800_dram_read75_write25(
        const uint4* source, uint4* destination,
        std::size_t warp_group_count, int iterations,
        std::uint32_t* read_sink) {
    directional_body<kOpsPerChunk, true, false, true, true>(
        source, destination, warp_group_count, iterations, read_sink);
}

extern "C" __global__ void a800_l2_read60_write40(
        const uint4* source, uint4* destination,
        std::size_t warp_group_count, int iterations,
        std::uint32_t* read_sink) {
    directional_body<kThreeTwoOpsPerChunk, true, false, true, false, true>(
        source, destination, warp_group_count, iterations, read_sink);
}

extern "C" __global__ void a800_dram_read50_write50(
        const uint4* source, uint4* destination,
        std::size_t warp_group_count, int iterations,
        std::uint32_t* read_sink) {
    directional_body<kOpsPerChunk, true, false, true, false>(
        source, destination, warp_group_count, iterations, read_sink);
}

extern "C" __global__ void a800_dram_read25_write75(
        const uint4* source, uint4* destination,
        std::size_t warp_group_count, int iterations,
        std::uint32_t* read_sink) {
    directional_body<kOpsPerChunk, true, false, false, false>(
        source, destination, warp_group_count, iterations, read_sink);
}

extern "C" __global__ void a800_dram_write(
        const uint4* source, uint4* destination,
        std::size_t warp_group_count, int iterations,
        std::uint32_t* read_sink) {
    directional_body<kOpsPerChunk, false, false, false, false>(
        source, destination, warp_group_count, iterations, read_sink);
}

template <bool IsRead>
__device__ __forceinline__ void validate_destination_slot(
        const uint4* destination, std::size_t index,
        std::uint32_t salt, unsigned long long* local_errors) {
    uint4 expected = make_uint4(
        kSentinelWord, kSentinelWord, kSentinelWord, kSentinelWord);
    if constexpr (!IsRead) {
        expected = make_value(static_cast<std::uint32_t>(index), salt);
    }
    uint4 observed = destination[index];
    *local_errors += (
        observed.x != expected.x || observed.y != expected.y
        || observed.z != expected.z || observed.w != expected.w);
}

template <int SlotCount, bool Read0, bool Read1, bool Read2, bool Read3,
          bool Read4 = false>
__global__ void validate_destination(
        const uint4* destination, std::size_t warp_group_count,
        int final_iteration, unsigned long long* errors) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t lane = threadIdx.x % kExpectedWarpSize;
    std::size_t warp = thread / kExpectedWarpSize;
    std::size_t warp_stride = (
        static_cast<std::size_t>(blockDim.x) * gridDim.x
        / kExpectedWarpSize);
    unsigned long long local_errors = 0;
    std::uint32_t salt = (
        kWriteSalt
        ^ static_cast<std::uint32_t>(final_iteration) * 0x85ebca6bU);
    for (std::size_t group = warp;
         group < warp_group_count; group += warp_stride) {
        std::size_t base = (
            group * kExpectedWarpSize * SlotCount + lane);
        validate_destination_slot<Read0>(
            destination, base, salt, &local_errors);
        validate_destination_slot<Read1>(
            destination, base + kExpectedWarpSize,
            salt, &local_errors);
        validate_destination_slot<Read2>(
            destination, base + 2 * kExpectedWarpSize,
            salt, &local_errors);
        validate_destination_slot<Read3>(
            destination, base + 3 * kExpectedWarpSize,
            salt, &local_errors);
        if constexpr (SlotCount == kThreeTwoOpsPerChunk) {
            validate_destination_slot<Read4>(
                destination, base + 4 * kExpectedWarpSize,
                salt, &local_errors);
        }
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

static void launch_mode(
        Mode mode, int grid, const uint4* source, uint4* destination,
        std::size_t warp_group_count, int iterations,
        std::uint32_t* read_sink) {
    if (mode == Mode::kRead) {
        a800_dram_read<<<grid, kBlockSize>>>(
            source, destination, warp_group_count, iterations, read_sink);
    } else if (mode == Mode::kRead75Write25) {
        a800_dram_read75_write25<<<grid, kBlockSize>>>(
            source, destination, warp_group_count, iterations, read_sink);
    } else if (mode == Mode::kRead60Write40) {
        a800_l2_read60_write40<<<grid, kBlockSize>>>(
            source, destination, warp_group_count, iterations, read_sink);
    } else if (mode == Mode::kRead50Write50) {
        a800_dram_read50_write50<<<grid, kBlockSize>>>(
            source, destination, warp_group_count, iterations, read_sink);
    } else if (mode == Mode::kRead25Write75) {
        a800_dram_read25_write75<<<grid, kBlockSize>>>(
            source, destination, warp_group_count, iterations, read_sink);
    } else if (mode == Mode::kWrite) {
        a800_dram_write<<<grid, kBlockSize>>>(
            source, destination, warp_group_count, iterations, read_sink);
    } else {
        std::fprintf(stderr, "unknown directional mode\n");
        std::exit(EXIT_FAILURE);
    }
}

template <int SlotCount, bool Read0, bool Read1, bool Read2, bool Read3,
          bool Read4 = false>
static void launch_validation(
        const uint4* destination, std::size_t warp_group_count,
        int final_iteration, unsigned long long* errors) {
    validate_destination<SlotCount, Read0, Read1, Read2, Read3, Read4>
        <<<kExpectedSms * 4, kBlockSize>>>(
            destination, warp_group_count, final_iteration, errors);
}

static void validate_mode(
        Mode mode, const uint4* destination, std::size_t warp_group_count,
        int final_iteration, unsigned long long* errors) {
    CUDA_CHECK(cudaMemset(errors, 0, sizeof(unsigned long long)));
    if (mode == Mode::kRead) {
        launch_validation<kOpsPerChunk, true, true, true, true>(
            destination, warp_group_count, final_iteration, errors);
    } else if (mode == Mode::kRead75Write25) {
        launch_validation<kOpsPerChunk, true, false, true, true>(
            destination, warp_group_count, final_iteration, errors);
    } else if (mode == Mode::kRead60Write40) {
        launch_validation<
            kThreeTwoOpsPerChunk, true, false, true, false, true>(
                destination, warp_group_count, final_iteration, errors);
    } else if (mode == Mode::kRead50Write50) {
        launch_validation<kOpsPerChunk, true, false, true, false>(
            destination, warp_group_count, final_iteration, errors);
    } else if (mode == Mode::kRead25Write75) {
        launch_validation<kOpsPerChunk, true, false, false, false>(
            destination, warp_group_count, final_iteration, errors);
    } else if (mode == Mode::kWrite) {
        launch_validation<kOpsPerChunk, false, false, false, false>(
            destination, warp_group_count, final_iteration, errors);
    } else {
        std::fprintf(stderr, "unknown validation mode\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long observed_errors = 0;
    CUDA_CHECK(cudaMemcpy(
        &observed_errors, errors, sizeof(observed_errors),
        cudaMemcpyDeviceToHost));
    if (observed_errors != 0) {
        std::fprintf(stderr, "destination validation found %llu errors\n",
                     observed_errors);
        std::exit(EXIT_FAILURE);
    }
}

static double measure_mode(
        Mode mode, int grid, const uint4* source, uint4* destination,
        std::size_t warp_group_count, std::uint32_t* read_sink) {
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    launch_mode(
        mode, grid, source, destination,
        warp_group_count, kIterations, read_sink);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
        std::fprintf(stderr, "invalid event time %.9g ms\n", elapsed_ms);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<double>(elapsed_ms);
}

static void run_case(
        const ModeSpec& spec, int grid,
        const uint4* source, uint4* destination,
        std::uint32_t* read_sink, unsigned long long* errors) {
    const int operation_count = spec.read_ops + spec.write_ops;
    const std::size_t element_count = kBufferBytes / sizeof(uint4);
    const std::size_t elements_per_warp_group = (
        kExpectedWarpSize * operation_count);
    const std::size_t warp_group_count = (
        element_count / elements_per_warp_group);
    if (element_count * sizeof(uint4) != kBufferBytes
            || warp_group_count * elements_per_warp_group != element_count) {
        std::fprintf(
            stderr, "buffer does not tile the selected directional mode\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemset(destination, 0xa5, kBufferBytes));
    launch_mode(
        spec.mode, grid, source, destination,
        warp_group_count, 1, read_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double bytes_per_iteration = static_cast<double>(kBufferBytes);
    const double read_fraction = (
        static_cast<double>(spec.read_ops) / operation_count);
    const double write_fraction = (
        static_cast<double>(spec.write_ops) / operation_count);
    for (int trial = 0; trial < kTrials; ++trial) {
        double elapsed_ms = measure_mode(
            spec.mode, grid, source, destination,
            warp_group_count, read_sink);
        double seconds = elapsed_ms * 1.0e-3;
        double total_bytes = bytes_per_iteration * kIterations;
        double read_gbs = total_bytes * read_fraction / seconds / 1.0e9;
        double write_gbs = total_bytes * write_fraction / seconds / 1.0e9;
        double total_gbs = total_bytes / seconds / 1.0e9;
        std::printf(
            "%s,%.2f,%d,%d,%d,%zu,%d,%.9f,%.6f,%.6f,%.6f\n",
            spec.name, read_fraction, grid, kBlockSize, kIterations,
            kBufferBytes, trial, elapsed_ms,
            read_gbs, write_gbs, total_gbs);
        std::fflush(stdout);
    }
    validate_mode(
        spec.mode, destination, warp_group_count,
        kIterations - 1, errors);
}

int main(int argc, char** argv) {
    bool single_case = false;
    const ModeSpec* selected_mode = nullptr;
    int selected_grid = 0;
    if (argc == 1) {
        single_case = false;
    } else if (argc == 4 && std::strcmp(argv[1], "--single") == 0) {
        single_case = true;
        selected_mode = &parse_mode(argv[2]);
        selected_grid = parse_positive_integer(argv[3], "grid");
    } else {
        std::fprintf(
            stderr, "usage: %s [--single MODE GRID]\n", argv[0]);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    constexpr std::size_t minimum_global_bytes = 79ULL * 1024 * 1024 * 1024;
    constexpr std::size_t maximum_global_bytes = 80ULL * 1024 * 1024 * 1024;
    if (std::strstr(properties.name, "A800") == nullptr
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kExpectedSms
            || properties.warpSize != kExpectedWarpSize
            || properties.l2CacheSize != kExpectedL2Bytes
            || properties.totalGlobalMem < minimum_global_bytes
            || properties.totalGlobalMem > maximum_global_bytes
            || properties.clockRate <= 0 || properties.memoryClockRate <= 0) {
        std::fprintf(
            stderr,
            "unsupported device name=%s cc=%d.%d sms=%d warp=%d l2=%d "
            "global=%zu clock_khz=%d memory_clock_khz=%d\n",
            properties.name, properties.major, properties.minor,
            properties.multiProcessorCount, properties.warpSize,
            properties.l2CacheSize, properties.totalGlobalMem,
            properties.clockRate, properties.memoryClockRate);
        return EXIT_FAILURE;
    }

    const std::size_t element_count = kBufferBytes / sizeof(uint4);
    if (element_count * sizeof(uint4) != kBufferBytes) {
        std::fprintf(stderr, "invalid buffer geometry\n");
        return EXIT_FAILURE;
    }
#if DIRECTIONAL_ENABLE_READ60_WRITE40
    const int grids[] = {
        kExpectedSms, kExpectedSms * 2, kExpectedSms * 4,
        kExpectedSms * 8, 1024, kExpectedSms * 16,
    };
#else
    const int grids[] = {
        kExpectedSms, kExpectedSms * 2, kExpectedSms * 4,
        kExpectedSms * 8, kExpectedSms * 16,
    };
#endif
    const int max_grid = kExpectedSms * 16;
    bool selected_grid_supported = false;
    for (int grid : grids) {
        if (selected_grid == grid) {
            selected_grid_supported = true;
        }
    }
    if (single_case && !selected_grid_supported) {
        std::fprintf(
            stderr,
            "single grid is not present in the compiled grid sweep\n");
        return EXIT_FAILURE;
    }

    uint4* source = nullptr;
    uint4* destination = nullptr;
    std::uint32_t* read_sink = nullptr;
    unsigned long long* errors = nullptr;
    CUDA_CHECK(cudaMalloc(&source, kBufferBytes));
    CUDA_CHECK(cudaMalloc(&destination, kBufferBytes));
    CUDA_CHECK(cudaMalloc(
        &read_sink,
        static_cast<std::size_t>(max_grid) * kBlockSize
            * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&errors, sizeof(unsigned long long)));

    initialize_source<<<kExpectedSms * 4, kBlockSize>>>(
        source, element_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(errors, 0, sizeof(unsigned long long)));
    validate_source<<<kExpectedSms * 4, kBlockSize>>>(
        source, element_count, errors);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long source_errors = 0;
    CUDA_CHECK(cudaMemcpy(
        &source_errors, errors, sizeof(source_errors),
        cudaMemcpyDeviceToHost));
    if (source_errors != 0) {
        std::fprintf(stderr, "source validation found %llu errors\n",
                     source_errors);
        return EXIT_FAILURE;
    }

    std::fprintf(
        stderr,
        "device=%s cc=%d.%d sms=%d warp=%d l2_bytes=%d "
        "global_bytes=%zu clock_khz=%d memory_clock_khz=%d "
        "buffer_bytes=%zu iterations=%d trials=%d sink_max_bytes=%zu\n",
        properties.name, properties.major, properties.minor,
        properties.multiProcessorCount, properties.warpSize,
        properties.l2CacheSize, properties.totalGlobalMem,
        properties.clockRate, properties.memoryClockRate,
        kBufferBytes, kIterations, kTrials,
        static_cast<std::size_t>(max_grid) * kBlockSize
            * sizeof(std::uint32_t));
    std::printf(
        "mode,read_fraction,grid,block,iterations,bytes_per_iteration,"
        "trial,event_ms,read_gbs,write_gbs,total_gbs\n");

    if (single_case) {
        run_case(
            *selected_mode, selected_grid, source, destination,
            read_sink, errors);
    } else {
        for (const ModeSpec& spec : kModes) {
            for (int grid : grids) {
                run_case(
                    spec, grid, source, destination,
                    read_sink, errors);
            }
        }
    }

    CUDA_CHECK(cudaFree(source));
    CUDA_CHECK(cudaFree(destination));
    CUDA_CHECK(cudaFree(read_sink));
    CUDA_CHECK(cudaFree(errors));
    std::fprintf(stderr, "validation=pass\n");
    return EXIT_SUCCESS;
}
