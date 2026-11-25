/*
 * MetaX C500 L2-request / DRAM-fill interference microbenchmark.
 *
 * Five host modes call one identical kernel with the same four float4 load
 * slots, address arithmetic, grid, block and total logical bytes.  The host
 * changes only each slot's base pointer and power-of-two index mask so that it
 * addresses a warmed 1-MiB hot set or a streaming 256-MiB cold set.  The cold
 * fractions are 0/25/50/75/100 percent.  Hot and cold slots are interleaved in
 * every wave; this is not a phased production-operator surrogate.
 *
 * Device events measure only the independent streaming kernel.  Source and
 * sink validation remain outside the timed region.  Final device code must
 * retain four equivalent global float4 loads for every mode before results
 * can be used as topology evidence.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       -o l2_fill_interference l2_fill_interference.cpp
 */

#include <mc_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

constexpr std::size_t kMiB = 1024ULL * 1024;
constexpr std::size_t kHotBytes = 1ULL * kMiB;
constexpr std::size_t kColdBytes = 256ULL * kMiB;
constexpr int kVectorBytes = 16;
constexpr int kSlotsPerGroup = 4;
constexpr int kBlockSize = 256;
constexpr int kGridSize = 104 * 16;
constexpr int kIterations = 32;
constexpr int kWarmupIterations = 32;
constexpr int kTrials = 7;
constexpr int kExpectedAps = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kExpectedL2Bytes = 8 * 1024 * 1024;
constexpr std::uint32_t kHotSalt = 0x243f6a88U;
constexpr std::uint32_t kColdSalt = 0x9e3779b9U;
static_assert(sizeof(float4) == kVectorBytes, "float4 must be 16 bytes");
static_assert(kHotBytes % (kSlotsPerGroup * kVectorBytes) == 0,
              "hot set must split equally across load slots");
static_assert(kColdBytes %
                  (kExpectedWaveSize * kSlotsPerGroup * kVectorBytes) == 0,
              "cold set must contain complete wave groups");

struct ModeSpec {
    const char* name;
    int cold_slots;
};

constexpr ModeSpec kModes[] = {
    {"cold0", 0},
    {"cold25", 1},
    {"cold50", 2},
    {"cold75", 3},
    {"cold100", 4},
};

__host__ __device__ __forceinline__ std::uint32_t mix_bits(
        std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__host__ __device__ __forceinline__ float finite_float(
        std::uint32_t bits) {
    union {
        std::uint32_t bits;
        float value;
    } converted;
    converted.bits = 0x3f000000U | (bits & 0x007fffffU);
    return converted.value;
}

__host__ __device__ __forceinline__ float4 make_value(
        std::uint32_t index, std::uint32_t salt) {
    std::uint32_t x = mix_bits(index ^ salt);
    std::uint32_t y = mix_bits(x ^ 0x68bc21ebU);
    std::uint32_t z = mix_bits(y ^ 0x02e5be93U);
    std::uint32_t w = mix_bits(z ^ 0x967a889bU);
    return make_float4(
        finite_float(x), finite_float(y), finite_float(z), finite_float(w));
}

__global__ void initialize_data(
        float4* data, long long elements, std::uint32_t salt) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    for (long long index = thread; index < elements; index += stride) {
        data[index] = make_value(static_cast<std::uint32_t>(index), salt);
    }
}

__global__ void validate_data(
        const float4* data, long long elements, std::uint32_t salt,
        unsigned long long* errors) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    unsigned long long local_errors = 0;
    for (long long index = thread; index < elements; index += stride) {
        float4 expected = make_value(
            static_cast<std::uint32_t>(index), salt);
        float4 observed = data[index];
        local_errors += (
            observed.x != expected.x || observed.y != expected.y
            || observed.z != expected.z || observed.w != expected.w);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

__global__ void validate_sink(
        const float* sink, long long elements,
        unsigned long long* errors) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    unsigned long long local_errors = 0;
    for (long long index = thread; index < elements; index += stride) {
        float value = sink[index];
        local_errors += (
            !(value > 0.0f) || !(value < 3.402823466e38f));
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

__device__ __forceinline__ void load_slot(
        const float4* source, long long index_mask,
        long long stream_index, float4* checksum) {
    float4 value = source[stream_index & index_mask];
    checksum->x += value.x;
    checksum->y += value.y;
    checksum->z += value.z;
    checksum->w += value.w;
    asm volatile("" ::: "memory");
}

__device__ __forceinline__ void interference_body(
        const float4* source0, const float4* source1,
        const float4* source2, const float4* source3,
        long long mask0, long long mask1,
        long long mask2, long long mask3,
        long long group_count, int iterations, float* sink) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long lane = thread % kExpectedWaveSize;
    long long wave = thread / kExpectedWaveSize;
    long long wave_stride = (
        static_cast<long long>(blockDim.x) * gridDim.x
        / kExpectedWaveSize);
    float4 checksum = make_float4(
        finite_float(mix_bits(static_cast<std::uint32_t>(thread))),
        0.5f, 0.75f, 0.625f);
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (long long group = wave;
             group < group_count; group += wave_stride) {
            long long stream_index = group * kExpectedWaveSize + lane;
            load_slot(source0, mask0, stream_index, &checksum);
            load_slot(source1, mask1, stream_index, &checksum);
            load_slot(source2, mask2, stream_index, &checksum);
            load_slot(source3, mask3, stream_index, &checksum);
        }
    }
    sink[thread] = checksum.x + checksum.y + checksum.z + checksum.w;
}

extern "C" __global__ void c500_l2_fill_interference(
        const float4* source0, const float4* source1,
        const float4* source2, const float4* source3,
        long long mask0, long long mask1,
        long long mask2, long long mask3,
        long long group_count, int iterations, float* sink) {
    interference_body(
        source0, source1, source2, source3,
        mask0, mask1, mask2, mask3,
        group_count, iterations, sink);
}

static const ModeSpec& parse_mode(const char* text) {
    for (const ModeSpec& mode : kModes) {
        if (std::strcmp(text, mode.name) == 0) {
            return mode;
        }
    }
    std::fprintf(stderr, "unsupported mode %s\n", text);
    std::exit(EXIT_FAILURE);
}

static void launch_mode(
        const ModeSpec& mode, const float4* hot, const float4* cold,
        long long group_count, int iterations, float* sink) {
    if (mode.cold_slots < 0 || mode.cold_slots > kSlotsPerGroup) {
        std::fprintf(stderr, "invalid cold-slot count %d\n", mode.cold_slots);
        std::exit(EXIT_FAILURE);
    }
    constexpr long long hot_slice_elements = (
        kHotBytes / sizeof(float4) / kSlotsPerGroup);
    constexpr long long cold_slice_elements = (
        kColdBytes / sizeof(float4) / kSlotsPerGroup);
    const float4* sources[kSlotsPerGroup];
    long long masks[kSlotsPerGroup];
    for (int slot = 0; slot < kSlotsPerGroup; ++slot) {
        bool use_cold = slot < mode.cold_slots;
        sources[slot] = use_cold
            ? cold + static_cast<long long>(slot) * cold_slice_elements
            : hot + static_cast<long long>(slot) * hot_slice_elements;
        masks[slot] = (
            use_cold ? cold_slice_elements : hot_slice_elements) - 1;
    }
    c500_l2_fill_interference<<<kGridSize, kBlockSize>>>(
        sources[0], sources[1], sources[2], sources[3],
        masks[0], masks[1], masks[2], masks[3],
        group_count, iterations, sink);
}

static void require_zero_errors(
        unsigned long long* errors, const char* context) {
    unsigned long long observed = 0;
    CHECK(mcMemcpy(
        &observed, errors, sizeof(observed), mcMemcpyDeviceToHost));
    if (observed != 0) {
        std::fprintf(stderr, "%s found %llu errors\n", context, observed);
        std::exit(EXIT_FAILURE);
    }
}

static void run_mode(
        const ModeSpec& mode, const float4* hot, const float4* cold,
        long long group_count, float* sink,
        unsigned long long* errors) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    const double logical_bytes = (
        static_cast<double>(kColdBytes) * kIterations);
    const double cold_fraction = (
        static_cast<double>(mode.cold_slots) / kSlotsPerGroup);
    for (int trial = 1; trial <= kTrials; ++trial) {
        launch_mode(
            kModes[0], hot, cold, group_count,
            kWarmupIterations, sink);
        CHECK(mcDeviceSynchronize());
        CHECK(mcEventRecord(start));
        launch_mode(mode, hot, cold, group_count, kIterations, sink);
        CHECK(mcEventRecord(end));
        CHECK(mcDeviceSynchronize());
        float elapsed_ms = 0.0f;
        CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
            std::fprintf(stderr, "invalid event time %.9g ms\n", elapsed_ms);
            std::exit(EXIT_FAILURE);
        }
        double logical_gbs = logical_bytes / (elapsed_ms * 1.0e6);
        double cold_gbs = logical_gbs * cold_fraction;
        std::printf(
            "%s,%.2f,%d,%d,%d,%zu,%zu,%d,%.9g,%.6f,%.6f\n",
            mode.name, cold_fraction, kGridSize, kBlockSize, kIterations,
            kColdBytes, static_cast<std::size_t>(
                kColdBytes * mode.cold_slots / kSlotsPerGroup),
            trial, elapsed_ms, logical_gbs, cold_gbs);
        std::fflush(stdout);
    }
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));

    CHECK(mcMemset(errors, 0, sizeof(unsigned long long)));
    validate_sink<<<kExpectedAps * 8, kBlockSize>>>(
        sink, static_cast<long long>(kGridSize) * kBlockSize, errors);
    CHECK(mcDeviceSynchronize());
    require_zero_errors(errors, "sink validation");
}

int main(int argc, char** argv) {
    bool single_mode = false;
    const ModeSpec* selected_mode = nullptr;
    if (argc == 1) {
        single_mode = false;
    } else if (argc == 3 && std::strcmp(argv[1], "--single") == 0) {
        single_mode = true;
        selected_mode = &parse_mode(argv[2]);
    } else {
        std::fprintf(stderr, "usage: %s [--single MODE]\n", argv[0]);
        return EXIT_FAILURE;
    }

    mcDeviceProp_t properties;
    CHECK(mcGetDeviceProperties(&properties, 0));
    constexpr std::size_t minimum_global_bytes = 63ULL * 1024 * 1024 * 1024;
    constexpr std::size_t maximum_global_bytes = 65ULL * 1024 * 1024 * 1024;
    if (std::strstr(properties.name, "C500") == nullptr
            || properties.multiProcessorCount != kExpectedAps
            || properties.warpSize != kExpectedWaveSize
            || properties.l2CacheSize != kExpectedL2Bytes
            || properties.totalGlobalMem < minimum_global_bytes
            || properties.totalGlobalMem > maximum_global_bytes
            || properties.clockRate <= 0 || properties.memoryClockRate <= 0) {
        std::fprintf(
            stderr,
            "unsupported device name=%s aps=%d wave=%d l2=%d global=%zu "
            "clock_khz=%d memory_clock_khz=%d\n",
            properties.name, properties.multiProcessorCount,
            properties.warpSize, properties.l2CacheSize,
            properties.totalGlobalMem, properties.clockRate,
            properties.memoryClockRate);
        return EXIT_FAILURE;
    }

    const long long hot_elements = kHotBytes / sizeof(float4);
    const long long cold_elements = kColdBytes / sizeof(float4);
    const long long group_count = (
        cold_elements / (kExpectedWaveSize * kSlotsPerGroup));
    const std::size_t sink_bytes = (
        static_cast<std::size_t>(kGridSize) * kBlockSize * sizeof(float));
    float4* hot = nullptr;
    float4* cold = nullptr;
    float* sink = nullptr;
    unsigned long long* errors = nullptr;
    CHECK(mcMalloc(&hot, kHotBytes));
    CHECK(mcMalloc(&cold, kColdBytes));
    CHECK(mcMalloc(&sink, sink_bytes));
    CHECK(mcMalloc(&errors, sizeof(unsigned long long)));
    initialize_data<<<kExpectedAps * 8, kBlockSize>>>(
        hot, hot_elements, kHotSalt);
    initialize_data<<<kExpectedAps * 8, kBlockSize>>>(
        cold, cold_elements, kColdSalt);
    CHECK(mcDeviceSynchronize());

    CHECK(mcMemset(errors, 0, sizeof(unsigned long long)));
    validate_data<<<kExpectedAps * 8, kBlockSize>>>(
        hot, hot_elements, kHotSalt, errors);
    validate_data<<<kExpectedAps * 8, kBlockSize>>>(
        cold, cold_elements, kColdSalt, errors);
    CHECK(mcDeviceSynchronize());
    require_zero_errors(errors, "source validation");

    std::fprintf(
        stderr,
        "device=%s aps=%d wave=%d l2_bytes=%d global_bytes=%zu "
        "clock_khz=%d memory_clock_khz=%d hot_bytes=%zu cold_bytes=%zu "
        "grid=%d block=%d iterations=%d trials=%d\n",
        properties.name, properties.multiProcessorCount, properties.warpSize,
        properties.l2CacheSize, properties.totalGlobalMem,
        properties.clockRate, properties.memoryClockRate,
        kHotBytes, kColdBytes, kGridSize, kBlockSize, kIterations, kTrials);
    std::printf(
        "mode,cold_fraction,grid,block,iterations,"
        "logical_bytes_per_iteration,cold_bytes_per_iteration,"
        "trial,event_ms,logical_gbs,cold_gbs\n");
    if (single_mode) {
        run_mode(*selected_mode, hot, cold, group_count, sink, errors);
    } else {
        for (const ModeSpec& mode : kModes) {
            run_mode(mode, hot, cold, group_count, sink, errors);
        }
    }

    CHECK(mcMemset(errors, 0, sizeof(unsigned long long)));
    validate_data<<<kExpectedAps * 8, kBlockSize>>>(
        hot, hot_elements, kHotSalt, errors);
    validate_data<<<kExpectedAps * 8, kBlockSize>>>(
        cold, cold_elements, kColdSalt, errors);
    CHECK(mcDeviceSynchronize());
    require_zero_errors(errors, "post-run source validation");
    CHECK(mcFree(errors));
    CHECK(mcFree(sink));
    CHECK(mcFree(cold));
    CHECK(mcFree(hot));
    std::fprintf(stderr, "validation=pass\n");
    return EXIT_SUCCESS;
}
