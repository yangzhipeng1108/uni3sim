/*
 * MetaX C500 directional DRAM streaming-capacity microbenchmark.
 *
 * Six static kernels issue 0/25/50/75/100-percent read mixtures plus the
 * exact two-read/one-write stream used by binary elementwise source DAGs over
 * disjoint 256-MiB source and destination arrays.  The original five modes
 * access one consecutive 16-byte float4 per lane.  The exact BF16 vec4 mode
 * accesses one consecutive 8-byte uint2 per lane, so each aligned C500
 * 16-lane subgroup spans one 128-byte line.  The source does not otherwise
 * assume how requests map to physical transactions.  The 256-MiB footprint
 * is much larger than the reported 8-MiB L2.
 *
 * The ordinary path uses MACA device events around only the independent
 * streaming kernel.  The optional profile-once path places exactly one target
 * launch inside mcProfilerStart/Stop so non-duration global-byte and DNoC
 * request counters can validate physical traffic.  Instrumented duration is
 * never a capacity input.  Full source/destination validation, static
 * mxobjdump instruction evidence, and live mx-smi telemetry are required
 * before any result can be promoted.  This benchmark does not reproduce a
 * production operator.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       -o directional_bandwidth directional_bandwidth.cpp
 *
 * Run the complete sweep or one mode/grid:
 *   MACA_VISIBLE_DEVICES=0 ./directional_bandwidth
 *   MACA_VISIBLE_DEVICES=0 ./directional_bandwidth \
 *       --single read50_write50 1664
 *   MACA_VISIBLE_DEVICES=0 ./directional_bandwidth \
 *       --single read2_write1 1664
 *   MACA_VISIBLE_DEVICES=0 ./directional_bandwidth \
 *       --profile-once write 104 5
 */

#include <mc_runtime.h>

#include <cerrno>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

constexpr std::size_t kBufferBytes = 256ULL * 1024 * 1024;
constexpr int kVectorBytes = 16;
constexpr int kOpsPerGroup = 4;
constexpr int kBlockSize = 256;
constexpr int kIterations = 32;
constexpr int kTrials = 7;
constexpr int kExpectedAps = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kExpectedTransactionLanes = 16;
constexpr int kExpectedL2Bytes = 8 * 1024 * 1024;
constexpr int kSentinelByte = 0x3f;
constexpr std::uint32_t kSentinelWord = 0x3f3f3f3fU;
constexpr std::uint32_t kSourceSalt = 0x31415926U;
constexpr std::uint32_t kWriteSalt = 0x9e3779b9U;
static_assert(sizeof(float4) == kVectorBytes, "float4 must be 16 bytes");
static_assert(
    kExpectedWaveSize % kExpectedTransactionLanes == 0,
    "wave must contain complete 16-lane transaction groups");

enum class Mode {
    kRead,
    kRead75Write25,
    kRead2Write1,
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
    {Mode::kRead2Write1, "read2_write1", 2, 1},
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

__global__ void initialize_source(float4* source, long long element_count) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    for (long long index = thread; index < element_count; index += stride) {
        source[index] = make_value(
            static_cast<std::uint32_t>(index), kSourceSalt);
    }
}

__global__ void validate_source(
        const float4* source, long long element_count,
        unsigned long long* errors) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    unsigned long long local_errors = 0;
    for (long long index = thread; index < element_count; index += stride) {
        float4 expected = make_value(
            static_cast<std::uint32_t>(index), kSourceSalt);
        float4 observed = source[index];
        local_errors += (
            observed.x != expected.x || observed.y != expected.y
            || observed.z != expected.z || observed.w != expected.w);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

__global__ void validate_read_sink(
        const float* read_sink, long long element_count,
        unsigned long long* errors) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    unsigned long long local_errors = 0;
    for (long long index = thread; index < element_count; index += stride) {
        float value = read_sink[index];
        local_errors += (!(value > 0.0f) || !(value < 3.402823466e38f));
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

template <bool IsRead>
__device__ __forceinline__ void execute_directional_slot(
        const float4* source, float4* destination,
        long long index, int iteration, float4* checksum) {
    if constexpr (IsRead) {
        float4 value = source[index];
        checksum->x += value.x;
        checksum->y += value.y;
        checksum->z += value.z;
        checksum->w += value.w;
    } else {
        std::uint32_t salt = (
            kWriteSalt
            ^ static_cast<std::uint32_t>(iteration) * 0x85ebca6bU);
        destination[index] = make_value(
            static_cast<std::uint32_t>(index), salt);
    }
    // Preserve the predeclared R/W slot order.  Final device code must still
    // be checked because this source barrier is not hardware evidence.
    asm volatile("" ::: "memory");
}

template <bool Read0, bool Read1, bool Read2, bool Read3>
__device__ __forceinline__ void directional_body(
        const float4* source, float4* destination,
        long long wave_group_count, int iterations, float* read_sink) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long lane = threadIdx.x % kExpectedWaveSize;
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
             group < wave_group_count; group += wave_stride) {
            long long base = (
                group * kExpectedWaveSize * kOpsPerGroup + lane);
            // Each slot is aligned to 64 float4 lanes (1024 bytes).  Thus the
            // four 16-lane subgroups each span two aligned 128-byte lines.
            execute_directional_slot<Read0>(
                source, destination, base, iteration, &checksum);
            execute_directional_slot<Read1>(
                source, destination,
                base + kExpectedWaveSize, iteration, &checksum);
            execute_directional_slot<Read2>(
                source, destination,
                base + 2 * kExpectedWaveSize, iteration, &checksum);
            execute_directional_slot<Read3>(
                source, destination,
                base + 3 * kExpectedWaveSize, iteration, &checksum);
        }
    }
    if constexpr (Read0 || Read1 || Read2 || Read3) {
        read_sink[thread] = (
            checksum.x + checksum.y + checksum.z + checksum.w);
    }
}

extern "C" __global__ void c500_dram_read(
        const float4* source, float4* destination,
        long long wave_group_count, int iterations, float* read_sink) {
    directional_body<true, true, true, true>(
        source, destination, wave_group_count, iterations, read_sink);
}

extern "C" __global__ void c500_dram_read75_write25(
        const float4* source, float4* destination,
        long long wave_group_count, int iterations, float* read_sink) {
    directional_body<true, false, true, true>(
        source, destination, wave_group_count, iterations, read_sink);
}

__host__ __device__ __forceinline__ uint2 make_packed_bf16_value(
        std::uint64_t index, int iteration) {
    std::uint32_t salt = (
        kWriteSalt
        ^ static_cast<std::uint32_t>(iteration) * 0x85ebca6bU);
    return make_uint2(
        static_cast<std::uint32_t>(index) ^ salt,
        static_cast<std::uint32_t>(index >> 32) ^ ~salt);
}

extern "C" __global__ void c500_dram_read2_write1(
        const float4* source, float4* destination,
        long long wave_group_count, int iterations, float* read_sink) {
    const uint2* packed_source = reinterpret_cast<const uint2*>(source);
    uint2* packed_destination = reinterpret_cast<uint2*>(destination);
    const long long packed_wave_group_count = 2 * wave_group_count;
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long lane = threadIdx.x % kExpectedWaveSize;
    long long wave = thread / kExpectedWaveSize;
    long long wave_stride = (
        static_cast<long long>(blockDim.x) * gridDim.x
        / kExpectedWaveSize);
    uint2 checksum = make_uint2(
        mix_bits(static_cast<std::uint32_t>(thread)), kSourceSalt);
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (long long group = wave;
             group < packed_wave_group_count; group += wave_stride) {
            long long base = (
                group * kExpectedWaveSize * kOpsPerGroup + lane);
            uint2 value_a = packed_source[base];
            checksum.x ^= value_a.x;
            checksum.y += value_a.y;
            asm volatile("" ::: "memory");
            uint2 value_b = packed_source[base + kExpectedWaveSize];
            checksum.x += value_b.x;
            checksum.y ^= value_b.y;
            asm volatile("" ::: "memory");
            packed_destination[base + 2 * kExpectedWaveSize] = (
                make_packed_bf16_value(
                    static_cast<std::uint64_t>(
                        base + 2 * kExpectedWaveSize),
                    iteration));
            asm volatile("" ::: "memory");
        }
    }
    read_sink[thread] = finite_float(checksum.x ^ checksum.y);
}

extern "C" __global__ void c500_dram_read50_write50(
        const float4* source, float4* destination,
        long long wave_group_count, int iterations, float* read_sink) {
    directional_body<true, false, true, false>(
        source, destination, wave_group_count, iterations, read_sink);
}

extern "C" __global__ void c500_dram_read25_write75(
        const float4* source, float4* destination,
        long long wave_group_count, int iterations, float* read_sink) {
    directional_body<true, false, false, false>(
        source, destination, wave_group_count, iterations, read_sink);
}

extern "C" __global__ void c500_dram_write(
        const float4* source, float4* destination,
        long long wave_group_count, int iterations, float* read_sink) {
    directional_body<false, false, false, false>(
        source, destination, wave_group_count, iterations, read_sink);
}

template <bool IsRead>
__device__ __forceinline__ void validate_destination_slot(
        const float4* destination, long long index,
        std::uint32_t salt, unsigned long long* local_errors) {
    // Construct the exact finite word written by mcMemset.
    union {
        std::uint32_t bits;
        float value;
    } converted;
    converted.bits = kSentinelWord;
    float sentinel = converted.value;
    float4 expected = make_float4(
        sentinel, sentinel, sentinel, sentinel);
    if constexpr (!IsRead) {
        expected = make_value(static_cast<std::uint32_t>(index), salt);
    }
    float4 observed = destination[index];
    *local_errors += (
        observed.x != expected.x || observed.y != expected.y
        || observed.z != expected.z || observed.w != expected.w);
}

template <bool Read0, bool Read1, bool Read2, bool Read3>
__global__ void validate_destination(
        const float4* destination, long long wave_group_count,
        int final_iteration, unsigned long long* errors) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long lane = threadIdx.x % kExpectedWaveSize;
    long long wave = thread / kExpectedWaveSize;
    long long wave_stride = (
        static_cast<long long>(blockDim.x) * gridDim.x
        / kExpectedWaveSize);
    unsigned long long local_errors = 0;
    std::uint32_t salt = (
        kWriteSalt
        ^ static_cast<std::uint32_t>(final_iteration) * 0x85ebca6bU);
    for (long long group = wave;
         group < wave_group_count; group += wave_stride) {
        long long base = (
            group * kExpectedWaveSize * kOpsPerGroup + lane);
        validate_destination_slot<Read0>(
            destination, base, salt, &local_errors);
        validate_destination_slot<Read1>(
            destination, base + kExpectedWaveSize, salt, &local_errors);
        validate_destination_slot<Read2>(
            destination, base + 2 * kExpectedWaveSize,
            salt, &local_errors);
        validate_destination_slot<Read3>(
            destination, base + 3 * kExpectedWaveSize,
            salt, &local_errors);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

static void launch_mode(
        Mode mode, int grid, const float4* source, float4* destination,
        long long wave_group_count, int iterations, float* read_sink) {
    if (mode == Mode::kRead) {
        c500_dram_read<<<grid, kBlockSize>>>(
            source, destination, wave_group_count, iterations, read_sink);
    } else if (mode == Mode::kRead75Write25) {
        c500_dram_read75_write25<<<grid, kBlockSize>>>(
            source, destination, wave_group_count, iterations, read_sink);
    } else if (mode == Mode::kRead2Write1) {
        c500_dram_read2_write1<<<grid, kBlockSize>>>(
            source, destination, wave_group_count, iterations, read_sink);
    } else if (mode == Mode::kRead50Write50) {
        c500_dram_read50_write50<<<grid, kBlockSize>>>(
            source, destination, wave_group_count, iterations, read_sink);
    } else if (mode == Mode::kRead25Write75) {
        c500_dram_read25_write75<<<grid, kBlockSize>>>(
            source, destination, wave_group_count, iterations, read_sink);
    } else if (mode == Mode::kWrite) {
        c500_dram_write<<<grid, kBlockSize>>>(
            source, destination, wave_group_count, iterations, read_sink);
    } else {
        std::fprintf(stderr, "unknown directional mode\n");
        std::exit(EXIT_FAILURE);
    }
}

template <bool Read0, bool Read1, bool Read2, bool Read3>
static void launch_validation(
        const float4* destination, long long wave_group_count,
        int final_iteration, unsigned long long* errors) {
    validate_destination<Read0, Read1, Read2, Read3>
        <<<kExpectedAps * 8, kBlockSize>>>(
            destination, wave_group_count, final_iteration, errors);
}

template <bool Written>
__device__ __forceinline__ void validate_packed_destination_slot(
        const uint2* destination, long long index,
        int final_iteration, unsigned long long* local_errors) {
    uint2 expected = make_uint2(kSentinelWord, kSentinelWord);
    if constexpr (Written) {
        expected = make_packed_bf16_value(
            static_cast<std::uint64_t>(index), final_iteration);
    }
    uint2 observed = destination[index];
    *local_errors += (
        observed.x != expected.x || observed.y != expected.y);
}

__global__ void validate_packed_read2_write1_destination(
        const float4* destination, long long wave_group_count,
        int final_iteration, unsigned long long* errors) {
    const uint2* packed_destination = (
        reinterpret_cast<const uint2*>(destination));
    const long long packed_wave_group_count = 2 * wave_group_count;
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long lane = threadIdx.x % kExpectedWaveSize;
    long long wave = thread / kExpectedWaveSize;
    long long wave_stride = (
        static_cast<long long>(blockDim.x) * gridDim.x
        / kExpectedWaveSize);
    unsigned long long local_errors = 0;
    for (long long group = wave;
         group < packed_wave_group_count; group += wave_stride) {
        long long base = (
            group * kExpectedWaveSize * kOpsPerGroup + lane);
        validate_packed_destination_slot<false>(
            packed_destination, base, final_iteration, &local_errors);
        validate_packed_destination_slot<false>(
            packed_destination, base + kExpectedWaveSize,
            final_iteration, &local_errors);
        validate_packed_destination_slot<true>(
            packed_destination, base + 2 * kExpectedWaveSize,
            final_iteration, &local_errors);
        validate_packed_destination_slot<false>(
            packed_destination, base + 3 * kExpectedWaveSize,
            final_iteration, &local_errors);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

static void validate_mode(
        Mode mode, const float4* destination, long long wave_group_count,
        int final_iteration, unsigned long long* errors) {
    CHECK(mcMemset(errors, 0, sizeof(unsigned long long)));
    if (mode == Mode::kRead) {
        launch_validation<true, true, true, true>(
            destination, wave_group_count, final_iteration, errors);
    } else if (mode == Mode::kRead75Write25) {
        launch_validation<true, false, true, true>(
            destination, wave_group_count, final_iteration, errors);
    } else if (mode == Mode::kRead2Write1) {
        validate_packed_read2_write1_destination
            <<<kExpectedAps * 8, kBlockSize>>>(
                destination, wave_group_count, final_iteration, errors);
    } else if (mode == Mode::kRead50Write50) {
        launch_validation<true, false, true, false>(
            destination, wave_group_count, final_iteration, errors);
    } else if (mode == Mode::kRead25Write75) {
        launch_validation<true, false, false, false>(
            destination, wave_group_count, final_iteration, errors);
    } else if (mode == Mode::kWrite) {
        launch_validation<false, false, false, false>(
            destination, wave_group_count, final_iteration, errors);
    } else {
        std::fprintf(stderr, "unknown validation mode\n");
        std::exit(EXIT_FAILURE);
    }
    CHECK(mcDeviceSynchronize());
    unsigned long long observed_errors = 0;
    CHECK(mcMemcpy(
        &observed_errors, errors, sizeof(observed_errors),
        mcMemcpyDeviceToHost));
    if (observed_errors != 0) {
        std::fprintf(stderr, "destination validation found %llu errors\n",
                     observed_errors);
        std::exit(EXIT_FAILURE);
    }
}

static double measure_mode(
        Mode mode, int grid, const float4* source, float4* destination,
        long long wave_group_count, float* read_sink) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    launch_mode(
        mode, grid, source, destination,
        wave_group_count, kIterations, read_sink);
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0f;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
        std::fprintf(stderr, "invalid event time %.9g ms\n", elapsed_ms);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<double>(elapsed_ms);
}

static void run_case(
        const ModeSpec& spec, int grid,
        const float4* source, float4* destination,
        long long wave_group_count, float* read_sink,
        std::size_t read_sink_bytes,
        unsigned long long* errors) {
    CHECK(mcMemset(destination, kSentinelByte, kBufferBytes));
    CHECK(mcMemset(read_sink, 0, read_sink_bytes));
    launch_mode(
        spec.mode, grid, source, destination,
        wave_group_count, 1, read_sink);
    CHECK(mcDeviceSynchronize());
    const double gross_read_fraction = (
        static_cast<double>(spec.read_ops) / kOpsPerGroup);
    const double gross_write_fraction = (
        static_cast<double>(spec.write_ops) / kOpsPerGroup);
    const double read_fraction = (
        static_cast<double>(spec.read_ops)
        / (spec.read_ops + spec.write_ops));
    const double bytes_per_iteration = (
        static_cast<double>(kBufferBytes)
        * (gross_read_fraction + gross_write_fraction));
    for (int trial = 0; trial < kTrials; ++trial) {
        double elapsed_ms = measure_mode(
            spec.mode, grid, source, destination,
            wave_group_count, read_sink);
        double seconds = elapsed_ms * 1.0e-3;
        double gross_bytes = (
            static_cast<double>(kBufferBytes) * kIterations);
        double read_gbs = (
            gross_bytes * gross_read_fraction / seconds / 1.0e9);
        double write_gbs = (
            gross_bytes * gross_write_fraction / seconds / 1.0e9);
        double total_bytes = bytes_per_iteration * kIterations;
        double total_gbs = total_bytes / seconds / 1.0e9;
        std::printf(
            "%s,%.2f,%d,%d,%d,%.0f,%d,%.9f,%.6f,%.6f,%.6f\n",
            spec.name, read_fraction, grid, kBlockSize, kIterations,
            bytes_per_iteration, trial, elapsed_ms,
            read_gbs, write_gbs, total_gbs);
        std::fflush(stdout);
    }
    validate_mode(
        spec.mode, destination, wave_group_count,
        kIterations - 1, errors);
    if (spec.read_ops > 0) {
        CHECK(mcMemset(errors, 0, sizeof(unsigned long long)));
        long long active_threads = (
            static_cast<long long>(grid) * kBlockSize);
        validate_read_sink<<<kExpectedAps * 8, kBlockSize>>>(
            read_sink, active_threads, errors);
        CHECK(mcDeviceSynchronize());
        unsigned long long observed_errors = 0;
        CHECK(mcMemcpy(
            &observed_errors, errors, sizeof(observed_errors),
            mcMemcpyDeviceToHost));
        if (observed_errors != 0) {
            std::fprintf(stderr,
                         "read-sink validation found %llu errors\n",
                         observed_errors);
            std::exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    bool single_case = false;
    bool profile_once = false;
    const ModeSpec* selected_mode = nullptr;
    int selected_grid = 0;
    int profiler_attach_wait_seconds = 0;
    if (argc == 1) {
        single_case = false;
    } else if (argc == 4 && std::strcmp(argv[1], "--single") == 0) {
        single_case = true;
        selected_mode = &parse_mode(argv[2]);
        selected_grid = parse_positive_integer(argv[3], "grid");
    } else if (argc == 5
            && std::strcmp(argv[1], "--profile-once") == 0) {
        single_case = true;
        profile_once = true;
        selected_mode = &parse_mode(argv[2]);
        selected_grid = parse_positive_integer(argv[3], "grid");
        profiler_attach_wait_seconds = parse_positive_integer(
            argv[4], "profiler attach wait seconds");
    } else {
        std::fprintf(
            stderr,
            "usage: %s [--single MODE GRID | "
            "--profile-once MODE GRID ATTACH_WAIT_SECONDS]\n",
            argv[0]);
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

    const long long element_count = static_cast<long long>(
        kBufferBytes / sizeof(float4));
    const long long elements_per_wave_group = (
        kExpectedWaveSize * kOpsPerGroup);
    const long long wave_group_count = (
        element_count / elements_per_wave_group);
    if (static_cast<std::size_t>(element_count) * sizeof(float4)
                != kBufferBytes
            || wave_group_count * elements_per_wave_group != element_count) {
        std::fprintf(stderr, "invalid buffer/wave-group geometry\n");
        return EXIT_FAILURE;
    }
    const int grids[] = {
        kExpectedAps, kExpectedAps * 2, kExpectedAps * 4,
        kExpectedAps * 8, kExpectedAps * 16,
    };
    const int max_grid = kExpectedAps * 16;
    const std::size_t read_sink_bytes = (
        static_cast<std::size_t>(max_grid) * kBlockSize * sizeof(float));
    bool selected_grid_supported = false;
    for (int grid : grids) {
        if (selected_grid == grid) {
            selected_grid_supported = true;
        }
    }
    if (single_case && !selected_grid_supported) {
        std::fprintf(
            stderr,
            "single grid must be one of %d,%d,%d,%d,%d\n",
            grids[0], grids[1], grids[2], grids[3], grids[4]);
        return EXIT_FAILURE;
    }

    float4* source = nullptr;
    float4* destination = nullptr;
    float* read_sink = nullptr;
    unsigned long long* errors = nullptr;
    CHECK(mcMalloc(&source, kBufferBytes));
    CHECK(mcMalloc(&destination, kBufferBytes));
    CHECK(mcMalloc(
        &read_sink, read_sink_bytes));
    CHECK(mcMalloc(&errors, sizeof(unsigned long long)));

    initialize_source<<<kExpectedAps * 8, kBlockSize>>>(
        source, element_count);
    CHECK(mcDeviceSynchronize());
    CHECK(mcMemset(errors, 0, sizeof(unsigned long long)));
    validate_source<<<kExpectedAps * 8, kBlockSize>>>(
        source, element_count, errors);
    CHECK(mcDeviceSynchronize());
    unsigned long long source_errors = 0;
    CHECK(mcMemcpy(
        &source_errors, errors, sizeof(source_errors),
        mcMemcpyDeviceToHost));
    if (source_errors != 0) {
        std::fprintf(stderr, "source validation found %llu errors\n",
                     source_errors);
        return EXIT_FAILURE;
    }

    std::fprintf(
        stderr,
        "device=%s aps=%d wave=%d transaction_lanes=%d l2_bytes=%d "
        "global_bytes=%zu clock_khz=%d memory_clock_khz=%d "
        "buffer_bytes=%zu iterations=%d trials=%d sink_max_bytes=%zu\n",
        properties.name, properties.multiProcessorCount, properties.warpSize,
        kExpectedTransactionLanes, properties.l2CacheSize,
        properties.totalGlobalMem, properties.clockRate,
        properties.memoryClockRate, kBufferBytes, kIterations, kTrials,
        read_sink_bytes);
    std::printf(
        "mode,read_fraction,grid,block,iterations,bytes_per_iteration,"
        "trial,event_ms,read_gbs,write_gbs,total_gbs\n");

    if (profile_once) {
        CHECK(mcMemset(destination, kSentinelByte, kBufferBytes));
        CHECK(mcMemset(read_sink, 0, read_sink_bytes));
        launch_mode(
            selected_mode->mode, selected_grid, source, destination,
            wave_group_count, 1, read_sink);
        CHECK(mcDeviceSynchronize());
        CHECK(mcMemset(destination, kSentinelByte, kBufferBytes));
        CHECK(mcMemset(read_sink, 0, read_sink_bytes));
        std::this_thread::sleep_for(
            std::chrono::seconds(profiler_attach_wait_seconds));
        CHECK(mcProfilerStart());
        launch_mode(
            selected_mode->mode, selected_grid, source, destination,
            wave_group_count, kIterations, read_sink);
        CHECK(mcDeviceSynchronize());
        CHECK(mcProfilerStop());
        validate_mode(
            selected_mode->mode, destination, wave_group_count,
            kIterations - 1, errors);
        const std::size_t logical_read_bytes = (
            static_cast<std::size_t>(selected_mode->read_ops)
            * kBufferBytes * kIterations / kOpsPerGroup);
        const std::size_t logical_write_bytes = (
            static_cast<std::size_t>(selected_mode->write_ops)
            * kBufferBytes * kIterations / kOpsPerGroup);
        std::printf(
            "profile_once,%s,%d,%d,%zu,%zu\n",
            selected_mode->name, selected_grid, kIterations,
            logical_read_bytes, logical_write_bytes);
        std::fflush(stdout);
    } else if (single_case) {
        run_case(
            *selected_mode, selected_grid, source, destination,
            wave_group_count, read_sink, read_sink_bytes, errors);
    } else {
        for (const ModeSpec& spec : kModes) {
            for (int grid : grids) {
                run_case(
                    spec, grid, source, destination,
                    wave_group_count, read_sink, read_sink_bytes, errors);
        }
    }

    CHECK(mcMemset(errors, 0, sizeof(unsigned long long)));
    validate_source<<<kExpectedAps * 8, kBlockSize>>>(
        source, element_count, errors);
    CHECK(mcDeviceSynchronize());
    source_errors = 0;
    CHECK(mcMemcpy(
        &source_errors, errors, sizeof(source_errors),
        mcMemcpyDeviceToHost));
    if (source_errors != 0) {
        std::fprintf(stderr,
                     "post-run source validation found %llu errors\n",
                     source_errors);
        return EXIT_FAILURE;
    }
    }

    CHECK(mcFree(source));
    CHECK(mcFree(destination));
    CHECK(mcFree(read_sink));
    CHECK(mcFree(errors));
    std::fprintf(stderr, "validation=pass\n");
    return EXIT_SUCCESS;
}
