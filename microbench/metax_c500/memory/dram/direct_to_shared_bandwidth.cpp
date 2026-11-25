/*
 * MetaX C500 global-to-shared read-capacity microbenchmark.
 *
 * The two static kernels traverse the same 256-MiB source with the same
 * grid, block, 64-KiB dynamic-shared-memory reservation, seven-request
 * window, and explicit gvm completion wait.  The register path provides an
 * instruction/address-matched control.  The direct path uses the documented
 * MACA ldg_b128-to-BSM builtin and never stages through registers.  Seven is
 * deliberately unrelated to a production GEMM tile or pipeline depth.
 *
 * Device Event duration is used only to derive this workload-independent
 * mechanism capacity.  The optional profile-once mode exists solely to
 * validate instruction and physical traffic counts; instrumented profiler
 * duration is not a capacity input.  Final device code must be checked with
 * mxobjdump before promotion.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       -o direct_to_shared_bandwidth direct_to_shared_bandwidth.cpp
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",               \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr std::size_t kBufferBytes = 256ULL * 1024 * 1024;
constexpr int kVectorBytes = 16;
constexpr int kBlockThreads = 256;
constexpr int kExpectedAps = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kExpectedL2Bytes = 8 * 1024 * 1024;
constexpr int kRequestWindow = 7;
constexpr int kIterations = 17;
constexpr int kWarmupLaunches = 3;
constexpr int kTrials = 7;
constexpr int kDynamicSharedBytes = 64 * 1024;

typedef __NATIVE_VECTOR__(4, std::uint32_t) v4u;

static_assert(sizeof(v4u) == kVectorBytes, "load vector must be 128 bits");
static_assert(kBlockThreads % kExpectedWaveSize == 0,
              "block must contain complete wave64 groups");
static_assert(kRequestWindow == 7,
              "request window must remain non-production and prime");
static_assert(
    kBlockThreads / kExpectedWaveSize * kRequestWindow
        * kExpectedWaveSize * kVectorBytes <= kDynamicSharedBytes,
    "direct-load destination ring must fit reserved shared memory");

enum class Mode { kRegister, kDirectToShared };

struct ModeSpec {
    Mode mode;
    const char* name;
    const char* kernel_identity;
};

constexpr ModeSpec kModes[] = {
    {Mode::kRegister, "register", "c500_global_to_register_read"},
    {Mode::kDirectToShared, "direct_to_shared",
     "c500_global_to_shared_read"},
};

static int parse_positive_integer(const char* text, const char* label) {
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value <= 0 || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a positive integer, got %s\n",
                     label, text);
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

__host__ __device__ __forceinline__ v4u make_value(std::uint32_t index) {
    std::uint32_t x = mix_bits(index ^ 0x31415926U);
    std::uint32_t y = mix_bits(x ^ 0x68bc21ebU);
    std::uint32_t z = mix_bits(y ^ 0x02e5be93U);
    std::uint32_t w = mix_bits(z ^ 0x967a889bU);
    return {x, y, z, w};
}

__global__ void initialize_source(v4u* source, long long element_count) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long index = thread; index < element_count; index += stride) {
        source[index] = make_value(static_cast<std::uint32_t>(index));
    }
}

__device__ __forceinline__ v4u load_register(const v4u* source) {
    return __builtin_mxc_ldg_b128(
        const_cast<v4u*>(source), 0, -1, true, true, false, false);
}

__device__ __forceinline__ void load_direct_to_shared(
        void* destination, const v4u* source) {
    __builtin_mxc_ldg_b128_bsm(
        destination, const_cast<v4u*>(source),
        0, -1, true, true, false, true);
}

extern "C" __global__ void c500_global_to_register_read(
        const v4u* source, long long rounds, int iterations, v4u* sink) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    v4u accum = make_value(static_cast<std::uint32_t>(thread));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (long long round = 0; round < rounds; ++round) {
            v4u values[kRequestWindow];
            #pragma unroll
            for (int slot = 0; slot < kRequestWindow; ++slot) {
                long long index = (
                    (round * kRequestWindow + slot) * stride + thread);
                values[slot] = load_register(source + index);
            }
            __builtin_mxc_arrive(64);
            #pragma unroll
            for (int slot = 0; slot < kRequestWindow; ++slot) {
                accum ^= values[slot];
            }
        }
    }
    sink[thread] = accum;
}

extern "C" __global__ void c500_global_to_shared_read(
        const v4u* source, long long rounds, int iterations, v4u* sink) {
    extern __shared__ std::uint8_t shared_bytes[];
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    int wave = static_cast<int>(threadIdx.x) / kExpectedWaveSize;
    int lane = static_cast<int>(threadIdx.x) % kExpectedWaveSize;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (long long round = 0; round < rounds; ++round) {
            #pragma unroll
            for (int slot = 0; slot < kRequestWindow; ++slot) {
                long long index = (
                    (round * kRequestWindow + slot) * stride + thread);
                std::size_t shared_offset = static_cast<std::size_t>(
                    (wave * kRequestWindow + slot)
                    * kExpectedWaveSize * kVectorBytes);
                load_direct_to_shared(shared_bytes + shared_offset,
                                      source + index);
            }
            __builtin_mxc_arrive(64);
        }
    }
    __builtin_mxc_barrier_inst();
    const std::size_t final_offset = static_cast<std::size_t>(
        (wave * kRequestWindow + (kRequestWindow - 1))
        * kExpectedWaveSize + lane);
    sink[thread] = reinterpret_cast<v4u*>(shared_bytes)[final_offset];
}

static void launch_mode(
        Mode mode, int grid, const v4u* source,
        long long rounds, int iterations, v4u* sink) {
    if (mode == Mode::kRegister) {
        c500_global_to_register_read
            <<<grid, kBlockThreads, kDynamicSharedBytes>>>(
                source, rounds, iterations, sink);
    } else if (mode == Mode::kDirectToShared) {
        c500_global_to_shared_read
            <<<grid, kBlockThreads, kDynamicSharedBytes>>>(
                source, rounds, iterations, sink);
    } else {
        std::fprintf(stderr, "unknown mode\n");
        std::exit(EXIT_FAILURE);
    }
}

__global__ void validate_direct_sink_kernel(
        const v4u* source, const v4u* sink, long long rounds,
        unsigned long long* errors) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    const long long stride = (
        static_cast<long long>(blockDim.x) * gridDim.x);
    const long long final_base = (
        (rounds * kRequestWindow - 1) * stride);
    const v4u expected = source[final_base + thread];
    const v4u observed = sink[thread];
    if (observed[0] != expected[0]
            || observed[1] != expected[1]
            || observed[2] != expected[2]
            || observed[3] != expected[3]) {
        atomicAdd(errors, 1ULL);
    }
}

static void validate_direct_sink(
        const v4u* source, const v4u* sink, int grid, long long rounds,
        unsigned long long* errors) {
    CHECK(mcMemset(errors, 0, sizeof(unsigned long long)));
    validate_direct_sink_kernel<<<grid, kBlockThreads>>>(
        source, sink, rounds, errors);
    CHECK(mcDeviceSynchronize());
    unsigned long long observed_errors = 0;
    CHECK(mcMemcpy(
        &observed_errors, errors, sizeof(observed_errors),
        mcMemcpyDeviceToHost));
    if (observed_errors != 0) {
        std::fprintf(stderr,
                     "direct-to-shared validation found %llu errors\n",
                     observed_errors);
        std::exit(EXIT_FAILURE);
    }
}

static double measure_mode(
        Mode mode, int grid, const v4u* source,
        long long rounds, v4u* sink) {
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    launch_mode(mode, grid, source, rounds, kIterations, sink);
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
        const ModeSpec& spec, int grid, const v4u* source,
        long long element_count, v4u* sink, unsigned long long* errors) {
    const long long thread_count = (
        static_cast<long long>(grid) * kBlockThreads);
    const long long rounds = (
        element_count / (thread_count * kRequestWindow));
    if (rounds <= 0) {
        std::fprintf(stderr, "grid leaves no complete request window\n");
        std::exit(EXIT_FAILURE);
    }
    const std::size_t bytes_per_iteration = static_cast<std::size_t>(
        rounds * thread_count * kRequestWindow * kVectorBytes);
    for (int warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        launch_mode(spec.mode, grid, source, rounds, 1, sink);
    }
    CHECK(mcDeviceSynchronize());
    for (int trial = 0; trial < kTrials; ++trial) {
        const double elapsed_ms = measure_mode(
            spec.mode, grid, source, rounds, sink);
        const double total_bytes = (
            static_cast<double>(bytes_per_iteration) * kIterations);
        const double bandwidth_gbs = total_bytes / elapsed_ms / 1.0e6;
        std::printf(
            "%s,%s,%d,%d,%d,%lld,%d,%zu,%d,%.9f,%.6f\n",
            spec.name, spec.kernel_identity, grid, kBlockThreads,
            kRequestWindow, rounds, kIterations, bytes_per_iteration,
            trial, elapsed_ms, bandwidth_gbs);
        std::fflush(stdout);
    }
    if (spec.mode == Mode::kDirectToShared) {
        launch_mode(spec.mode, grid, source, rounds, 1, sink);
        CHECK(mcDeviceSynchronize());
        validate_direct_sink(source, sink, grid, rounds, errors);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool single_case = false;
    bool profile_once = false;
    const ModeSpec* selected_mode = nullptr;
    int selected_grid = 0;
    int attach_wait_seconds = 0;
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
        attach_wait_seconds = parse_positive_integer(
            argv[4], "attach wait seconds");
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
    if (std::strstr(properties.name, "C500") == nullptr
            || properties.multiProcessorCount != kExpectedAps
            || properties.warpSize != kExpectedWaveSize
            || properties.l2CacheSize != kExpectedL2Bytes) {
        std::fprintf(
            stderr, "unsupported device name=%s aps=%d wave=%d l2=%d\n",
            properties.name, properties.multiProcessorCount,
            properties.warpSize, properties.l2CacheSize);
        return EXIT_FAILURE;
    }

    const int grids[] = {
        kExpectedAps, kExpectedAps * 2, kExpectedAps * 4,
        kExpectedAps * 8, kExpectedAps * 16,
    };
    bool selected_grid_supported = false;
    for (int grid : grids) {
        selected_grid_supported |= selected_grid == grid;
    }
    if (single_case && !selected_grid_supported) {
        std::fprintf(stderr, "unsupported grid %d\n", selected_grid);
        return EXIT_FAILURE;
    }

    const long long element_count = static_cast<long long>(
        kBufferBytes / sizeof(v4u));
    const int max_grid = grids[sizeof(grids) / sizeof(grids[0]) - 1];
    const std::size_t sink_bytes = static_cast<std::size_t>(
        max_grid) * kBlockThreads * sizeof(v4u);
    v4u* source = nullptr;
    v4u* sink = nullptr;
    unsigned long long* errors = nullptr;
    CHECK(mcMalloc(&source, kBufferBytes));
    CHECK(mcMalloc(&sink, sink_bytes));
    CHECK(mcMalloc(&errors, sizeof(unsigned long long)));
    initialize_source<<<kExpectedAps * 8, kBlockThreads>>>(
        source, element_count);
    CHECK(mcDeviceSynchronize());

    for (const ModeSpec& spec : kModes) {
        const void* kernel = (
            spec.mode == Mode::kRegister
            ? reinterpret_cast<const void*>(c500_global_to_register_read)
            : reinterpret_cast<const void*>(c500_global_to_shared_read));
        mcFuncAttributes attributes;
        CHECK(mcFuncGetAttributes(&attributes, kernel));
        int resident_blocks = 0;
        CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, kernel, kBlockThreads, kDynamicSharedBytes));
        if (attributes.sharedSizeBytes != 0 || resident_blocks != 1) {
            std::fprintf(
                stderr,
                "resource contract failed for %s: registers=%d "
                "static_shared=%zu dynamic_shared=%d resident_blocks=%d\n",
                spec.name, attributes.numRegs,
                static_cast<std::size_t>(attributes.sharedSizeBytes),
                kDynamicSharedBytes, resident_blocks);
            return EXIT_FAILURE;
        }
        std::fprintf(
            stderr,
            "resource,%s,%s,registers=%d,static_shared=%zu,"
            "dynamic_shared=%d,resident_blocks_per_ap=%d\n",
            spec.name, spec.kernel_identity, attributes.numRegs,
            static_cast<std::size_t>(attributes.sharedSizeBytes),
            kDynamicSharedBytes, resident_blocks);
    }

    std::fprintf(
        stderr,
        "device=%s aps=%d wave=%d l2_bytes=%d buffer_bytes=%zu "
        "window=%d iterations=%d trials=%d\n",
        properties.name, properties.multiProcessorCount,
        properties.warpSize, properties.l2CacheSize, kBufferBytes,
        kRequestWindow, kIterations, kTrials);
    std::printf(
        "mode,kernel,grid,block,request_window,rounds,iterations,"
        "bytes_per_iteration,trial,event_ms,bandwidth_gbs\n");

    if (profile_once) {
        const long long thread_count = (
            static_cast<long long>(selected_grid) * kBlockThreads);
        const long long rounds = (
            element_count / (thread_count * kRequestWindow));
        launch_mode(selected_mode->mode, selected_grid, source, rounds, 1, sink);
        CHECK(mcDeviceSynchronize());
        std::this_thread::sleep_for(std::chrono::seconds(attach_wait_seconds));
        CHECK(mcProfilerStart());
        launch_mode(
            selected_mode->mode, selected_grid, source,
            rounds, kIterations, sink);
        CHECK(mcDeviceSynchronize());
        CHECK(mcProfilerStop());
        if (selected_mode->mode == Mode::kDirectToShared) {
            launch_mode(
                selected_mode->mode, selected_grid, source, rounds, 1, sink);
            CHECK(mcDeviceSynchronize());
            validate_direct_sink(
                source, sink, selected_grid, rounds, errors);
        }
        std::printf(
            "profile_once,%s,%s,%d,%d,%d,%lld,%d,%lld\n",
            selected_mode->name, selected_mode->kernel_identity,
            selected_grid, kBlockThreads, kRequestWindow, rounds,
            kIterations,
            rounds * thread_count * kRequestWindow
                * kVectorBytes * kIterations);
    } else if (single_case) {
        run_case(
            *selected_mode, selected_grid, source, element_count, sink,
            errors);
    } else {
        for (const ModeSpec& spec : kModes) {
            for (int grid : grids) {
                run_case(
                    spec, grid, source, element_count, sink, errors);
            }
        }
    }

    CHECK(mcFree(source));
    CHECK(mcFree(sink));
    CHECK(mcFree(errors));
    std::fprintf(stderr, "validation=pass\n");
    return EXIT_SUCCESS;
}
