/*
 * MetaX C500 active-wave SiLU-like dependency scoreboard benchmark.
 *
 * This is a mechanism benchmark, not an implementation of a production
 * operator.  The timed dependency round is a fixed register DAG containing
 * arithmetic -> exp_f32 -> arithmetic -> rcp_f32 -> arithmetic.  The two
 * memory variants additionally execute aligned ldg(A), ldg(B), and stg(C)
 * operations over a synthetic 4-MiB or 64-MiB power-of-two footprint.
 *
 * Every planned sample is executed twice with the same binary/configuration.
 * The --timing path has no profiler API and publishes AP-local clock64 XCORE
 * cycles.  The --profile-once path places exactly one launch in the profiler
 * marker and supplies counters only; its instrumented cycles and duration are
 * never used by the published C_round signature.
 *
 * Build in the prescribed MACA 3.7.2 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       -o silu_wave_scoreboard silu_wave_scoreboard.cpp
 */

#include <mc_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kDagRounds = 40;
constexpr int kWarmupLaunches = 4;
constexpr std::size_t kL2WorkingSetBytes = 4ULL * 1024 * 1024;
constexpr std::size_t kStreamingWorkingSetBytes = 64ULL * 1024 * 1024;

enum class Configuration : int {
    kRegisterQ1 = 0,
    kRegisterQ8 = 1,
    kRegisterQ32 = 2,
    kL2Q32 = 3,
    kStreamingQ32 = 4,
};

struct ConfigurationSpec {
    Configuration configuration;
    const char* name;
    const char* cache_state;
    int grid;
    int block;
    int waves_per_ap;
    std::size_t working_set_bytes;
};

constexpr ConfigurationSpec kConfigurations[] = {
    {Configuration::kRegisterQ1, "register_q1", "register_only",
     kExpectedApCount, 64, 1, 0},
    {Configuration::kRegisterQ8, "register_q8", "register_only",
     kExpectedApCount, 512, 8, 0},
    {Configuration::kRegisterQ32, "register_q32", "register_only",
     kExpectedApCount * 4, 512, 32, 0},
    {Configuration::kL2Q32, "l2_q32", "l2_resident_4mib",
     kExpectedApCount * 4, 512, 32, kL2WorkingSetBytes},
    {Configuration::kStreamingQ32, "streaming_q32", "streaming_64mib",
     kExpectedApCount * 4, 512, 32, kStreamingWorkingSetBytes},
};

__device__ __forceinline__ float exp_f32(float value) {
    float output;
    asm volatile("exp_f32 %0, neg(%1)" : "=r"(output) : "r"(value));
    return output;
}

__device__ __forceinline__ float rcp_f32(float value) {
    float output;
    asm volatile("rcp_f32 %0, %1" : "=r"(output) : "r"(value));
    return output;
}

__device__ __forceinline__ float dependency_round(
        float value, float injected) {
    const float shifted = value * 0.125f + injected * 0.03125f + 0.25f;
    const float exponential = exp_f32(shifted);
    const float reciprocal = rcp_f32(1.0f + exponential);
    return shifted * reciprocal + 0.000244140625f;
}

template <bool use_memory>
__device__ __forceinline__ void execute_scoreboard(
        const float* input_a, const float* input_b, float* output_c,
        std::size_t input_element_mask, std::size_t output_element_mask,
        std::uint64_t* elapsed_cycles, float* sink) {
    const int wave_in_cta = static_cast<int>(threadIdx.x) / kExpectedWaveSize;
    const int waves_per_cta = static_cast<int>(blockDim.x) / kExpectedWaveSize;
    const std::size_t global_wave = (
        static_cast<std::size_t>(blockIdx.x) * waves_per_cta + wave_in_cta);
    const std::size_t global_thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    const std::size_t thread_count = (
        static_cast<std::size_t>(gridDim.x) * blockDim.x);
    float value = 0.125f
        + static_cast<float>(threadIdx.x & 15U) * 0.0009765625f;
    asm volatile("" : "+r"(value) : : "memory");
    const std::uint64_t start = clock64();
    #pragma unroll 1
    for (int round = 0; round < kDagRounds; ++round) {
        float injected = 0.5f;
        std::size_t index = 0;
        if constexpr (use_memory) {
            index = (global_thread
                     + static_cast<std::size_t>(round) * thread_count)
                & input_element_mask;
            injected = input_a[index] + input_b[index];
        }
        value = dependency_round(value, injected);
        if constexpr (use_memory) {
            const std::size_t destination = (
                global_thread
                + static_cast<std::size_t>(round) * thread_count)
                & output_element_mask;
            output_c[destination] = value;
        }
    }
    asm volatile("" : : "r"(value) : "memory");
    const std::uint64_t end = clock64();
    if ((threadIdx.x & (kExpectedWaveSize - 1)) == 0) {
        elapsed_cycles[global_wave] = end - start;
        sink[global_wave] = value;
    }
}

__global__ void silu_wave_reg_q1(
        std::uint64_t* elapsed_cycles, float* sink) {
    execute_scoreboard<false>(
        nullptr, nullptr, nullptr, 0, 0, elapsed_cycles, sink);
}

__global__ void silu_wave_reg_q8(
        std::uint64_t* elapsed_cycles, float* sink) {
    execute_scoreboard<false>(
        nullptr, nullptr, nullptr, 0, 0, elapsed_cycles, sink);
}

__global__ void silu_wave_reg_q32(
        std::uint64_t* elapsed_cycles, float* sink) {
    execute_scoreboard<false>(
        nullptr, nullptr, nullptr, 0, 0, elapsed_cycles, sink);
}

__global__ void silu_wave_l2_q32(
        const float* input_a, const float* input_b, float* output_c,
        std::size_t input_element_mask, std::size_t output_element_mask,
        std::uint64_t* elapsed_cycles, float* sink) {
    execute_scoreboard<true>(
        input_a, input_b, output_c, input_element_mask, output_element_mask,
        elapsed_cycles, sink);
}

__global__ void silu_wave_dram_q32(
        const float* input_a, const float* input_b, float* output_c,
        std::size_t input_element_mask, std::size_t output_element_mask,
        std::uint64_t* elapsed_cycles, float* sink) {
    execute_scoreboard<true>(
        input_a, input_b, output_c, input_element_mask, output_element_mask,
        elapsed_cycles, sink);
}

const ConfigurationSpec& parse_configuration(const char* text) {
    for (const ConfigurationSpec& spec : kConfigurations) {
        if (std::strcmp(spec.name, text) == 0) {
            return spec;
        }
    }
    std::fprintf(stderr, "unsupported configuration: %s\n", text);
    std::exit(EXIT_FAILURE);
}

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

void launch(
        const ConfigurationSpec& spec, float* buffer,
        std::uint64_t* elapsed_cycles, float* sink) {
    const std::size_t input_elements = (
        spec.working_set_bytes / 4 / sizeof(float));
    const std::size_t output_elements = (
        spec.working_set_bytes / 2 / sizeof(float));
    const std::size_t input_element_mask = (
        input_elements == 0 ? 0 : input_elements - 1);
    const std::size_t output_element_mask = (
        output_elements == 0 ? 0 : output_elements - 1);
    const float* input_a = buffer;
    const float* input_b = (
        buffer == nullptr ? nullptr : buffer + input_elements);
    float* output_c = (
        buffer == nullptr ? nullptr : buffer + 2 * input_elements);
    switch (spec.configuration) {
        case Configuration::kRegisterQ1:
            silu_wave_reg_q1<<<spec.grid, spec.block>>>(elapsed_cycles, sink);
            return;
        case Configuration::kRegisterQ8:
            silu_wave_reg_q8<<<spec.grid, spec.block>>>(elapsed_cycles, sink);
            return;
        case Configuration::kRegisterQ32:
            silu_wave_reg_q32<<<spec.grid, spec.block>>>(elapsed_cycles, sink);
            return;
        case Configuration::kL2Q32:
            silu_wave_l2_q32<<<spec.grid, spec.block>>>(
                input_a, input_b, output_c,
                input_element_mask, output_element_mask,
                elapsed_cycles, sink);
            return;
        case Configuration::kStreamingQ32:
            silu_wave_dram_q32<<<spec.grid, spec.block>>>(
                input_a, input_b, output_c,
                input_element_mask, output_element_mask,
                elapsed_cycles, sink);
            return;
        default:
            std::fprintf(stderr, "unknown configuration\n");
            std::exit(EXIT_FAILURE);
    }
}

void read_resources(
        const ConfigurationSpec& spec, mcFuncAttributes* attributes,
        int* resident_ctas_per_ap) {
    switch (spec.configuration) {
        case Configuration::kRegisterQ1:
            CHECK(mcFuncGetAttributes(
                attributes,
                reinterpret_cast<const void*>(silu_wave_reg_q1)));
            CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
                resident_ctas_per_ap, silu_wave_reg_q1, spec.block, 0));
            return;
        case Configuration::kRegisterQ8:
            CHECK(mcFuncGetAttributes(
                attributes,
                reinterpret_cast<const void*>(silu_wave_reg_q8)));
            CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
                resident_ctas_per_ap, silu_wave_reg_q8, spec.block, 0));
            return;
        case Configuration::kRegisterQ32:
            CHECK(mcFuncGetAttributes(
                attributes,
                reinterpret_cast<const void*>(silu_wave_reg_q32)));
            CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
                resident_ctas_per_ap, silu_wave_reg_q32, spec.block, 0));
            return;
        case Configuration::kL2Q32:
            CHECK(mcFuncGetAttributes(
                attributes,
                reinterpret_cast<const void*>(silu_wave_l2_q32)));
            CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
                resident_ctas_per_ap, silu_wave_l2_q32, spec.block, 0));
            return;
        case Configuration::kStreamingQ32:
            CHECK(mcFuncGetAttributes(
                attributes,
                reinterpret_cast<const void*>(silu_wave_dram_q32)));
            CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
                resident_ctas_per_ap, silu_wave_dram_q32, spec.block, 0));
            return;
        default:
            std::fprintf(stderr, "unknown configuration\n");
            std::exit(EXIT_FAILURE);
    }
}

void run_timing(
        const ConfigurationSpec& spec, float* device_buffer,
        std::uint64_t* device_cycles, float* device_sink) {
    launch(spec, device_buffer, device_cycles, device_sink);
    CHECK(mcDeviceSynchronize());
}

void run_profile_once(
        const ConfigurationSpec& spec, float* device_buffer,
        std::uint64_t* device_cycles, float* device_sink,
        int attach_wait_seconds) {
    std::this_thread::sleep_for(std::chrono::seconds(attach_wait_seconds));
    CHECK(mcProfilerStart());
    launch(spec, device_buffer, device_cycles, device_sink);
    CHECK(mcDeviceSynchronize());
    CHECK(mcProfilerStop());
}

}  // namespace

int main(int argc, char** argv) {
    const bool timing_mode = (
        argc == 10 && std::string(argv[1]) == "--timing"
        && std::string(argv[2]) == "--device-id"
        && std::string(argv[4]) == "--trial"
        && std::string(argv[6]) == "--order-position"
        && std::string(argv[8]) == "--configuration");
    const bool profile_mode = (
        argc == 12 && std::string(argv[1]) == "--profile-once"
        && std::string(argv[2]) == "--device-id"
        && std::string(argv[4]) == "--trial"
        && std::string(argv[6]) == "--order-position"
        && std::string(argv[8]) == "--configuration"
        && std::string(argv[10]) == "--attach-wait-seconds");
    if (!timing_mode && !profile_mode) {
        std::fprintf(
            stderr,
            "usage: %s --timing --device-id ID --trial TRIAL "
            "--order-position POS --configuration CONFIG | "
            "--profile-once --device-id ID --trial TRIAL "
            "--order-position POS --configuration CONFIG "
            "--attach-wait-seconds SECONDS\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[3], "device-id");
    const int trial = parse_nonnegative_int(argv[5], "trial");
    const int order_position = parse_nonnegative_int(
        argv[7], "order-position");
    const ConfigurationSpec& spec = parse_configuration(argv[9]);
    const int attach_wait_seconds = (
        timing_mode ? 0
        : parse_nonnegative_int(argv[11], "attach-wait-seconds"));
    if (trial < 1 || trial > 3 || order_position >= 5
            || (profile_mode && attach_wait_seconds <= 0)) {
        std::fprintf(stderr, "trial/order-position/attach wait is invalid\n");
        return EXIT_FAILURE;
    }

    CHECK(mcSetDevice(device_id));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device_id));
    if (std::strstr(properties.name, "C500") == nullptr
            || properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize
            || properties.l2CacheSize != 8 * 1024 * 1024
            || properties.clockRate <= 0) {
        std::fprintf(
            stderr, "unsupported device name=%s aps=%d wave=%d l2=%d\n",
            properties.name, properties.multiProcessorCount,
            properties.warpSize, properties.l2CacheSize);
        return EXIT_FAILURE;
    }
    if ((spec.working_set_bytes != 0)
            && ((spec.working_set_bytes & (spec.working_set_bytes - 1)) != 0
                || spec.working_set_bytes % sizeof(float) != 0)) {
        std::fprintf(stderr, "working set must be a float-aligned power of two\n");
        return EXIT_FAILURE;
    }

    mcFuncAttributes attributes {};
    int resident_ctas_per_ap = 0;
    read_resources(spec, &attributes, &resident_ctas_per_ap);
    const int required_ctas_per_ap = spec.grid / kExpectedApCount;
    if (resident_ctas_per_ap < required_ctas_per_ap) {
        std::fprintf(
            stderr, "configuration requires %d resident CTAs/AP but has %d\n",
            required_ctas_per_ap, resident_ctas_per_ap);
        return EXIT_FAILURE;
    }

    const std::size_t wave_count = (
        static_cast<std::size_t>(spec.grid) * spec.block / kExpectedWaveSize);
    std::uint64_t* device_cycles = nullptr;
    float* device_sink = nullptr;
    float* device_buffer = nullptr;
    CHECK(mcMalloc(&device_cycles, wave_count * sizeof(std::uint64_t)));
    CHECK(mcMalloc(&device_sink, wave_count * sizeof(float)));
    if (spec.working_set_bytes != 0) {
        CHECK(mcMalloc(&device_buffer, spec.working_set_bytes));
        CHECK(mcMemset(device_buffer, 0, spec.working_set_bytes));
    }

    for (int warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        launch(spec, device_buffer, device_cycles, device_sink);
    }
    CHECK(mcDeviceSynchronize());
    if (timing_mode) {
        run_timing(spec, device_buffer, device_cycles, device_sink);
    } else if (profile_mode) {
        run_profile_once(
            spec, device_buffer, device_cycles, device_sink,
            attach_wait_seconds);
    } else {
        std::abort();
    }

    std::vector<std::uint64_t> cycles(wave_count);
    std::vector<float> sink(wave_count);
    CHECK(mcMemcpy(
        cycles.data(), device_cycles, wave_count * sizeof(std::uint64_t),
        mcMemcpyDeviceToHost));
    CHECK(mcMemcpy(
        sink.data(), device_sink, wave_count * sizeof(float),
        mcMemcpyDeviceToHost));
    if (std::any_of(cycles.begin(), cycles.end(),
                    [](std::uint64_t value) { return value == 0; })) {
        std::fprintf(stderr, "one or more waves reported zero cycles\n");
        return EXIT_FAILURE;
    }
    std::sort(cycles.begin(), cycles.end());
    const std::uint64_t median_cycles = cycles[cycles.size() / 2];
    double checksum = 0.0;
    for (float value : sink) {
        checksum += static_cast<double>(value);
    }
    if (!(checksum > 0.0)) {
        std::fprintf(stderr, "validation checksum is not positive\n");
        return EXIT_FAILURE;
    }

    std::printf(
        "metadata,2,%s,MetaX C500,%d,%d,%d,%d,%d,%d\n",
        timing_mode ? "timing" : "profile_once",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, 5);
    std::printf(
        "%s,%d,%s,%s,%d,%d,%d,%d,%d,%zu,%zu,%llu,%llu,%llu,%.9g,%d,%zu\n",
        timing_mode ? "timing_sample" : "profile_sample",
        order_position, spec.name, spec.cache_state, spec.grid, spec.block,
        spec.block / kExpectedWaveSize, spec.waves_per_ap, kDagRounds,
        spec.working_set_bytes, wave_count,
        static_cast<unsigned long long>(median_cycles),
        static_cast<unsigned long long>(cycles.front()),
        static_cast<unsigned long long>(cycles.back()), checksum,
        attributes.numRegs, attributes.sharedSizeBytes);
    std::fflush(stdout);

    if (device_buffer != nullptr) {
        CHECK(mcFree(device_buffer));
    }
    CHECK(mcFree(device_sink));
    CHECK(mcFree(device_cycles));
    return EXIT_SUCCESS;
}
