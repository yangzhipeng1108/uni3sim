/*
 * MetaX C500 source-level scalar-math dependency benchmark.
 *
 * The two functions are operator-independent semantic stages used by common
 * activations: expf(-x) and 1/(1+x).  One runtime loop shared by every chain
 * length lets adjacent-count differences remove launch, event, loop-entry,
 * and terminal-store overhead without comparing differently unrolled
 * binaries.  Event batches cover both one 512-thread block and 416 blocks
 * (four blocks per C500 AP), exposing dependency latency and saturated
 * service throughput without any production tensor, production kernel, or
 * evaluation duration.  Inputs stay in a bounded range under repetition.
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kBlockThreads = 512;
constexpr int kWarmupsPerConfiguration = 8;
constexpr int kSamplesPerConfiguration = 31;
constexpr int kLaunchesPerBatch = 16;
constexpr int kSingleGrid = 1;
constexpr int kSaturatedGrid = 4 * kExpectedApCount;

enum class SemanticFunction : int {
    kExpNegF32 = 0,
    kReciprocalOnePlusF32 = 1,
};

template <SemanticFunction function>
struct Function;

template <>
struct Function<SemanticFunction::kExpNegF32> {
    __device__ __forceinline__ static float apply(float value) {
        return __builtin_expf(-value);
    }
};

template <>
struct Function<SemanticFunction::kReciprocalOnePlusF32> {
    __device__ __forceinline__ static float apply(float value) {
        return 1.0F / (1.0F + value);
    }
};

template <SemanticFunction function>
__global__ void dependent_kernel(float* sink, int count) {
    float value = 1.0009765625F
        + static_cast<float>(threadIdx.x & 1U) * 0.0009765625F;
    asm volatile("" : : : "memory");
    #pragma unroll 1
    for (int iteration = 0; iteration < count; ++iteration) {
        value = Function<function>::apply(value);
    }
    asm volatile("" : : "r"(value) : "memory");
    sink[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

const char* function_name(SemanticFunction function) {
    switch (function) {
        case SemanticFunction::kExpNegF32: return "exp_neg_f32";
        case SemanticFunction::kReciprocalOnePlusF32:
            return "reciprocal_one_plus_f32";
        default:
            std::fprintf(stderr, "unknown semantic function\n");
            std::exit(EXIT_FAILURE);
    }
}

template <SemanticFunction function>
void launch_count(int grid, int count, float* sink) {
    if (count != 64 && count != 128 && count != 256 && count != 512
            && count != 1024) {
        std::fprintf(stderr, "unsupported chain count: %d\n", count);
        std::exit(EXIT_FAILURE);
    }
    dependent_kernel<function><<<grid, kBlockThreads>>>(sink, count);
}

void launch(SemanticFunction function, int grid, int count, float* sink) {
    switch (function) {
        case SemanticFunction::kExpNegF32:
            launch_count<SemanticFunction::kExpNegF32>(grid, count, sink);
            return;
        case SemanticFunction::kReciprocalOnePlusF32:
            launch_count<SemanticFunction::kReciprocalOnePlusF32>(
                grid, count, sink);
            return;
        default:
            std::fprintf(stderr, "unknown semantic function\n");
            std::exit(EXIT_FAILURE);
    }
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--device-id"
            || std::string(argv[3]) != "--trial") {
        std::fprintf(
            stderr, "usage: %s --device-id DEVICE_ID --trial TRIAL\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[2], "device-id");
    const int trial = parse_nonnegative_int(argv[4], "trial");
    if (trial < 1 || trial > 3) {
        std::fprintf(stderr, "trial must be one of 1, 2, or 3\n");
        return EXIT_FAILURE;
    }
    CHECK(mcSetDevice(device_id));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device_id));
    if (properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize
            || properties.clockRate <= 0) {
        std::fprintf(
            stderr, "expected C500 104 AP/wave64 and a positive clock\n");
        return EXIT_FAILURE;
    }

    float* device_sink = nullptr;
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&device_sink),
        kSaturatedGrid * kBlockThreads * sizeof(float)));

    const SemanticFunction functions[] = {
        SemanticFunction::kExpNegF32,
        SemanticFunction::kReciprocalOnePlusF32,
    };
    // Keep every adjacent difference well above the event timer's observed
    // quantization floor.  This remains one runtime-counted binary: larger
    // counts improve measurement resolution without changing the mechanism.
    const int counts[] = {64, 128, 256, 512, 1024};
    for (SemanticFunction function : functions) {
        for (int grid : {kSingleGrid, kSaturatedGrid}) {
            for (int count : counts) {
                for (int warmup = 0;
                     warmup < kWarmupsPerConfiguration; ++warmup) {
                    launch(function, grid, count, device_sink);
                }
            }
        }
    }
    CHECK(mcDeviceSynchronize());

    std::vector<std::tuple<SemanticFunction, int, int, int>> plan;
    for (SemanticFunction function : functions) {
        for (int grid : {kSingleGrid, kSaturatedGrid}) {
            for (int count : counts) {
                for (int sample = 0;
                     sample < kSamplesPerConfiguration; ++sample) {
                    plan.emplace_back(function, grid, count, sample);
                }
            }
        }
    }
    std::mt19937 generator(500719U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,3,MetaX C500,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, kBlockThreads, trial,
        kSamplesPerConfiguration, kLaunchesPerBatch, kSingleGrid,
        kSaturatedGrid);
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (std::size_t order = 0; order < plan.size(); ++order) {
        SemanticFunction function;
        int grid = 0;
        int count = 0;
        int sample = 0;
        std::tie(function, grid, count, sample) = plan[order];
        CHECK(mcEventRecord(start));
        for (int launch_index = 0;
             launch_index < kLaunchesPerBatch; ++launch_index) {
            launch(function, grid, count, device_sink);
        }
        CHECK(mcEventRecord(end));
        CHECK(mcDeviceSynchronize());
        float elapsed_ms = 0.0F;
        CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
            std::fprintf(stderr, "semantic dependency event time invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "sample,%zu,%s,%d,%d,%d,%a\n",
            order, function_name(function), grid, count, sample,
            static_cast<double>(elapsed_ms));
    }
    std::fflush(stdout);

    CHECK(mcEventDestroy(end));
    CHECK(mcEventDestroy(start));
    CHECK(mcFree(device_sink));
    return EXIT_SUCCESS;
}
