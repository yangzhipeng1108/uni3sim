/*
 * MetaX C500 source-level integer division/remainder dependency benchmark.
 *
 * Runtime divisors and register-resident dependent chains isolate the scalar
 * integer mechanisms used by address calculations.  This benchmark contains
 * no production tensor, operator, launch geometry, or evaluation duration.
 */

#include <mc_runtime.h>

#include <algorithm>
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
constexpr int kBlockThreads = 64;
constexpr int kWarmupsPerConfiguration = 8;
constexpr int kSamplesPerConfiguration = 31;

enum class SemanticFunction : int {
    kDivU32 = 0,
    kRemU32 = 1,
    kDivU64 = 2,
    kRemU64 = 3,
};

template <SemanticFunction function>
struct Function;

template <>
struct Function<SemanticFunction::kDivU32> {
    using Value = std::uint32_t;
    __device__ __forceinline__ static Value apply(Value value, Value divisor) {
        return value / divisor;
    }
};

template <>
struct Function<SemanticFunction::kRemU32> {
    using Value = std::uint32_t;
    __device__ __forceinline__ static Value apply(Value value, Value divisor) {
        return value % divisor;
    }
};

template <>
struct Function<SemanticFunction::kDivU64> {
    using Value = std::uint64_t;
    __device__ __forceinline__ static Value apply(Value value, Value divisor) {
        return value / divisor;
    }
};

template <>
struct Function<SemanticFunction::kRemU64> {
    using Value = std::uint64_t;
    __device__ __forceinline__ static Value apply(Value value, Value divisor) {
        return value % divisor;
    }
};

template <SemanticFunction function>
__global__ void dependent_kernel(
        std::uint64_t* elapsed_cycles, std::uint64_t* sink,
        typename Function<function>::Value divisor, int count) {
    using Value = typename Function<function>::Value;
    Value value = static_cast<Value>(0x6f12345U + threadIdx.x * 17U);
    asm volatile("" : : : "memory");
    const std::uint64_t start = clock64();
    #pragma unroll 1
    for (int iteration = 0; iteration < count; ++iteration) {
        value = Function<function>::apply(value, divisor);
    }
    asm volatile("" : : "r"(value) : "memory");
    const std::uint64_t end = clock64();
    if (threadIdx.x == 0) {
        elapsed_cycles[0] = end - start;
        sink[0] = static_cast<std::uint64_t>(value);
    }
}

const char* function_name(SemanticFunction function) {
    switch (function) {
        case SemanticFunction::kDivU32: return "div_u32";
        case SemanticFunction::kRemU32: return "rem_u32";
        case SemanticFunction::kDivU64: return "div_u64";
        case SemanticFunction::kRemU64: return "rem_u64";
        default:
            std::fprintf(stderr, "unknown semantic function\n");
            std::exit(EXIT_FAILURE);
    }
}

template <SemanticFunction function>
void launch_count(
        int count, std::uint64_t* cycles, std::uint64_t* sink) {
    using Value = typename Function<function>::Value;
    const Value divisor = static_cast<Value>(127U);
    if (count != 4 && count != 8 && count != 16 && count != 32
            && count != 64) {
        std::fprintf(stderr, "unsupported chain count: %d\n", count);
        std::exit(EXIT_FAILURE);
    }
    dependent_kernel<function><<<1, kBlockThreads>>>(
        cycles, sink, divisor, count);
}

void launch(
        SemanticFunction function, int count,
        std::uint64_t* cycles, std::uint64_t* sink) {
    switch (function) {
        case SemanticFunction::kDivU32:
            launch_count<SemanticFunction::kDivU32>(count, cycles, sink);
            return;
        case SemanticFunction::kRemU32:
            launch_count<SemanticFunction::kRemU32>(count, cycles, sink);
            return;
        case SemanticFunction::kDivU64:
            launch_count<SemanticFunction::kDivU64>(count, cycles, sink);
            return;
        case SemanticFunction::kRemU64:
            launch_count<SemanticFunction::kRemU64>(count, cycles, sink);
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

    std::uint64_t* device_cycles = nullptr;
    std::uint64_t* device_sink = nullptr;
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&device_cycles), sizeof(std::uint64_t)));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&device_sink), sizeof(std::uint64_t)));

    const SemanticFunction functions[] = {
        SemanticFunction::kDivU32,
        SemanticFunction::kRemU32,
        SemanticFunction::kDivU64,
        SemanticFunction::kRemU64,
    };
    const int counts[] = {4, 8, 16, 32, 64};
    for (SemanticFunction function : functions) {
        for (int count : counts) {
            for (int warmup = 0;
                 warmup < kWarmupsPerConfiguration; ++warmup) {
                launch(function, count, device_cycles, device_sink);
            }
        }
    }
    CHECK(mcDeviceSynchronize());

    std::vector<std::tuple<SemanticFunction, int, int>> plan;
    for (SemanticFunction function : functions) {
        for (int count : counts) {
            for (int sample = 0;
                 sample < kSamplesPerConfiguration; ++sample) {
                plan.emplace_back(function, count, sample);
            }
        }
    }
    std::mt19937 generator(500729U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,1,MetaX C500,%d,%d,%d,%d,%d,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerConfiguration);
    for (std::size_t order = 0; order < plan.size(); ++order) {
        SemanticFunction function;
        int count = 0;
        int sample = 0;
        std::tie(function, count, sample) = plan[order];
        launch(function, count, device_cycles, device_sink);
        CHECK(mcDeviceSynchronize());
        std::uint64_t elapsed_cycles = 0;
        CHECK(mcMemcpy(
            &elapsed_cycles, device_cycles, sizeof(elapsed_cycles),
            mcMemcpyDeviceToHost));
        if (elapsed_cycles == 0) {
            std::fprintf(stderr, "integer dependency elapsed zero cycles\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "sample,%zu,%s,%d,%d,%llu\n",
            order, function_name(function), count, sample,
            static_cast<unsigned long long>(elapsed_cycles));
    }
    std::fflush(stdout);

    CHECK(mcFree(device_sink));
    CHECK(mcFree(device_cycles));
    return EXIT_SUCCESS;
}
