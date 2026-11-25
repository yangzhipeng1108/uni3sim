/*
 * MetaX C500 register-only SFU dependency benchmark.
 *
 * This benchmark uses statically unrolled dependent chains and clock64 so a
 * compiler loop/control recurrence is not mistaken for primitive latency.
 * It does not execute a production operator, touch tensor memory, use shared
 * memory, or consume a production/evaluation duration.
 *
 * Build in the prescribed MACA 3.7.2 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 primitive_dependency.cpp \
 *       -o primitive_dependency
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
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
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

enum class Primitive : int {
    kExpF32 = 0,
    kRcpF32 = 1,
    kRsqF32 = 2,
    kRcpiF32 = 3,
};

template <Primitive primitive>
struct Instruction;

template <>
struct Instruction<Primitive::kExpF32> {
    __device__ __forceinline__ static float apply(float value) {
        float output;
        asm volatile("exp_f32 %0, neg(%1)"
                     : "=r"(output) : "r"(value));
        return output;
    }
};

template <>
struct Instruction<Primitive::kRcpF32> {
    __device__ __forceinline__ static float apply(float value) {
        float output;
        asm volatile("rcp_f32 %0, %1" : "=r"(output) : "r"(value));
        return output;
    }
};

template <>
struct Instruction<Primitive::kRsqF32> {
    __device__ __forceinline__ static float apply(float value) {
        float output;
        asm volatile("rsq_f32 %0, %1" : "=r"(output) : "r"(value));
        return output;
    }
};

template <>
struct Instruction<Primitive::kRcpiF32> {
    __device__ __forceinline__ static float apply(float value) {
        float output;
        asm volatile("rcpi_f32 %0, %1" : "=r"(output) : "r"(value));
        return output;
    }
};

template <Primitive primitive, int count>
struct DependentChain {
    __device__ __forceinline__ static float run(float value) {
        return DependentChain<primitive, count - 1>::run(
            Instruction<primitive>::apply(value));
    }
};

template <Primitive primitive>
struct DependentChain<primitive, 0> {
    __device__ __forceinline__ static float run(float value) {
        return value;
    }
};

template <Primitive primitive, int count>
__global__ void dependent_kernel(
        std::uint64_t* elapsed_cycles, float* sink) {
    static_assert(
        count == 4 || count == 8 || count == 16 || count == 32
        || count == 64,
        "unsupported primitive dependency chain length");
    float value = 1.0009765625f
        + static_cast<float>(threadIdx.x & 1U) * 0.0009765625f;
    asm volatile("" : : : "memory");
    const std::uint64_t start = clock64();
    value = DependentChain<primitive, count>::run(value);
    asm volatile("" : : "r"(value) : "memory");
    const std::uint64_t end = clock64();
    if (threadIdx.x == 0) {
        elapsed_cycles[0] = end - start;
        sink[0] = value;
    }
}

const char* primitive_name(Primitive primitive) {
    switch (primitive) {
        case Primitive::kExpF32: return "exp_f32";
        case Primitive::kRcpF32: return "rcp_f32";
        case Primitive::kRsqF32: return "rsq_f32";
        case Primitive::kRcpiF32: return "rcpi_f32";
        default:
            std::fprintf(stderr, "unknown primitive\n");
            std::exit(EXIT_FAILURE);
    }
}

template <Primitive primitive>
void launch_count(int count, std::uint64_t* cycles, float* sink) {
    switch (count) {
        case 4:
            dependent_kernel<primitive, 4><<<1, kBlockThreads>>>(
                cycles, sink);
            return;
        case 8:
            dependent_kernel<primitive, 8><<<1, kBlockThreads>>>(
                cycles, sink);
            return;
        case 16:
            dependent_kernel<primitive, 16><<<1, kBlockThreads>>>(
                cycles, sink);
            return;
        case 32:
            dependent_kernel<primitive, 32><<<1, kBlockThreads>>>(
                cycles, sink);
            return;
        case 64:
            dependent_kernel<primitive, 64><<<1, kBlockThreads>>>(
                cycles, sink);
            return;
        default:
            std::fprintf(stderr, "unsupported chain count: %d\n", count);
            std::exit(EXIT_FAILURE);
    }
}

void launch(
        Primitive primitive, int count, std::uint64_t* cycles, float* sink) {
    switch (primitive) {
        case Primitive::kExpF32:
            launch_count<Primitive::kExpF32>(count, cycles, sink); return;
        case Primitive::kRcpF32:
            launch_count<Primitive::kRcpF32>(count, cycles, sink); return;
        case Primitive::kRsqF32:
            launch_count<Primitive::kRsqF32>(count, cycles, sink); return;
        case Primitive::kRcpiF32:
            launch_count<Primitive::kRcpiF32>(count, cycles, sink); return;
        default:
            std::fprintf(stderr, "unknown primitive\n");
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
            stderr,
            "usage: %s --device-id DEVICE_ID --trial TRIAL\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[2], "device-id");
    const int trial = parse_nonnegative_int(argv[4], "trial");
    if (trial == 0) {
        std::fprintf(stderr, "trial must be positive\n");
        return EXIT_FAILURE;
    }
    CHECK(mcSetDevice(device_id));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device_id));
    if (properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize) {
        std::fprintf(
            stderr, "expected C500 104 AP/wave64, got %d AP/wave%d\n",
            properties.multiProcessorCount, properties.warpSize);
        return EXIT_FAILURE;
    }

    std::uint64_t* device_cycles = nullptr;
    float* device_sink = nullptr;
    CHECK(mcMalloc(&device_cycles, sizeof(std::uint64_t)));
    CHECK(mcMalloc(&device_sink, sizeof(float)));

    const Primitive primitives[] = {
        Primitive::kExpF32, Primitive::kRcpF32,
        Primitive::kRsqF32, Primitive::kRcpiF32,
    };
    const int counts[] = {4, 8, 16, 32, 64};
    for (Primitive primitive : primitives) {
        for (int count : counts) {
            for (int warmup = 0;
                    warmup < kWarmupsPerConfiguration; ++warmup) {
                launch(primitive, count, device_cycles, device_sink);
            }
        }
    }
    CHECK(mcDeviceSynchronize());

    std::vector<std::tuple<Primitive, int, int>> plan;
    for (Primitive primitive : primitives) {
        for (int count : counts) {
            for (int sample = 0;
                    sample < kSamplesPerConfiguration; ++sample) {
                plan.emplace_back(primitive, count, sample);
            }
        }
    }
    std::mt19937 generator(500373U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,1,MetaX C500,%d,%d,%d,%d,%d,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerConfiguration);
    for (std::size_t order = 0; order < plan.size(); ++order) {
        Primitive primitive;
        int count = 0;
        int sample = 0;
        std::tie(primitive, count, sample) = plan[order];
        launch(primitive, count, device_cycles, device_sink);
        CHECK(mcDeviceSynchronize());
        std::uint64_t elapsed_cycles = 0;
        CHECK(mcMemcpy(
            &elapsed_cycles, device_cycles, sizeof(elapsed_cycles),
            mcMemcpyDeviceToHost));
        if (elapsed_cycles == 0) {
            std::fprintf(stderr, "primitive dependency elapsed zero cycles\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "sample,%zu,%s,%d,%d,%llu\n",
            order, primitive_name(primitive), count, sample,
            static_cast<unsigned long long>(elapsed_cycles));
    }
    std::fflush(stdout);

    CHECK(mcFree(device_sink));
    CHECK(mcFree(device_cycles));
    return EXIT_SUCCESS;
}
