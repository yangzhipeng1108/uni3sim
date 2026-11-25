/*
 * MetaX C500 BF16 vector-memory dependency-round benchmark.
 *
 * The two source rounds are generic BF16 vector primitives: BF16-to-FP32
 * conversion plus add/square accumulation, and BF16-to-FP32 conversion plus
 * scale/multiply and FP32-to-BF16 conversion. Both perform two aligned
 * 16-byte loads and one aligned 16-byte store per thread.
 * A runtime repetition count removes launch and fixed task cost.
 * The bounded warm working set isolates the CTA-local vector-memory service;
 * it is not a production operator and contains no reduction or model shape.
 */

#include <mc_runtime.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
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
constexpr int kMaximumThreads = 1024;
constexpr int kWarmupsPerGeometry = 4;
constexpr int kSamplesPerCell = 21;
constexpr int kLaunchesPerBatch = 8;

union PackedBf16Pair {
    std::uint32_t bits;
    __nv_bfloat162 values;
};

template <int Function>
__device__ __forceinline__ std::uint32_t vector_pair(
        std::uint32_t left_bits, std::uint32_t right_bits, float scale,
        float* carry) {
    PackedBf16Pair left;
    PackedBf16Pair right;
    PackedBf16Pair result;
    left.bits = left_bits;
    right.bits = right_bits;
    if constexpr (Function == 0) {
        const float x = (
            __low2float(left.values) + __low2float(right.values));
        const float y = (
            __high2float(left.values) + __high2float(right.values));
        *carry += (x * x + y * y) * 0.00000011920928955078125F;
        result.values = __floats2bfloat162_rn(x, y);
    } else if constexpr (Function == 1) {
        const float x = (
            __low2float(left.values) * scale
            * __low2float(right.values));
        const float y = (
            __high2float(left.values) * scale
            * __high2float(right.values));
        result.values = __floats2bfloat162_rn(x, y);
    } else {
        static_assert(Function == 0 || Function == 1, "unknown function");
    }
    return result.bits;
}

template <int Function>
__global__ void bf16_vector_memory_round(
        const volatile uint4* input_a, const volatile uint4* input_b,
        volatile uint4* output, int repetitions, int active_threads) {
    if (threadIdx.x >= active_threads) {
        return;
    }
    const int local_block = blockIdx.x % kExpectedApCount;
    const int index = local_block * blockDim.x + threadIdx.x;
    float carry = 1.0F
        + static_cast<float>(threadIdx.x & 7U) * 0.0009765625F;
    #pragma unroll 1
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        const uint4 left = make_uint4(
            input_a[index].x, input_a[index].y,
            input_a[index].z, input_a[index].w);
        const uint4 right = make_uint4(
            input_b[index].x, input_b[index].y,
            input_b[index].z, input_b[index].w);
        const float scale = 0.9990234375F + carry * 0.00000095367431640625F;
        uint4 result;
        result.x = vector_pair<Function>(left.x, right.x, scale, &carry);
        result.y = vector_pair<Function>(left.y, right.y, scale, &carry);
        result.z = vector_pair<Function>(left.z, right.z, scale, &carry);
        result.w = vector_pair<Function>(left.w, right.w, scale, &carry);
        output[index].x = result.x;
        output[index].y = result.y;
        output[index].z = result.z;
        output[index].w = result.w;
    }
    asm volatile("" : : "r"(carry) : "memory");
}

struct Cell {
    int function;
    int threads;
    int grid;
    int repetitions;
    int sample;
};

struct ResourceMetadata {
    int registers_per_thread;
    std::size_t shared_bytes_per_cta;
    int resident_256;
    int resident_512;
    int resident_1024;
};

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
        int function, int grid, int active_threads, int repetitions,
        const uint4* input_a, const uint4* input_b, uint4* output) {
    if (function == 0) {
        bf16_vector_memory_round<0><<<grid, kMaximumThreads>>>(
            input_a, input_b, output, repetitions, active_threads);
    } else if (function == 1) {
        bf16_vector_memory_round<1><<<grid, kMaximumThreads>>>(
            input_a, input_b, output, repetitions, active_threads);
    } else {
        std::fprintf(stderr, "unknown BF16 vector function: %d\n", function);
        std::exit(EXIT_FAILURE);
    }
}

float record_batch(
        mcEvent_t start, mcEvent_t end, int function, int grid, int threads,
        int repetitions, const uint4* input_a, const uint4* input_b,
        uint4* output) {
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(function, grid, threads, repetitions, input_a, input_b, output);
    }
    CHECK(mcEventRecord(start));
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(function, grid, threads, repetitions, input_a, input_b, output);
    }
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0F;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
        std::fprintf(stderr, "invalid mcEvent duration: %.9g ms\n", elapsed_ms);
        std::exit(EXIT_FAILURE);
    }
    return elapsed_ms;
}

template <int Function>
ResourceMetadata resource_metadata() {
    mcFuncAttributes attributes {};
    CHECK(mcFuncGetAttributes(
        &attributes,
        reinterpret_cast<const void*>(bf16_vector_memory_round<Function>)));
    ResourceMetadata result {
        attributes.numRegs,
        attributes.sharedSizeBytes,
        0,
        0,
        0,
    };
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.resident_1024, bf16_vector_memory_round<Function>, 1024, 0));
    result.resident_256 = result.resident_1024;
    result.resident_512 = result.resident_1024;
    return result;
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

    const ResourceMetadata add_square_resources = resource_metadata<0>();
    const ResourceMetadata scale_multiply_resources = resource_metadata<1>();
    if (add_square_resources.resident_256 != 2
            || add_square_resources.resident_512 != 2
            || add_square_resources.resident_1024 != 2
            || scale_multiply_resources.resident_256 != 2
            || scale_multiply_resources.resident_512 != 2
            || scale_multiply_resources.resident_1024 != 2) {
        std::fprintf(
            stderr,
            "expected production-equivalent occupancy 8/4/2, got "
            "add_square=%d/%d/%d scale_multiply=%d/%d/%d\n",
            add_square_resources.resident_256,
            add_square_resources.resident_512,
            add_square_resources.resident_1024,
            scale_multiply_resources.resident_256,
            scale_multiply_resources.resident_512,
            scale_multiply_resources.resident_1024);
        return EXIT_FAILURE;
    }

    const std::size_t elements =
        static_cast<std::size_t>(kExpectedApCount) * kMaximumThreads;
    const std::size_t bytes = elements * sizeof(uint4);
    std::vector<uint4> host(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        host[index] = make_uint4(
            0x3f803f80U, 0x3f003f00U, 0x3e803e80U, 0x3e003e00U);
    }
    uint4* input_a = nullptr;
    uint4* input_b = nullptr;
    uint4* output = nullptr;
    CHECK(mcMalloc(reinterpret_cast<void**>(&input_a), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&input_b), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&output), bytes));
    CHECK(mcMemcpy(input_a, host.data(), bytes, mcMemcpyHostToDevice));
    CHECK(mcMemcpy(input_b, host.data(), bytes, mcMemcpyHostToDevice));

    const int functions[] = {0, 1};
    const int blocks[] = {256, 512, 1024};
    const int grids[] = {
        1, 2, 4, 8, 16, 32, 64, 104, 208, 416, 832, 1664};
    const int repetition_counts[] = {4, 8, 16, 32};
    for (int function : functions) {
        for (int threads : blocks) {
            for (int grid : grids) {
                for (int repetitions : repetition_counts) {
                    for (int warmup = 0;
                         warmup < kWarmupsPerGeometry; ++warmup) {
                        launch(
                            function, grid, threads, repetitions,
                            input_a, input_b, output);
                    }
                }
            }
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (int function : functions) {
        for (int threads : blocks) {
            for (int grid : grids) {
                for (int repetitions : repetition_counts) {
                    for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                        plan.push_back({
                            function, threads, grid, repetitions, sample});
                    }
                }
            }
        }
    }
    std::mt19937 generator(500839U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,5,MetaX C500,%d,%d,%d,%d,%d,%d,%d,"
        "%d,%zu,%d,%d,%d,%d,%zu,%d,%d,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, kLaunchesPerBatch,
        add_square_resources.registers_per_thread,
        add_square_resources.shared_bytes_per_cta,
        add_square_resources.resident_256,
        add_square_resources.resident_512,
        add_square_resources.resident_1024,
        scale_multiply_resources.registers_per_thread,
        scale_multiply_resources.shared_bytes_per_cta,
        scale_multiply_resources.resident_256,
        scale_multiply_resources.resident_512,
        scale_multiply_resources.resident_1024);
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& item = plan[order];
        const float duration_ms = record_batch(
            start, end, item.function, item.grid, item.threads,
            item.repetitions,
            input_a, input_b, output);
        std::printf(
            "sample,%zu,%s,%d,%d,%d,%d,%a\n",
            order,
            item.function == 0 ? "add_square" : "scale_multiply",
            item.threads, item.grid, item.repetitions, item.sample,
            static_cast<double>(duration_ms));
    }
    CHECK(mcEventDestroy(end));
    CHECK(mcEventDestroy(start));

    launch(0, 1, 256, 4, input_a, input_b, output);
    CHECK(mcDeviceSynchronize());
    CHECK(mcMemcpy(host.data(), output, bytes, mcMemcpyDeviceToHost));
    if (host[0].x == 0U && host[0].y == 0U
            && host[0].z == 0U && host[0].w == 0U) {
        std::fprintf(stderr, "BF16 vector-memory output was not written\n");
        return EXIT_FAILURE;
    }

    CHECK(mcFree(output));
    CHECK(mcFree(input_b));
    CHECK(mcFree(input_a));
    return EXIT_SUCCESS;
}
