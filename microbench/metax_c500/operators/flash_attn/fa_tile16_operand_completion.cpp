/*
 * MetaX C500 tile16 single-operand completion mechanism probe.
 *
 * This is not FlashAttention. It contains one source-shaped global-load,
 * CTA-barrier, and BF16-MMA consumer chain and excludes every other operator
 * stage and every production tensor shape.
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",               \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kThreads = kExpectedWaveSize;
constexpr int kWarmups = 3;
constexpr int kTrials = 7;

typedef __NATIVE_VECTOR__(2, int32_t) v2i;
typedef __NATIVE_VECTOR__(4, float) v4f;

struct alignas(16) Vector128 {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint32_t w;
};

static_assert(sizeof(Vector128) == 16);

__device__ __forceinline__ v4f mma_bf16(
        v2i lhs, v2i rhs, v4f accumulator) {
    return __builtin_mxc_mma_16x16x16bf16(lhs, rhs, accumulator);
}

__device__ __forceinline__ v2i low_half(const Vector128& value) {
    return {static_cast<int32_t>(value.x), static_cast<int32_t>(value.y)};
}

__device__ __forceinline__ v2i high_half(const Vector128& value) {
    return {static_cast<int32_t>(value.z), static_cast<int32_t>(value.w)};
}

template <int HeadDim>
__global__ void operand_completion(
        const Vector128* input, float* output,
        unsigned long long* block_cycles, int repetitions) {
    constexpr int vectors_per_operand = HeadDim == 64 ? 2 : 4;
    constexpr int mma_per_operand = 2 * vectors_per_operand;
    const int lane = static_cast<int>(threadIdx.x);
    const v2i rhs = {0x3b803c00, 0x3a803b00};
    v4f accumulators[mma_per_operand];
    #pragma unroll
    for (int index = 0; index < mma_per_operand; ++index) {
        const float seed = (
            0.125f + lane * 0.0009765625f + index * 0.015625f);
        accumulators[index] = {
            seed, seed + 0.125f, seed + 0.25f, seed + 0.5f};
    }
    __shared__ unsigned long long begin_cycle;
    if (lane == 0) begin_cycle = clock64();
    __syncthreads();
    #pragma unroll 1
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const std::size_t base = (
            (static_cast<std::size_t>(blockIdx.x) * repetitions
             + static_cast<std::size_t>(repetition)) * kThreads
            + static_cast<std::size_t>(lane)) * vectors_per_operand;
        Vector128 operand[vectors_per_operand];
        #pragma unroll
        for (int index = 0; index < vectors_per_operand; ++index) {
            operand[index] = input[base + index];
        }
        __syncthreads();
        #pragma unroll
        for (int index = 0; index < vectors_per_operand; ++index) {
            accumulators[2 * index] = mma_bf16(
                low_half(operand[index]), rhs, accumulators[2 * index]);
            accumulators[2 * index + 1] = mma_bf16(
                high_half(operand[index]), rhs,
                accumulators[2 * index + 1]);
        }
    }
    __syncthreads();
    if (lane == 0) block_cycles[blockIdx.x] = clock64() - begin_cycle;
    float checksum = 0.0f;
    #pragma unroll
    for (int index = 0; index < mma_per_operand; ++index) {
        checksum += accumulators[index].x + accumulators[index].y;
    }
    output[blockIdx.x * kThreads + lane] = checksum;
}

void require_device(int device) {
    CHECK(mcSetDevice(device));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device));
    if (properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize) {
        std::fprintf(stderr, "expected C500 AP=104 and wave=64\n");
        std::exit(EXIT_FAILURE);
    }
}

template <int HeadDim>
void run(int grid, int repetitions) {
    constexpr int vectors_per_operand = HeadDim == 64 ? 2 : 4;
    const std::size_t input_vectors = (
        static_cast<std::size_t>(grid) * repetitions * kThreads
        * vectors_per_operand);
    const std::size_t output_values = static_cast<std::size_t>(grid) * kThreads;
    Vector128* input = nullptr;
    float* output = nullptr;
    unsigned long long* block_cycles = nullptr;
    CHECK(mcMalloc(&input, input_vectors * sizeof(Vector128)));
    CHECK(mcMalloc(&output, output_values * sizeof(float)));
    CHECK(mcMalloc(&block_cycles, grid * sizeof(unsigned long long)));
    CHECK(mcMemset(input, 0x3b, input_vectors * sizeof(Vector128)));
    CHECK(mcMemset(output, 0, output_values * sizeof(float)));
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        operand_completion<HeadDim><<<grid, kThreads>>>(
            input, output, block_cycles, repetitions);
        CHECK(mcDeviceSynchronize());
    }
    std::vector<double> median_cycles;
    std::vector<double> maximum_cycles;
    std::vector<unsigned long long> host_cycles(grid);
    for (int trial = 0; trial < kTrials; ++trial) {
        operand_completion<HeadDim><<<grid, kThreads>>>(
            input, output, block_cycles, repetitions);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
        CHECK(mcMemcpy(host_cycles.data(), block_cycles,
                       grid * sizeof(unsigned long long),
                       mcMemcpyDeviceToHost));
        std::sort(host_cycles.begin(), host_cycles.end());
        median_cycles.push_back(
            static_cast<double>(host_cycles[host_cycles.size() / 2])
            / repetitions);
        maximum_cycles.push_back(
            static_cast<double>(host_cycles.back()) / repetitions);
    }
    std::sort(median_cycles.begin(), median_cycles.end());
    std::sort(maximum_cycles.begin(), maximum_cycles.end());
    std::printf("grid=%d head_dim=%d repetitions=%d input_bytes=%zu "
                "median_block_cycles_per_operand=%.6f "
                "maximum_block_cycles_per_operand=%.6f trials=",
                grid, HeadDim, repetitions,
                input_vectors * sizeof(Vector128),
                median_cycles[median_cycles.size() / 2],
                maximum_cycles[maximum_cycles.size() / 2]);
    for (std::size_t index = 0; index < median_cycles.size(); ++index) {
        std::printf("%s%.6f", index == 0 ? "" : ",", median_cycles[index]);
    }
    std::printf("\n");
    CHECK(mcFree(block_cycles));
    CHECK(mcFree(output));
    CHECK(mcFree(input));
}

int parse_positive(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 4096) {
        std::fprintf(stderr, "%s must be in [1, 4096]\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s DEVICE GRID HEAD_DIM REPETITIONS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = parse_positive(argv[1], "DEVICE") - 1;
    const int grid = parse_positive(argv[2], "GRID");
    const int head_dim = parse_positive(argv[3], "HEAD_DIM");
    const int repetitions = parse_positive(argv[4], "REPETITIONS");
    if (grid > kExpectedApCount) {
        std::fprintf(stderr, "GRID must not exceed 104 APs\n");
        return EXIT_FAILURE;
    }
    require_device(device);
    if (head_dim == 64) {
        run<64>(grid, repetitions);
    } else if (head_dim == 128) {
        run<128>(grid, repetitions);
    } else {
        std::fprintf(stderr, "HEAD_DIM must be 64 or 128\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
