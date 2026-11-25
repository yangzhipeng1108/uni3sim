/*
 * MetaX C500 physical-block256 scalar AddRMSNorm source-round benchmark.
 *
 * This is not a complete production operator.  It isolates the two scalar
 * BF16 loops on a real 256-thread CTA, without BlockReduce, reciprocal square
 * root, epsilon, or the publication barrier.  Runtime round count and grid
 * depth identify first, resident, and queued CTA-local loop services without
 * consuming production/evaluation duration.
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
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kThreads = 256;
constexpr int kMaximumRounds = 64;
constexpr int kWarmupsPerGeometry = 2;
constexpr int kSamplesPerCell = 15;
constexpr int kLaunchesPerBatch = 8;

union ScalarBf16 {
    std::uint16_t bits;
    __nv_bfloat16 value;
};

__device__ __forceinline__ __nv_bfloat16 load_bf16(
        const volatile std::uint16_t* source, int index) {
    ScalarBf16 packed;
    packed.bits = source[index];
    return packed.value;
}

__device__ __forceinline__ std::uint16_t store_bf16(__nv_bfloat16 value) {
    ScalarBf16 packed;
    packed.value = value;
    return packed.bits;
}

template <int Function>
__global__ void add_rmsnorm_block256_scalar_round(
        volatile std::uint16_t* input,
        volatile std::uint16_t* residual,
        const volatile std::uint16_t* weight,
        volatile float* checksum, int rounds, int active_threads,
        float scale) {
    static_assert(Function == 0 || Function == 1, "unknown function");
    if (threadIdx.x >= active_threads) {
        return;
    }
    const int local_block = blockIdx.x % kExpectedApCount;
    const int row_offset = local_block * kMaximumRounds * kThreads;
    float variance = 0.0F;
    #pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
        const int index = row_offset + round * kThreads + threadIdx.x;
        if constexpr (Function == 0) {
            __nv_bfloat16 z = load_bf16(input, index);
            z += load_bf16(residual, index);
            const float x = __bfloat162float(z);
            variance += x * x;
            residual[index] = store_bf16(z);
        } else {
            const float x = __bfloat162float(load_bf16(residual, index));
            const float gamma = __bfloat162float(
                load_bf16(weight, round * kThreads + threadIdx.x));
            input[index] = store_bf16(
                __float2bfloat16(x * scale) * __float2bfloat16(gamma));
        }
    }
    if constexpr (Function == 0) {
        checksum[local_block * kThreads + threadIdx.x] = variance;
    }
}

struct Cell {
    int function;
    int active_threads;
    int grid;
    int rounds;
    int sample;
};

struct ResourceMetadata {
    int registers_per_thread;
    std::size_t shared_bytes_per_cta;
    int resident_ctas_per_ap;
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

template <int Function>
ResourceMetadata resource_metadata() {
    mcFuncAttributes attributes {};
    CHECK(mcFuncGetAttributes(
        &attributes, reinterpret_cast<const void*>(
            add_rmsnorm_block256_scalar_round<Function>)));
    ResourceMetadata result {attributes.numRegs, attributes.sharedSizeBytes, 0};
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.resident_ctas_per_ap,
        add_rmsnorm_block256_scalar_round<Function>, kThreads, 0));
    return result;
}

void launch(
        int function, int grid, int active_threads, int rounds,
        std::uint16_t* input, std::uint16_t* residual,
        const std::uint16_t* weight, float* checksum, float scale) {
    if (function == 0) {
        add_rmsnorm_block256_scalar_round<0><<<grid, kThreads>>>(
            input, residual, weight, checksum, rounds, active_threads, scale);
    } else if (function == 1) {
        add_rmsnorm_block256_scalar_round<1><<<grid, kThreads>>>(
            input, residual, weight, checksum, rounds, active_threads, scale);
    } else {
        std::fprintf(stderr, "unknown function: %d\n", function);
        std::exit(EXIT_FAILURE);
    }
}

float record_batch(
        mcEvent_t start, mcEvent_t end, const Cell& cell,
        std::uint16_t* input, std::uint16_t* residual,
        const std::uint16_t* weight, float* checksum, float scale) {
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(cell.function, cell.grid, cell.active_threads, cell.rounds,
               input, residual, weight, checksum, scale);
    }
    CHECK(mcEventRecord(start));
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(cell.function, cell.grid, cell.active_threads, cell.rounds,
               input, residual, weight, checksum, scale);
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

const char* function_name(int function) {
    if (function == 0) {
        return "add_square";
    }
    if (function == 1) {
        return "scale_multiply";
    }
    std::fprintf(stderr, "unknown function: %d\n", function);
    std::exit(EXIT_FAILURE);
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
        std::fprintf(stderr, "expected C500 104 AP/wave64 and positive clock\n");
        return EXIT_FAILURE;
    }
    const ResourceMetadata resources[] = {
        resource_metadata<0>(), resource_metadata<1>()};
    for (const ResourceMetadata& resource : resources) {
        if (resource.resident_ctas_per_ap != 8) {
            std::fprintf(
                stderr, "expected physical block256 occupancy eight, got %d\n",
                resource.resident_ctas_per_ap);
            return EXIT_FAILURE;
        }
    }

    const std::size_t row_elements =
        static_cast<std::size_t>(kMaximumRounds) * kThreads;
    const std::size_t elements = kExpectedApCount * row_elements;
    const std::size_t bytes = elements * sizeof(std::uint16_t);
    const std::size_t weight_bytes = row_elements * sizeof(std::uint16_t);
    const std::size_t checksum_elements =
        static_cast<std::size_t>(kExpectedApCount) * kThreads;
    std::vector<std::uint16_t> host(elements, 0x3f00U);
    std::vector<std::uint16_t> host_weight(row_elements, 0x3f40U);
    std::uint16_t* input = nullptr;
    std::uint16_t* residual = nullptr;
    std::uint16_t* weight = nullptr;
    float* checksum = nullptr;
    CHECK(mcMalloc(reinterpret_cast<void**>(&input), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&residual), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&weight), weight_bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&checksum),
                   checksum_elements * sizeof(float)));
    CHECK(mcMemcpy(input, host.data(), bytes, mcMemcpyHostToDevice));
    CHECK(mcMemcpy(residual, host.data(), bytes, mcMemcpyHostToDevice));
    CHECK(mcMemcpy(weight, host_weight.data(), weight_bytes,
                   mcMemcpyHostToDevice));

    const int functions[] = {0, 1};
    const int active_thread_counts[] = {64, 128, 256};
    const int grids[] = {1, 2, 4, 8, 16, 32, 64, 104, 208, 416, 832, 1664};
    const int round_counts[] = {4, 8, 16, 32, 64};
    for (int function : functions) {
        for (int active_threads : active_thread_counts) {
            for (int grid : grids) {
                for (int rounds : round_counts) {
                    for (int warmup = 0; warmup < kWarmupsPerGeometry;
                         ++warmup) {
                        launch(function, grid, active_threads, rounds, input,
                               residual, weight, checksum, 0.9990234375F);
                    }
                }
            }
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (int function : functions) {
        for (int active_threads : active_thread_counts) {
            for (int grid : grids) {
                for (int rounds : round_counts) {
                    for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                        plan.push_back(
                            {function, active_threads, grid, rounds, sample});
                    }
                }
            }
        }
    }
    std::mt19937 generator(256911U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,1,MetaX C500,%d,%d,%d,%d,%d,%d,%d,"
        "%d,%zu,%d,%d,%zu,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, kLaunchesPerBatch,
        resources[0].registers_per_thread,
        resources[0].shared_bytes_per_cta,
        resources[0].resident_ctas_per_ap,
        resources[1].registers_per_thread,
        resources[1].shared_bytes_per_cta,
        resources[1].resident_ctas_per_ap);
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& cell = plan[order];
        const float scale = 0.9990234375F
            + static_cast<float>(trial) * 0.00000095367431640625F;
        const float duration_ms = record_batch(
            start, end, cell, input, residual, weight, checksum, scale);
        std::printf(
            "sample,%zu,%s,%d,%d,%d,%d,%a\n", order,
            function_name(cell.function), cell.active_threads, cell.grid,
            cell.rounds, cell.sample, static_cast<double>(duration_ms));
    }
    CHECK(mcEventDestroy(end));
    CHECK(mcEventDestroy(start));
    CHECK(mcFree(checksum));
    CHECK(mcFree(weight));
    CHECK(mcFree(residual));
    CHECK(mcFree(input));
    return EXIT_SUCCESS;
}
