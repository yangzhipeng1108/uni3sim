/*
 * MetaX C500 scalar BF16 RMSNorm source-round benchmark.
 *
 * This isolates the two scalar loops around the production CUB reduction.
 * The input loop keeps the source idx += blockDim.x address progression and
 * loop-carried square accumulation. The output loop keeps the intermediate
 * BF16 rounding boundary before the BF16 weight multiply. The reduction,
 * launch framework, and any production tensor shape are deliberately absent.
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
constexpr int kMaximumThreads = 1024;
constexpr int kMaximumGrid = 3328;
constexpr int kMaximumRounds = 32;
constexpr int kWarmupsPerGeometry = 2;
constexpr int kSamplesPerCell = 15;
constexpr int kLaunchesPerBatch = 64;

union ScalarBf16 {
    std::uint16_t bits;
    __nv_bfloat16 value;
};

__device__ __forceinline__ float load_bf16(
        const std::uint16_t* source, int index) {
    ScalarBf16 packed;
    packed.bits = source[index];
    return __bfloat162float(packed.value);
}

__device__ __forceinline__ std::uint16_t store_bf16(float value) {
    ScalarBf16 packed;
    packed.value = __float2bfloat16(value);
    return packed.bits;
}

template <int Function>
__global__ void rmsnorm_scalar_round(
        const std::uint16_t* __restrict__ input,
        const std::uint16_t* __restrict__ weight,
        std::uint16_t* __restrict__ output,
        float* __restrict__ checksum, int rounds, float scale) {
    const int hidden_size = rounds * blockDim.x;
    const int row_offset = blockIdx.x * hidden_size;
    float variance = 0.0F;
    #pragma unroll 1
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        const float x = load_bf16(input, row_offset + idx);
        if constexpr (Function == 0) {
            variance += x * x;
        } else if constexpr (Function == 1) {
            ScalarBf16 intermediate;
            intermediate.value = __float2bfloat16(x * scale);
            const float rounded = __bfloat162float(intermediate.value);
            const float gamma = load_bf16(weight, idx);
            output[row_offset + idx] = store_bf16(rounded * gamma);
        } else {
            static_assert(Function == 0 || Function == 1,
                          "unknown RMSNorm scalar function");
        }
    }
    if constexpr (Function == 0) {
        checksum[blockIdx.x * blockDim.x + threadIdx.x] = variance;
    }
}

struct Cell {
    int function;
    int threads;
    int grid;
    int rounds;
    int sample;
};

struct ResourceMetadata {
    int registers_per_thread;
    std::size_t shared_bytes_per_cta;
    int resident_128;
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

template <int Function>
void launch_function(
        int threads, int grid, int rounds, const std::uint16_t* input,
        const std::uint16_t* weight, std::uint16_t* output,
        float* checksum, float scale) {
    rmsnorm_scalar_round<Function><<<grid, threads>>>(
        input, weight, output, checksum, rounds, scale);
}

void launch(
        int function, int threads, int grid, int rounds,
        const std::uint16_t* input, const std::uint16_t* weight,
        std::uint16_t* output, float* checksum, float scale) {
    if (function == 0) {
        launch_function<0>(
            threads, grid, rounds, input, weight, output, checksum, scale);
    } else if (function == 1) {
        launch_function<1>(
            threads, grid, rounds, input, weight, output, checksum, scale);
    } else {
        std::fprintf(stderr, "unknown RMSNorm scalar function: %d\n", function);
        std::exit(EXIT_FAILURE);
    }
}

float record_batch(
        mcEvent_t start, mcEvent_t end, int function, int threads,
        int grid, int rounds,
        const std::uint16_t* input, const std::uint16_t* weight,
        std::uint16_t* output, float* checksum, float scale) {
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(
            function, threads, grid, rounds,
            input, weight, output, checksum, scale);
    }
    CHECK(mcEventRecord(start));
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(
            function, threads, grid, rounds,
            input, weight, output, checksum, scale);
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
        &attributes, reinterpret_cast<const void*>(
            rmsnorm_scalar_round<Function>)));
    ResourceMetadata result {
        attributes.numRegs,
        attributes.sharedSizeBytes,
        0,
        0,
    };
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.resident_128,
        rmsnorm_scalar_round<Function>, 128, 0));
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.resident_1024,
        rmsnorm_scalar_round<Function>, 1024, 0));
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
        std::fprintf(stderr, "expected C500 104 AP/wave64 and positive clock\n");
        return EXIT_FAILURE;
    }

    const ResourceMetadata input_resources = resource_metadata<0>();
    const ResourceMetadata output_resources = resource_metadata<1>();
    if (input_resources.resident_128 != 16
            || output_resources.resident_128 != 16
            || input_resources.resident_1024 != 2
            || output_resources.resident_1024 != 2) {
        std::fprintf(
            stderr, "unexpected occupancy input128=%d output128=%d "
                    "input1024=%d output1024=%d\n",
            input_resources.resident_128, output_resources.resident_128,
            input_resources.resident_1024, output_resources.resident_1024);
        return EXIT_FAILURE;
    }

    const std::size_t row_elements =
        static_cast<std::size_t>(kMaximumRounds) * kMaximumThreads;
    const std::size_t elements = kMaximumGrid * row_elements;
    const std::size_t bytes = elements * sizeof(std::uint16_t);
    const std::size_t weight_bytes = row_elements * sizeof(std::uint16_t);
    const std::size_t checksum_elements =
        static_cast<std::size_t>(kMaximumGrid) * kMaximumThreads;
    std::vector<std::uint16_t> host(elements, 0x3f00U);
    std::vector<std::uint16_t> host_weight(row_elements, 0x3f40U);
    std::uint16_t* input = nullptr;
    std::uint16_t* weight = nullptr;
    std::uint16_t* output = nullptr;
    float* checksum = nullptr;
    CHECK(mcMalloc(reinterpret_cast<void**>(&input), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&weight), weight_bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&output), bytes));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&checksum),
        checksum_elements * sizeof(float)));
    CHECK(mcMemcpy(input, host.data(), bytes, mcMemcpyHostToDevice));
    CHECK(mcMemcpy(
        weight, host_weight.data(), weight_bytes, mcMemcpyHostToDevice));

    const int functions[] = {0, 1};
    const int blocks[] = {128, 1024};
    const int grids_128[] = {
        1, 2, 4, 8, 16, 32, 64, 104, 208, 416, 832, 1664, 3328};
    const int grids_1024[] = {
        1, 2, 4, 8, 16, 32, 64, 104, 208, 312, 416, 832};
    const int round_counts[] = {2, 3, 4, 8, 16, 32};
    const float scale = 0.9990234375F
        + static_cast<float>(trial) * 0.00000095367431640625F;
    for (int function : functions) {
        for (int threads : blocks) {
          const int* grids = threads == 128 ? grids_128 : grids_1024;
          const int grid_count = threads == 128
              ? static_cast<int>(sizeof(grids_128) / sizeof(grids_128[0]))
              : static_cast<int>(sizeof(grids_1024) / sizeof(grids_1024[0]));
          for (int grid_index = 0; grid_index < grid_count; ++grid_index) {
            const int grid = grids[grid_index];
            for (int rounds : round_counts) {
                for (int warmup = 0; warmup < kWarmupsPerGeometry; ++warmup) {
                    launch(
                        function, threads, grid, rounds, input, weight, output,
                        checksum, scale);
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
          const int* grids = threads == 128 ? grids_128 : grids_1024;
          const int grid_count = threads == 128
              ? static_cast<int>(sizeof(grids_128) / sizeof(grids_128[0]))
              : static_cast<int>(sizeof(grids_1024) / sizeof(grids_1024[0]));
          for (int grid_index = 0; grid_index < grid_count; ++grid_index) {
            const int grid = grids[grid_index];
            for (int rounds : round_counts) {
                for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                    plan.push_back({function, threads, grid, rounds, sample});
                }
            }
          }
        }
    }
    std::mt19937 generator(910163U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,2,MetaX C500,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%d,%d,%d,%zu,%d,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, kLaunchesPerBatch,
        input_resources.registers_per_thread,
        input_resources.shared_bytes_per_cta,
        input_resources.resident_128,
        input_resources.resident_1024,
        output_resources.registers_per_thread,
        output_resources.shared_bytes_per_cta,
        output_resources.resident_128,
        output_resources.resident_1024);
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& item = plan[order];
        const float duration_ms = record_batch(
            start, end, item.function, item.threads, item.grid, item.rounds,
            input, weight, output, checksum, scale);
        std::printf(
            "sample,%zu,%s,%d,%d,%d,%d,%a\n",
            order, item.function == 0 ? "input_square" : "output_scale",
            item.threads, item.grid, item.rounds, item.sample,
            static_cast<double>(duration_ms));
    }
    CHECK(mcEventDestroy(end));
    CHECK(mcEventDestroy(start));

    launch(1, 128, 1, 4, input, weight, output, checksum, scale);
    CHECK(mcDeviceSynchronize());
    CHECK(mcMemcpy(host.data(), output, bytes, mcMemcpyDeviceToHost));
    if (host[0] == 0U) {
        std::fprintf(stderr, "RMSNorm scalar output was not written\n");
        return EXIT_FAILURE;
    }

    CHECK(mcFree(checksum));
    CHECK(mcFree(output));
    CHECK(mcFree(weight));
    CHECK(mcFree(input));
    return EXIT_SUCCESS;
}
