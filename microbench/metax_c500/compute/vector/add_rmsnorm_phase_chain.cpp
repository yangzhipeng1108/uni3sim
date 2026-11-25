/*
 * MetaX C500 scalar BF16 AddRMSNorm phase-chain benchmark.
 *
 * This is not a production AddRMSNorm implementation: it omits the block
 * reduction, reciprocal square root, and epsilon operation. It compares the
 * two source-identical scalar loops in isolation with the same loops joined
 * by the source publication barrier. Adjacent round-count differences and an
 * independent scheduler envelope separate steady service from fixed phase
 * entry without consuming production or evaluation duration.
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
constexpr int kThreads = 1024;
constexpr int kMaximumGrid = 104;
constexpr int kMaximumRounds = 32;
constexpr int kWarmupsPerGeometry = 2;
constexpr int kSamplesPerCell = 15;
constexpr int kLaunchesPerBatch = 8;

union ScalarBf16 {
    std::uint16_t bits;
    __nv_bfloat16 value;
};

__device__ __forceinline__ __nv_bfloat16 load_bf16(
        const std::uint16_t* source, int index) {
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
__global__ void add_rmsnorm_phase_chain(
        std::uint16_t* __restrict__ input,
        std::uint16_t* __restrict__ residual,
        const std::uint16_t* __restrict__ weight,
        float* __restrict__ checksum, int rounds, float scale) {
    static_assert(Function >= 0 && Function <= 2, "unknown function");
    __shared__ float published_scale;
    const int hidden_size = rounds * blockDim.x;
    const int row_offset = blockIdx.x * hidden_size;
    float variance = 0.0F;
    if constexpr (Function == 0 || Function == 2) {
        #pragma unroll 1
        for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
            const int element = row_offset + idx;
            __nv_bfloat16 z = load_bf16(input, element);
            z += load_bf16(residual, element);
            const float x = __bfloat162float(z);
            variance += x * x;
            residual[element] = store_bf16(z);
        }
    }
    if constexpr (Function == 0) {
        checksum[blockIdx.x * blockDim.x + threadIdx.x] = variance;
    } else {
        if constexpr (Function == 2) {
            if (threadIdx.x == 0) {
                published_scale = scale + variance * 0x1p-24F;
            }
            __syncthreads();
        }
        const float chained_scale = Function == 2 ? published_scale : scale;
        #pragma unroll 1
        for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
            const int element = row_offset + idx;
            const float x = __bfloat162float(load_bf16(residual, element));
            const float gamma = __bfloat162float(load_bf16(weight, idx));
            input[element] = store_bf16(
                __float2bfloat16(x * chained_scale) *
                __float2bfloat16(gamma));
        }
    }
}

struct Cell {
    int function;
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
void launch_function(
        int grid, int rounds, std::uint16_t* input,
        std::uint16_t* residual, const std::uint16_t* weight,
        float* checksum, float scale) {
    add_rmsnorm_phase_chain<Function><<<grid, kThreads>>>(
        input, residual, weight, checksum, rounds, scale);
}

void launch(
        int function, int grid, int rounds, std::uint16_t* input,
        std::uint16_t* residual, const std::uint16_t* weight,
        float* checksum, float scale) {
    if (function == 0) {
        launch_function<0>(
            grid, rounds, input, residual, weight, checksum, scale);
    } else if (function == 1) {
        launch_function<1>(
            grid, rounds, input, residual, weight, checksum, scale);
    } else if (function == 2) {
        launch_function<2>(
            grid, rounds, input, residual, weight, checksum, scale);
    } else {
        std::fprintf(stderr, "unknown AddRMSNorm function: %d\n", function);
        std::exit(EXIT_FAILURE);
    }
}

float record_batch(
        mcEvent_t start, mcEvent_t end, int function, int grid, int rounds,
        std::uint16_t* input, std::uint16_t* residual,
        const std::uint16_t* weight, float* checksum, float scale) {
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(
            function, grid, rounds, input, residual, weight, checksum, scale);
    }
    CHECK(mcEventRecord(start));
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(
            function, grid, rounds, input, residual, weight, checksum, scale);
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
            add_rmsnorm_phase_chain<Function>)));
    ResourceMetadata result {
        attributes.numRegs,
        attributes.sharedSizeBytes,
        0,
    };
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.resident_ctas_per_ap,
        add_rmsnorm_phase_chain<Function>, kThreads, 0));
    return result;
}

const char* function_name(int function) {
    if (function == 0) {
        return "input_add_square";
    } else if (function == 1) {
        return "output_scale";
    } else if (function == 2) {
        return "phase_chain";
    }
    std::fprintf(stderr, "unknown AddRMSNorm function: %d\n", function);
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
        resource_metadata<0>(),
        resource_metadata<1>(),
        resource_metadata<2>(),
    };
    for (const ResourceMetadata& resource : resources) {
        if (resource.resident_ctas_per_ap != 2) {
            std::fprintf(
                stderr, "expected block1024 occupancy two, got %d\n",
                resource.resident_ctas_per_ap);
            return EXIT_FAILURE;
        }
    }

    const std::size_t row_elements =
        static_cast<std::size_t>(kMaximumRounds) * kThreads;
    const std::size_t elements = kMaximumGrid * row_elements;
    const std::size_t bytes = elements * sizeof(std::uint16_t);
    const std::size_t weight_bytes = row_elements * sizeof(std::uint16_t);
    const std::size_t checksum_elements =
        static_cast<std::size_t>(kMaximumGrid) * kThreads;
    std::vector<std::uint16_t> host(elements, 0x3f00U);
    std::vector<std::uint16_t> host_weight(row_elements, 0x3f40U);
    std::uint16_t* input = nullptr;
    std::uint16_t* residual = nullptr;
    std::uint16_t* weight = nullptr;
    float* checksum = nullptr;
    CHECK(mcMalloc(reinterpret_cast<void**>(&input), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&residual), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&weight), weight_bytes));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&checksum),
        checksum_elements * sizeof(float)));
    CHECK(mcMemcpy(input, host.data(), bytes, mcMemcpyHostToDevice));
    CHECK(mcMemcpy(residual, host.data(), bytes, mcMemcpyHostToDevice));
    CHECK(mcMemcpy(
        weight, host_weight.data(), weight_bytes, mcMemcpyHostToDevice));

    const int functions[] = {0, 1, 2};
    const int grids[] = {1, 2, 4, 8, 16, 32, 64, 104};
    const int round_counts[] = {8, 16, 32};
    const float scale = 0.9990234375F
        + static_cast<float>(trial) * 0.00000095367431640625F;
    for (int function : functions) {
        for (int grid : grids) {
            for (int rounds : round_counts) {
                for (int warmup = 0; warmup < kWarmupsPerGeometry; ++warmup) {
                    launch(
                        function, grid, rounds, input, residual, weight,
                        checksum, scale);
                }
            }
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (int function : functions) {
        for (int grid : grids) {
            for (int rounds : round_counts) {
                for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                    plan.push_back({function, grid, rounds, sample});
                }
            }
        }
    }
    std::mt19937 generator(910271U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,1,MetaX C500,%d,%d,%d,%d,%d,%d,%d"
        ",%d,%zu,%d,%d,%zu,%d,%d,%zu,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, kLaunchesPerBatch,
        resources[0].registers_per_thread,
        resources[0].shared_bytes_per_cta,
        resources[0].resident_ctas_per_ap,
        resources[1].registers_per_thread,
        resources[1].shared_bytes_per_cta,
        resources[1].resident_ctas_per_ap,
        resources[2].registers_per_thread,
        resources[2].shared_bytes_per_cta,
        resources[2].resident_ctas_per_ap);
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& item = plan[order];
        const float duration_ms = record_batch(
            start, end, item.function, item.grid, item.rounds,
            input, residual, weight, checksum, scale);
        std::printf(
            "sample,%zu,%s,%d,%d,%d,%a\n",
            order, function_name(item.function), item.grid, item.rounds,
            item.sample, static_cast<double>(duration_ms));
    }
    CHECK(mcEventDestroy(end));
    CHECK(mcEventDestroy(start));

    launch(2, 1, 8, input, residual, weight, checksum, scale);
    CHECK(mcDeviceSynchronize());
    CHECK(mcMemcpy(host.data(), input, bytes, mcMemcpyDeviceToHost));
    if (host[0] == 0U) {
        std::fprintf(stderr, "AddRMSNorm phase-chain output was not written\n");
        return EXIT_FAILURE;
    }

    CHECK(mcFree(checksum));
    CHECK(mcFree(weight));
    CHECK(mcFree(residual));
    CHECK(mcFree(input));
    return EXIT_SUCCESS;
}
