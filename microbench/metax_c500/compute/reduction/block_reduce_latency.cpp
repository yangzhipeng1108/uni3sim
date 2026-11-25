/*
 * MetaX C500 block-reduction dependency benchmark.
 *
 * One runtime-counted kernel repeats the source-level NORM collective:
 * CUB BlockReduce<float, 1024>, variance division, epsilon addition,
 * reciprocal square root, and scalar publication.  Adjacent repeat-count
 * differences remove launch and fixed task overhead without comparing a
 * candidate kernel against a different control program.  No production
 * tensor, production kernel duration, or machine-instruction count is used.
 */

#include <mc_runtime.h>

#include <cub/cub.cuh>

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
constexpr int kHiddenSize = 4096;
constexpr float kEpsilon = 1.0e-5F;
constexpr int kWarmupsPerGeometry = 4;
constexpr int kSamplesPerCell = 31;
constexpr int kLaunchesPerBatch = 16;

__global__ void reduction_dependency(
        float* output, int hidden_size, int repetitions) {
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduce_store;
    __shared__ float inverse_root;

    float value = 1.0F + static_cast<float>(threadIdx.x & 7U) * 0.125F;
    #pragma unroll 1
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        const float variance = BlockReduce(reduce_store).Reduce(
            value, cub::Sum{}, blockDim.x);
        if (threadIdx.x == 0) {
            inverse_root = rsqrtf(variance / hidden_size + kEpsilon);
        }
        // This is both the production scalar-publication edge and the CUB
        // TempStorage reuse barrier required before the next iteration.
        __syncthreads();
        value = value + inverse_root * 0.0009765625F;
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = inverse_root + value;
    }
}

struct Cell {
    int threads;
    int grid;
    int repetitions;
    int sample;
};

struct ResourceMetadata {
    int registers_per_thread;
    std::size_t shared_bytes_per_cta;
    int resident_128;
    int resident_256;
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

void launch(int grid, int threads, int repetitions, float* output) {
    reduction_dependency<<<grid, threads>>>(
        output, kHiddenSize, repetitions);
}

void launch_batch(
        int grid, int threads, int repetitions, float* output) {
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(grid, threads, repetitions, output);
    }
    CHECK(mcGetLastError());
}

float record_batch(
        mcEvent_t start, mcEvent_t end, int grid, int threads,
        int repetitions, float* output) {
    launch_batch(grid, threads, repetitions, output);
    CHECK(mcEventRecord(start));
    launch_batch(grid, threads, repetitions, output);
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

ResourceMetadata resource_metadata() {
    mcFuncAttributes attributes {};
    CHECK(mcFuncGetAttributes(
        &attributes, reinterpret_cast<const void*>(reduction_dependency)));
    ResourceMetadata result {
        attributes.numRegs,
        attributes.sharedSizeBytes,
        0,
        0,
        0,
    };
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.resident_128, reduction_dependency, 128, 0));
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.resident_256, reduction_dependency, 256, 0));
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.resident_1024, reduction_dependency, 1024, 0));
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

    const ResourceMetadata resources = resource_metadata();
    if (resources.resident_128 <= 0 || resources.resident_256 <= 0
            || resources.resident_1024 <= 0) {
        std::fprintf(stderr, "invalid block-reduction occupancy\n");
        return EXIT_FAILURE;
    }

    constexpr int maximum_grid = 32 * kExpectedApCount;
    float* output = nullptr;
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&output),
        maximum_grid * sizeof(float)));

    const int blocks[] = {128, 256, 1024};
    const int grids[] = {
        1, 2, 4, 8, 16, 32, 64, 104, 208, 416, 832, 1664, 3328};
    const int repetition_counts[] = {4, 8, 16, 32};
    for (int threads : blocks) {
        for (int grid : grids) {
            for (int repetitions : repetition_counts) {
                for (int warmup = 0;
                     warmup < kWarmupsPerGeometry; ++warmup) {
                    launch(grid, threads, repetitions, output);
                }
            }
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (int threads : blocks) {
        for (int grid : grids) {
            for (int repetitions : repetition_counts) {
                for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                    plan.push_back({threads, grid, repetitions, sample});
                }
            }
        }
    }
    std::mt19937 generator(500827U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,6,MetaX C500,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%d,%d,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, kLaunchesPerBatch,
        resources.registers_per_thread, resources.shared_bytes_per_cta,
        resources.resident_128, resources.resident_256,
        resources.resident_1024);

    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& item = plan[order];
        const float duration_ms = record_batch(
            start, end, item.grid, item.threads,
            item.repetitions, output);
        std::printf(
            "sample,%zu,%d,%d,%d,%d,%a\n",
            order, item.threads, item.grid, item.repetitions, item.sample,
            static_cast<double>(duration_ms));
    }
    CHECK(mcEventDestroy(end));
    CHECK(mcEventDestroy(start));

    launch(maximum_grid, 1024, 4, output);
    CHECK(mcDeviceSynchronize());
    std::vector<float> host_output(maximum_grid);
    CHECK(mcMemcpy(
        host_output.data(), output,
        maximum_grid * sizeof(float), mcMemcpyDeviceToHost));
    for (int block = 0; block < maximum_grid; ++block) {
        if (!std::isfinite(host_output[block])) {
            std::fprintf(
                stderr, "invalid block-reduction output for block %d\n",
                block);
            return EXIT_FAILURE;
        }
    }

    CHECK(mcFree(output));
    return EXIT_SUCCESS;
}
