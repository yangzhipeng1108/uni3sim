/*
 * MetaX C500 paired warm global-memory dependency-task benchmark.
 *
 * One lane per hardware wave executes one ordinary global load, one dependent
 * integer add, and one ordinary global store. The body is operator-independent
 * and contains no production tensor shape or operator implementation. Repeated
 * launches over the same small allocation establish the deliberately weak
 * warm-default-cache contract; they do not claim exclusive L2 residency.
 * Every sample measures a batch of empty tasks and a batch of dependency
 * tasks back-to-back with mcEvent.  Their per-launch difference isolates the
 * generic memory-dependency increment from common launch/scheduler drift.
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <time.h>
#include <tuple>
#include <utility>
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
constexpr int kMaximumGridSize = 16384;
constexpr int kMaximumBlockSize = 1024;
constexpr int kWarmupLaunchesPerGeometry = 4;
constexpr int kSamplesPerCell = 31;
constexpr int kLaunchesPerBatch = 128;

struct Geometry {
    int grid_x;
    int grid_y;
    int block;
};

struct Sample {
    Geometry geometry;
    int repeat;
};

std::vector<Geometry> campaign_geometries() {
    std::vector<Geometry> geometries;
    for (int grid : {
            1, 16, 64, 104, 128, 256, 512, 1024,
            2048, 4096, 8192, 16384}) {
        for (int block : {64, 256, 512, 1024}) {
            geometries.push_back({grid, 1, block});
        }
    }
    // Generic two-dimensional factorizations are a rank holdout.  They are
    // deliberately powers of two rather than production-operator shapes.
    for (const auto& axes : {
            std::pair<int, int>{2, 2},
            std::pair<int, int>{4, 8},
            std::pair<int, int>{8, 16},
            std::pair<int, int>{16, 32},
            std::pair<int, int>{16, 128},
            std::pair<int, int>{16, 512}}) {
        geometries.push_back({axes.first, axes.second, 512});
    }
    return geometries;
}

__global__ void empty_task() {}

__global__ void memory_dependency_task(
        const std::uint32_t* input, std::uint32_t* output) {
    const int lane = threadIdx.x & (kExpectedWaveSize - 1);
    if (lane != 0) {
        return;
    }
    const int waves_per_block = (
        blockDim.x + kExpectedWaveSize - 1) / kExpectedWaveSize;
    const int wave = threadIdx.x / kExpectedWaveSize;
    const int flattened_block = blockIdx.y * gridDim.x + blockIdx.x;
    const int index = flattened_block * waves_per_block + wave;
    output[index] = input[index] + 1U;
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

void launch_batch(
        bool dependency, const Geometry& geometry,
        const std::uint32_t* input, std::uint32_t* output) {
    const dim3 grid(geometry.grid_x, geometry.grid_y, 1);
    const dim3 block(geometry.block, 1, 1);
    for (int launch = 0; launch < kLaunchesPerBatch; ++launch) {
        if (dependency) {
            memory_dependency_task<<<grid, block>>>(input, output);
        } else {
            empty_task<<<grid, block>>>();
        }
    }
    CHECK(mcGetLastError());
}

void record_batch(
        mcEvent_t start, mcEvent_t end, bool dependency,
        const Geometry& geometry,
        const std::uint32_t* input, std::uint32_t* output) {
    // Prime every timed arm with the same task type on the same stream.  This
    // keeps the ABBA comparison while removing the device's type-dependent
    // first-batch state from both event intervals.
    launch_batch(dependency, geometry, input, output);
    CHECK(mcEventRecord(start));
    launch_batch(dependency, geometry, input, output);
    CHECK(mcEventRecord(end));
}

float elapsed_batch_ms(mcEvent_t start, mcEvent_t end) {
    float elapsed_ms = 0.0F;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
        std::fprintf(
            stderr, "invalid paired mcEvent duration: %.9g ms\n",
            elapsed_ms);
        std::exit(EXIT_FAILURE);
    }
    return elapsed_ms;
}

long long monotonic_raw_ns() {
    timespec now {};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        std::perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        std::exit(EXIT_FAILURE);
    }
    return (static_cast<long long>(now.tv_sec) * 1000000000LL
            + static_cast<long long>(now.tv_nsec));
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
            || properties.warpSize != kExpectedWaveSize) {
        std::fprintf(
            stderr, "expected C500 104 AP/wave64, got %d AP/wave%d\n",
            properties.multiProcessorCount, properties.warpSize);
        return EXIT_FAILURE;
    }

    constexpr int maximum_waves = (
        kMaximumGridSize * (kMaximumBlockSize / kExpectedWaveSize));
    std::uint32_t* input = nullptr;
    std::uint32_t* output = nullptr;
    CHECK(mcMalloc(&input, maximum_waves * sizeof(*input)));
    CHECK(mcMalloc(&output, maximum_waves * sizeof(*output)));
    CHECK(mcMemset(input, 1, maximum_waves * sizeof(*input)));
    CHECK(mcMemset(output, 0, maximum_waves * sizeof(*output)));

    const std::vector<Geometry> geometries = campaign_geometries();
    for (const Geometry& geometry : geometries) {
        const dim3 grid(geometry.grid_x, geometry.grid_y, 1);
        const dim3 block(geometry.block, 1, 1);
        for (int launch = 0;
             launch < kWarmupLaunchesPerGeometry; ++launch) {
            empty_task<<<grid, block>>>();
            memory_dependency_task<<<grid, block>>>(input, output);
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Sample> samples;
    for (const Geometry& geometry : geometries) {
        for (int repeat = 0; repeat < kSamplesPerCell; ++repeat) {
            samples.push_back({geometry, repeat});
        }
    }
    std::mt19937 generator(500441U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(samples.begin(), samples.end(), generator);

    std::printf(
        "metadata,5,MetaX C500,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, kLaunchesPerBatch,
        kMaximumGridSize, 2);
    std::printf(
        "sample_header,order,grid_x,grid_y,grid_z,grid_size,"
        "block,repeat,pair_order,"
        "host_start_ns,host_end_ns,"
        "empty_batch_ns,dependent_batch_ns,empty_per_launch_ns,"
        "dependent_per_launch_ns,increment_per_launch_ns,"
        "empty_first_elapsed_ms_hex,empty_second_elapsed_ms_hex,"
        "dependent_first_elapsed_ms_hex,dependent_second_elapsed_ms_hex,"
        "empty_event_status,dependent_event_status\n");
    mcEvent_t empty_start[2];
    mcEvent_t empty_end[2];
    mcEvent_t dependent_start[2];
    mcEvent_t dependent_end[2];
    for (int arm = 0; arm < 2; ++arm) {
        CHECK(mcEventCreate(&empty_start[arm]));
        CHECK(mcEventCreate(&empty_end[arm]));
        CHECK(mcEventCreate(&dependent_start[arm]));
        CHECK(mcEventCreate(&dependent_end[arm]));
    }
    for (std::size_t order = 0; order < samples.size(); ++order) {
        const Sample& sample = samples[order];
        const Geometry& geometry = sample.geometry;
        const bool dependency_first = ((order + trial) & 1U) != 0U;
        const long long host_start_ns = monotonic_raw_ns();
        if (dependency_first) {
            record_batch(
                dependent_start[0], dependent_end[0], true,
                geometry, input, output);
            record_batch(
                empty_start[0], empty_end[0], false,
                geometry, input, output);
            record_batch(
                empty_start[1], empty_end[1], false,
                geometry, input, output);
            record_batch(
                dependent_start[1], dependent_end[1], true,
                geometry, input, output);
        } else {
            record_batch(
                empty_start[0], empty_end[0], false,
                geometry, input, output);
            record_batch(
                dependent_start[0], dependent_end[0], true,
                geometry, input, output);
            record_batch(
                dependent_start[1], dependent_end[1], true,
                geometry, input, output);
            record_batch(
                empty_start[1], empty_end[1], false,
                geometry, input, output);
        }
        CHECK(mcDeviceSynchronize());
        const long long host_end_ns = monotonic_raw_ns();
        const float empty_first_ms = elapsed_batch_ms(
            empty_start[0], empty_end[0]);
        const float empty_second_ms = elapsed_batch_ms(
            empty_start[1], empty_end[1]);
        const float dependent_first_ms = elapsed_batch_ms(
            dependent_start[0], dependent_end[0]);
        const float dependent_second_ms = elapsed_batch_ms(
            dependent_start[1], dependent_end[1]);
        const double empty_batch_ns = (
            (static_cast<double>(empty_first_ms)
             + static_cast<double>(empty_second_ms))
            * 500000.0);
        const double dependent_batch_ns = (
            (static_cast<double>(dependent_first_ms)
             + static_cast<double>(dependent_second_ms))
            * 500000.0);
        const long long host_elapsed_ns = host_end_ns - host_start_ns;
        if (host_elapsed_ns <= 0
                || static_cast<double>(host_elapsed_ns)
                < 2.0 * (empty_batch_ns + dependent_batch_ns)) {
            std::fprintf(
                stderr, "host pair is shorter than device batches: "
                "%lld < %.9f ns\n", host_elapsed_ns,
                2.0 * (empty_batch_ns + dependent_batch_ns));
            return EXIT_FAILURE;
        }
        const double empty_per_launch_ns = (
            empty_batch_ns / kLaunchesPerBatch);
        const double dependent_per_launch_ns = (
            dependent_batch_ns / kLaunchesPerBatch);
        const int grid_size = geometry.grid_x * geometry.grid_y;
        std::printf(
            "sample,%zu,%d,%d,1,%d,%d,%d,%s,%lld,%lld,"
            "%.9f,%.9f,%.9f,%.9f,%.9f,"
            "%a,%a,%a,%a,resolved,resolved\n",
            order, geometry.grid_x, geometry.grid_y, grid_size,
            geometry.block, sample.repeat,
            dependency_first ? "dependent_first" : "empty_first",
            host_start_ns, host_end_ns,
            empty_batch_ns, dependent_batch_ns,
            empty_per_launch_ns, dependent_per_launch_ns,
            dependent_per_launch_ns - empty_per_launch_ns,
            empty_first_ms, empty_second_ms,
            dependent_first_ms, dependent_second_ms);
    }
    for (int arm = 0; arm < 2; ++arm) {
        CHECK(mcEventDestroy(empty_start[arm]));
        CHECK(mcEventDestroy(empty_end[arm]));
        CHECK(mcEventDestroy(dependent_start[arm]));
        CHECK(mcEventDestroy(dependent_end[arm]));
    }
    CHECK(mcFree(input));
    CHECK(mcFree(output));
    return EXIT_SUCCESS;
}
