/*
 * MetaX C500 CTA barrier service benchmark.
 *
 * A 256-thread CTA repeats only the source-level __syncthreads primitive.
 * Runtime repetition-count differences remove the fixed clock64 and CTA
 * entry/exit span.  The launch reserves 32 KiB dynamic shared memory so the
 * compiler/runtime occupancy query fixes the production-relevant two resident
 * CTAs per AP.  This benchmark does not invoke or time FlashAttention.
 *
 * Build in the MACA 3.7.2 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *     -O3 -std=c++17 cta_barrier_service.cpp -o cta_barrier_service
 */

#include <mc_runtime.h>

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
constexpr int kThreads = 256;
constexpr std::size_t kDynamicSharedBytes = 32 * 1024;
constexpr int kSamplesPerCell = 31;

struct Cell {
    int grid;
    int repetitions;
    int sample;
};

__global__ void cta_barrier_cycles(
        unsigned long long* block_cycles, int repetitions) {
    extern __shared__ unsigned char reserved_shared[];
    if (threadIdx.x == 0) {
        reserved_shared[0] = static_cast<unsigned char>(blockIdx.x);
    }
    __syncthreads();
    const unsigned long long start = clock64();
    #pragma unroll 1
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        __syncthreads();
    }
    const unsigned long long end = clock64();
    if (threadIdx.x == 0) {
        block_cycles[blockIdx.x] = end - start;
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

unsigned long long median(std::vector<unsigned long long> values) {
    if (values.empty()) {
        std::fprintf(stderr, "cannot take median of an empty cycle vector\n");
        std::exit(EXIT_FAILURE);
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
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
        std::fprintf(stderr, "trial must be 1, 2, or 3\n");
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
    int resident_ctas = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, cta_barrier_cycles, kThreads, kDynamicSharedBytes));
    if (resident_ctas != 2) {
        std::fprintf(
            stderr, "expected two resident 256-thread/32-KiB CTAs, got %d\n",
            resident_ctas);
        return EXIT_FAILURE;
    }

    constexpr int maximum_grid = 2 * kExpectedApCount;
    unsigned long long* device_cycles = nullptr;
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&device_cycles),
        maximum_grid * sizeof(unsigned long long)));
    std::vector<unsigned long long> host_cycles(maximum_grid);

    const int grids[] = {1, kExpectedApCount, 2 * kExpectedApCount};
    const int repetition_counts[] = {32, 64, 128, 256};
    for (int grid : grids) {
        for (int repetitions : repetition_counts) {
            cta_barrier_cycles<<<grid, kThreads, kDynamicSharedBytes>>>(
                device_cycles, repetitions);
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (int grid : grids) {
        for (int repetitions : repetition_counts) {
            for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                plan.push_back({grid, repetitions, sample});
            }
        }
    }
    std::mt19937 generator(500829U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,1,MetaX C500,%d,%d,%d,%d,%d,%d,%d,%zu\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, resident_ctas,
        kDynamicSharedBytes);
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& cell = plan[order];
        cta_barrier_cycles<<<
            cell.grid, kThreads, kDynamicSharedBytes>>>(
                device_cycles, cell.repetitions);
        CHECK(mcGetLastError());
        CHECK(mcMemcpy(
            host_cycles.data(), device_cycles,
            cell.grid * sizeof(unsigned long long), mcMemcpyDeviceToHost));
        std::vector<unsigned long long> observed(
            host_cycles.begin(), host_cycles.begin() + cell.grid);
        if (std::any_of(observed.begin(), observed.end(),
                        [](unsigned long long value) { return value == 0; })) {
            std::fprintf(stderr, "barrier probe produced a zero cycle span\n");
            return EXIT_FAILURE;
        }
        const auto bounds = std::minmax_element(
            observed.begin(), observed.end());
        std::printf(
            "sample,%zu,%d,%d,%d,%llu,%llu,%llu\n",
            order, cell.grid, cell.repetitions, cell.sample,
            *bounds.first, median(observed), *bounds.second);
    }

    CHECK(mcFree(device_cycles));
    return EXIT_SUCCESS;
}
