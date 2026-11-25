/*
 * MetaX C500 CTA-local AddRMSNorm grid64 input-loop cycle probe.
 *
 * The probe preserves the generic BF16 input loop's two reads, add, square
 * accumulation, residual write, and blockDim-stride address progression. It
 * omits reduction/output work and records only a CTA-local clock64 interval;
 * launch, admission, and production-operator duration are outside the timer.
 */

#include <mc_runtime.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",               \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kThreads = 1024;
constexpr int kMaximumGrid = 64;
constexpr int kMaximumDepth = 32;
constexpr int kSamplesPerCell = 31;
constexpr int kWarmupsPerMode = 4;

union ScalarBf16 {
    std::uint16_t bits;
    __nv_bfloat16 value;
};

__device__ __forceinline__ __nv_bfloat16 load_bf16(
        const volatile std::uint16_t* source, std::size_t index) {
    ScalarBf16 packed;
    packed.bits = source[index];
    return packed.value;
}

__device__ __forceinline__ std::uint16_t store_bf16(__nv_bfloat16 value) {
    ScalarBf16 packed;
    packed.value = value;
    return packed.bits;
}

template <bool Progress>
__global__ void add_rmsnorm_input_progression_cycles(
        const volatile std::uint16_t* input,
        const volatile std::uint16_t* residual,
        volatile std::uint16_t* output,
        unsigned long long* block_cycles, int depth) {
    const std::size_t row_stride =
        static_cast<std::size_t>(kMaximumDepth) * kThreads;
    const std::size_t row_offset =
        static_cast<std::size_t>(blockIdx.x) * row_stride;
    float carry = 1.0F;
    __syncthreads();
    const unsigned long long start = clock64();
    #pragma unroll 1
    for (int round = 0; round < depth; ++round) {
        const std::size_t round_offset = Progress
            ? static_cast<std::size_t>(round) * kThreads : 0U;
        const std::size_t element =
            row_offset + round_offset + threadIdx.x;
        __nv_bfloat16 z = load_bf16(input, element);
        z += load_bf16(residual, element);
        const float x = __bfloat162float(z);
        carry += x * x * 0x1p-24F;
        output[element] = store_bf16(z);
    }
    asm volatile("" : : "r"(carry) : "memory");
    __syncthreads();
    const unsigned long long end = clock64();
    if (threadIdx.x == 0) {
        block_cycles[blockIdx.x] = end - start;
    }
}

struct Cell {
    int depth;
    bool progress;
    int sample;
    int grid;
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

template <bool Progress>
void launch_typed(
        int grid, int depth, const std::uint16_t* input,
        const std::uint16_t* residual, std::uint16_t* output,
        unsigned long long* block_cycles) {
    add_rmsnorm_input_progression_cycles<Progress>
        <<<grid, kThreads>>>(
            input, residual, output, block_cycles, depth);
}

void launch(
        bool progress, int grid, int depth, const std::uint16_t* input,
        const std::uint16_t* residual, std::uint16_t* output,
        unsigned long long* block_cycles) {
    if (progress) {
        launch_typed<true>(
            grid, depth, input, residual, output, block_cycles);
    } else {
        launch_typed<false>(
            grid, depth, input, residual, output, block_cycles);
    }
}

unsigned long long median(std::vector<unsigned long long> values) {
    if (values.empty()) {
        std::fprintf(stderr, "cannot take median of empty cycles\n");
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
    int resident_ctas = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas,
        add_rmsnorm_input_progression_cycles<true>, kThreads, 0));
    if (resident_ctas != 2) {
        std::fprintf(stderr, "expected block1024 occupancy 2, got %d\n",
                     resident_ctas);
        return EXIT_FAILURE;
    }

    const std::size_t row_stride =
        static_cast<std::size_t>(kMaximumDepth) * kThreads;
    const std::size_t elements = kMaximumGrid * row_stride;
    std::vector<std::uint16_t> host(elements, 0x3f80U);
    std::uint16_t* input = nullptr;
    std::uint16_t* residual = nullptr;
    std::uint16_t* output = nullptr;
    unsigned long long* device_cycles = nullptr;
    CHECK(mcMalloc(reinterpret_cast<void**>(&input), elements * sizeof(*input)));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&residual), elements * sizeof(*residual)));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&output), elements * sizeof(*output)));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&device_cycles),
        kMaximumGrid * sizeof(*device_cycles)));
    CHECK(mcMemcpy(
        input, host.data(), elements * sizeof(*input), mcMemcpyHostToDevice));
    CHECK(mcMemcpy(
        residual, host.data(), elements * sizeof(*residual),
        mcMemcpyHostToDevice));

    const int depths[] = {4, 5, 8, 16, 32};
    const int grids[] = {64};
    for (bool progress : {false, true}) {
        for (int warmup = 0; warmup < kWarmupsPerMode; ++warmup) {
            launch(
                progress, kMaximumGrid, kMaximumDepth,
                input, residual, output, device_cycles);
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (bool progress : {false, true}) {
        for (int grid : grids) {
            for (int depth : depths) {
                for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                    plan.push_back({depth, progress, sample, grid});
                }
            }
        }
    }
    std::mt19937 generator(500036U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);
    std::vector<unsigned long long> host_cycles(kMaximumGrid);

    std::printf(
        "metadata,1,MetaX C500,%d,%d,%d,%d,%d,%d,%d\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, resident_ctas);
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& cell = plan[order];
        launch(
            cell.progress, cell.grid, cell.depth,
            input, residual, output, device_cycles);
        CHECK(mcGetLastError());
        CHECK(mcMemcpy(
            host_cycles.data(), device_cycles,
            cell.grid * sizeof(*device_cycles), mcMemcpyDeviceToHost));
        std::vector<unsigned long long> observed(
            host_cycles.begin(), host_cycles.begin() + cell.grid);
        if (std::any_of(
                observed.begin(), observed.end(),
                [](unsigned long long value) { return value == 0; })) {
            std::fprintf(stderr, "progression probe emitted zero cycles\n");
            return EXIT_FAILURE;
        }
        const auto bounds = std::minmax_element(
            observed.begin(), observed.end());
        std::printf(
            "sample,%zu,%d,%d,%s,%d,%llu,%llu,%llu\n",
            order, cell.grid, cell.depth,
            cell.progress ? "progressing" : "same", cell.sample,
            *bounds.first, median(observed), *bounds.second);
    }

    CHECK(mcMemcpy(
        host.data(), output, elements * sizeof(*output),
        mcMemcpyDeviceToHost));
    if (host[0] == 0U) {
        std::fprintf(stderr, "progression output was not written\n");
        return EXIT_FAILURE;
    }
    CHECK(mcFree(device_cycles));
    CHECK(mcFree(output));
    CHECK(mcFree(residual));
    CHECK(mcFree(input));
    return EXIT_SUCCESS;
}
