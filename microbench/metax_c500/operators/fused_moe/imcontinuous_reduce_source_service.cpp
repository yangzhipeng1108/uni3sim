/*
 * MetaX C500 imcontinuous-reduction source-service benchmark.
 *
 * This is an independent vec4 reduction primitive, not a PyTorch or FusedMoE
 * operator.  It preserves the static MACA source structure that matters to
 * the hardware service: one 512-thread CTA, a strided per-thread vec4 input
 * fold, shared-memory inter-wave combines, CTA barriers, and a vec4 output.
 * Adjacent runtime repetition counts remove launch and fixed clock64 work.
 * Multiple reduction depths and grids separate the source-program service
 * from one production workload shape.
 *
 * Build in the MACA 3.7.2 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *     -O3 -std=c++17 imcontinuous_reduce_source_service.cpp \
 *     -o imcontinuous_reduce_source_service
 */

#include <mc_runtime.h>

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
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kThreads = 512;
constexpr std::size_t kDynamicSharedBytes = 4104;
constexpr int kSamplesPerCell = 17;

struct alignas(16) Vec4 {
    float value[4];
};

struct Program {
    int block_x;
    int block_y;
    int depth;
};

struct Cell {
    Program program;
    int grid;
    int repetitions;
    int sample;
};

__device__ __forceinline__ Vec4 add_vec4(Vec4 left, const Vec4& right) {
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        left.value[lane] += right.value[lane];
    }
    return left;
}

__global__ void imcontinuous_reduce_source_cycles(
        const Vec4* input, Vec4* output,
        unsigned long long* block_cycles, int depth, int repetitions) {
    extern __shared__ unsigned char shared_bytes[];
    Vec4* shared = reinterpret_cast<Vec4*>(shared_bytes);
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int output_vectors = static_cast<int>(gridDim.x * blockDim.x);
    const int output_vector = static_cast<int>(blockIdx.x * blockDim.x) + tx;
    const int wave_size =
        blockDim.x > kExpectedWaveSize ? blockDim.x : kExpectedWaveSize;
    const int wave_y_lanes = wave_size / static_cast<int>(blockDim.x);
    const int wave_id = ty / wave_y_lanes;

    __syncthreads();
    const unsigned long long start = clock64();
    Vec4 retained {{0.0f, 0.0f, 0.0f, 0.0f}};
    #pragma unroll 1
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        Vec4 thread_value {{0.0f, 0.0f, 0.0f, 0.0f}};
        #pragma unroll 1
        for (int input_index = ty; input_index < depth;
             input_index += static_cast<int>(blockDim.y)) {
            thread_value = add_vec4(
                thread_value,
                input[input_index * output_vectors + output_vector]);
        }

        __syncthreads();
        for (int offset = static_cast<int>(blockDim.y) >> 1;
             offset >= wave_y_lanes; offset >>= 1) {
            if (ty >= offset && ty < 2 * offset) {
                const int shared_index = (ty - offset)
                    * static_cast<int>(blockDim.x) + tx;
                shared[shared_index] = thread_value;
            }
            __syncthreads();
            if (ty < offset) {
                const int shared_index = ty
                    * static_cast<int>(blockDim.x) + tx;
                thread_value = add_vec4(thread_value, shared[shared_index]);
            }
            __syncthreads();
        }

        if (wave_id == 0) {
            for (int offset = wave_size >> 1;
                 offset >= static_cast<int>(blockDim.x); offset >>= 1) {
                Vec4 other;
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    other.value[lane] = __shfl_down(
                        thread_value.value[lane], offset, wave_size);
                }
                thread_value = add_vec4(thread_value, other);
            }
        }
        __syncthreads();
        if (ty == 0) {
            output[output_vector] = thread_value;
            retained = add_vec4(retained, thread_value);
        }
    }
    const unsigned long long end = clock64();
    if (tx == 0 && ty == 0) {
        block_cycles[blockIdx.x] = end - start
            + static_cast<unsigned long long>(retained.value[0] == -1.0f);
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
        std::fprintf(stderr, "expected C500 104 AP/wave64 and positive clock\n");
        return EXIT_FAILURE;
    }

    const Program programs[] = {
        {256, 2, 2},
        {128, 4, 4},
        {64, 8, 8},
        {64, 8, 10},
        {32, 16, 16},
    };
    const int grids[] = {1, 8, kExpectedApCount};
    const int repetition_counts[] = {4, 8, 16, 32};
    constexpr int maximum_depth = 16;
    constexpr int maximum_grid = kExpectedApCount;
    constexpr int maximum_block_x = 256;
    const std::size_t maximum_vectors = static_cast<std::size_t>(
        maximum_depth) * maximum_grid * maximum_block_x;
    Vec4* input = nullptr;
    Vec4* output = nullptr;
    unsigned long long* device_cycles = nullptr;
    CHECK(mcMalloc(reinterpret_cast<void**>(&input),
                   maximum_vectors * sizeof(Vec4)));
    CHECK(mcMalloc(reinterpret_cast<void**>(&output),
                   maximum_grid * maximum_block_x * sizeof(Vec4)));
    CHECK(mcMalloc(reinterpret_cast<void**>(&device_cycles),
                   maximum_grid * sizeof(unsigned long long)));
    CHECK(mcMemset(input, 0, maximum_vectors * sizeof(Vec4)));

    for (const Program& program : programs) {
        const dim3 block(program.block_x, program.block_y, 1);
        for (int grid : grids) {
            for (int repetitions : repetition_counts) {
                imcontinuous_reduce_source_cycles<<<
                    grid, block, kDynamicSharedBytes>>>(
                        input, output, device_cycles,
                        program.depth, repetitions);
            }
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (const Program& program : programs) {
        for (int grid : grids) {
            for (int repetitions : repetition_counts) {
                for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                    plan.push_back({program, grid, repetitions, sample});
                }
            }
        }
    }
    std::mt19937 generator(500907U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);
    std::vector<unsigned long long> host_cycles(maximum_grid);

    std::printf(
        "metadata,1,MetaX C500,%d,%d,%d,%d,%d,%d,%zu\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kSamplesPerCell, kDynamicSharedBytes);
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& cell = plan[order];
        const dim3 block(cell.program.block_x, cell.program.block_y, 1);
        imcontinuous_reduce_source_cycles<<<
            cell.grid, block, kDynamicSharedBytes>>>(
                input, output, device_cycles,
                cell.program.depth, cell.repetitions);
        CHECK(mcGetLastError());
        CHECK(mcMemcpy(
            host_cycles.data(), device_cycles,
            cell.grid * sizeof(unsigned long long), mcMemcpyDeviceToHost));
        std::vector<unsigned long long> observed(
            host_cycles.begin(), host_cycles.begin() + cell.grid);
        if (std::any_of(observed.begin(), observed.end(),
                        [](unsigned long long value) { return value == 0; })) {
            std::fprintf(stderr, "reduction probe produced a zero cycle span\n");
            return EXIT_FAILURE;
        }
        const auto bounds = std::minmax_element(
            observed.begin(), observed.end());
        std::printf(
            "sample,%zu,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu\n",
            order, cell.program.block_x, cell.program.block_y,
            cell.program.depth, cell.grid, cell.repetitions, cell.sample,
            *bounds.first, median(observed), *bounds.second);
    }

    CHECK(mcFree(device_cycles));
    CHECK(mcFree(output));
    CHECK(mcFree(input));
    return EXIT_SUCCESS;
}
