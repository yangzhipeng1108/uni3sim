/*
 * Independent SM80 staged CTA-overlap mechanism probe.
 *
 * This is not a SiLU implementation.  Separate specializations isolate a
 * cached pair of generic U16 loads, one register-only EX2-to-RCP dependency,
 * and one generic U16 store.  No specialization contains the complete SiLU
 * dataflow or emits a production-like operator duration.  The live clock
 * values hold each stage in the production 24-register residency class.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do {                                               \
    cudaError_t error = (call);                                             \
    if (error != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

constexpr int kThreads = 1024;
constexpr int kSmCount = 108;
constexpr int kResidentCtas = 2;
constexpr int kRounds = 512;
constexpr int kBufferSlots = 8;
constexpr int kWarmups = 2;
constexpr int kCellCount = 12;
constexpr int kMaximumGrid = 1024;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";
constexpr const char* kHeader =
    "run_rotation,trial,order_position,stage,grid_blocks,block_threads,"
    "rounds,resident_cta_limit,active_sms,minimum_ctas_per_sm,"
    "maximum_ctas_per_sm,maximum_sm_span_ns,median_block_ns,stage_units,"
    "sm_cta_counts,sm_spans_ns,checksum";

enum class Stage : int { CACHED_TWO_U16_LOAD = 0, EX2_RCP = 1, U16_STORE = 2 };

struct Cell { Stage stage; int grid_blocks; };

__device__ __forceinline__ unsigned int read_smid() {
    unsigned int value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ std::uint64_t read_globaltimer() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ unsigned int read_clock() {
    unsigned int value;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(value));
    return value;
}

__device__ __forceinline__ float ex2_approx(float value) {
    float output;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(output) : "f"(value));
    return output;
}

__device__ __forceinline__ float rcp_approx(float value) {
    float output;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(output) : "f"(value));
    return output;
}

template <Stage stage>
__global__ __launch_bounds__(kThreads, kResidentCtas)
void silu_cta_overlap_stage(
        const __nv_bfloat16* __restrict__ input_a,
        const __nv_bfloat16* __restrict__ input_b,
        __nv_bfloat16* __restrict__ output,
        float* __restrict__ sink,
        std::uint64_t* __restrict__ start_ns,
        std::uint64_t* __restrict__ end_ns,
        unsigned int* __restrict__ smids) {
    const int linear_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int slot_stride = gridDim.x * blockDim.x;
    float state = 0.625f + static_cast<float>(threadIdx.x & 31) * 0.001953125f;
    const float lifetime0 = static_cast<float>(read_clock());
    const float lifetime1 = static_cast<float>(read_clock());
    const float lifetime2 = static_cast<float>(read_clock());
    const float lifetime3 = static_cast<float>(read_clock());
    const float lifetime4 = static_cast<float>(read_clock());
    const float lifetime5 = static_cast<float>(read_clock());
    const float lifetime6 = static_cast<float>(read_clock());
    const float lifetime7 = static_cast<float>(read_clock());
    const float lifetime8 = static_cast<float>(read_clock());
    const float lifetime9 = lifetime8 + state;

    __syncthreads();
    if (threadIdx.x == 0) {
        smids[blockIdx.x] = read_smid();
        start_ns[blockIdx.x] = read_globaltimer();
    }
    __syncthreads();

#pragma unroll 1
    for (int round = 0; round < kRounds; ++round) {
        const int slot = round & (kBufferSlots - 1);
        const int offset = linear_thread + slot * slot_stride;
        if constexpr (stage == Stage::CACHED_TWO_U16_LOAD) {
            const __nv_bfloat16 left = input_a[offset];
            const __nv_bfloat16 right = input_b[offset];
            state += (__bfloat162float(left) + __bfloat162float(right))
                * 0.00000011920928955078125f;
        } else if constexpr (stage == Stage::EX2_RCP) {
            const float exponent = (
                static_cast<float>((round + threadIdx.x) & 31) * 0.03125f
                - 0.5f);
            const float exponential = ex2_approx(exponent);
            const float reciprocal = rcp_approx(exponential + 1.0f);
            state += reciprocal * 0.00000011920928955078125f;
        } else {
            output[offset] = __float2bfloat16_rn(state);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) end_ns[blockIdx.x] = read_globaltimer();
    float sink_value = (
        state + lifetime0 + lifetime1 + lifetime2 + lifetime3 + lifetime4
        + lifetime5 + lifetime6 + lifetime7 + lifetime8);
    if constexpr (stage != Stage::U16_STORE) sink_value += lifetime9;
    sink[linear_thread] = sink_value;
}

template <Stage stage>
int resident_cta_limit() {
    int blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, silu_cta_overlap_stage<stage>, kThreads, 0));
    return blocks;
}

template <Stage stage>
void launch(const Cell& cell, const __nv_bfloat16* input_a,
            const __nv_bfloat16* input_b, __nv_bfloat16* output,
            float* sink,
            std::uint64_t* starts, std::uint64_t* ends,
            unsigned int* smids) {
    silu_cta_overlap_stage<stage><<<cell.grid_blocks, kThreads>>>(
        input_a, input_b, output, sink, starts, ends, smids);
}

const char* stage_name(Stage stage) {
    if (stage == Stage::CACHED_TWO_U16_LOAD) return "cached_two_u16_load";
    if (stage == Stage::EX2_RCP) return "register_ex2_rcp";
    if (stage == Stage::U16_STORE) return "u16_store";
    std::fprintf(stderr, "unknown stage\n");
    std::exit(EXIT_FAILURE);
}

int resident_limit(Stage stage) {
    if (stage == Stage::CACHED_TWO_U16_LOAD) {
        return resident_cta_limit<Stage::CACHED_TWO_U16_LOAD>();
    }
    if (stage == Stage::EX2_RCP) return resident_cta_limit<Stage::EX2_RCP>();
    if (stage == Stage::U16_STORE) return resident_cta_limit<Stage::U16_STORE>();
    std::fprintf(stderr, "unknown stage\n");
    std::exit(EXIT_FAILURE);
}

void launch_cell(const Cell& cell, const __nv_bfloat16* input_a,
                 const __nv_bfloat16* input_b, __nv_bfloat16* output,
                 float* sink,
                 std::uint64_t* starts, std::uint64_t* ends,
                 unsigned int* smids) {
    if (cell.stage == Stage::CACHED_TWO_U16_LOAD) {
        launch<Stage::CACHED_TWO_U16_LOAD>(
            cell, input_a, input_b, output, sink, starts, ends, smids);
    } else if (cell.stage == Stage::EX2_RCP) {
        launch<Stage::EX2_RCP>(
            cell, input_a, input_b, output, sink, starts, ends, smids);
    } else if (cell.stage == Stage::U16_STORE) {
        launch<Stage::U16_STORE>(
            cell, input_a, input_b, output, sink, starts, ends, smids);
    } else {
        std::fprintf(stderr, "unknown stage\n");
        std::exit(EXIT_FAILURE);
    }
}

void run_cell(const Cell& cell, int run_rotation, int trial,
              int order_position, const __nv_bfloat16* input_a,
              const __nv_bfloat16* input_b, __nv_bfloat16* output,
              float* sink,
              std::uint64_t* starts, std::uint64_t* ends,
              unsigned int* smids) {
    const int limit = resident_limit(cell.stage);
    if (limit != kResidentCtas) {
        std::fprintf(stderr, "stage does not admit two resident CTAs: %s=%d\n",
                     stage_name(cell.stage), limit);
        std::exit(EXIT_FAILURE);
    }
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        launch_cell(cell, input_a, input_b, output, sink, starts, ends, smids);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    launch_cell(cell, input_a, input_b, output, sink, starts, ends, smids);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<std::uint64_t> host_starts(cell.grid_blocks);
    std::vector<std::uint64_t> host_ends(cell.grid_blocks);
    std::vector<unsigned int> host_smids(cell.grid_blocks);
    CHECK_CUDA(cudaMemcpy(host_starts.data(), starts,
                          cell.grid_blocks * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_ends.data(), ends,
                          cell.grid_blocks * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_smids.data(), smids,
                          cell.grid_blocks * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    std::vector<std::uint64_t> sm_start(kSmCount, UINT64_MAX);
    std::vector<std::uint64_t> sm_end(kSmCount, 0);
    std::vector<std::uint64_t> block_ns;
    std::vector<int> ctas_per_sm(kSmCount, 0);
    for (int block = 0; block < cell.grid_blocks; ++block) {
        const unsigned int sm = host_smids[block];
        if (sm >= static_cast<unsigned int>(kSmCount)
                || host_ends[block] <= host_starts[block]) {
            std::fprintf(stderr, "invalid block timing record\n");
            std::exit(EXIT_FAILURE);
        }
        ++ctas_per_sm[sm];
        sm_start[sm] = std::min(sm_start[sm], host_starts[block]);
        sm_end[sm] = std::max(sm_end[sm], host_ends[block]);
        block_ns.push_back(host_ends[block] - host_starts[block]);
    }
    const int active_sms = static_cast<int>(std::count_if(
        ctas_per_sm.begin(), ctas_per_sm.end(), [](int count) { return count > 0; }));
    const int minimum_ctas = *std::min_element(
        ctas_per_sm.begin(), ctas_per_sm.end());
    const int maximum_ctas = *std::max_element(
        ctas_per_sm.begin(), ctas_per_sm.end());
    std::uint64_t maximum_sm_span = 0;
    for (int sm = 0; sm < kSmCount; ++sm) {
        if (ctas_per_sm[sm] > 0) {
            maximum_sm_span = std::max(
                maximum_sm_span, sm_end[sm] - sm_start[sm]);
        }
    }
    std::sort(block_ns.begin(), block_ns.end());
    __nv_bfloat16 checksum_value;
    CHECK_CUDA(cudaMemcpy(&checksum_value, output,
                          sizeof(checksum_value), cudaMemcpyDeviceToHost));
    const unsigned int checksum =
        *reinterpret_cast<const unsigned short*>(&checksum_value);
    const std::uint64_t stage_units = static_cast<std::uint64_t>(
        cell.grid_blocks) * kThreads * kRounds;
    std::string sm_cta_counts;
    std::string sm_spans_ns;
    for (int sm = 0; sm < kSmCount; ++sm) {
        if (sm > 0) {
            sm_cta_counts += ';';
            sm_spans_ns += ';';
        }
        sm_cta_counts += std::to_string(ctas_per_sm[sm]);
        sm_spans_ns += std::to_string(
            ctas_per_sm[sm] > 0 ? sm_end[sm] - sm_start[sm] : 0);
    }
    std::printf(
        "%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%s,%s,%u\n",
        run_rotation, trial, order_position, stage_name(cell.stage),
        cell.grid_blocks, kThreads, kRounds, limit, active_sms,
        minimum_ctas, maximum_ctas,
        static_cast<unsigned long long>(maximum_sm_span),
        static_cast<unsigned long long>(block_ns[block_ns.size() / 2]),
        static_cast<unsigned long long>(stage_units), sm_cta_counts.c_str(),
        sm_spans_ns.c_str(), checksum);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS RUN_ROTATION\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::stoi(argv[1]);
    const int trials = std::stoi(argv[2]);
    const int run_rotation = std::stoi(argv[3]);
    if (device < 0 || trials != 3 || run_rotation < 0 || run_rotation >= 3) {
        std::fprintf(stderr, "device>=0, trials=3, and run rotation in [0,2] required\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != kExpectedDevice
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "unexpected device geometry\n");
        return EXIT_FAILURE;
    }

    const int grids[] = {108, 216, 324, 1024};
    const Stage stages[] = {
        Stage::CACHED_TWO_U16_LOAD, Stage::EX2_RCP, Stage::U16_STORE};
    std::vector<Cell> cells;
    for (Stage stage : stages) {
        for (int grid : grids) cells.push_back({stage, grid});
    }
    if (cells.size() != kCellCount) return EXIT_FAILURE;

    const std::size_t values = static_cast<std::size_t>(
        kMaximumGrid) * kThreads * kBufferSlots;
    std::vector<__nv_bfloat16> host_input(values);
    for (std::size_t index = 0; index < values; ++index) {
        host_input[index] = __float2bfloat16_rn(
            0.25f + static_cast<float>(index & 127) * 0.001953125f);
    }
    __nv_bfloat16 *input_a = nullptr, *input_b = nullptr, *output = nullptr;
    float* sink = nullptr;
    std::uint64_t *starts = nullptr, *ends = nullptr;
    unsigned int* smids = nullptr;
    CHECK_CUDA(cudaMalloc(&input_a, values * sizeof(*input_a)));
    CHECK_CUDA(cudaMalloc(&input_b, values * sizeof(*input_b)));
    CHECK_CUDA(cudaMalloc(&output, values * sizeof(*output)));
    CHECK_CUDA(cudaMalloc(&sink,
                          static_cast<std::size_t>(kMaximumGrid) * kThreads
                          * sizeof(*sink)));
    CHECK_CUDA(cudaMalloc(&starts, kMaximumGrid * sizeof(*starts)));
    CHECK_CUDA(cudaMalloc(&ends, kMaximumGrid * sizeof(*ends)));
    CHECK_CUDA(cudaMalloc(&smids, kMaximumGrid * sizeof(*smids)));
    CHECK_CUDA(cudaMemcpy(input_a, host_input.data(), values * sizeof(*input_a),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(input_b, host_input.data(), values * sizeof(*input_b),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(output, 0, values * sizeof(*output)));

    std::puts(kHeader);
    for (int trial = 0; trial < trials; ++trial) {
        for (int position = 0; position < kCellCount; ++position) {
            const int cell_index = (
                position + 5 * trial + 4 * run_rotation) % kCellCount;
            run_cell(cells[cell_index], run_rotation, trial, position,
                     input_a, input_b, output, sink, starts, ends, smids);
        }
    }

    CHECK_CUDA(cudaFree(smids));
    CHECK_CUDA(cudaFree(ends));
    CHECK_CUDA(cudaFree(starts));
    CHECK_CUDA(cudaFree(output));
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(input_b));
    CHECK_CUDA(cudaFree(input_a));
    return EXIT_SUCCESS;
}
