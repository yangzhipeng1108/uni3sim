/*
 * Independent SM80 BF16 load/SFU/store pipeline probe.
 *
 * This is not a SiLU implementation.  It preserves only the instruction
 * classes and per-round global-memory widths needed to test whether two
 * 1024-thread CTAs can hand load, SFU, and store work across resident slots.
 * The independent specialization constructs every SFU input from the current
 * round's loads; the control deliberately adds a loop-carried SFU dependency.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
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
constexpr int kMaximumGridBlocks = 3 * kSmCount;
constexpr int kShortRounds = 64;
constexpr int kSteadyRounds = 512;
constexpr int kWarmups = 3;
constexpr int kBytesLoadedPerThreadRound = 4;
constexpr int kBytesStoredPerThreadRound = 2;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";
constexpr const char* kHeader =
    "trial,cell,dependency,grid_blocks,block_threads,rounds,"
    "resident_cta_limit,active_sms,loaded_bytes,stored_bytes,"
    "minimum_ctas_per_sm,maximum_ctas_per_sm,maximum_sm_span_cycles,"
    "median_block_cycles,checksum";


enum Dependency : int {
    INDEPENDENT_ROUNDS = 0,
    LOOP_CARRIED_SFU = 1,
};


struct Cell {
    const char* name;
    int grid_blocks;
    int rounds;
    Dependency dependency;
};


__device__ __forceinline__ unsigned int read_smid() {
    unsigned int value;
    asm volatile("mov.u32 %0, %smid;" : "=r"(value));
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


template <Dependency dependency>
__global__ __launch_bounds__(kThreads, 2)
void bf16_cta_pipeline(const std::uint32_t* __restrict__ input,
                       __nv_bfloat16* __restrict__ output,
                       int rounds,
                       std::uint64_t* __restrict__ start_cycles,
                       std::uint64_t* __restrict__ end_cycles,
                       unsigned int* __restrict__ smids) {
    const int linear_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    float state = 0.75f + static_cast<float>(threadIdx.x & 31) * 0.0009765625f;

    __syncthreads();
    if (threadIdx.x == 0) {
        smids[blockIdx.x] = read_smid();
        start_cycles[blockIdx.x] = clock64();
    }
    __syncthreads();

#pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
        const std::uint32_t packed = input[linear_thread + round * stride];
        const __nv_bfloat16 low = *reinterpret_cast<const __nv_bfloat16*>(&packed);
        const __nv_bfloat16 high = *(reinterpret_cast<const __nv_bfloat16*>(&packed) + 1);
        const float a = __bfloat162float(low);
        const float b = __bfloat162float(high);
        float exponent;
        if constexpr (dependency == INDEPENDENT_ROUNDS) {
            exponent = a * 0.125f + b * 0.0625f - 0.5f;
        } else {
            exponent = state * 0.5f + a * 0.0625f;
        }
        const float exponential = ex2_approx(exponent);
        state = rcp_approx(exponential + 1.0f) + b * 0.0009765625f;
        output[linear_thread + round * stride] = __float2bfloat16_rn(state);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        end_cycles[blockIdx.x] = clock64();
    }
}


template <Dependency dependency>
int resident_cta_limit() {
    int blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, bf16_cta_pipeline<dependency>, kThreads, 0));
    return blocks;
}


template <Dependency dependency>
void launch_cell(const Cell& cell,
                 const std::uint32_t* input,
                 __nv_bfloat16* output,
                 std::uint64_t* starts,
                 std::uint64_t* ends,
                 unsigned int* smids) {
    bf16_cta_pipeline<dependency><<<cell.grid_blocks, kThreads>>>(
        input, output, cell.rounds, starts, ends, smids);
}


void run_cell(const Cell& cell,
              int trials,
              const std::uint32_t* input,
              __nv_bfloat16* output,
              std::uint64_t* starts,
              std::uint64_t* ends,
              unsigned int* smids) {
    const int resident_limit = cell.dependency == INDEPENDENT_ROUNDS
        ? resident_cta_limit<INDEPENDENT_ROUNDS>()
        : resident_cta_limit<LOOP_CARRIED_SFU>();
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        if (cell.dependency == INDEPENDENT_ROUNDS) {
            launch_cell<INDEPENDENT_ROUNDS>(cell, input, output, starts, ends, smids);
        } else {
            launch_cell<LOOP_CARRIED_SFU>(cell, input, output, starts, ends, smids);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<std::uint64_t> host_starts(cell.grid_blocks);
    std::vector<std::uint64_t> host_ends(cell.grid_blocks);
    std::vector<unsigned int> host_smids(cell.grid_blocks);
    for (int trial = 0; trial < trials; ++trial) {
        if (cell.dependency == INDEPENDENT_ROUNDS) {
            launch_cell<INDEPENDENT_ROUNDS>(cell, input, output, starts, ends, smids);
        } else {
            launch_cell<LOOP_CARRIED_SFU>(cell, input, output, starts, ends, smids);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(host_starts.data(), starts,
                              cell.grid_blocks * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_ends.data(), ends,
                              cell.grid_blocks * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_smids.data(), smids,
                              cell.grid_blocks * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        std::vector<std::uint64_t> block_cycles;
        std::vector<std::uint64_t> sm_start(kSmCount, UINT64_MAX);
        std::vector<std::uint64_t> sm_end(kSmCount, 0);
        std::vector<bool> active(kSmCount, false);
        for (int block = 0; block < cell.grid_blocks; ++block) {
            if (host_smids[block] >= static_cast<unsigned int>(kSmCount)
                    || host_ends[block] <= host_starts[block]) {
                std::fprintf(stderr, "invalid block timing record\n");
                std::exit(EXIT_FAILURE);
            }
            const unsigned int sm = host_smids[block];
            active[sm] = true;
            sm_start[sm] = std::min(sm_start[sm], host_starts[block]);
            sm_end[sm] = std::max(sm_end[sm], host_ends[block]);
            block_cycles.push_back(host_ends[block] - host_starts[block]);
        }
        const int active_sms = static_cast<int>(
            std::count(active.begin(), active.end(), true));
        std::uint64_t maximum_sm_span = 0;
        std::vector<int> ctas_per_sm(kSmCount, 0);
        for (int block = 0; block < cell.grid_blocks; ++block) {
            ++ctas_per_sm[host_smids[block]];
        }
        const int minimum_ctas_per_sm = *std::min_element(
            ctas_per_sm.begin(), ctas_per_sm.end());
        const int maximum_ctas_per_sm = *std::max_element(
            ctas_per_sm.begin(), ctas_per_sm.end());
        for (int sm = 0; sm < kSmCount; ++sm) {
            if (active[sm]) {
                maximum_sm_span = std::max(maximum_sm_span,
                                             sm_end[sm] - sm_start[sm]);
            }
        }
        std::sort(block_cycles.begin(), block_cycles.end());
        const std::uint64_t median_block = block_cycles[block_cycles.size() / 2];
        __nv_bfloat16 checksum_value;
        CHECK_CUDA(cudaMemcpy(&checksum_value, output,
                              sizeof(checksum_value), cudaMemcpyDeviceToHost));
        const unsigned int checksum =
            *reinterpret_cast<const unsigned short*>(&checksum_value);
        const std::uint64_t thread_rounds =
            static_cast<std::uint64_t>(cell.grid_blocks) * kThreads * cell.rounds;
        std::printf(
                    "%d,%s,%s,%d,%d,%d,%d,%d,%llu,%llu,%d,%d,%llu,%llu,%u\n",
                    trial, cell.name,
                    cell.dependency == INDEPENDENT_ROUNDS
                        ? "independent_rounds" : "loop_carried_sfu",
                    cell.grid_blocks, kThreads, cell.rounds, resident_limit,
                    active_sms,
                    static_cast<unsigned long long>(
                        thread_rounds * kBytesLoadedPerThreadRound),
                    static_cast<unsigned long long>(
                        thread_rounds * kBytesStoredPerThreadRound),
                    minimum_ctas_per_sm, maximum_ctas_per_sm,
                    static_cast<unsigned long long>(maximum_sm_span),
                    static_cast<unsigned long long>(median_block), checksum);
    }
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::stoi(argv[1]);
    const int trials = std::stoi(argv[2]);
    if (device < 0 || trials < 3) {
        std::fprintf(stderr, "device must be nonnegative and trials >= 3\n");
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
    const std::vector<Cell> cells = {
        {"single_cta_steady", kSmCount, kSteadyRounds, INDEPENDENT_ROUNDS},
        {"dual_cta_short", 2 * kSmCount, kShortRounds, INDEPENDENT_ROUNDS},
        {"dual_cta_steady", 2 * kSmCount, kSteadyRounds, INDEPENDENT_ROUNDS},
        {"dual_cta_next_wave_steady", 3 * kSmCount, kSteadyRounds,
         INDEPENDENT_ROUNDS},
        {"single_cta_dependent_control", kSmCount, kSteadyRounds, LOOP_CARRIED_SFU},
    };
    const std::size_t maximum_thread_rounds = static_cast<std::size_t>(
        kMaximumGridBlocks) * kThreads * kSteadyRounds;
    std::vector<std::uint32_t> host_input(maximum_thread_rounds);
    for (std::size_t index = 0; index < host_input.size(); ++index) {
        const unsigned short low = static_cast<unsigned short>(
            0x3f00u + (index & 0x3fu));
        const unsigned short high = static_cast<unsigned short>(
            0x3e80u + ((index >> 6) & 0x3fu));
        host_input[index] = static_cast<std::uint32_t>(low)
            | (static_cast<std::uint32_t>(high) << 16);
    }
    std::uint32_t* input = nullptr;
    __nv_bfloat16* output = nullptr;
    std::uint64_t* starts = nullptr;
    std::uint64_t* ends = nullptr;
    unsigned int* smids = nullptr;
    CHECK_CUDA(cudaMalloc(&input, maximum_thread_rounds * sizeof(*input)));
    CHECK_CUDA(cudaMalloc(&output, maximum_thread_rounds * sizeof(*output)));
    CHECK_CUDA(cudaMalloc(&starts, kMaximumGridBlocks * sizeof(*starts)));
    CHECK_CUDA(cudaMalloc(&ends, kMaximumGridBlocks * sizeof(*ends)));
    CHECK_CUDA(cudaMalloc(&smids, kMaximumGridBlocks * sizeof(*smids)));
    CHECK_CUDA(cudaMemcpy(input, host_input.data(),
                          maximum_thread_rounds * sizeof(*input),
                          cudaMemcpyHostToDevice));
    std::puts(kHeader);
    for (const Cell& cell : cells) {
        run_cell(cell, trials, input, output, starts, ends, smids);
    }
    CHECK_CUDA(cudaFree(smids));
    CHECK_CUDA(cudaFree(ends));
    CHECK_CUDA(cudaFree(starts));
    CHECK_CUDA(cudaFree(output));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
