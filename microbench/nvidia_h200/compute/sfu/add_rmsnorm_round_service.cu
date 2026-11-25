/*
 * H200 owner copy of microbench/nvidia_a800/compute/sfu/add_rmsnorm_round_service.cu.
 * Kernel body, 512 rounds, 8-slot buffers, warm-ups, cells and CSV fields are
 * unchanged; the loops remain the vLLM 0.11.0 BF16 vec8 Add-RMSNorm source
 * rounds.  Hardware coupling only: kSmCount 108 -> 132 (cells first = 132
 * CTAs, resident = 8 x 132 = 1056 CTAs), device identity "NVIDIA H200", built
 * for sm_90, and kMaximumGridBlocks 1024 -> 1252 = round(1024 * 132 / 108) so
 * the 1056-CTA resident cell fits the buffers (the optional queued_wave_floor
 * mode, not used by this package, scales the same way).
 */
/*
 * Independent SM80 BF16 Add-RMSNorm source-round service probe.
 *
 * This is not an Add-RMSNorm implementation. The two specializations isolate
 * the width-eight work on either side of the block reduction. There is no
 * reduction, reciprocal square root, epsilon, or production tensor shape.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
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

constexpr int kThreads = 256;
constexpr int kSmCount = 132;
constexpr int kResidentCtas = 8;
constexpr int kMaximumGridBlocks = 1252;
constexpr int kBufferSlots = 8;
constexpr int kRounds = 512;
constexpr int kWarmups = 3;
constexpr int kQueuedCampaignWarmups = 16;
constexpr const char* kExpectedDevice = "NVIDIA H200";
constexpr const char* kHeader =
    "trial,function,cell,grid_blocks,block_threads,rounds,"
    "resident_cta_limit,active_sms,minimum_ctas_per_sm,"
    "maximum_ctas_per_sm,maximum_sm_span_ns,median_block_ns,checksum";

enum class RoundFunction : int { ADD_SQUARE = 0, SCALE_MULTIPLY = 1 };

struct Cell { const char* name; int grid_blocks; };

struct alignas(16) Pack {
    union { uint4 words; __nv_bfloat16 values[8]; };
};

static_assert(sizeof(Pack) == 16);

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

template <RoundFunction function>
__global__ __launch_bounds__(kThreads, kResidentCtas)
void add_rmsnorm_round_service(
        const Pack* __restrict__ input, Pack* __restrict__ residual,
        const Pack* __restrict__ gamma, Pack* __restrict__ output,
        float* __restrict__ sink, int rounds,
        std::uint64_t* __restrict__ start_ns,
        std::uint64_t* __restrict__ end_ns,
        unsigned int* __restrict__ smids) {
    float state = 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        smids[blockIdx.x] = read_smid();
        start_ns[blockIdx.x] = read_globaltimer();
    }
    __syncthreads();
#pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
        const int slot = round & (kBufferSlots - 1);
        const int offset = (
            (blockIdx.x * kBufferSlots + slot) * blockDim.x + threadIdx.x);
        Pack result;
        if constexpr (function == RoundFunction::ADD_SQUARE) {
            const Pack left = input[offset];
            const Pack right = residual[offset];
#pragma unroll
            for (int lane = 0; lane < 8; ++lane) {
                const float value = (
                    __bfloat162float(left.values[lane])
                    + __bfloat162float(right.values[lane]));
                state = fmaf(value, value, state);
                result.values[lane] = __float2bfloat16_rn(value);
            }
            residual[offset] = result;
        } else {
            const Pack value = residual[offset];
            const Pack weight = gamma[threadIdx.x];
#pragma unroll
            for (int lane = 0; lane < 8; ++lane) {
                const float product = (
                    __bfloat162float(value.values[lane]) * 0.875f
                    * __bfloat162float(weight.values[lane]));
                state += product * 0.00000011920928955078125f;
                result.values[lane] = __float2bfloat16_rn(product);
            }
            output[offset] = result;
        }
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = state;
    __syncthreads();
    if (threadIdx.x == 0) end_ns[blockIdx.x] = read_globaltimer();
}

template <RoundFunction function>
int resident_cta_limit() {
    int blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, add_rmsnorm_round_service<function>, kThreads, 0));
    return blocks;
}

template <RoundFunction function>
void launch(const Cell& cell, const Pack* input, Pack* residual,
            const Pack* gamma, Pack* output, float* sink,
            std::uint64_t* starts, std::uint64_t* ends, unsigned int* smids) {
    add_rmsnorm_round_service<function><<<cell.grid_blocks, kThreads>>>(
        input, residual, gamma, output, sink, kRounds, starts, ends, smids);
}

void warm_queued_campaign(
        const Cell& resident, const Cell& queued,
        const Pack* input, Pack* residual, const Pack* gamma, Pack* output,
        float* sink, std::uint64_t* starts, std::uint64_t* ends,
        unsigned int* smids) {
    for (int warmup = 0; warmup < kQueuedCampaignWarmups; ++warmup) {
        launch<RoundFunction::ADD_SQUARE>(
            queued, input, residual, gamma, output, sink, starts, ends, smids);
        launch<RoundFunction::ADD_SQUARE>(
            resident, input, residual, gamma, output, sink, starts, ends, smids);
        launch<RoundFunction::SCALE_MULTIPLY>(
            queued, input, residual, gamma, output, sink, starts, ends, smids);
        launch<RoundFunction::SCALE_MULTIPLY>(
            resident, input, residual, gamma, output, sink, starts, ends, smids);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

void run_cell(RoundFunction function, const Cell& cell, int trials,
              int trial_offset,
              const Pack* input, Pack* residual, const Pack* gamma,
              Pack* output, float* sink, std::uint64_t* starts,
              std::uint64_t* ends, unsigned int* smids) {
    const int resident_limit = function == RoundFunction::ADD_SQUARE
        ? resident_cta_limit<RoundFunction::ADD_SQUARE>()
        : resident_cta_limit<RoundFunction::SCALE_MULTIPLY>();
    if (resident_limit != kResidentCtas) {
        std::fprintf(stderr, "unexpected resident CTA limit: %d\n", resident_limit);
        std::exit(EXIT_FAILURE);
    }
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        if (function == RoundFunction::ADD_SQUARE) {
            launch<RoundFunction::ADD_SQUARE>(
                cell, input, residual, gamma, output, sink, starts, ends, smids);
        } else {
            launch<RoundFunction::SCALE_MULTIPLY>(
                cell, input, residual, gamma, output, sink, starts, ends, smids);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<std::uint64_t> host_starts(cell.grid_blocks);
    std::vector<std::uint64_t> host_ends(cell.grid_blocks);
    std::vector<unsigned int> host_smids(cell.grid_blocks);
    std::vector<float> host_sink(cell.grid_blocks * kThreads);
    for (int local_trial = 0; local_trial < trials; ++local_trial) {
        const int trial = trial_offset + local_trial;
        if (function == RoundFunction::ADD_SQUARE) {
            launch<RoundFunction::ADD_SQUARE>(
                cell, input, residual, gamma, output, sink, starts, ends, smids);
        } else {
            launch<RoundFunction::SCALE_MULTIPLY>(
                cell, input, residual, gamma, output, sink, starts, ends, smids);
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
        CHECK_CUDA(cudaMemcpy(host_sink.data(), sink,
                              host_sink.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::vector<std::uint64_t> block_ns;
        std::vector<std::uint64_t> sm_start(kSmCount, UINT64_MAX);
        std::vector<std::uint64_t> sm_end(kSmCount, 0);
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
            ctas_per_sm.begin(), ctas_per_sm.end(),
            [](int count) { return count > 0; }));
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
        double checksum = 0.0;
        for (float value : host_sink) checksum += std::isfinite(value) ? value : 0.0;
        const char* function_name = function == RoundFunction::ADD_SQUARE
            ? "add_square" : "scale_multiply";
        std::printf(
            "%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%.9g\n",
            trial, function_name, cell.name, cell.grid_blocks, kThreads,
            kRounds, resident_limit, active_sms, minimum_ctas, maximum_ctas,
            static_cast<unsigned long long>(maximum_sm_span),
            static_cast<unsigned long long>(block_ns[block_ns.size() / 2]),
            checksum);
    }
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(
            stderr, "usage: %s DEVICE TRIALS [queued_wave_floor]\n", argv[0]);
        return EXIT_FAILURE;
    }
    char* device_end = nullptr;
    char* trials_end = nullptr;
    const long device_value = std::strtol(argv[1], &device_end, 10);
    const long trials_value = std::strtol(argv[2], &trials_end, 10);
    const bool queued_wave_floor = argc == 4;
    if (*argv[1] == '\0' || *device_end != '\0'
            || *argv[2] == '\0' || *trials_end != '\0'
            || device_value < 0 || device_value > INT_MAX
            || trials_value <= 0 || trials_value > INT_MAX
            || (queued_wave_floor
                && std::string(argv[3]) != "queued_wave_floor")) {
        return EXIT_FAILURE;
    }
    const int device = static_cast<int>(device_value);
    const int trials = static_cast<int>(trials_value);
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != kExpectedDevice
            || properties.multiProcessorCount != kSmCount) return EXIT_FAILURE;

    const std::size_t packs = static_cast<std::size_t>(
        kMaximumGridBlocks) * kBufferSlots * kThreads;
    Pack *input = nullptr, *residual = nullptr, *gamma = nullptr, *output = nullptr;
    float* sink = nullptr;
    std::uint64_t *starts = nullptr, *ends = nullptr;
    unsigned int* smids = nullptr;
    CHECK_CUDA(cudaMalloc(&input, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&residual, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&gamma, kThreads * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&output, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&sink, kMaximumGridBlocks * kThreads * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&starts, kMaximumGridBlocks * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&ends, kMaximumGridBlocks * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&smids, kMaximumGridBlocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(input, 0x3f, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(residual, 0x3e, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(gamma, 0x3f, kThreads * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(output, 0, packs * sizeof(Pack)));

    std::puts(kHeader);
    if (queued_wave_floor) {
        const Cell resident{"resident", kResidentCtas * kSmCount};
        const Cell queued{"queued_wave", kMaximumGridBlocks};
        warm_queued_campaign(
            resident, queued, input, residual, gamma, output, sink,
            starts, ends, smids);
        for (int trial = 0; trial < trials; ++trial) {
            for (RoundFunction function : {
                     RoundFunction::ADD_SQUARE,
                     RoundFunction::SCALE_MULTIPLY}) {
                if (trial % 2 == 0) {
                    run_cell(function, queued, 1, trial, input, residual,
                             gamma, output, sink, starts, ends, smids);
                    run_cell(function, resident, 1, trial, input, residual,
                             gamma, output, sink, starts, ends, smids);
                } else {
                    run_cell(function, resident, 1, trial, input, residual,
                             gamma, output, sink, starts, ends, smids);
                    run_cell(function, queued, 1, trial, input, residual,
                             gamma, output, sink, starts, ends, smids);
                }
            }
        }
    } else {
        const Cell cells[] = {
            {"first", kSmCount},
            {"resident", kResidentCtas * kSmCount},
        };
        for (RoundFunction function : {
                 RoundFunction::ADD_SQUARE,
                 RoundFunction::SCALE_MULTIPLY}) {
            for (const Cell& cell : cells) {
                run_cell(function, cell, trials, 0, input, residual, gamma,
                         output, sink, starts, ends, smids);
            }
        }
    }
    CHECK_CUDA(cudaFree(smids));
    CHECK_CUDA(cudaFree(ends));
    CHECK_CUDA(cudaFree(starts));
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(output));
    CHECK_CUDA(cudaFree(gamma));
    CHECK_CUDA(cudaFree(residual));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
