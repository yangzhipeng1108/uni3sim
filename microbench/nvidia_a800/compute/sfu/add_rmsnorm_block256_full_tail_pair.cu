#define main standalone_tail_probe_main
/*
 * Independent SM80 BF16 Add-RMSNorm partial source-round service probe.
 *
 * This is not Add-RMSNorm. It repeats only one width-eight source loop for
 * the exact 64- and 128-thread tails produced by H=2560 and H=5120 with a
 * 256-thread block. There is no reduction, reciprocal square root, epsilon,
 * or complete production operator. The timed region excludes the resource
 * identity padding while the compiled CTA retains the production 32-register
 * and 168-byte static-shared-memory allocation.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do {                                               \
    const cudaError_t error = (call);                                       \
    if (error != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

constexpr int kThreads = 256;
constexpr int kResidentCtas = 8;
constexpr int kStaticSharedBytes = 168;
constexpr int kSmCount = 108;
constexpr int kMaximumGridBlocks = kResidentCtas * kSmCount;
constexpr int kBufferSlots = 8;
constexpr int kRounds = 8192;
constexpr int kWarmups = 3;
constexpr int kSamplesPerTrial = 5;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";
constexpr const char* kHeader =
    "trial,function,cell,grid_blocks,block_threads,active_threads,rounds,"
    "registers_per_thread,static_shared_bytes,resident_cta_limit,active_sms,"
    "minimum_ctas_per_sm,maximum_ctas_per_sm,maximum_sm_span_ns,"
    "median_block_ns,checksum";

enum class RoundFunction : int { ADD_SQUARE = 0, SCALE_MULTIPLY = 1 };

struct Cell { const char* name; int grid_blocks; };

struct alignas(16) Pack {
    union { uint4 words; __nv_bfloat16 values[8]; };
};

struct SharedStorage {
    float variance_scale;
    unsigned char occupancy_padding[kStaticSharedBytes - sizeof(float)];
};

static_assert(sizeof(Pack) == 16);
static_assert(sizeof(SharedStorage) == kStaticSharedBytes);

__device__ __forceinline__ unsigned int read_smid() {
    unsigned int value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}
__device__ __forceinline__ std::uint64_t read_globaltimer() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

__device__ __forceinline__ float enforce_resource_identity(
        const SharedStorage* shared) {
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    float sink;
    asm volatile(
        "{ .reg .f32 t<30>;"
        "ld.shared.f32 t0, [%1+0]; ld.shared.f32 t1, [%1+4];"
        "ld.shared.f32 t2, [%1+8]; ld.shared.f32 t3, [%1+12];"
        "ld.shared.f32 t4, [%1+16]; ld.shared.f32 t5, [%1+20];"
        "ld.shared.f32 t6, [%1+24]; ld.shared.f32 t7, [%1+28];"
        "ld.shared.f32 t8, [%1+32]; ld.shared.f32 t9, [%1+36];"
        "ld.shared.f32 t10, [%1+40]; ld.shared.f32 t11, [%1+44];"
        "ld.shared.f32 t12, [%1+48]; ld.shared.f32 t13, [%1+52];"
        "ld.shared.f32 t14, [%1+56]; ld.shared.f32 t15, [%1+60];"
        "ld.shared.f32 t16, [%1+64]; ld.shared.f32 t17, [%1+68];"
        "ld.shared.f32 t18, [%1+72]; ld.shared.f32 t19, [%1+76];"
        "ld.shared.f32 t20, [%1+80]; ld.shared.f32 t21, [%1+84];"
        "ld.shared.f32 t22, [%1+88]; ld.shared.f32 t23, [%1+92];"
        "ld.shared.f32 t24, [%1+96]; ld.shared.f32 t25, [%1+100];"
        "ld.shared.f32 t26, [%1+104]; ld.shared.f32 t27, [%1+108];"
        "ld.shared.f32 t28, [%1+112]; ld.shared.f32 t29, [%1+116];"
        "membar.cta;"
        "add.f32 t0, t0, t1; add.f32 t2, t2, t3;"
        "add.f32 t4, t4, t5; add.f32 t6, t6, t7;"
        "add.f32 t8, t8, t9; add.f32 t10, t10, t11;"
        "add.f32 t12, t12, t14; add.f32 t16, t16, t18;"
        "add.f32 t20, t20, t22; add.f32 t0, t0, t2;"
        "add.f32 t4, t4, t6; add.f32 t8, t8, t10;"
        "add.f32 t12, t12, t14; add.f32 t16, t16, t18;"
        "add.f32 t20, t20, t22; add.f32 t24, t24, t25;"
        "add.f32 t26, t26, t27; add.f32 t28, t28, t29;"
        "add.f32 t24, t24, t26; add.f32 t24, t24, t28;"
        "add.f32 t0, t0, t4;"
        "add.f32 t8, t8, t12; add.f32 t16, t16, t20;"
        "add.f32 t0, t0, t8; add.f32 t0, t0, t24;"
        "add.f32 %0, t0, t16; }"
        : "=f"(sink) : "r"(address) : "memory");
    return sink;
}

template <RoundFunction function, int active_threads>
__global__ __launch_bounds__(kThreads, kResidentCtas)
void add_rmsnorm_block256_tail_round(
        const Pack* __restrict__ input, Pack* __restrict__ residual,
        const Pack* __restrict__ gamma, Pack* __restrict__ output,
        float* __restrict__ sink, std::uint64_t* __restrict__ start_ns,
        std::uint64_t* __restrict__ end_ns,
        unsigned int* __restrict__ smids) {
    static_assert(active_threads == 64 || active_threads == 128);
    __shared__ __align__(16) SharedStorage shared;
    if (threadIdx.x < kStaticSharedBytes) {
        reinterpret_cast<unsigned char*>(&shared)[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        shared.variance_scale = 0.875f;
        smids[blockIdx.x] = read_smid();
    }
    __syncthreads();
    if (threadIdx.x == 0) start_ns[blockIdx.x] = read_globaltimer();

    float state = 0.0f;
#pragma unroll 1
    for (int round = 0; round < kRounds; ++round) {
        if (threadIdx.x < active_threads) {
            const int slot = round & (kBufferSlots - 1);
            const int offset = (
                (blockIdx.x * kBufferSlots + slot) * active_threads
                + threadIdx.x);
            Pack result;
            if constexpr (function == RoundFunction::ADD_SQUARE) {
                const Pack left = input[offset];
                const Pack right = residual[offset];
                const auto* left_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(&left);
                const auto* right_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(&right);
                auto* result_pairs = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const __nv_bfloat162 added =
                        __hadd2(left_pairs[pair], right_pairs[pair]);
                    result_pairs[pair] = added;
                    const float2 values = __bfloat1622float2(added);
                    state += values.x * values.x + values.y * values.y;
                }
                residual[offset] = result;
            } else {
                const Pack value = residual[offset];
                const Pack weight = gamma[threadIdx.x];
                const auto* value_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(&value);
                const auto* weight_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(&weight);
                auto* result_pairs = reinterpret_cast<__nv_bfloat162*>(&result);
                const float scale =
                    *reinterpret_cast<volatile float*>(&shared.variance_scale);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    float2 scaled = __bfloat1622float2(value_pairs[pair]);
                    scaled.x *= scale;
                    scaled.y *= scale;
                    const __nv_bfloat162 rounded =
                        __floats2bfloat162_rn(scaled.x, scaled.y);
                    result_pairs[pair] = __hmul2(rounded, weight_pairs[pair]);
                }
                output[offset] = result;
                state += __bfloat162float(result.values[0]);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) end_ns[blockIdx.x] = read_globaltimer();
    const float resource_identity = enforce_resource_identity(&shared);
    sink[blockIdx.x * kThreads + threadIdx.x] = state + resource_identity;
}

template <RoundFunction function, int active_threads>
cudaFuncAttributes attributes() {
    cudaFuncAttributes result{};
    CHECK_CUDA(cudaFuncGetAttributes(
        &result, add_rmsnorm_block256_tail_round<function, active_threads>));
    return result;
}

template <RoundFunction function, int active_threads>
int resident_cta_limit() {
    int blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, add_rmsnorm_block256_tail_round<function, active_threads>,
        kThreads, 0));
    return blocks;
}

template <RoundFunction function, int active_threads>
void run_cell(
        const Cell& cell, int trials, const Pack* input, Pack* residual,
        const Pack* gamma, Pack* output, float* sink,
        std::uint64_t* starts, std::uint64_t* ends, unsigned int* smids) {
    const cudaFuncAttributes resource = attributes<function, active_threads>();
    const int resident_limit = resident_cta_limit<function, active_threads>();
    if (resource.numRegs != 32
            || resource.sharedSizeBytes != kStaticSharedBytes
            || resource.localSizeBytes != 0
            || resident_limit != kResidentCtas) {
        std::fprintf(
            stderr, "resource identity mismatch: regs=%d smem=%zu local=%zu "
            "resident=%d\n", resource.numRegs, resource.sharedSizeBytes,
            resource.localSizeBytes, resident_limit);
        std::exit(EXIT_FAILURE);
    }
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        add_rmsnorm_block256_tail_round<function, active_threads>
            <<<cell.grid_blocks, kThreads>>>(
                input, residual, gamma, output, sink, starts, ends, smids);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<std::uint64_t> host_starts(cell.grid_blocks);
    std::vector<std::uint64_t> host_ends(cell.grid_blocks);
    std::vector<unsigned int> host_smids(cell.grid_blocks);
    std::vector<float> host_sink(cell.grid_blocks * kThreads);
    for (int trial = 0; trial < trials; ++trial) {
        std::vector<std::uint64_t> maximum_sm_span_samples;
        std::vector<std::uint64_t> median_block_samples;
        std::vector<double> checksum_samples;
        int trial_active_sms = -1;
        int trial_minimum_ctas = -1;
        int trial_maximum_ctas = -1;
        for (int sample = 0; sample < kSamplesPerTrial; ++sample) {
            add_rmsnorm_block256_tail_round<function, active_threads>
                <<<cell.grid_blocks, kThreads>>>(
                    input, residual, gamma, output, sink, starts, ends, smids);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(
                host_starts.data(), starts,
                host_starts.size() * sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(
                host_ends.data(), ends,
                host_ends.size() * sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(
                host_smids.data(), smids,
                host_smids.size() * sizeof(unsigned int),
                cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(
                host_sink.data(), sink,
                host_sink.size() * sizeof(float), cudaMemcpyDeviceToHost));

            std::array<std::uint64_t, kSmCount> sm_start{};
            std::array<std::uint64_t, kSmCount> sm_end{};
            std::array<int, kSmCount> ctas_per_sm{};
            sm_start.fill(UINT64_MAX);
            std::vector<std::uint64_t> block_ns;
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
            if (sample == 0) {
                trial_active_sms = active_sms;
                trial_minimum_ctas = minimum_ctas;
                trial_maximum_ctas = maximum_ctas;
            } else if (active_sms != trial_active_sms
                    || minimum_ctas != trial_minimum_ctas
                    || maximum_ctas != trial_maximum_ctas) {
                std::fprintf(stderr, "CTA topology changed within trial\n");
                std::exit(EXIT_FAILURE);
            }
            std::uint64_t maximum_sm_span = 0;
            for (int sm = 0; sm < kSmCount; ++sm) {
                if (ctas_per_sm[sm] > 0) {
                    maximum_sm_span = std::max(
                        maximum_sm_span, sm_end[sm] - sm_start[sm]);
                }
            }
            std::sort(block_ns.begin(), block_ns.end());
            double checksum = 0.0;
            for (const float value : host_sink) {
                if (!std::isfinite(value)) {
                    std::fprintf(stderr, "non-finite validation sink\n");
                    std::exit(EXIT_FAILURE);
                }
                checksum += value;
            }
            maximum_sm_span_samples.push_back(maximum_sm_span);
            median_block_samples.push_back(block_ns[block_ns.size() / 2]);
            checksum_samples.push_back(checksum);
        }
        std::sort(maximum_sm_span_samples.begin(), maximum_sm_span_samples.end());
        std::sort(median_block_samples.begin(), median_block_samples.end());
        std::sort(checksum_samples.begin(), checksum_samples.end());
        const char* name = function == RoundFunction::ADD_SQUARE
            ? "add_square" : "scale_multiply";
        std::printf(
            "%d,%s,%s,%d,%d,%d,%d,%d,%zu,%d,%d,%d,%d,%llu,%llu,%.17g\n",
            trial, name, cell.name, cell.grid_blocks, kThreads, active_threads,
            kRounds, resource.numRegs, resource.sharedSizeBytes,
            resident_limit, trial_active_sms, trial_minimum_ctas,
            trial_maximum_ctas,
            static_cast<unsigned long long>(
                maximum_sm_span_samples[kSamplesPerTrial / 2]),
            static_cast<unsigned long long>(
                median_block_samples[kSamplesPerTrial / 2]),
            checksum_samples[kSamplesPerTrial / 2]);
    }
}

template <int active_threads>
void run_width(
        int trials, const Pack* input, Pack* residual, const Pack* gamma,
        Pack* output, float* sink, std::uint64_t* starts,
        std::uint64_t* ends, unsigned int* smids) {
    const Cell cells[] = {
        {"first", kSmCount},
        {"resident", kResidentCtas * kSmCount},
    };
    for (const Cell& cell : cells) {
        run_cell<RoundFunction::ADD_SQUARE, active_threads>(
            cell, trials, input, residual, gamma, output, sink,
            starts, ends, smids);
        run_cell<RoundFunction::SCALE_MULTIPLY, active_threads>(
            cell, trials, input, residual, gamma, output, sink,
            starts, ends, smids);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS\n", argv[0]);
        return EXIT_FAILURE;
    }
    char* device_end = nullptr;
    char* trials_end = nullptr;
    const long device_value = std::strtol(argv[1], &device_end, 10);
    const long trials_value = std::strtol(argv[2], &trials_end, 10);
    if (*argv[1] == '\0' || *device_end != '\0'
            || *argv[2] == '\0' || *trials_end != '\0'
            || device_value < 0 || device_value > INT_MAX
            || trials_value <= 0 || trials_value > INT_MAX) {
        return EXIT_FAILURE;
    }
    const int device = static_cast<int>(device_value);
    const int trials = static_cast<int>(trials_value);
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != kExpectedDevice
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "target is not the pinned A800 identity\n");
        return EXIT_FAILURE;
    }

    constexpr int kMaximumActiveThreads = 128;
    const std::size_t packs = static_cast<std::size_t>(
        kMaximumGridBlocks) * kBufferSlots * kMaximumActiveThreads;
    Pack *input = nullptr, *residual = nullptr, *gamma = nullptr, *output = nullptr;
    float* sink = nullptr;
    std::uint64_t *starts = nullptr, *ends = nullptr;
    unsigned int* smids = nullptr;
    CHECK_CUDA(cudaMalloc(&input, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&residual, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&gamma, kMaximumActiveThreads * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&output, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&sink, kMaximumGridBlocks * kThreads * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&starts, kMaximumGridBlocks * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&ends, kMaximumGridBlocks * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&smids, kMaximumGridBlocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(input, 0x3f, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(residual, 0x3e, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(gamma, 0x3f, kMaximumActiveThreads * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(output, 0, packs * sizeof(Pack)));

    std::puts(kHeader);
    run_width<64>(trials, input, residual, gamma, output, sink, starts, ends, smids);
    run_width<128>(trials, input, residual, gamma, output, sink, starts, ends, smids);

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

#undef main

#include <fstream>
#include <numeric>

template <RoundFunction function, int tail_threads>
__global__ __launch_bounds__(kThreads, kResidentCtas)
void add_rmsnorm_block256_full_tail_pair(
        const Pack* __restrict__ input, Pack* __restrict__ residual,
        const Pack* __restrict__ gamma, Pack* __restrict__ output,
        float* __restrict__ sink, std::uint64_t* __restrict__ start_ns,
        std::uint64_t* __restrict__ end_ns,
        unsigned int* __restrict__ smids) {
    static_assert(tail_threads == 64 || tail_threads == 128);
    constexpr int pair_threads = kThreads + tail_threads;
    __shared__ __align__(16) SharedStorage shared;
    if (threadIdx.x < kStaticSharedBytes) {
        reinterpret_cast<unsigned char*>(&shared)[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        shared.variance_scale = 0.875f;
        smids[blockIdx.x] = read_smid();
    }
    __syncthreads();
    if (threadIdx.x == 0) start_ns[blockIdx.x] = read_globaltimer();

    float state = 0.0f;
#pragma unroll 1
    for (int round = 0; round < kRounds; ++round) {
        const int slot = round & (kBufferSlots - 1);
#pragma unroll 1
        for (int idx = threadIdx.x; idx < pair_threads; idx += kThreads) {
            const int offset = (
                (blockIdx.x * kBufferSlots + slot) * pair_threads + idx);
            Pack result;
            if constexpr (function == RoundFunction::ADD_SQUARE) {
                const Pack left = input[offset];
                const Pack right = residual[offset];
                const auto* left_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(&left);
                const auto* right_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(&right);
                auto* result_pairs = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const __nv_bfloat162 added =
                        __hadd2(left_pairs[pair], right_pairs[pair]);
                    result_pairs[pair] = added;
                    const float2 values = __bfloat1622float2(added);
                    state += values.x * values.x + values.y * values.y;
                }
                residual[offset] = result;
            } else {
                const Pack value = residual[offset];
                const Pack weight = gamma[idx];
                const auto* value_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(&value);
                const auto* weight_pairs =
                    reinterpret_cast<const __nv_bfloat162*>(&weight);
                auto* result_pairs = reinterpret_cast<__nv_bfloat162*>(&result);
                const float scale =
                    *reinterpret_cast<volatile float*>(&shared.variance_scale);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    float2 scaled = __bfloat1622float2(value_pairs[pair]);
                    scaled.x *= scale;
                    scaled.y *= scale;
                    const __nv_bfloat162 rounded =
                        __floats2bfloat162_rn(scaled.x, scaled.y);
                    result_pairs[pair] = __hmul2(
                        rounded, weight_pairs[pair]);
                }
                output[offset] = result;
                state += __bfloat162float(result.values[0]);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) end_ns[blockIdx.x] = read_globaltimer();
    const float resource_identity = enforce_resource_identity(&shared);
    sink[blockIdx.x * kThreads + threadIdx.x] = state + resource_identity;
}

template <RoundFunction function, int tail_threads>
void run_pair_cell(
        const Cell& cell, const Pack* input, Pack* residual,
        const Pack* gamma, Pack* output, float* sink,
        std::uint64_t* starts, std::uint64_t* ends, unsigned int* smids,
        std::ofstream& raw) {
    cudaFuncAttributes attributes{};
    CHECK_CUDA(cudaFuncGetAttributes(
        &attributes,
        add_rmsnorm_block256_full_tail_pair<function, tail_threads>));
    int resident_limit = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_limit,
        add_rmsnorm_block256_full_tail_pair<function, tail_threads>,
        kThreads, 0));
    if (attributes.numRegs != 32
            || attributes.sharedSizeBytes != kStaticSharedBytes
            || attributes.localSizeBytes != 0
            || resident_limit != kResidentCtas) {
        std::fprintf(
            stderr, "resource identity mismatch: regs=%d smem=%zu local=%zu "
            "resident=%d\n", attributes.numRegs, attributes.sharedSizeBytes,
            attributes.localSizeBytes, resident_limit);
        std::exit(EXIT_FAILURE);
    }
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        add_rmsnorm_block256_full_tail_pair<function, tail_threads>
            <<<cell.grid_blocks, kThreads>>>(
                input, residual, gamma, output, sink, starts, ends, smids);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<std::uint64_t> host_starts(cell.grid_blocks);
    std::vector<std::uint64_t> host_ends(cell.grid_blocks);
    std::vector<unsigned int> host_smids(cell.grid_blocks);
    for (int sample = 0; sample < kSamplesPerTrial; ++sample) {
        add_rmsnorm_block256_full_tail_pair<function, tail_threads>
            <<<cell.grid_blocks, kThreads>>>(
                input, residual, gamma, output, sink, starts, ends, smids);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(
            host_starts.data(), starts,
            host_starts.size() * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            host_ends.data(), ends,
            host_ends.size() * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            host_smids.data(), smids,
            host_smids.size() * sizeof(unsigned int),
            cudaMemcpyDeviceToHost));
        std::vector<std::uint64_t> sm_start(kSmCount, UINT64_MAX);
        std::vector<std::uint64_t> sm_end(kSmCount, 0);
        std::vector<int> ctas_per_sm(kSmCount, 0);
        for (int block = 0; block < cell.grid_blocks; ++block) {
            const unsigned int sm = host_smids[block];
            if (sm >= static_cast<unsigned int>(kSmCount)
                    || host_ends[block] <= host_starts[block]) {
                std::fprintf(stderr, "invalid timing record\n");
                std::exit(EXIT_FAILURE);
            }
            ++ctas_per_sm[sm];
            sm_start[sm] = std::min(sm_start[sm], host_starts[block]);
            sm_end[sm] = std::max(sm_end[sm], host_ends[block]);
        }
        const int active_sms = static_cast<int>(std::count_if(
            ctas_per_sm.begin(), ctas_per_sm.end(),
            [](int count) { return count > 0; }));
        const int minimum_ctas = *std::min_element(
            ctas_per_sm.begin(), ctas_per_sm.end());
        const int maximum_ctas = *std::max_element(
            ctas_per_sm.begin(), ctas_per_sm.end());
        std::uint64_t maximum_span = 0;
        for (int sm = 0; sm < kSmCount; ++sm) {
            if (ctas_per_sm[sm] > 0) {
                maximum_span = std::max(
                    maximum_span, sm_end[sm] - sm_start[sm]);
            }
        }
        const int expected_ctas = cell.grid_blocks / kSmCount;
        if (active_sms != kSmCount || minimum_ctas != expected_ctas
                || maximum_ctas != expected_ctas || maximum_span == 0) {
            std::fprintf(stderr, "unexpected CTA placement\n");
            std::exit(EXIT_FAILURE);
        }
        raw << (function == RoundFunction::ADD_SQUARE
                    ? "add_square" : "scale_multiply")
            << ',' << tail_threads << ',' << cell.name << ',' << sample
            << ',' << maximum_span << '\n';
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s DEVICE_ID OUTPUT_CSV\n", argv[0]);
        return EXIT_FAILURE;
    }
    char* device_end = nullptr;
    const long device_value = std::strtol(argv[1], &device_end, 10);
    if (*argv[1] == '\0' || *device_end != '\0'
            || device_value < 0 || device_value > INT_MAX) {
        std::fprintf(stderr, "invalid device id\n");
        return EXIT_FAILURE;
    }
    const int device = static_cast<int>(device_value);
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != kExpectedDevice
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "unexpected device identity\n");
        return EXIT_FAILURE;
    }
    constexpr int kMaximumPairThreads = 384;
    const std::size_t packs = static_cast<std::size_t>(
        kMaximumGridBlocks) * kBufferSlots * kMaximumPairThreads;
    Pack *input = nullptr, *residual = nullptr, *gamma = nullptr, *output = nullptr;
    float* sink = nullptr;
    std::uint64_t *starts = nullptr, *ends = nullptr;
    unsigned int* smids = nullptr;
    CHECK_CUDA(cudaMalloc(&input, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&residual, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&gamma, kMaximumPairThreads * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&output, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&sink, kMaximumGridBlocks * kThreads * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&starts, kMaximumGridBlocks * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&ends, kMaximumGridBlocks * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&smids, kMaximumGridBlocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(input, 0x3f, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(residual, 0x3f, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(gamma, 0x3f, kMaximumPairThreads * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(output, 0, packs * sizeof(Pack)));
    std::ofstream raw(argv[2]);
    if (!raw) {
        std::fprintf(stderr, "cannot open output CSV\n");
        return EXIT_FAILURE;
    }
    raw << "function,tail_threads,cell,sample,maximum_sm_span_ns\n";
    const Cell cells[] = {{"first", kSmCount}, {"resident", kMaximumGridBlocks}};
    for (const Cell& cell : cells) {
        run_pair_cell<RoundFunction::ADD_SQUARE, 64>(
            cell, input, residual, gamma, output, sink, starts, ends, smids, raw);
        run_pair_cell<RoundFunction::SCALE_MULTIPLY, 64>(
            cell, input, residual, gamma, output, sink, starts, ends, smids, raw);
        run_pair_cell<RoundFunction::ADD_SQUARE, 128>(
            cell, input, residual, gamma, output, sink, starts, ends, smids, raw);
        run_pair_cell<RoundFunction::SCALE_MULTIPLY, 128>(
            cell, input, residual, gamma, output, sink, starts, ends, smids, raw);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
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
