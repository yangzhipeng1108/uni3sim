/*
 * H200 owner copy of the A800 exact-H2560 block-1024 Add-RMSNorm source-round
 * probe.  A800 origin: the H2048 source above; the A800 H2560 source (capture add-rmsnorm-h2560-block1024-1adea53d-device6-20260904, source sha256 b99f24b8...) is not in the tree or on disk. The five retained A800 width probes (H2048/H4096/H5120/H8192) differ from each other only in kHiddenSize, the derived static_asserts, the kernel name and the grid list, so H2560 is reconstructed with kHiddenSize = 2560 (320 active threads, 10 active warps) and the A800 H2560 grid list {1, 4, 16, 108}.
 * Kernel bodies, resource-identity padding, timer placement, warm-ups, trials,
 * samples, schedule and CSV fields are unchanged; the source remains the
 * vLLM 0.11.0 fused_add_rms_norm vec8 source loops (not vLLM 0.24).
 * Hardware coupling only: kSmCount and kMaximumGrid 108 -> 132, the full-SM
 * grid cell 108 -> 132, device identity "NVIDIA H200" with major 9, built for
 * sm_90.  The 32-register / 168-byte / 2-CTA resource identity check is kept.
 */
/*
 * Exact-H2560 SM80 BF16 vec8 Add-RMSNorm source-round service probe.
 *
 * This is not Add-RMSNorm. The specializations repeat only one of the two source loops surrounding the production reduction. The timed paths contain no reduction, reciprocal square root, epsilon path, checksum math, or production-operator call.
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
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

constexpr int kHiddenSize = 2560;
constexpr int kVectorWidth = 8;
constexpr int kThreads = 1024;
constexpr int kActiveThreads = kHiddenSize / kVectorWidth;
constexpr int kActiveWarps = kActiveThreads / 32;
constexpr int kResidentCtas = 2;
constexpr int kStaticSharedBytes = 168;
constexpr int kSmCount = 132;
constexpr int kMaximumGrid = 132;
constexpr int kWarmups = 2;
constexpr int kTrials = 3;
constexpr int kSamples = 200;
constexpr const char* kExpectedDevice = "NVIDIA H200";
constexpr const char* kHeader =
    "trial,position,sample,function,grid_blocks,block_threads,hidden_size,"
    "active_threads,active_warps,registers_per_thread,"
    "static_shared_bytes,resident_cta_limit,active_sms,"
    "minimum_ctas_per_active_sm,maximum_ctas_per_active_sm,"
    "median_block_clock64_cycles,checksum";

enum class RoundFunction : int { ADD_SQUARE = 0, SCALE_MULTIPLY = 1 };

struct Cell {
    RoundFunction function;
    int grid_blocks;
};

struct alignas(16) Pack {
    union {
        uint4 words;
        __nv_bfloat16 values[kVectorWidth];
    };
};

struct SharedStorage {
    float s_variance;
    unsigned char occupancy_padding[kStaticSharedBytes - sizeof(float)];
};

static_assert(sizeof(Pack) == 16);
static_assert(sizeof(SharedStorage) == kStaticSharedBytes);
static_assert(kThreads == 1024);
static_assert(kActiveThreads == 320);
static_assert(kActiveWarps == 10);
static_assert(kResidentCtas == 2);

__device__ __forceinline__ unsigned int read_smid() {
    unsigned int value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ std::uint64_t read_clock64() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
    return value;
}

__device__ __forceinline__ float enforce_resource_identity(
        const SharedStorage* shared) {
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    float sink;
    asm volatile(
        "{ .reg .f32 t<24>;"
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
        "membar.cta;"
        "add.f32 t0, t0, t1; add.f32 t2, t2, t3;"
        "add.f32 t4, t4, t5; add.f32 t6, t6, t7;"
        "add.f32 t8, t8, t9; add.f32 t10, t10, t11;"
        "add.f32 t12, t12, t13; add.f32 t14, t14, t15;"
        "add.f32 t16, t16, t17; add.f32 t18, t18, t19;"
        "add.f32 t20, t20, t21; add.f32 t22, t22, t23;"
        "add.f32 t0, t0, t2; add.f32 t4, t4, t6;"
        "add.f32 t8, t8, t10; add.f32 t12, t12, t14;"
        "add.f32 t16, t16, t18; add.f32 t20, t20, t22;"
        "add.f32 t0, t0, t4; add.f32 t8, t8, t12;"
        "add.f32 t16, t16, t20; add.f32 t0, t0, t8;"
        "add.f32 %0, t0, t16; }"
        : "=f"(sink) : "r"(address) : "memory");
    return sink;
}

template <RoundFunction function>
__global__ __launch_bounds__(kThreads, kResidentCtas)
void add_rmsnorm_round_service_1024_h2560(
        Pack* __restrict__ input,
        Pack* __restrict__ residual,
        const Pack* __restrict__ gamma,
        float runtime_variance_scale,
        float* __restrict__ sink,
        std::uint64_t* __restrict__ start_cycles,
        std::uint64_t* __restrict__ end_cycles,
        unsigned int* __restrict__ smids) {
    __shared__ __align__(16) SharedStorage shared;
    if (threadIdx.x < kStaticSharedBytes) {
        reinterpret_cast<unsigned char*>(&shared)[threadIdx.x] =
            static_cast<unsigned char>(0);
    }
    if (threadIdx.x == 0) shared.s_variance = runtime_variance_scale;
    if (threadIdx.x == 0) smids[blockIdx.x] = read_smid();
    __syncthreads();

    std::uint64_t begin_cycles = 0;
    std::uint64_t finish_cycles = 0;
    float variance = 0.0f;
    const int offset = blockIdx.x * kActiveThreads + threadIdx.x;
    if (threadIdx.x < kActiveThreads) {
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        if (lane == 0) begin_cycles = read_clock64();

        Pack result;
        if constexpr (function == RoundFunction::ADD_SQUARE) {
            const Pack left = input[offset];
            const Pack right = residual[offset];
            const auto* left_pairs =
                reinterpret_cast<const __nv_bfloat162*>(&left);
            const auto* right_pairs =
                reinterpret_cast<const __nv_bfloat162*>(&right);
            auto* result_pairs =
                reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
            for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
                const __nv_bfloat162 added =
                    __hadd2(left_pairs[pair], right_pairs[pair]);
                result_pairs[pair] = added;
                const float2 values = __bfloat1622float2(added);
                variance += values.x * values.x + values.y * values.y;
            }
            residual[offset] = result;
        } else {
            const Pack value = residual[offset];
            const Pack weight = gamma[threadIdx.x];
            const auto* value_pairs =
                reinterpret_cast<const __nv_bfloat162*>(&value);
            const auto* weight_pairs =
                reinterpret_cast<const __nv_bfloat162*>(&weight);
            auto* result_pairs =
                reinterpret_cast<__nv_bfloat162*>(&result);
            const float scale =
                *reinterpret_cast<volatile float*>(&shared.s_variance);
#pragma unroll
            for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
                float2 scaled = __bfloat1622float2(value_pairs[pair]);
                scaled.x *= scale;
                scaled.y *= scale;
                const __nv_bfloat162 rounded =
                    __floats2bfloat162_rn(scaled.x, scaled.y);
                result_pairs[pair] = __hmul2(rounded, weight_pairs[pair]);
            }
            input[offset] = result;
        }

        if (lane == 0) finish_cycles = read_clock64();
    }

    __syncthreads();
    if (threadIdx.x < kActiveThreads) {
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int timer_offset = blockIdx.x * kActiveWarps + warp;
        if (lane == 0) {
            start_cycles[timer_offset] = begin_cycles;
            end_cycles[timer_offset] = finish_cycles;
        }
        /* This exact-resource padding is strictly after the timed path. */
        const float resource_identity = enforce_resource_identity(&shared);
        if constexpr (function == RoundFunction::ADD_SQUARE) {
            sink[blockIdx.x * kActiveThreads + threadIdx.x] =
                variance + resource_identity;
        } else {
            sink[blockIdx.x * kActiveThreads + threadIdx.x] =
                __bfloat162float(input[offset].values[0])
                + resource_identity;
        }
    }
}

template <RoundFunction function>
void launch(const Cell& cell, Pack* input, Pack* residual,
            const Pack* gamma, float runtime_variance_scale, float* sink,
            std::uint64_t* starts, std::uint64_t* ends,
            unsigned int* smids) {
    add_rmsnorm_round_service_1024_h2560<function><<<cell.grid_blocks, kThreads>>>(
        input, residual, gamma,
        runtime_variance_scale, sink, starts, ends, smids);
}

template <RoundFunction function>
cudaFuncAttributes attributes() {
    cudaFuncAttributes result{};
    CHECK_CUDA(cudaFuncGetAttributes(
        &result, add_rmsnorm_round_service_1024_h2560<function>));
    return result;
}

template <RoundFunction function>
int resident_cta_limit() {
    int blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, add_rmsnorm_round_service_1024_h2560<function>, kThreads, 0));
    return blocks;
}

const char* function_name(RoundFunction function) {
    if (function == RoundFunction::ADD_SQUARE) return "add_square";
    if (function == RoundFunction::SCALE_MULTIPLY) return "scale_multiply";
    std::fprintf(stderr, "unsupported round function\n");
    std::exit(EXIT_FAILURE);
}

void run_cell(
        const Cell& cell, int trial, int position, float variance_scale,
        Pack* input, Pack* residual, const Pack* gamma, float* sink,
        std::uint64_t* starts, std::uint64_t* ends, unsigned int* smids) {
    const cudaFuncAttributes resource =
        cell.function == RoundFunction::ADD_SQUARE
        ? attributes<RoundFunction::ADD_SQUARE>()
        : attributes<RoundFunction::SCALE_MULTIPLY>();
    const int resident_limit =
        cell.function == RoundFunction::ADD_SQUARE
        ? resident_cta_limit<RoundFunction::ADD_SQUARE>()
        : resident_cta_limit<RoundFunction::SCALE_MULTIPLY>();
    if (resource.numRegs != 32
            || resource.sharedSizeBytes != kStaticSharedBytes
            || resident_limit != kResidentCtas) {
        std::fprintf(
            stderr, "resource identity mismatch: regs=%d smem=%zu resident=%d\n",
            resource.numRegs, resource.sharedSizeBytes, resident_limit);
        std::exit(EXIT_FAILURE);
    }

    const std::size_t cell_bytes = static_cast<std::size_t>(
        cell.grid_blocks) * kActiveThreads * sizeof(Pack);
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        CHECK_CUDA(cudaMemset(input, 1, cell_bytes));
        CHECK_CUDA(cudaMemset(residual, 2, cell_bytes));
        if (cell.function == RoundFunction::ADD_SQUARE) {
            launch<RoundFunction::ADD_SQUARE>(
                cell, input, residual, gamma, variance_scale, sink,
                starts, ends, smids);
        } else {
            launch<RoundFunction::SCALE_MULTIPLY>(
                cell, input, residual, gamma, variance_scale, sink,
                starts, ends, smids);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int sample = 0; sample < kSamples; ++sample) {
        CHECK_CUDA(cudaMemset(input, 1, cell_bytes));
        CHECK_CUDA(cudaMemset(residual, 2, cell_bytes));
        if (cell.function == RoundFunction::ADD_SQUARE) {
            launch<RoundFunction::ADD_SQUARE>(
                cell, input, residual, gamma, variance_scale, sink,
                starts, ends, smids);
        } else {
            launch<RoundFunction::SCALE_MULTIPLY>(
                cell, input, residual, gamma, variance_scale, sink,
                starts, ends, smids);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        const std::size_t timer_records =
            static_cast<std::size_t>(cell.grid_blocks) * kActiveWarps;
        std::vector<std::uint64_t> host_starts(timer_records);
        std::vector<std::uint64_t> host_ends(timer_records);
        std::vector<unsigned int> host_smids(cell.grid_blocks);
        std::vector<float> host_sink(
            static_cast<std::size_t>(cell.grid_blocks) * kActiveThreads);
        CHECK_CUDA(cudaMemcpy(
            host_starts.data(), starts,
            host_starts.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            host_ends.data(), ends,
            host_ends.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            host_smids.data(), smids,
            host_smids.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            host_sink.data(), sink,
            host_sink.size() * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<std::uint64_t> durations;
        std::array<int, kSmCount> ctas_per_sm{};
        for (int block = 0; block < cell.grid_blocks; ++block) {
            if (host_smids[block] >= static_cast<unsigned int>(kSmCount)) {
                std::fprintf(stderr, "invalid block SM identity\n");
                std::exit(EXIT_FAILURE);
            }
            ++ctas_per_sm[host_smids[block]];
            std::uint64_t first_start = UINT64_MAX;
            std::uint64_t last_end = 0;
            for (int warp = 0; warp < kActiveWarps; ++warp) {
                const int offset = block * kActiveWarps + warp;
                if (host_ends[offset] < host_starts[offset]) {
                    std::fprintf(
                        stderr, "invalid active-warp timing identity\n");
                    std::exit(EXIT_FAILURE);
                }
                first_start = std::min(first_start, host_starts[offset]);
                last_end = std::max(last_end, host_ends[offset]);
            }
            durations.push_back(last_end - first_start);
        }
        std::sort(durations.begin(), durations.end());
        int active_sms = 0;
        int minimum_ctas = INT_MAX;
        int maximum_ctas = 0;
        for (const int count : ctas_per_sm) {
            if (count > 0) {
                ++active_sms;
                minimum_ctas = std::min(minimum_ctas, count);
                maximum_ctas = std::max(maximum_ctas, count);
            }
        }
        if (active_sms != cell.grid_blocks
                || minimum_ctas != 1 || maximum_ctas != 1) {
            std::fprintf(
                stderr,
                "topology mismatch: grid=%d active_sms=%d min=%d max=%d\n",
                cell.grid_blocks, active_sms, minimum_ctas, maximum_ctas);
            std::exit(EXIT_FAILURE);
        }
        double checksum = 0.0;
        for (const float value : host_sink) {
            if (!std::isfinite(value)) {
                std::fprintf(stderr, "non-finite validation sink\n");
                std::exit(EXIT_FAILURE);
            }
            checksum += value;
        }
        std::printf(
            "%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%.17g\n",
            trial, position, sample, function_name(cell.function),
            cell.grid_blocks, kThreads, kHiddenSize, kActiveThreads,
            kActiveWarps, resource.numRegs,
            static_cast<int>(resource.sharedSizeBytes), resident_limit,
            active_sms, minimum_ctas, maximum_ctas,
            static_cast<unsigned long long>(durations[durations.size() / 2]),
            checksum);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s DEVICE VARIANCE_SCALE\n", argv[0]);
        return EXIT_FAILURE;
    }
    char* device_end = nullptr;
    char* scale_end = nullptr;
    const long device_value = std::strtol(argv[1], &device_end, 10);
    const float variance_scale = std::strtof(argv[2], &scale_end);
    if (*argv[1] == '\0' || *device_end != '\0'
            || *argv[2] == '\0' || *scale_end != '\0'
            || device_value < 0 || device_value > INT_MAX
            || !std::isfinite(variance_scale) || variance_scale <= 0.0f) {
        return EXIT_FAILURE;
    }
    const int device = static_cast<int>(device_value);
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::string(properties.name) != kExpectedDevice
            || properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "target is not the pinned H200 identity\n");
        return EXIT_FAILURE;
    }

    const std::size_t packs = static_cast<std::size_t>(
        kMaximumGrid) * kActiveThreads;
    Pack *input = nullptr, *residual = nullptr, *gamma = nullptr;
    float* sink = nullptr;
    std::uint64_t *starts = nullptr, *ends = nullptr;
    unsigned int* smids = nullptr;
    CHECK_CUDA(cudaMalloc(&input, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&residual, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(&gamma, kActiveThreads * sizeof(Pack)));
    CHECK_CUDA(cudaMalloc(
        &sink, kMaximumGrid * kActiveThreads * sizeof(float)));
    CHECK_CUDA(cudaMalloc(
        &starts, kMaximumGrid * kActiveWarps * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(
        &ends, kMaximumGrid * kActiveWarps * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&smids, kMaximumGrid * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(input, 1, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(residual, 2, packs * sizeof(Pack)));
    CHECK_CUDA(cudaMemset(gamma, 3, kActiveThreads * sizeof(Pack)));

    constexpr std::array<Cell, 8> cells{{
        {RoundFunction::ADD_SQUARE, 1},
        {RoundFunction::SCALE_MULTIPLY, 1},
        {RoundFunction::ADD_SQUARE, 4},
        {RoundFunction::SCALE_MULTIPLY, 4},
        {RoundFunction::ADD_SQUARE, 16},
        {RoundFunction::SCALE_MULTIPLY, 16},
        {RoundFunction::ADD_SQUARE, 132},
        {RoundFunction::SCALE_MULTIPLY, 132},
    }};
    std::puts(kHeader);
    for (int trial = 0; trial < kTrials; ++trial) {
        const int offset = (trial * 3) % static_cast<int>(cells.size());
        for (int position = 0; position < static_cast<int>(cells.size());
             ++position) {
            const Cell& cell = cells[(position + offset) % cells.size()];
            run_cell(
                cell, trial, position, variance_scale, input,
                residual, gamma, sink,
                starts, ends, smids);
        }
    }

    CHECK_CUDA(cudaFree(smids));
    CHECK_CUDA(cudaFree(ends));
    CHECK_CUDA(cudaFree(starts));
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(gamma));
    CHECK_CUDA(cudaFree(residual));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
