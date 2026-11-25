/*
 * MetaX C500 isolated FA2 split-combine dependency fragments.
 *
 * The probe measures source-level fragments separately with clock64 and
 * adjacent runtime repetition counts.  It never executes a complete
 * FlashAttention reducer and consumes no production/evaluation duration.
 * The compile-time geometry is the MACA 3.7.2 block256 source specialization.
 * It defaults to kMaxSplits=2, kBlockM=16, and D=64; the exact source
 * specialization is selected with FA_PROBE_MAX_SPLITS, FA_PROBE_BLOCK_M,
 * and FA_PROBE_HEAD_DIM.  In particular, a runtime five-way SplitKV reducer
 * dispatches the padded kMaxSplits=8 program, not the two-way program.
 */

#include <mc_runtime.h>
#include <cuda_bf16.h>

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
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kThreads = 256;
#ifndef FA_PROBE_BLOCK_M
#define FA_PROBE_BLOCK_M 16
#endif
#ifndef FA_PROBE_HEAD_DIM
#define FA_PROBE_HEAD_DIM 64
#endif
#ifndef FA_PROBE_MAX_SPLITS
#define FA_PROBE_MAX_SPLITS 2
#endif
#ifndef FA_PROBE_NUM_SPLITS
#define FA_PROBE_NUM_SPLITS FA_PROBE_MAX_SPLITS
#endif
#ifndef FA_PROBE_REPETITION_BASE
#define FA_PROBE_REPETITION_BASE 16
#endif
#ifndef FA_PROBE_STEADY_MAX_GRID
#define FA_PROBE_STEADY_MAX_GRID 104
#endif
constexpr int kBlockM = FA_PROBE_BLOCK_M;
constexpr int kHeadDim = FA_PROBE_HEAD_DIM;
constexpr int kMaxSplits = FA_PROBE_MAX_SPLITS;
constexpr int kNumSplits = FA_PROBE_NUM_SPLITS;
constexpr int kRepetitionBase = FA_PROBE_REPETITION_BASE;
constexpr int kSteadyMaxGrid = FA_PROBE_STEADY_MAX_GRID;
constexpr int kRowsPerLoadLse = kThreads / kBlockM;
constexpr int kRowsPerLoadTranspose = std::min(
    kRowsPerLoadLse, kMaxSplits);
constexpr int kValuesPerThread = kBlockM * kHeadDim / kThreads;
constexpr int kWarmupsPerCell = 4;
constexpr int kSamplesPerCell = 31;
constexpr unsigned long long kWaveMask = 0xffffffffffffffffULL;

static_assert(kMaxSplits == 2 || kMaxSplits == 8);
static_assert(kNumSplits > 0 && kNumSplits <= kMaxSplits);
static_assert(kRepetitionBase > 0);
static_assert(kSteadyMaxGrid == kExpectedApCount);
static_assert(kValuesPerThread == 4);
static_assert(kBlockM * kHeadDim == 1024);
static_assert(kMaxSplits * kBlockM <= kThreads);

enum class Mode : int {
    kControl = 0,
    kLseSharedTranspose = 1,
    kMaxAllreduce = 2,
    kPostMaxExpSum = 3,
    kSumAllreduce = 4,
    kPostSumLog = 5,
    kScaleSharedBarrier = 6,
    kOaccumLoadFma = 7,
    kOutputConversion = 8,
    kSourceDependencyChain = 9,
};

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::kControl: return "control";
        case Mode::kLseSharedTranspose: return "lse_shared_transpose";
        case Mode::kMaxAllreduce:
            return kRowsPerLoadTranspose == 2
                ? "max_allreduce_width2" : "max_allreduce_width8";
        case Mode::kPostMaxExpSum: return "post_max_exp_sum";
        case Mode::kSumAllreduce:
            return kRowsPerLoadTranspose == 2
                ? "sum_allreduce_width2" : "sum_allreduce_width8";
        case Mode::kPostSumLog: return "post_sum_log";
        case Mode::kScaleSharedBarrier: return "scale_shared_barrier";
        case Mode::kOaccumLoadFma: return "oaccum_load_fma";
        case Mode::kOutputConversion: return "output_conversion";
        case Mode::kSourceDependencyChain:
            return "source_dependency_chain";
        default:
            std::fprintf(stderr, "unknown mode\n");
            std::exit(EXIT_FAILURE);
    }
}

struct Result {
    unsigned long long cycles;
    std::uint32_t checksum;
};

struct Resources {
    int registers_per_thread;
    std::size_t static_shared_bytes;
    int resident_ctas_per_ap;
};

template <Mode mode>
__global__ void dependency_fragment(
        const float4* oaccum, Result* results, int repetitions) {
    __shared__ float s_lse[kMaxSplits][kBlockM + 1];
    __shared__ unsigned long long begin_ticks[kThreads];
    __shared__ unsigned long long end_ticks[kThreads];
    __shared__ std::uint32_t checksums[kThreads];

    const int lane = threadIdx.x % kExpectedWaveSize;
    const int row = threadIdx.x % kRowsPerLoadTranspose;
    const int column = threadIdx.x / kRowsPerLoadTranspose;
    float value = 1.0F + static_cast<float>(threadIdx.x & 15) * 0.015625F;
    float4 output = make_float4(value, value + 0.25F, value + 0.5F,
                                value + 0.75F);
    __nv_bfloat162 converted0;
    __nv_bfloat162 converted1;

    if (threadIdx.x < kMaxSplits * kBlockM) {
        const int source_row = threadIdx.x / kBlockM;
        s_lse[source_row][threadIdx.x % kBlockM] = (
            source_row < kNumSplits ? value : -INFINITY);
    }
    __syncthreads();
    const unsigned long long begin = clock64();
    asm volatile("" ::: "memory");
    #pragma unroll 1
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        if constexpr (mode == Mode::kControl) {
            asm volatile("" : "+r"(value) : : "memory");
        } else if constexpr (mode == Mode::kLseSharedTranspose) {
            const int source_row = threadIdx.x / kBlockM;
            const int source_column = threadIdx.x % kBlockM;
            if (source_row < kMaxSplits) {
                s_lse[source_row][source_column] = (
                    source_row < kNumSplits ? value : -INFINITY);
            }
            __syncthreads();
            value = (row < kMaxSplits && column < kBlockM)
                ? s_lse[row][column] : -INFINITY;
        } else if constexpr (mode == Mode::kMaxAllreduce) {
            #pragma unroll
            for (int delta = 1; delta < kRowsPerLoadTranspose;
                 delta *= 2) {
                const float peer = __shfl_xor_sync(
                    kWaveMask, value, delta, kExpectedWaveSize);
                value = fmaxf(value, peer);
            }
        } else if constexpr (mode == Mode::kPostMaxExpSum) {
            value = expf(value - 0.5F) + 0.125F;
        } else if constexpr (mode == Mode::kSumAllreduce) {
            #pragma unroll
            for (int delta = 1; delta < kRowsPerLoadTranspose;
                 delta *= 2) {
                const float peer = __shfl_xor_sync(
                    kWaveMask, value, delta, kExpectedWaveSize);
                value += peer;
            }
        } else if constexpr (mode == Mode::kPostSumLog) {
            const float positive = fabsf(value) + 1.0F;
            value = (positive == 0.0F || positive != positive)
                ? INFINITY : logf(positive) + 0.5F;
        } else if constexpr (mode == Mode::kScaleSharedBarrier) {
            if (row < kNumSplits && column < kBlockM) {
                s_lse[row][column] = expf(value - 0.75F);
            }
            __syncthreads();
            value += s_lse[row][threadIdx.x % kBlockM] * 0.0009765625F;
        } else if constexpr (mode == Mode::kOaccumLoadFma) {
            const std::size_t block_stride = (
                static_cast<std::size_t>(kThreads) * kMaxSplits);
            const std::size_t block_offset = (
                static_cast<std::size_t>(blockIdx.x) * block_stride);
            #pragma unroll
            for (int split = 0; split < kNumSplits; ++split) {
                const float4 loaded = oaccum[
                    block_offset + split * kThreads + threadIdx.x];
                const float scale = s_lse[split][threadIdx.x % kBlockM];
                output.x = fmaf(scale, loaded.x, output.x);
                output.y = fmaf(scale, loaded.y, output.y);
                output.z = fmaf(scale, loaded.z, output.z);
                output.w = fmaf(scale, loaded.w, output.w);
            }
            value = output.x + output.y + output.z + output.w;
        } else if constexpr (mode == Mode::kOutputConversion) {
            converted0 = __floats2bfloat162_rn(output.x, output.y);
            converted1 = __floats2bfloat162_rn(output.z, output.w);
            std::uint32_t packed0 =
                *reinterpret_cast<const std::uint32_t*>(&converted0);
            std::uint32_t packed1 =
                *reinterpret_cast<const std::uint32_t*>(&converted1);
            asm volatile("" : "+r"(packed0), "+r"(packed1));
            value = __uint_as_float(
                (__float_as_uint(value) & 0xffff0000U)
                ^ ((packed0 ^ packed1) & 0x0000ffffU));
        } else if constexpr (mode == Mode::kSourceDependencyChain) {
            // One source-ordered reducer prefix: retain the exact static
            // LSE/reduce/exp/log/scale/conversion sequence but omit the
            // Oaccum global-copy/FMA loop and output stores.  This checks
            // whether isolated stage envelopes can be summed without using a
            // complete production FlashAttention duration.
            const int source_row = threadIdx.x / kBlockM;
            const int source_column = threadIdx.x % kBlockM;
            if (source_row < kMaxSplits) {
                s_lse[source_row][source_column] = (
                    source_row < kNumSplits ? value : -INFINITY);
            }
            __syncthreads();
            value = (row < kMaxSplits && column < kBlockM)
                ? s_lse[row][column] : -INFINITY;
            #pragma unroll
            for (int delta = 1; delta < kRowsPerLoadTranspose;
                 delta *= 2) {
                const float peer = __shfl_xor_sync(
                    kWaveMask, value, delta, kExpectedWaveSize);
                value = fmaxf(value, peer);
            }
            value = value == -INFINITY ? 0.0F : value;
            const float exp_value = expf(value - 0.5F);
            value = exp_value + 0.125F;
            #pragma unroll
            for (int delta = 1; delta < kRowsPerLoadTranspose;
                 delta *= 2) {
                const float peer = __shfl_xor_sync(
                    kWaveMask, value, delta, kExpectedWaveSize);
                value += peer;
            }
            const float positive = fabsf(value) + 1.0F;
            value = (positive == 0.0F || positive != positive)
                ? INFINITY : logf(positive) + 0.5F;
            if (row < kNumSplits && column < kBlockM) {
                s_lse[row][column] = expf(value - 0.75F);
            }
            __syncthreads();
            converted0 = __floats2bfloat162_rn(
                output.x + value, output.y + value);
            converted1 = __floats2bfloat162_rn(
                output.z + value, output.w + value);
            const std::uint32_t packed0 =
                *reinterpret_cast<const std::uint32_t*>(&converted0);
            const std::uint32_t packed1 =
                *reinterpret_cast<const std::uint32_t*>(&converted1);
            value = __uint_as_float(
                (__float_as_uint(value) & 0xffff0000U)
                ^ ((packed0 ^ packed1) & 0x0000ffffU));
        }
    }
    asm volatile("" : : "r"(value) : "memory");
    const unsigned long long end = clock64();
    begin_ticks[threadIdx.x] = begin;
    end_ticks[threadIdx.x] = end;
    checksums[threadIdx.x] = __float_as_uint(value);
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long first = begin_ticks[0];
        unsigned long long last = end_ticks[0];
        std::uint32_t checksum = 0;
        #pragma unroll
        for (int thread = 0; thread < kThreads; ++thread) {
            first = min(first, begin_ticks[thread]);
            last = max(last, end_ticks[thread]);
            checksum ^= checksums[thread]
                + static_cast<std::uint32_t>(thread);
        }
        results[blockIdx.x] = {last - first, checksum};
    }
}

void launch(
        Mode mode, int grid, int repetitions, const float4* oaccum,
        Result* results) {
    switch (mode) {
        case Mode::kControl:
            dependency_fragment<Mode::kControl><<<grid, kThreads>>>(
                oaccum, results, repetitions); return;
        case Mode::kLseSharedTranspose:
            dependency_fragment<Mode::kLseSharedTranspose>
                <<<grid, kThreads>>>(oaccum, results, repetitions); return;
        case Mode::kMaxAllreduce:
            dependency_fragment<Mode::kMaxAllreduce><<<grid, kThreads>>>(
                oaccum, results, repetitions); return;
        case Mode::kPostMaxExpSum:
            dependency_fragment<Mode::kPostMaxExpSum><<<grid, kThreads>>>(
                oaccum, results, repetitions); return;
        case Mode::kSumAllreduce:
            dependency_fragment<Mode::kSumAllreduce><<<grid, kThreads>>>(
                oaccum, results, repetitions); return;
        case Mode::kPostSumLog:
            dependency_fragment<Mode::kPostSumLog><<<grid, kThreads>>>(
                oaccum, results, repetitions); return;
        case Mode::kScaleSharedBarrier:
            dependency_fragment<Mode::kScaleSharedBarrier>
                <<<grid, kThreads>>>(oaccum, results, repetitions); return;
        case Mode::kOaccumLoadFma:
            dependency_fragment<Mode::kOaccumLoadFma><<<grid, kThreads>>>(
                oaccum, results, repetitions); return;
        case Mode::kOutputConversion:
            dependency_fragment<Mode::kOutputConversion>
                <<<grid, kThreads>>>(oaccum, results, repetitions); return;
        case Mode::kSourceDependencyChain:
            dependency_fragment<Mode::kSourceDependencyChain>
                <<<grid, kThreads>>>(oaccum, results, repetitions); return;
        default:
            std::fprintf(stderr, "unknown launch mode\n");
            std::exit(EXIT_FAILURE);
    }
}

template <Mode mode>
Resources kernel_resources() {
    mcFuncAttributes attributes {};
    CHECK(mcFuncGetAttributes(
        &attributes,
        reinterpret_cast<const void*>(dependency_fragment<mode>)));
    int resident = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, dependency_fragment<mode>, kThreads, 0));
    return {attributes.numRegs, attributes.sharedSizeBytes, resident};
}

Resources resources(Mode mode) {
    switch (mode) {
        case Mode::kControl:
            return kernel_resources<Mode::kControl>();
        case Mode::kLseSharedTranspose:
            return kernel_resources<Mode::kLseSharedTranspose>();
        case Mode::kMaxAllreduce:
            return kernel_resources<Mode::kMaxAllreduce>();
        case Mode::kPostMaxExpSum:
            return kernel_resources<Mode::kPostMaxExpSum>();
        case Mode::kSumAllreduce:
            return kernel_resources<Mode::kSumAllreduce>();
        case Mode::kPostSumLog:
            return kernel_resources<Mode::kPostSumLog>();
        case Mode::kScaleSharedBarrier:
            return kernel_resources<Mode::kScaleSharedBarrier>();
        case Mode::kOaccumLoadFma:
            return kernel_resources<Mode::kOaccumLoadFma>();
        case Mode::kOutputConversion:
            return kernel_resources<Mode::kOutputConversion>();
        case Mode::kSourceDependencyChain:
            return kernel_resources<Mode::kSourceDependencyChain>();
        default:
            std::fprintf(stderr, "unknown resource mode\n");
            std::exit(EXIT_FAILURE);
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

struct Cell {
    Mode mode;
    int grid;
    int repetitions;
    int sample;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--device-id"
            || std::string(argv[3]) != "--trial") {
        std::fprintf(stderr, "usage: %s --device-id ID --trial TRIAL\n",
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
        std::fprintf(stderr, "expected C500 104 AP/wave64\n");
        return EXIT_FAILURE;
    }

    constexpr int maximum_grid = kSteadyMaxGrid;
    constexpr std::size_t input_items = (
        static_cast<std::size_t>(maximum_grid) * kThreads * kMaxSplits);
    std::vector<float4> host_input(input_items);
    for (std::size_t index = 0; index < input_items; ++index) {
        const float value = 1.0F + static_cast<float>(index & 31) * 0.015625F;
        host_input[index] = make_float4(
            value, value + 0.25F, value + 0.5F, value + 0.75F);
    }
    float4* device_input = nullptr;
    Result* device_results = nullptr;
    CHECK(mcMalloc(reinterpret_cast<void**>(&device_input),
                   input_items * sizeof(float4)));
    CHECK(mcMalloc(reinterpret_cast<void**>(&device_results),
                   maximum_grid * sizeof(Result)));
    CHECK(mcMemcpy(device_input, host_input.data(),
                   input_items * sizeof(float4), mcMemcpyHostToDevice));

    const Mode modes[] = {
        Mode::kControl, Mode::kLseSharedTranspose, Mode::kMaxAllreduce,
        Mode::kPostMaxExpSum, Mode::kSumAllreduce, Mode::kPostSumLog,
        Mode::kScaleSharedBarrier, Mode::kOaccumLoadFma,
        Mode::kOutputConversion, Mode::kSourceDependencyChain,
    };
    // These are steady source-program grids: a bounded partial AP wave and
    // one CTA per AP.  The pipeline separately schedules all 224 Graph CTAs
    // over those 104 independent source workers; launch-first-CTA work is
    // outside this control-subtracted per-CTA dependency measurement.
    const int grids[] = {32, kSteadyMaxGrid};
    const int repetition_counts[] = {
        kRepetitionBase, 2 * kRepetitionBase, 4 * kRepetitionBase};
    for (Mode mode : modes) {
        for (int grid : grids) {
            for (int repetitions : repetition_counts) {
                for (int warmup = 0; warmup < kWarmupsPerCell; ++warmup) {
                    launch(mode, grid, repetitions, device_input,
                           device_results);
                }
            }
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (Mode mode : modes) {
        for (int grid : grids) {
            for (int repetitions : repetition_counts) {
                for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                    plan.push_back({mode, grid, repetitions, sample});
                }
            }
        }
    }
    std::mt19937 generator(500907U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);
    std::vector<Result> host_results(maximum_grid);
    std::printf("metadata,2,MetaX C500,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                device_id, properties.multiProcessorCount,
                properties.warpSize, properties.clockRate, trial,
                kSamplesPerCell, kNumSplits, kMaxSplits, kBlockM, kHeadDim,
                kRepetitionBase);
    for (Mode mode : modes) {
        const Resources observed = resources(mode);
        if (observed.registers_per_thread <= 0
                || observed.resident_ctas_per_ap <= 0) {
            std::fprintf(stderr, "invalid kernel resources\n");
            return EXIT_FAILURE;
        }
        std::printf("resource,%s,%d,%zu,%d\n", mode_name(mode),
                    observed.registers_per_thread,
                    observed.static_shared_bytes,
                    observed.resident_ctas_per_ap);
    }
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& cell = plan[order];
        launch(cell.mode, cell.grid, cell.repetitions,
               device_input, device_results);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
        CHECK(mcMemcpy(host_results.data(), device_results,
                       cell.grid * sizeof(Result), mcMemcpyDeviceToHost));
        for (int block = 0; block < cell.grid; ++block) {
            if (host_results[block].cycles == 0) {
                std::fprintf(stderr, "zero cycle result\n");
                return EXIT_FAILURE;
            }
            std::printf("sample,%zu,%s,%d,%d,%d,%d,%llu,%u\n",
                        order, mode_name(cell.mode), cell.grid,
                        cell.repetitions, cell.sample, block,
                        host_results[block].cycles,
                        host_results[block].checksum);
        }
    }

    CHECK(mcFree(device_results));
    CHECK(mcFree(device_input));
    return EXIT_SUCCESS;
}
