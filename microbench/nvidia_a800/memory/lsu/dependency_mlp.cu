/*
 * NVIDIA A800 finite-source dependent-load MLP mechanism benchmark.
 *
 * This benchmark is independent of every production/evaluation kernel.  It
 * varies exactly three physical axes:
 *
 *   producer warps / SM:                 1, 4, 8, 16, 32
 *   independent pointer chains / warp:   1, 2, 4, 8
 *   cache tier:                          L1 hit, L2 hit, DRAM miss
 *
 * Lane zero of each producer warp advances C mutually independent pointer
 * recurrences.  Each recurrence step is one warp-global-load instruction
 * requesting one aligned 32-byte sector.  The next address of a chain is the
 * value returned by its preceding load; chains never consume one another's
 * values.  A post-interval checksum store keeps all chains observable.
 *
 * One 96-KiB CTA is admitted per active SM.  Every CTA records its own
 * clock64 interval and SM ID, and the host rejects duplicate SM IDs.  The
 * reported unit is 32-byte dependent-load requests / active SM / SM cycle.
 * It is a mechanism observation, not a hardware hard limit or a production
 * latency coefficient.
 *
 * Cache contracts:
 *
 *   l1_warm_default: ordinary ld.global, one private self-loop sector per
 *                    chain, warmed by the target CTA before timing;
 *   l2_warm_cg:      ld.global.cg, the same sub-MiB device footprint, warmed
 *                    by the target CTA before timing;
 *   dram_cg:         ld.global.cg over one incompressible randomized 256-MiB
 *                    ring (> 6x the required 40-MiB A800 L2), with evenly
 *                    spaced chain heads and an 80-MiB pre-trial L2 eviction.
 *
 * NCU traffic/cache counters must certify these tier claims before the native
 * sweep may be interpreted.  Native clock64 timing is the only duration
 * consumed by this independent mechanism benchmark.
 *
 * Build:
 *   /usr/local/cuda-12.2/bin/nvcc -O3 -std=c++14 -arch=sm_80 \
 *     dependency_mlp.cu -o dependency_mlp
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>


namespace {

constexpr int kWarpSize = 32;
constexpr int kMaximumWarps = 32;
constexpr int kMaximumChains = 8;
constexpr int kDynamicSharedBytes = 96 * 1024;
constexpr int kExpectedSmCount = 108;
constexpr int kExpectedL2Bytes = 40 * 1024 * 1024;
constexpr int kSectorBytes = 32;
constexpr int kWordsPerSector = kSectorBytes / sizeof(std::uint32_t);
constexpr std::size_t kDramRingBytes = 256ULL * 1024 * 1024;
constexpr std::size_t kEvictionBytes = 80ULL * 1024 * 1024;
constexpr std::uint64_t kWarmTargetAdvancesPerSm = 1ULL << 20;
constexpr int kDefaultTrials = 9;
constexpr int kWarpPoints[] = {1, 4, 8, 16, 32};
constexpr int kChainPoints[] = {1, 2, 4, 8};


void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "%s failed: %s\n", expression,
            cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA(call) check_cuda((call), #call)


enum class CacheTier {
    kL1WarmDefault,
    kL2WarmCg,
    kDramCg,
};


struct alignas(32) BlockResult {
    std::uint64_t cycles;
    std::uint64_t sm_id;
    std::uint64_t checksum;
    std::uint64_t reserved;
};


static_assert(sizeof(BlockResult) == kSectorBytes,
              "one result must occupy one 32-byte sector");


template <CacheTier tier>
__device__ __forceinline__ std::uint32_t dependent_load(
        const std::uint32_t* address) {
    std::uint32_t value;
    if (tier == CacheTier::kL1WarmDefault) {
        asm volatile(
            "ld.global.ca.u32 %0, [%1];"
            : "=r"(value)
            : "l"(address)
            : "memory");
    } else {
        asm volatile(
            "ld.global.cg.u32 %0, [%1];"
            : "=r"(value)
            : "l"(address)
            : "memory");
    }
    return value;
}


template <CacheTier tier, int chains>
__device__ __forceinline__ void advance_chains(
        const std::uint32_t* nodes, std::uint32_t (&pointers)[chains]) {
    static_assert(
        chains == 1 || chains == 2 || chains == 4 || chains == 8,
        "chain count must be one of the frozen sweep points");
    #pragma unroll
    for (int chain = 0; chain < chains; ++chain) {
        pointers[chain] = dependent_load<tier>(
            nodes + static_cast<std::size_t>(pointers[chain])
                    * kWordsPerSector);
    }
}


template <CacheTier tier, int chains>
__device__ __forceinline__ void dependency_mlp_body(
        const std::uint32_t* nodes, const std::uint32_t* heads,
        std::uint32_t* sink, BlockResult* results, int steps) {
    extern __shared__ unsigned char occupancy_storage[];
    __shared__ std::uint64_t start_cycle;
    __shared__ std::uint64_t elapsed_cycles;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int warps = blockDim.x / kWarpSize;
    std::uint32_t pointers[chains];
    std::uint32_t checksum = 0;

    if (lane == 0) {
        const std::size_t head_base = (
            (static_cast<std::size_t>(blockIdx.x) * kMaximumWarps + warp)
            * kMaximumChains);
        #pragma unroll
        for (int chain = 0; chain < chains; ++chain) {
            pointers[chain] = heads[head_base + chain];
        }
        if (tier != CacheTier::kDramCg) {
            // One source-visible recurrence advance brings each private
            // self-loop sector into the selected cache before the timed
            // interval.  Its returned pointer feeds the timed recurrence, so
            // the compiler cannot discard the warm access as dead work.
            advance_chains<tier, chains>(nodes, pointers);
        }
    }

    if (threadIdx.x == 0) {
        occupancy_storage[0] = static_cast<unsigned char>(blockIdx.x);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        start_cycle = clock64();
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll 1
        for (int step = 0; step < steps; ++step) {
            advance_chains<tier, chains>(nodes, pointers);
        }
        #pragma unroll
        for (int chain = 0; chain < chains; ++chain) {
            checksum ^= pointers[chain] + static_cast<std::uint32_t>(chain);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        elapsed_cycles = clock64() - start_cycle;
    }
    __syncthreads();

    if (lane == 0) {
        sink[static_cast<std::size_t>(blockIdx.x) * kMaximumWarps + warp] = (
            checksum);
    }
    if (threadIdx.x == 0) {
        std::uint32_t sm_id32;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id32));
        BlockResult* result = results + blockIdx.x;
        result->cycles = elapsed_cycles;
        result->sm_id = sm_id32;
        result->checksum = checksum ^ static_cast<std::uint32_t>(warps);
        result->reserved = 0;
    }
}


}  // namespace


#define DEFINE_DEPENDENCY_MLP_KERNEL(name, tier, chains)                 \
extern "C" __global__ __launch_bounds__(1024, 1)                        \
void name(                                                               \
        const std::uint32_t* nodes, const std::uint32_t* heads,           \
        std::uint32_t* sink, BlockResult* results, int steps) {           \
    dependency_mlp_body<tier, chains>(                                   \
        nodes, heads, sink, results, steps);                              \
}


DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_l1_warm_default_c1, CacheTier::kL1WarmDefault, 1)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_l1_warm_default_c2, CacheTier::kL1WarmDefault, 2)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_l1_warm_default_c4, CacheTier::kL1WarmDefault, 4)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_l1_warm_default_c8, CacheTier::kL1WarmDefault, 8)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_l2_warm_cg_c1, CacheTier::kL2WarmCg, 1)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_l2_warm_cg_c2, CacheTier::kL2WarmCg, 2)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_l2_warm_cg_c4, CacheTier::kL2WarmCg, 4)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_l2_warm_cg_c8, CacheTier::kL2WarmCg, 8)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_dram_cg_c1, CacheTier::kDramCg, 1)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_dram_cg_c2, CacheTier::kDramCg, 2)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_dram_cg_c4, CacheTier::kDramCg, 4)
DEFINE_DEPENDENCY_MLP_KERNEL(
    dependency_mlp_dram_cg_c8, CacheTier::kDramCg, 8)


#undef DEFINE_DEPENDENCY_MLP_KERNEL


namespace {

using Kernel = void (*)(
    const std::uint32_t*, const std::uint32_t*, std::uint32_t*,
    BlockResult*, int);


struct Variant {
    const char* tier_name;
    const char* kernel_name;
    CacheTier tier;
    int chains;
    Kernel kernel;
};


const Variant kVariants[] = {
    {"l1_warm_default", "dependency_mlp_l1_warm_default_c1",
     CacheTier::kL1WarmDefault, 1, dependency_mlp_l1_warm_default_c1},
    {"l1_warm_default", "dependency_mlp_l1_warm_default_c2",
     CacheTier::kL1WarmDefault, 2, dependency_mlp_l1_warm_default_c2},
    {"l1_warm_default", "dependency_mlp_l1_warm_default_c4",
     CacheTier::kL1WarmDefault, 4, dependency_mlp_l1_warm_default_c4},
    {"l1_warm_default", "dependency_mlp_l1_warm_default_c8",
     CacheTier::kL1WarmDefault, 8, dependency_mlp_l1_warm_default_c8},
    {"l2_warm_cg", "dependency_mlp_l2_warm_cg_c1",
     CacheTier::kL2WarmCg, 1, dependency_mlp_l2_warm_cg_c1},
    {"l2_warm_cg", "dependency_mlp_l2_warm_cg_c2",
     CacheTier::kL2WarmCg, 2, dependency_mlp_l2_warm_cg_c2},
    {"l2_warm_cg", "dependency_mlp_l2_warm_cg_c4",
     CacheTier::kL2WarmCg, 4, dependency_mlp_l2_warm_cg_c4},
    {"l2_warm_cg", "dependency_mlp_l2_warm_cg_c8",
     CacheTier::kL2WarmCg, 8, dependency_mlp_l2_warm_cg_c8},
    {"dram_cg", "dependency_mlp_dram_cg_c1",
     CacheTier::kDramCg, 1, dependency_mlp_dram_cg_c1},
    {"dram_cg", "dependency_mlp_dram_cg_c2",
     CacheTier::kDramCg, 2, dependency_mlp_dram_cg_c2},
    {"dram_cg", "dependency_mlp_dram_cg_c4",
     CacheTier::kDramCg, 4, dependency_mlp_dram_cg_c4},
    {"dram_cg", "dependency_mlp_dram_cg_c8",
     CacheTier::kDramCg, 8, dependency_mlp_dram_cg_c8},
};


struct Options {
    const char* tier;
    int chains;
    int warps;
    int trials;
    bool profile;
};


void print_usage(const char* executable) {
    std::fprintf(
        stderr,
        "usage: %s [--tier l1_warm_default|l2_warm_cg|dram_cg] "
        "[--chains 1|2|4|8] [--warps 1|4|8|16|32] "
        "[--trials N] [--profile]\n",
        executable);
}


int parse_positive_int(const char* text, const char* option) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0'
            || value <= 0 || value > INT_MAX) {
        std::fprintf(stderr, "%s requires a positive int, got %s\n",
                     option, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}


bool in_points(int value, const int* values, int count) {
    for (int index = 0; index < count; ++index) {
        if (value == values[index]) {
            return true;
        }
    }
    return false;
}


Options parse_options(int argc, char** argv) {
    Options options{nullptr, 0, 0, kDefaultTrials, false};
    for (int index = 1; index < argc; ++index) {
        const char* option = argv[index];
        if (std::strcmp(option, "--tier") == 0) {
            if (++index == argc || options.tier != nullptr) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.tier = argv[index];
        } else if (std::strcmp(option, "--chains") == 0) {
            if (++index == argc || options.chains != 0) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.chains = parse_positive_int(argv[index], option);
        } else if (std::strcmp(option, "--warps") == 0) {
            if (++index == argc || options.warps != 0) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.warps = parse_positive_int(argv[index], option);
        } else if (std::strcmp(option, "--trials") == 0) {
            if (++index == argc) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.trials = parse_positive_int(argv[index], option);
        } else if (std::strcmp(option, "--profile") == 0) {
            options.profile = true;
        } else if (std::strcmp(option, "--help") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::fprintf(stderr, "unknown option %s\n", option);
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (options.chains != 0
            && !in_points(
                options.chains, kChainPoints,
                sizeof(kChainPoints) / sizeof(kChainPoints[0]))) {
        std::fprintf(stderr, "--chains must be 1, 2, 4, or 8\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.warps != 0
            && !in_points(
                options.warps, kWarpPoints,
                sizeof(kWarpPoints) / sizeof(kWarpPoints[0]))) {
        std::fprintf(stderr, "--warps must be 1, 4, 8, 16, or 32\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.profile) {
        if (options.tier == nullptr
                || options.chains == 0 || options.warps == 0) {
            std::fprintf(
                stderr,
                "--profile requires --tier, --chains, and --warps\n");
            std::exit(EXIT_FAILURE);
        }
        options.trials = 1;
    }
    return options;
}


const Variant* find_variant(const char* tier, int chains) {
    for (const Variant& variant : kVariants) {
        if (std::strcmp(tier, variant.tier_name) == 0
                && chains == variant.chains) {
            return &variant;
        }
    }
    std::fprintf(
        stderr,
        "unknown tier/chains pair %s/%d; expected a frozen matrix cell\n",
        tier, chains);
    std::exit(EXIT_FAILURE);
}


void configure_kernel(const Variant& variant, int threads) {
    CHECK_CUDA(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(variant.kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int active_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, variant.kernel, threads, kDynamicSharedBytes));
    if (active_blocks != 1) {
        std::fprintf(
            stderr,
            "%s at %d threads must admit exactly one CTA/SM; got %d\n",
            variant.kernel_name, threads, active_blocks);
        std::exit(EXIT_FAILURE);
    }
}


std::uint64_t ceil_div_u64(std::uint64_t numerator,
                           std::uint64_t denominator) {
    return numerator / denominator + (numerator % denominator != 0);
}


int steps_for_cell(CacheTier tier, int grid, int warps, int chains) {
    std::uint64_t target;
    if (tier == CacheTier::kDramCg) {
        target = kDramRingBytes / kSectorBytes;
    } else {
        target = static_cast<std::uint64_t>(grid)
                 * kWarmTargetAdvancesPerSm;
    }
    const std::uint64_t producers = (
        static_cast<std::uint64_t>(grid) * warps * chains);
    const std::uint64_t steps = ceil_div_u64(target, producers);
    if (steps == 0 || steps > static_cast<std::uint64_t>(INT_MAX)) {
        std::fprintf(stderr, "cell step count is outside int range\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(steps);
}


std::uint64_t target_advances(
        int grid, int warps, int chains, int steps) {
    const std::uint64_t first = static_cast<std::uint64_t>(grid) * warps;
    if (first > std::numeric_limits<std::uint64_t>::max() / chains
            || first * chains
               > std::numeric_limits<std::uint64_t>::max() / steps) {
        std::fprintf(stderr, "target advance count overflows uint64\n");
        std::exit(EXIT_FAILURE);
    }
    return first * chains * steps;
}


void launch(
        const Variant& variant, int grid, int threads,
        const std::uint32_t* nodes, const std::uint32_t* heads,
        std::uint32_t* sink, BlockResult* results, int steps) {
    const std::uint32_t* node_argument = nodes;
    const std::uint32_t* head_argument = heads;
    std::uint32_t* sink_argument = sink;
    BlockResult* result_argument = results;
    int step_argument = steps;
    void* arguments[] = {
        &node_argument, &head_argument, &sink_argument,
        &result_argument, &step_argument,
    };
    CHECK_CUDA(cudaLaunchKernel(
        reinterpret_cast<const void*>(variant.kernel),
        dim3(grid), dim3(threads), arguments,
        kDynamicSharedBytes, nullptr));
    CHECK_CUDA(cudaGetLastError());
}


struct TrialResult {
    std::uint64_t sum_cycles;
    std::uint64_t minimum_cycles;
    std::uint64_t maximum_cycles;
    double requests_per_active_sm_cycle;
};


TrialResult run_trial(
        const Variant& variant, int grid, int threads, int warps, int chains,
        int steps, const std::uint32_t* nodes, const std::uint32_t* heads,
        std::uint32_t* sink, BlockResult* device_results,
        std::vector<BlockResult>* host_results,
        std::uint32_t* eviction, std::size_t eviction_bytes, int trial) {
    if (variant.tier == CacheTier::kDramCg) {
        CHECK_CUDA(cudaMemset(
            eviction, static_cast<unsigned char>(trial + 1), eviction_bytes));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaMemset(
        device_results, 0xff,
        static_cast<std::size_t>(grid) * sizeof(BlockResult)));
    launch(
        variant, grid, threads, nodes, heads, sink, device_results, steps);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(
        host_results->data(), device_results,
        static_cast<std::size_t>(grid) * sizeof(BlockResult),
        cudaMemcpyDeviceToHost));

    std::set<std::uint64_t> sm_ids;
    std::uint64_t sum_cycles = 0;
    std::uint64_t minimum_cycles = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t maximum_cycles = 0;
    for (int block = 0; block < grid; ++block) {
        const BlockResult& result = (*host_results)[block];
        if (result.cycles == 0
                || result.cycles == std::numeric_limits<std::uint64_t>::max()
                || result.sm_id == std::numeric_limits<std::uint64_t>::max()) {
            std::fprintf(stderr, "block %d did not record a valid result\n",
                         block);
            std::exit(EXIT_FAILURE);
        }
        if (!sm_ids.insert(result.sm_id).second) {
            std::fprintf(
                stderr,
                "launch did not map one CTA to each SM: duplicate SM ID %llu\n",
                static_cast<unsigned long long>(result.sm_id));
            std::exit(EXIT_FAILURE);
        }
        if (sum_cycles
                > std::numeric_limits<std::uint64_t>::max() - result.cycles) {
            std::fprintf(stderr, "sum of SM cycles overflows uint64\n");
            std::exit(EXIT_FAILURE);
        }
        sum_cycles += result.cycles;
        minimum_cycles = std::min(minimum_cycles, result.cycles);
        maximum_cycles = std::max(maximum_cycles, result.cycles);
    }
    if (static_cast<int>(sm_ids.size()) != grid) {
        std::fprintf(stderr, "expected %d distinct SM IDs, got %zu\n",
                     grid, sm_ids.size());
        std::exit(EXIT_FAILURE);
    }
    const std::uint64_t advances = target_advances(
        grid, warps, chains, steps);
    return TrialResult{
        sum_cycles,
        minimum_cycles,
        maximum_cycles,
        static_cast<double>(advances) / static_cast<double>(sum_cycles),
    };
}


double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}


void fill_small_self_loops(
        int grid, std::vector<std::uint32_t>* nodes,
        std::vector<std::uint32_t>* heads) {
    const std::size_t producer_slots = (
        static_cast<std::size_t>(grid) * kMaximumWarps * kMaximumChains);
    nodes->assign(producer_slots * kWordsPerSector, 0);
    heads->assign(producer_slots, 0);
    std::mt19937 random(0x31af20dU);
    for (std::size_t slot = 0; slot < producer_slots; ++slot) {
        const std::uint32_t sector = static_cast<std::uint32_t>(slot);
        (*heads)[slot] = sector;
        const std::size_t word = slot * kWordsPerSector;
        (*nodes)[word] = sector;
        for (int index = 1; index < kWordsPerSector; ++index) {
            (*nodes)[word + index] = random();
        }
    }
}


void fill_dram_ring(
        std::vector<std::uint32_t>* nodes,
        std::vector<std::uint32_t>* cycle) {
    const std::size_t sectors = kDramRingBytes / kSectorBytes;
    const std::size_t words = kDramRingBytes / sizeof(std::uint32_t);
    nodes->resize(words);
    std::mt19937 random(0x7c9e51aU);
    for (std::uint32_t& value : *nodes) {
        value = random();
    }
    cycle->resize(sectors);
    std::iota(cycle->begin(), cycle->end(), 0U);
    std::shuffle(cycle->begin(), cycle->end(), random);
    for (std::size_t position = 0; position < sectors; ++position) {
        const std::uint32_t current = (*cycle)[position];
        const std::uint32_t next = (*cycle)[(position + 1) % sectors];
        (*nodes)[static_cast<std::size_t>(current) * kWordsPerSector] = next;
    }
}


void make_dram_heads(
        int grid, int warps, int chains,
        const std::vector<std::uint32_t>& cycle,
        std::vector<std::uint32_t>* heads) {
    const std::size_t head_slots = (
        static_cast<std::size_t>(grid) * kMaximumWarps * kMaximumChains);
    heads->assign(head_slots, 0);
    const std::size_t producers = (
        static_cast<std::size_t>(grid) * warps * chains);
    std::size_t producer = 0;
    for (int block = 0; block < grid; ++block) {
        for (int warp = 0; warp < warps; ++warp) {
            for (int chain = 0; chain < chains; ++chain) {
                const std::size_t position = (
                    producer * cycle.size() / producers);
                const std::size_t head_index = (
                    (static_cast<std::size_t>(block) * kMaximumWarps + warp)
                    * kMaximumChains + chain);
                (*heads)[head_index] = cycle[position];
                ++producer;
            }
        }
    }
    if (producer != producers) {
        std::fprintf(stderr, "DRAM head construction lost producers\n");
        std::exit(EXIT_FAILURE);
    }
}


void benchmark_cell(
        const Variant& variant, const Options& options, int grid, int warps,
        const std::uint32_t* small_nodes, const std::uint32_t* small_heads,
        const std::uint32_t* dram_nodes,
        const std::vector<std::uint32_t>& dram_cycle,
        std::uint32_t* device_heads, std::size_t head_bytes,
        std::uint32_t* sink, BlockResult* device_results,
        std::vector<BlockResult>* host_results,
        std::uint32_t* eviction) {
    const int threads = warps * kWarpSize;
    const int chains = variant.chains;
    const int steps = steps_for_cell(variant.tier, grid, warps, chains);
    configure_kernel(variant, threads);

    const std::uint32_t* nodes = small_nodes;
    const std::uint32_t* heads = small_heads;
    std::vector<std::uint32_t> host_dram_heads;
    if (variant.tier == CacheTier::kDramCg) {
        make_dram_heads(
            grid, warps, chains, dram_cycle, &host_dram_heads);
        CHECK_CUDA(cudaMemcpy(
            device_heads, host_dram_heads.data(), head_bytes,
            cudaMemcpyHostToDevice));
        nodes = dram_nodes;
        heads = device_heads;
    }

    const std::uint64_t advances = target_advances(
        grid, warps, chains, steps);
    const std::uint64_t target_bytes = advances * kSectorBytes;
    const std::uint64_t producer_chains = (
        static_cast<std::uint64_t>(grid) * warps * chains);
    // Every producer first loads its chain head.  L1/L2 variants then issue
    // one source-visible target warm access before timing; DRAM deliberately
    // does not warm its unique traversal.
    const std::uint64_t untimed_load_rounds = (
        variant.tier == CacheTier::kDramCg ? 1 : 2);
    const std::uint64_t counter_load_instructions = (
        advances + producer_chains * untimed_load_rounds);
    std::printf(
        "expected,%s,%s,%d,%d,%d,%d,%llu,%llu,%llu,%llu\n",
        variant.tier_name, variant.kernel_name, grid, warps, chains, steps,
        static_cast<unsigned long long>(advances),
        static_cast<unsigned long long>(target_bytes),
        static_cast<unsigned long long>(counter_load_instructions),
        static_cast<unsigned long long>(counter_load_instructions));

    std::vector<double> rates;
    rates.reserve(options.trials);
    for (int trial = 0; trial < options.trials; ++trial) {
        const TrialResult result = run_trial(
            variant, grid, threads, warps, chains, steps,
            nodes, heads, sink, device_results, host_results,
            eviction, kEvictionBytes, trial);
        rates.push_back(result.requests_per_active_sm_cycle);
        std::printf(
            "raw,%s,%s,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%.12f\n",
            variant.tier_name, variant.kernel_name, grid, warps, chains,
            steps, trial,
            static_cast<unsigned long long>(advances),
            static_cast<unsigned long long>(result.sum_cycles),
            static_cast<unsigned long long>(result.minimum_cycles),
            static_cast<unsigned long long>(result.maximum_cycles),
            result.requests_per_active_sm_cycle);
        std::fflush(stdout);
    }
    std::printf(
        "summary,%s,%s,%d,%d,%d,%d,%.12f\n",
        variant.tier_name, variant.kernel_name, grid, warps, chains, steps,
        median(rates));
}


}  // namespace


int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    int device = -1;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 8 || properties.minor != 0
            || properties.warpSize != kWarpSize
            || properties.multiProcessorCount != kExpectedSmCount
            || properties.l2CacheSize != kExpectedL2Bytes) {
        std::fprintf(
            stderr,
            "expected A800/GA100 sm_80, warp32, 108 SM, 40-MiB L2; got "
            "sm_%d%d, warp%d, %d SM, %d-byte L2\n",
            properties.major, properties.minor, properties.warpSize,
            properties.multiProcessorCount, properties.l2CacheSize);
        return EXIT_FAILURE;
    }
    if (properties.maxThreadsPerBlock < kMaximumWarps * kWarpSize
            || properties.sharedMemPerMultiprocessor
                < kDynamicSharedBytes + 2 * sizeof(std::uint64_t)
            || properties.sharedMemPerBlockOptin
                < kDynamicSharedBytes + 2 * sizeof(std::uint64_t)) {
        std::fprintf(
            stderr,
            "device cannot support the frozen one-CTA-per-SM contract\n");
        return EXIT_FAILURE;
    }
    const int grid = properties.multiProcessorCount;

    std::vector<std::uint32_t> host_small_nodes;
    std::vector<std::uint32_t> host_small_heads;
    fill_small_self_loops(grid, &host_small_nodes, &host_small_heads);
    std::vector<std::uint32_t> host_dram_nodes;
    std::vector<std::uint32_t> host_dram_cycle;
    fill_dram_ring(&host_dram_nodes, &host_dram_cycle);

    std::uint32_t* small_nodes = nullptr;
    std::uint32_t* small_heads = nullptr;
    std::uint32_t* dram_nodes = nullptr;
    std::uint32_t* device_heads = nullptr;
    std::uint32_t* sink = nullptr;
    std::uint32_t* eviction = nullptr;
    BlockResult* device_results = nullptr;
    const std::size_t small_node_bytes = (
        host_small_nodes.size() * sizeof(std::uint32_t));
    const std::size_t head_bytes = (
        host_small_heads.size() * sizeof(std::uint32_t));
    const std::size_t sink_bytes = (
        static_cast<std::size_t>(grid) * kMaximumWarps
        * sizeof(std::uint32_t));
    CHECK_CUDA(cudaMalloc(&small_nodes, small_node_bytes));
    CHECK_CUDA(cudaMalloc(&small_heads, head_bytes));
    CHECK_CUDA(cudaMalloc(&dram_nodes, kDramRingBytes));
    CHECK_CUDA(cudaMalloc(&device_heads, head_bytes));
    CHECK_CUDA(cudaMalloc(&sink, sink_bytes));
    CHECK_CUDA(cudaMalloc(&eviction, kEvictionBytes));
    CHECK_CUDA(cudaMalloc(
        &device_results,
        static_cast<std::size_t>(grid) * sizeof(BlockResult)));
    CHECK_CUDA(cudaMemcpy(
        small_nodes, host_small_nodes.data(), small_node_bytes,
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        small_heads, host_small_heads.data(), head_bytes,
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        dram_nodes, host_dram_nodes.data(), kDramRingBytes,
        cudaMemcpyHostToDevice));
    std::vector<BlockResult> host_results(grid);

    const cudaUUID_t& uuid = properties.uuid;
    char uuid_text[48]{};
    char pci_bus_id[32]{};
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));
    std::snprintf(
        uuid_text, sizeof(uuid_text),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
        "%02x%02x%02x%02x%02x%02x",
        static_cast<unsigned char>(uuid.bytes[0]),
        static_cast<unsigned char>(uuid.bytes[1]),
        static_cast<unsigned char>(uuid.bytes[2]),
        static_cast<unsigned char>(uuid.bytes[3]),
        static_cast<unsigned char>(uuid.bytes[4]),
        static_cast<unsigned char>(uuid.bytes[5]),
        static_cast<unsigned char>(uuid.bytes[6]),
        static_cast<unsigned char>(uuid.bytes[7]),
        static_cast<unsigned char>(uuid.bytes[8]),
        static_cast<unsigned char>(uuid.bytes[9]),
        static_cast<unsigned char>(uuid.bytes[10]),
        static_cast<unsigned char>(uuid.bytes[11]),
        static_cast<unsigned char>(uuid.bytes[12]),
        static_cast<unsigned char>(uuid.bytes[13]),
        static_cast<unsigned char>(uuid.bytes[14]),
        static_cast<unsigned char>(uuid.bytes[15]));
    std::printf("meta,device_index,%d\n", device);
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,device_uuid,%s\n", uuid_text);
    std::printf("meta,pci_bus_id,%s\n", pci_bus_id);
    std::printf("meta,compute_capability,sm_%d%d\n",
                properties.major, properties.minor);
    std::printf("meta,sm_count,%d\n", grid);
    std::printf("meta,l2_bytes,%d\n", properties.l2CacheSize);
    std::printf("meta,dynamic_shared_bytes_per_cta,%d\n",
                kDynamicSharedBytes);
    std::printf("meta,producer_warps_per_sm_points,1|4|8|16|32\n");
    std::printf("meta,independent_chains_per_warp_points,1|2|4|8\n");
    std::printf(
        "meta,cache_tiers,l1_warm_default|l2_warm_cg|dram_cg\n");
    std::printf("meta,sector_bytes,%d\n", kSectorBytes);
    std::printf("meta,dram_ring_bytes,%zu\n", kDramRingBytes);
    std::printf("meta,eviction_bytes,%zu\n", kEvictionBytes);
    std::printf(
        "meta,rate_unit,32B_dependent_load_requests_per_active_SM_per_SM_cycle\n");
    std::printf("meta,production_or_evaluation_duration_consumed,false\n");
    std::printf("meta,production_parameter_published,false\n");
    std::printf(
        "expected,tier,kernel,grid,producer_warps_per_sm,"
        "independent_chains_per_warp,steps_per_chain,target_chain_advances,"
        "target_request_bytes,counter_load_warp_instructions,"
        "counter_load_sectors\n");
    std::printf(
        "raw,tier,kernel,grid,producer_warps_per_sm,"
        "independent_chains_per_warp,steps_per_chain,trial,"
        "target_chain_advances,sum_cta_sm_cycles,min_cta_sm_cycles,"
        "max_cta_sm_cycles,requests_per_active_sm_cycle\n");

    std::vector<const Variant*> variants;
    if (options.tier != nullptr && options.chains != 0) {
        variants.push_back(find_variant(options.tier, options.chains));
    } else {
        for (const Variant& variant : kVariants) {
            if (options.tier != nullptr
                    && std::strcmp(options.tier, variant.tier_name) != 0) {
                continue;
            }
            if (options.chains != 0 && options.chains != variant.chains) {
                continue;
            }
            variants.push_back(&variant);
        }
    }
    if (variants.empty()) {
        std::fprintf(stderr, "selection contains no benchmark variants\n");
        return EXIT_FAILURE;
    }

    for (const Variant* variant : variants) {
        if (options.warps != 0) {
            benchmark_cell(
                *variant, options, grid, options.warps,
                small_nodes, small_heads, dram_nodes, host_dram_cycle,
                device_heads, head_bytes, sink, device_results,
                &host_results, eviction);
        } else {
            for (int warps : kWarpPoints) {
                benchmark_cell(
                    *variant, options, grid, warps,
                    small_nodes, small_heads, dram_nodes, host_dram_cycle,
                    device_heads, head_bytes, sink, device_results,
                    &host_results, eviction);
            }
        }
    }

    CHECK_CUDA(cudaFree(device_results));
    CHECK_CUDA(cudaFree(eviction));
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(device_heads));
    CHECK_CUDA(cudaFree(dram_nodes));
    CHECK_CUDA(cudaFree(small_heads));
    CHECK_CUDA(cudaFree(small_nodes));
    return EXIT_SUCCESS;
}
