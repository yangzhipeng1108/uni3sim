/*
 * NVIDIA A800 LSU 32-byte-sector request-rate mechanism benchmark.
 *
 * This is a workload-independent frontend sweep, not a production-kernel
 * calibration.  Each target warp memory instruction has exactly 8, 16, or 32
 * contiguous active lanes, so it requests exactly 1, 2, or 4 aligned 32-byte
 * sectors.  The nine extern "C" entry points make every access-kind/sector
 * variant independently selectable and give NCU stable kernel names.
 *
 * One CTA is launched on each active SM at each of 128, 256, 512, and 1024
 * threads (4, 8, 16, and 32 producer warps/SM).  Ninety-six KiB of dynamic
 * shared memory limits residency to one CTA/SM at every point; the program
 * also rejects any launch that does not report one distinct SM ID per CTA.
 * The optional grid-block axis changes the number of active SM producers
 * without changing the per-SM instruction stream.  Each CTA records its own
 * clock64 interval, and total target sectors divided by the sum of CTA cycles
 * has units 32-byte sectors / active SM / SM-cycle.  A per-SM rate that falls
 * with the active-SM count is shared-backend evidence, not an LSU frontend
 * ceiling.
 *
 * The result is explicitly an achieved lower bound: loop/control/barrier and
 * load-checksum ALU cost remain in the measured interval, and cache/store-path
 * backpressure may bind before the LSU frontend.  Load, store, and mixed
 * results must remain separate if the independently observed ceilings differ.
 *
 * Build:
 *   nvcc -O3 -std=c++14 -arch=sm_80 request_rate.cu -o request_rate
 *
 * Benchmark all variants (nine raw trials and a median per variant):
 *   ./request_rate
 *
 * Backend-binding grid sweep (one example access class):
 *   ./request_rate --variant store_s4 --threads 1024 --grid-blocks 1
 *
 * Formal NCU contract (CLI variant -> stable extern-C kernel):
 *   load_s1  -> request_rate_load_s1
 *   load_s2  -> request_rate_load_s2
 *   load_s4  -> request_rate_load_s4
 *   store_s1 -> request_rate_store_s1
 *   store_s2 -> request_rate_store_s2
 *   store_s4 -> request_rate_store_s4
 *   mixed_s1 -> request_rate_mixed_s1
 *   mixed_s2 -> request_rate_mixed_s2
 *   mixed_s4 -> request_rate_mixed_s4
 *
 * Counter-check each of those nine variants at each producer point with
 * exactly one target launch (substitute KERNEL, VARIANT, and THREADS):
 *   ncu --kernel-name-base function \
 *     --kernel-name regex:"^KERNEL$" --launch-count 1 \
 *     --metrics \
 * l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
 * l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
 * smsp__sass_inst_executed_op_global_ld.sum,\
 * smsp__sass_inst_executed_op_global_st.sum \
 *     ./request_rate --profile VARIANT --threads THREADS --iterations 4096
 *
 * The target PTX uses ordinary ld.global/st.global semantics: it does not use
 * PTX .volatile, cache operators, or explicit memory scopes.  The inline-asm
 * volatile qualifier prevents the CUDA frontend from deleting the source asm;
 * it does not add the PTX .volatile modifier.  Pure-load addresses alternate
 * between two L1-resident banks through a runtime kernel argument, which keeps
 * PTXAS from hoisting invariant loads out of the loop.  Loaded values feed a
 * checksum that is stored after the timed interval, so the loads remain
 * observable without changing their memory semantic.  CUDA 12.2 sm_80 emits
 * ordinary LDG.E/STG.E for these operations (no STRONG scope suffix).
 *
 * A load or mixed kernel writes one checksum store per active warp after the
 * measured interval.  Every kernel also writes one 16-byte timing/SM-ID record
 * from lane 0.  The checksum store has the selected 1/2/4-sector geometry; the
 * aligned record occupies one sector.  The program prints target-only and full
 * expected NCU counter values, including both post-interval stores, so neither
 * can be mistaken for the timed mechanism.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <vector>


namespace {

constexpr int kMaximumThreads = 1024;
constexpr int kWarpSize = 32;
constexpr int kMaximumWarpsPerBlock = kMaximumThreads / kWarpSize;
constexpr int kTargetInstructionsPerIteration = 8;
constexpr int kWordsPerSector = 8;
constexpr int kWordsPerSlot = 32;
constexpr int kSlotsPerWarp = 8;
constexpr int kInputBanksPerWarp = 2;
constexpr int kWordsPerInputBank = kSlotsPerWarp * kWordsPerSlot;
constexpr int kWordsPerWarp = kInputBanksPerWarp * kWordsPerInputBank;
constexpr int kWordsPerBlock = kMaximumWarpsPerBlock * kWordsPerWarp;
constexpr int kDynamicSharedBytes = 96 * 1024;
constexpr int kDefaultIterations = 16384;
constexpr int kDefaultTrials = 9;
constexpr int kWarmupIterations = 256;
constexpr int kThreadPoints[] = {128, 256, 512, 1024};


void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "%s failed: %s\n", expression,
            cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA(call) check_cuda((call), #call)


struct alignas(32) BlockResult {
    std::uint64_t cycles;
    std::uint64_t sm_id;
    std::uint64_t reserved0;
    std::uint64_t reserved1;
};


static_assert(sizeof(BlockResult) == 32, "one result must occupy one sector");


enum class AccessKind {
    kLoad,
    kStore,
    kMixed,
};


template <int sectors>
__device__ __forceinline__ bool target_lane(int lane) {
    static_assert(sectors == 1 || sectors == 2 || sectors == 4,
                  "sector variant must be 1, 2, or 4");
    return lane < sectors * kWordsPerSector;
}


__device__ __forceinline__ std::uint32_t load_global(
        const std::uint32_t* address) {
    std::uint32_t value;
    asm volatile(
        "ld.global.u32 %0, [%1];"
        : "=r"(value)
        : "l"(address)
        : "memory");
    return value;
}


__device__ __forceinline__ void store_global(
        std::uint32_t* address, std::uint32_t value) {
    asm volatile(
        "st.global.u32 [%0], %1;"
        :
        : "l"(address), "r"(value)
        : "memory");
}


__device__ __forceinline__ std::uint32_t issue_eight_loads(
        const std::uint32_t* base, int lane, std::uint32_t checksum) {
    const std::uint32_t value0 = load_global(
        base + 0 * kWordsPerSlot + lane);
    const std::uint32_t value1 = load_global(
        base + 1 * kWordsPerSlot + lane);
    const std::uint32_t value2 = load_global(
        base + 2 * kWordsPerSlot + lane);
    const std::uint32_t value3 = load_global(
        base + 3 * kWordsPerSlot + lane);
    const std::uint32_t value4 = load_global(
        base + 4 * kWordsPerSlot + lane);
    const std::uint32_t value5 = load_global(
        base + 5 * kWordsPerSlot + lane);
    const std::uint32_t value6 = load_global(
        base + 6 * kWordsPerSlot + lane);
    const std::uint32_t value7 = load_global(
        base + 7 * kWordsPerSlot + lane);
    return (checksum ^ value0 ^ value1 ^ value2 ^ value3
            ^ value4 ^ value5 ^ value6 ^ value7);
}


__device__ __forceinline__ void issue_eight_stores(
        std::uint32_t* base, int lane, std::uint32_t value) {
    store_global(base + 0 * kWordsPerSlot + lane, value);
    store_global(base + 1 * kWordsPerSlot + lane, value);
    store_global(base + 2 * kWordsPerSlot + lane, value);
    store_global(base + 3 * kWordsPerSlot + lane, value);
    store_global(base + 4 * kWordsPerSlot + lane, value);
    store_global(base + 5 * kWordsPerSlot + lane, value);
    store_global(base + 6 * kWordsPerSlot + lane, value);
    store_global(base + 7 * kWordsPerSlot + lane, value);
}


__device__ __forceinline__ std::uint32_t issue_four_loads(
        const std::uint32_t* base, int lane, std::uint32_t checksum) {
    const std::uint32_t value0 = load_global(
        base + 0 * kWordsPerSlot + lane);
    const std::uint32_t value1 = load_global(
        base + 1 * kWordsPerSlot + lane);
    const std::uint32_t value2 = load_global(
        base + 2 * kWordsPerSlot + lane);
    const std::uint32_t value3 = load_global(
        base + 3 * kWordsPerSlot + lane);
    return checksum ^ value0 ^ value1 ^ value2 ^ value3;
}


__device__ __forceinline__ void issue_four_stores(
        std::uint32_t* base, int lane, std::uint32_t value) {
    store_global(base + 0 * kWordsPerSlot + lane, value);
    store_global(base + 1 * kWordsPerSlot + lane, value);
    store_global(base + 2 * kWordsPerSlot + lane, value);
    store_global(base + 3 * kWordsPerSlot + lane, value);
}


template <AccessKind kind>
__device__ __forceinline__ std::uint32_t issue_target_instructions(
        const std::uint32_t* input, std::uint32_t* output,
        int lane, int iteration, int input_bank_mask,
        std::uint32_t store_value, std::uint32_t checksum) {
    if (kind == AccessKind::kLoad) {
        const std::uint32_t* selected_input = (
            input
            + (iteration & input_bank_mask) * kWordsPerInputBank);
        return issue_eight_loads(selected_input, lane, checksum);
    } else if (kind == AccessKind::kStore) {
        issue_eight_stores(output, lane, store_value);
        return checksum;
    } else if (kind == AccessKind::kMixed) {
        const std::uint32_t* selected_input = (
            input
            + (iteration & input_bank_mask) * kWordsPerInputBank);
        checksum = issue_four_loads(selected_input, lane, checksum);
        issue_four_stores(output, lane, store_value);
        return checksum;
    }
    return checksum;
}


template <AccessKind kind, int sectors>
__device__ __forceinline__ void request_rate_body(
        const std::uint32_t* input, std::uint32_t* output,
        BlockResult* results, int iterations, int input_bank_mask) {
    extern __shared__ unsigned char occupancy_storage[];
    __shared__ std::uint64_t start_cycle;
    __shared__ std::uint64_t elapsed_cycles;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const std::size_t block_word = (
        static_cast<std::size_t>(blockIdx.x) * kWordsPerBlock);
    const std::size_t warp_word = (
        block_word + static_cast<std::size_t>(warp) * kWordsPerWarp);
    const std::uint32_t* warp_input = input + warp_word;
    std::uint32_t* warp_output = output + warp_word;
    const std::uint32_t store_value = (
        0x9e3779b9U ^ static_cast<std::uint32_t>(threadIdx.x));
    std::uint32_t checksum = store_value;

    if (threadIdx.x == 0) {
        occupancy_storage[0] = static_cast<unsigned char>(blockIdx.x);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        start_cycle = clock64();
    }
    __syncthreads();

    if (target_lane<sectors>(lane)) {
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            checksum = issue_target_instructions<kind>(
                warp_input, warp_output, lane, iteration, input_bank_mask,
                store_value, checksum);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        elapsed_cycles = clock64() - start_cycle;
    }
    __syncthreads();
    if (kind != AccessKind::kStore && target_lane<sectors>(lane)) {
        // This post-interval store makes every active lane's load checksum
        // observable.  Its one warp instruction and selected sector geometry
        // are reported separately in the expected NCU counter contract.
        store_global(warp_output + lane, checksum);
    }
    if (threadIdx.x == 0) {
        std::uint32_t sm_id32;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id32));
        const std::uint64_t sm_id = sm_id32;
        BlockResult* result = results + blockIdx.x;
        asm volatile(
            "st.global.v2.u64 [%0], {%1, %2};"
            :
            : "l"(result), "l"(elapsed_cycles), "l"(sm_id)
            : "memory");
    }
}


}  // namespace


#define DEFINE_REQUEST_RATE_KERNEL(name, kind, sectors)                    \
extern "C" __global__ __launch_bounds__(1024)                             \
void name(const std::uint32_t* input, std::uint32_t* output,               \
          BlockResult* results, int iterations, int input_bank_mask) {     \
    request_rate_body<kind, sectors>(                                     \
        input, output, results, iterations, input_bank_mask);              \
}


DEFINE_REQUEST_RATE_KERNEL(
    request_rate_load_s1, AccessKind::kLoad, 1)
DEFINE_REQUEST_RATE_KERNEL(
    request_rate_load_s2, AccessKind::kLoad, 2)
DEFINE_REQUEST_RATE_KERNEL(
    request_rate_load_s4, AccessKind::kLoad, 4)
DEFINE_REQUEST_RATE_KERNEL(
    request_rate_store_s1, AccessKind::kStore, 1)
DEFINE_REQUEST_RATE_KERNEL(
    request_rate_store_s2, AccessKind::kStore, 2)
DEFINE_REQUEST_RATE_KERNEL(
    request_rate_store_s4, AccessKind::kStore, 4)
DEFINE_REQUEST_RATE_KERNEL(
    request_rate_mixed_s1, AccessKind::kMixed, 1)
DEFINE_REQUEST_RATE_KERNEL(
    request_rate_mixed_s2, AccessKind::kMixed, 2)
DEFINE_REQUEST_RATE_KERNEL(
    request_rate_mixed_s4, AccessKind::kMixed, 4)


#undef DEFINE_REQUEST_RATE_KERNEL


namespace {

using Kernel = void (*)(
    const std::uint32_t*, std::uint32_t*, BlockResult*, int, int);


struct Variant {
    const char* name;
    const char* kernel_name;
    AccessKind access_kind;
    int sectors_per_instruction;
    int load_instructions_per_iteration;
    int store_instructions_per_iteration;
    Kernel kernel;
};


const Variant kVariants[] = {
    {"load_s1", "request_rate_load_s1", AccessKind::kLoad,
     1, 8, 0, request_rate_load_s1},
    {"load_s2", "request_rate_load_s2", AccessKind::kLoad,
     2, 8, 0, request_rate_load_s2},
    {"load_s4", "request_rate_load_s4", AccessKind::kLoad,
     4, 8, 0, request_rate_load_s4},
    {"store_s1", "request_rate_store_s1", AccessKind::kStore,
     1, 0, 8, request_rate_store_s1},
    {"store_s2", "request_rate_store_s2", AccessKind::kStore,
     2, 0, 8, request_rate_store_s2},
    {"store_s4", "request_rate_store_s4", AccessKind::kStore,
     4, 0, 8, request_rate_store_s4},
    {"mixed_s1", "request_rate_mixed_s1", AccessKind::kMixed,
     1, 4, 4, request_rate_mixed_s1},
    {"mixed_s2", "request_rate_mixed_s2", AccessKind::kMixed,
     2, 4, 4, request_rate_mixed_s2},
    {"mixed_s4", "request_rate_mixed_s4", AccessKind::kMixed,
     4, 4, 4, request_rate_mixed_s4},
};


const char* access_kind_name(AccessKind kind) {
    if (kind == AccessKind::kLoad) {
        return "load";
    } else if (kind == AccessKind::kStore) {
        return "store";
    } else if (kind == AccessKind::kMixed) {
        return "mixed";
    }
    std::fprintf(stderr, "unknown access kind\n");
    std::exit(EXIT_FAILURE);
}


const Variant* find_variant(const char* name) {
    for (const Variant& variant : kVariants) {
        if (std::strcmp(name, variant.name) == 0) {
            return &variant;
        }
    }
    std::fprintf(stderr, "unknown variant %s; expected", name);
    for (const Variant& variant : kVariants) {
        std::fprintf(stderr, " %s", variant.name);
    }
    std::fprintf(stderr, "\n");
    std::exit(EXIT_FAILURE);
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


struct Options {
    bool profile;
    const Variant* selected;
    int iterations;
    int trials;
    int threads;
    int grid_blocks;
};


void print_usage(const char* executable) {
    std::fprintf(
        stderr,
        "usage: %s [--variant NAME | --profile NAME] "
        "[--threads 128|256|512|1024] "
        "[--grid-blocks N] [--iterations N] [--trials N]\n",
        executable);
}


bool is_thread_point(int threads) {
    for (int point : kThreadPoints) {
        if (threads == point) {
            return true;
        }
    }
    return false;
}


Options parse_options(int argc, char** argv) {
    Options options{
        false, nullptr, kDefaultIterations, kDefaultTrials, 0, 0};
    for (int index = 1; index < argc; ++index) {
        const char* option = argv[index];
        if (std::strcmp(option, "--variant") == 0) {
            if (++index == argc || options.selected != nullptr) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.selected = find_variant(argv[index]);
        } else if (std::strcmp(option, "--profile") == 0) {
            if (++index == argc || options.selected != nullptr) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.profile = true;
            options.selected = find_variant(argv[index]);
            options.trials = 1;
        } else if (std::strcmp(option, "--iterations") == 0) {
            if (++index == argc) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.iterations = parse_positive_int(argv[index], option);
        } else if (std::strcmp(option, "--threads") == 0) {
            if (++index == argc || options.threads != 0) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.threads = parse_positive_int(argv[index], option);
            if (!is_thread_point(options.threads)) {
                std::fprintf(
                    stderr,
                    "--threads must be 128, 256, 512, or 1024; got %d\n",
                    options.threads);
                std::exit(EXIT_FAILURE);
            }
        } else if (std::strcmp(option, "--trials") == 0) {
            if (++index == argc || options.profile) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.trials = parse_positive_int(argv[index], option);
        } else if (std::strcmp(option, "--grid-blocks") == 0) {
            if (++index == argc || options.grid_blocks != 0) {
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            options.grid_blocks = parse_positive_int(argv[index], option);
        } else if (std::strcmp(option, "--help") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::fprintf(stderr, "unknown option %s\n", option);
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (options.profile
            && (options.selected == nullptr || options.threads == 0)) {
        std::fprintf(
            stderr,
            "--profile requires exactly one variant and --threads point\n");
        std::exit(EXIT_FAILURE);
    }
    return options;
}


std::uint64_t checked_multiply(
        std::uint64_t left, std::uint64_t right, const char* label) {
    if (right != 0
            && left > std::numeric_limits<std::uint64_t>::max() / right) {
        std::fprintf(stderr, "%s overflows uint64\n", label);
        std::exit(EXIT_FAILURE);
    }
    return left * right;
}


struct ExpectedCounts {
    std::uint64_t target_load_instructions;
    std::uint64_t target_store_instructions;
    std::uint64_t target_load_sectors;
    std::uint64_t target_store_sectors;
    std::uint64_t checksum_store_instructions;
    std::uint64_t checksum_store_sectors;
    std::uint64_t metadata_store_instructions;
    std::uint64_t metadata_store_sectors;
    std::uint64_t counter_load_instructions;
    std::uint64_t counter_store_instructions;
    std::uint64_t counter_load_sectors;
    std::uint64_t counter_store_sectors;
};


ExpectedCounts expected_counts(
        const Variant& variant, int grid, int threads, int iterations) {
    const int warps = threads / kWarpSize;
    const std::uint64_t warp_iterations = checked_multiply(
        checked_multiply(
            static_cast<std::uint64_t>(grid), warps,
            "warp iterations"),
        static_cast<std::uint64_t>(iterations), "warp iterations");
    const std::uint64_t target_load_instructions = checked_multiply(
        warp_iterations, variant.load_instructions_per_iteration,
        "target load instructions");
    const std::uint64_t target_store_instructions = checked_multiply(
        warp_iterations, variant.store_instructions_per_iteration,
        "target store instructions");
    const std::uint64_t target_load_sectors = checked_multiply(
        target_load_instructions, variant.sectors_per_instruction,
        "target load sectors");
    const std::uint64_t target_store_sectors = checked_multiply(
        target_store_instructions, variant.sectors_per_instruction,
        "target store sectors");
    const std::uint64_t checksum_store_instructions = (
        variant.load_instructions_per_iteration == 0
        ? 0
        : checked_multiply(
            static_cast<std::uint64_t>(grid), warps,
            "checksum store instructions"));
    const std::uint64_t checksum_store_sectors = checked_multiply(
        checksum_store_instructions, variant.sectors_per_instruction,
        "checksum store sectors");
    // One post-interval vector store from warp 0 records each CTA result.
    const std::uint64_t metadata_store_instructions = grid;
    const std::uint64_t metadata_store_sectors = grid;
    const std::uint64_t instrumentation_store_instructions = (
        checksum_store_instructions + metadata_store_instructions);
    const std::uint64_t instrumentation_store_sectors = (
        checksum_store_sectors + metadata_store_sectors);
    if (instrumentation_store_instructions < checksum_store_instructions
            || instrumentation_store_sectors < checksum_store_sectors
            || target_store_instructions
                > std::numeric_limits<std::uint64_t>::max()
                  - instrumentation_store_instructions
            || target_store_sectors
                > std::numeric_limits<std::uint64_t>::max()
                  - instrumentation_store_sectors) {
        std::fprintf(stderr, "instrumented store count overflows uint64\n");
        std::exit(EXIT_FAILURE);
    }
    return ExpectedCounts{
        target_load_instructions,
        target_store_instructions,
        target_load_sectors,
        target_store_sectors,
        checksum_store_instructions,
        checksum_store_sectors,
        metadata_store_instructions,
        metadata_store_sectors,
        target_load_instructions,
        target_store_instructions + instrumentation_store_instructions,
        target_load_sectors,
        target_store_sectors + instrumentation_store_sectors,
    };
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
            "%s at %d threads must admit exactly one CTA/SM, "
            "occupancy API reports %d\n",
            variant.kernel_name, threads, active_blocks);
        std::exit(EXIT_FAILURE);
    }
}


void launch(
        const Variant& variant, int grid, int threads,
        const std::uint32_t* input, std::uint32_t* output,
        BlockResult* results, int iterations) {
    const std::uint32_t* input_argument = input;
    std::uint32_t* output_argument = output;
    BlockResult* result_argument = results;
    int iteration_argument = iterations;
    int input_bank_mask_argument = kInputBanksPerWarp - 1;
    void* arguments[] = {
        &input_argument, &output_argument, &result_argument,
        &iteration_argument, &input_bank_mask_argument,
    };
    CHECK_CUDA(cudaLaunchKernel(
        reinterpret_cast<const void*>(variant.kernel),
        dim3(grid), dim3(threads), arguments,
        kDynamicSharedBytes, nullptr));
    CHECK_CUDA(cudaGetLastError());
}


struct TrialResult {
    std::uint64_t sum_sm_cycles;
    std::uint64_t minimum_sm_cycles;
    std::uint64_t maximum_sm_cycles;
    double load_sectors_per_sm_cycle;
    double store_sectors_per_sm_cycle;
    double total_sectors_per_sm_cycle;
};


TrialResult run_trial(
        const Variant& variant, int grid, int threads, int iterations,
        const ExpectedCounts& expected, const std::uint32_t* input,
        std::uint32_t* output, BlockResult* device_results,
        std::vector<BlockResult>* host_results) {
    CHECK_CUDA(cudaMemset(
        device_results, 0xff,
        static_cast<std::size_t>(grid) * sizeof(BlockResult)));
    launch(
        variant, grid, threads, input, output, device_results, iterations);
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

    const double denominator = static_cast<double>(sum_cycles);
    const double load_rate = expected.target_load_sectors / denominator;
    const double store_rate = expected.target_store_sectors / denominator;
    return TrialResult{
        sum_cycles,
        minimum_cycles,
        maximum_cycles,
        load_rate,
        store_rate,
        load_rate + store_rate,
    };
}


double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}


void print_expected(
        const Variant& variant, int grid, int threads, int iterations,
        const ExpectedCounts& expected) {
    const int warps = threads / kWarpSize;
    std::printf(
        "expected,%s,%s,%d,%d,%d,%d,%d,"
        "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
        "%llu,%llu,%llu,%llu\n",
        variant.name,
        variant.kernel_name,
        variant.sectors_per_instruction,
        grid,
        threads,
        warps,
        iterations,
        static_cast<unsigned long long>(expected.target_load_instructions),
        static_cast<unsigned long long>(expected.target_store_instructions),
        static_cast<unsigned long long>(expected.target_load_sectors),
        static_cast<unsigned long long>(expected.target_store_sectors),
        static_cast<unsigned long long>(expected.checksum_store_instructions),
        static_cast<unsigned long long>(expected.checksum_store_sectors),
        static_cast<unsigned long long>(expected.metadata_store_instructions),
        static_cast<unsigned long long>(expected.metadata_store_sectors),
        static_cast<unsigned long long>(expected.counter_load_instructions),
        static_cast<unsigned long long>(expected.counter_store_instructions),
        static_cast<unsigned long long>(expected.counter_load_sectors),
        static_cast<unsigned long long>(expected.counter_store_sectors));
}


void benchmark_variant(
        const Variant& variant, const Options& options, int grid, int threads,
        const std::uint32_t* input, std::uint32_t* output,
        BlockResult* device_results,
        std::vector<BlockResult>* host_results) {
    configure_kernel(variant, threads);
    const ExpectedCounts expected = expected_counts(
        variant, grid, threads, options.iterations);
    print_expected(variant, grid, threads, options.iterations, expected);

    if (!options.profile) {
        launch(
            variant, grid, threads, input, output, device_results,
            kWarmupIterations);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::vector<double> load_rates;
    std::vector<double> store_rates;
    std::vector<double> total_rates;
    load_rates.reserve(options.trials);
    store_rates.reserve(options.trials);
    total_rates.reserve(options.trials);
    for (int trial = 0; trial < options.trials; ++trial) {
        const TrialResult result = run_trial(
            variant, grid, threads, options.iterations, expected,
            input, output, device_results, host_results);
        load_rates.push_back(result.load_sectors_per_sm_cycle);
        store_rates.push_back(result.store_sectors_per_sm_cycle);
        total_rates.push_back(result.total_sectors_per_sm_cycle);
        std::printf(
            "raw,%s,%s,%d,%d,%d,%d,%d,%d,"
            "%llu,%llu,%llu,%.9f,%.9f,%.9f,true\n",
            variant.name,
            access_kind_name(variant.access_kind),
            variant.sectors_per_instruction,
            threads,
            threads / kWarpSize,
            trial,
            grid,
            options.iterations,
            static_cast<unsigned long long>(result.sum_sm_cycles),
            static_cast<unsigned long long>(result.minimum_sm_cycles),
            static_cast<unsigned long long>(result.maximum_sm_cycles),
            result.load_sectors_per_sm_cycle,
            result.store_sectors_per_sm_cycle,
            result.total_sectors_per_sm_cycle);
        std::fflush(stdout);
    }
    std::printf(
        "summary,%s,%s,%d,%d,%d,%.9f,%.9f,%.9f,true\n",
        variant.name,
        access_kind_name(variant.access_kind),
        variant.sectors_per_instruction,
        threads,
        threads / kWarpSize,
        median(load_rates),
        median(store_rates),
        median(total_rates));
}


}  // namespace


int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    int device = -1;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 8 || properties.minor != 0
            || properties.warpSize != kWarpSize) {
        std::fprintf(
            stderr,
            "expected GA100 sm_80/warp32, got sm_%d%d/warp%d\n",
            properties.major, properties.minor, properties.warpSize);
        return EXIT_FAILURE;
    }
    if (properties.multiProcessorCount <= 0
            || properties.maxThreadsPerBlock < kMaximumThreads
            || properties.sharedMemPerMultiprocessor
                < kDynamicSharedBytes + 2 * sizeof(std::uint64_t)
            || properties.sharedMemPerBlockOptin
                < kDynamicSharedBytes + 2 * sizeof(std::uint64_t)) {
        std::fprintf(
            stderr,
            "device cannot support the one-CTA-per-SM launch contract\n");
        return EXIT_FAILURE;
    }

    const int grid = (
        options.grid_blocks == 0
        ? properties.multiProcessorCount
        : options.grid_blocks);
    if (grid > properties.multiProcessorCount) {
        std::fprintf(
            stderr,
            "--grid-blocks cannot exceed physical SM count %d; got %d\n",
            properties.multiProcessorCount, grid);
        return EXIT_FAILURE;
    }
    const std::size_t words = (
        static_cast<std::size_t>(grid) * kWordsPerBlock);
    const std::size_t bytes = words * sizeof(std::uint32_t);
    std::uint32_t* input = nullptr;
    std::uint32_t* output = nullptr;
    BlockResult* device_results = nullptr;
    CHECK_CUDA(cudaMalloc(&input, bytes));
    CHECK_CUDA(cudaMalloc(&output, bytes));
    CHECK_CUDA(cudaMalloc(
        &device_results,
        static_cast<std::size_t>(grid) * sizeof(BlockResult)));
    CHECK_CUDA(cudaMemset(input, 0x5a, bytes));
    CHECK_CUDA(cudaMemset(output, 0xa5, bytes));
    std::vector<BlockResult> host_results(grid);

    const cudaUUID_t& device_uuid = properties.uuid;
    char pci_bus_id[32]{};
    char uuid_text[48]{};
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));
    std::snprintf(
        uuid_text, sizeof(uuid_text),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
        "%02x%02x%02x%02x%02x%02x",
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[0])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[1])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[2])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[3])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[4])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[5])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[6])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[7])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[8])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[9])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[10])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[11])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[12])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[13])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[14])),
        static_cast<unsigned int>(
            static_cast<unsigned char>(device_uuid.bytes[15])));

    std::printf("meta,device_index,%d\n", device);
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,device_uuid,%s\n", uuid_text);
    std::printf("meta,pci_bus_id,%s\n", pci_bus_id);
    std::printf("meta,compute_capability,sm_%d%d\n",
                properties.major, properties.minor);
    std::printf("meta,sm_count,%d\n", properties.multiProcessorCount);
    std::printf("meta,active_grid_blocks,%d\n", grid);
    std::printf("meta,producer_threads_points,128|256|512|1024\n");
    std::printf("meta,producer_warps_per_sm_points,4|8|16|32\n");
    std::printf("meta,dynamic_shared_bytes_per_cta,%d\n",
                kDynamicSharedBytes);
    std::printf("meta,target_instructions_per_warp_iteration,%d\n",
                kTargetInstructionsPerIteration);
    std::printf("meta,input_banks_per_warp,%d\n", kInputBanksPerWarp);
    std::printf("meta,sector_bytes,32\n");
    std::printf("meta,rate_unit,32B_sectors_per_SM_per_SM_cycle\n");
    std::printf(
        "meta,target_memory_semantics,ordinary_PTX_ld_global_st_global\n");
    std::printf(
        "meta,cuda_12_2_sm80_sass_memory_ops,LDG_E_STG_E_no_STRONG_scope\n");
    std::printf("meta,interpretation,achieved_lower_bound_not_peak\n");
    std::printf(
        "expected,variant,kernel,sectors_per_instruction,grid,threads,warps,"
        "iterations,"
        "target_load_warp_instructions,target_store_warp_instructions,"
        "target_load_sectors,target_store_sectors,"
        "checksum_store_warp_instructions,checksum_store_sectors,"
        "metadata_store_warp_instructions,metadata_store_sectors,"
        "counter_load_warp_instructions,counter_store_warp_instructions,"
        "counter_load_sectors,counter_store_sectors\n");
    std::printf(
        "raw,variant,access_kind,sectors_per_instruction,threads,warps,"
        "trial,grid,"
        "iterations,sum_sm_cycles,min_cta_sm_cycles,max_cta_sm_cycles,"
        "achieved_load_sectors_per_sm_cycle_lower_bound,"
        "achieved_store_sectors_per_sm_cycle_lower_bound,"
        "achieved_total_sectors_per_sm_cycle_lower_bound,lower_bound\n");

    if (options.selected != nullptr) {
        if (options.threads != 0) {
            benchmark_variant(
                *options.selected, options, grid, options.threads,
                input, output, device_results, &host_results);
        } else {
            for (int threads : kThreadPoints) {
                benchmark_variant(
                    *options.selected, options, grid, threads,
                    input, output, device_results, &host_results);
            }
        }
    } else {
        for (const Variant& variant : kVariants) {
            if (options.threads != 0) {
                benchmark_variant(
                    variant, options, grid, options.threads,
                    input, output, device_results, &host_results);
            } else {
                for (int threads : kThreadPoints) {
                    benchmark_variant(
                        variant, options, grid, threads,
                        input, output, device_results, &host_results);
                }
            }
        }
    }

    CHECK_CUDA(cudaFree(device_results));
    CHECK_CUDA(cudaFree(output));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
