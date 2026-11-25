/*
 * NVIDIA A800 two-stage LDGSTS loader-service mechanism benchmark.
 *
 * This is not an attention kernel.  It contains no tensor-core work,
 * softmax, output accumulation, page-table lookup, or production shape.  It
 * isolates the static global-to-shared transaction stream used by the pinned
 * grouped-MLA stage-1 Triton loop on SM80:
 *
 *   128 threads / CTA;
 *   a prologue copy followed by wait/consume/next-copy software pipelining;
 *   three committed streams per stage: K-nope 512 x 32 x BF16, K-rope
 *   64 x 32 x BF16, and V 32 x 512 x BF16;
 *   one compiler-reused shared panel inside the two-stage software pipeline.
 *
 * Each thread issues 18 K copies and 16 V copies of 16 bytes.  Therefore one
 * stage contains 34 dynamic LDGSTS warp instructions per warp, 136 per CTA,
 * and 69,632 payload bytes.  K and V use distinct copy streams but alias the
 * same 576-wide source record, as they do in the pinned Triton call.  NCU,
 * rather than source arithmetic, establishes the mapping from those
 * instructions to the Xbar's initial-transaction counter.  The native
 * clock64 interval includes pipeline fill and drain.
 *
 * The default source footprint is exactly two K panels per CTA (less than 8
 * MiB at 108 CTAs), and three untimed launches warm it before every measured
 * cell.  The separate reuse_profile_case mode launches the same loader without
 * a timer or warmup as a cold-cache counter probe.  It freezes the production
 * logical grid (batch/source x four HQ64/BLOCK_H16 head groups x four KV
 * splits) and sweeps only a synthetic per-split source address footprint.
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
#include <set>
#include <string>
#include <vector>


namespace {

constexpr int kExpectedSmCount = 108;
constexpr int kExpectedL2Bytes = 40 * 1024 * 1024;
constexpr int kBlockThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerCta = kBlockThreads / kWarpSize;
constexpr int kPipelineStages = 2;
constexpr int kHeadGroupCount = 4;
constexpr int kKvSplitCount = 4;
constexpr int kCopyBytesPerThread = 16;
constexpr int kQkHeadDim = 576;
constexpr int kValueHeadDim = 512;
constexpr int kKvTile = 32;
constexpr int kElementBytes = 2;
constexpr int kKPanelBytes = kQkHeadDim * kKvTile * kElementBytes;
constexpr int kVPanelBytes = kKvTile * kValueHeadDim * kElementBytes;
constexpr int kStageBytes = kKPanelBytes + kVPanelBytes;
// The pinned Triton artifact reports 89,088 B.  Its one 69,632-B async panel
// shares the allocation with Q and softmax scratch; the extra bytes are kept
// solely to reproduce the one-CTA-per-SM resource envelope.
constexpr int kDynamicSharedBytes = 89088;
// Frozen TTGIR allocation offsets. K-nope and K-rope use #shared1
// (order=[0,1]); V uses #shared (order=[1,0]). The intervening range belongs
// to other production-kernel scratch and remains untouched here.
constexpr int kKNopeSharedBase = 0;
constexpr int kVSharedBase = 32768;
constexpr int kKRopeSharedBase = 81920;
constexpr int kChecksumScratchBase = 86016;
constexpr int kTimerScratchBase = 86032;
constexpr int kSwizzleVectorElements = 8;
constexpr int kSwizzlePerPhase = 1;
constexpr int kSwizzleMaxPhase = 8;
constexpr int kKCopiesPerThread =
    kKPanelBytes / (kBlockThreads * kCopyBytesPerThread);
constexpr int kKNopeCopiesPerThread =
    kValueHeadDim * kKvTile * kElementBytes
    / (kBlockThreads * kCopyBytesPerThread);
constexpr int kKRopeCopiesPerThread =
    (kQkHeadDim - kValueHeadDim) * kKvTile * kElementBytes
    / (kBlockThreads * kCopyBytesPerThread);
constexpr int kVCopiesPerThread =
    kVPanelBytes / (kBlockThreads * kCopyBytesPerThread);
constexpr int kCopiesPerThreadPerStage =
    kKCopiesPerThread + kVCopiesPerThread;
constexpr int kLdgstsWarpInstructionsPerStage =
    kWarpsPerCta * kCopiesPerThreadPerStage;
constexpr int kWarmupLaunches = 3;
constexpr int kDefaultTrials = 7;
constexpr int kMinimumCandidateTrials = 7;
constexpr int kReuseActiveSources = 108;
constexpr int kReuseTilePoints[] = {1, 4, 8, 10, 11, 12, 16};
constexpr int kGridPoints[] = {1, 2, 8, 32, 64, 108};
constexpr int kTilePoints[] = {2, 4, 16, 64};

static_assert(kKPanelBytes == 36864, "K panel geometry changed");
static_assert(kVPanelBytes == 32768, "V panel geometry changed");
static_assert(kStageBytes == 69632, "stage payload changed");
static_assert(kDynamicSharedBytes > kStageBytes, "shared envelope is too small");
static_assert(kKCopiesPerThread == 18, "K copy count changed");
static_assert(kKNopeCopiesPerThread == 16, "K-nope copy count changed");
static_assert(kKRopeCopiesPerThread == 2, "K-rope copy count changed");
static_assert(kVCopiesPerThread == 16, "V copy count changed");
static_assert(kLdgstsWarpInstructionsPerStage == 136,
              "LDGSTS instruction geometry changed");
static_assert(kHeadGroupCount * 16 == 64,
              "BLOCK_H=16 must form four HQ64 head groups");
static_assert(kKvSplitCount == 4, "KV split count changed");
static_assert(kKNopeSharedBase + kKNopeCopiesPerThread * kBlockThreads
                  * kCopyBytesPerThread == kVSharedBase,
              "K-nope shared allocation changed");
static_assert(kVSharedBase + kVPanelBytes <= kKRopeSharedBase,
              "V shared allocation overlaps production scratch");
static_assert(kKRopeSharedBase + kKRopeCopiesPerThread * kBlockThreads
                  * kCopyBytesPerThread == kChecksumScratchBase,
              "K-rope shared allocation changed");
static_assert(kChecksumScratchBase + kWarpsPerCta * sizeof(std::uint32_t)
                  == kTimerScratchBase,
              "checksum and timer scratch layout changed");
static_assert(kTimerScratchBase + 2 * kWarpsPerCta * sizeof(std::uint64_t)
                  <= kDynamicSharedBytes,
              "validation scratch exceeds the frozen shared envelope");


// Triton's swizzled_shared encoding XORs the vector coordinate by a phase
// selected from the orthogonal coordinate. These wrappers name both logical
// orders explicitly: K/K-rope [feature,token], V [token,value].
__host__ __device__ constexpr int triton_swizzled_contiguous_element(
        int contiguous_element, int orthogonal_element) {
    return contiguous_element ^ (
        ((orthogonal_element / kSwizzlePerPhase) % kSwizzleMaxPhase)
        * kSwizzleVectorElements);
}


__host__ __device__ constexpr int triton_k_shared_byte_offset(
        int base, int feature_element, int token, int feature_extent) {
    return base + (
        token * feature_extent
        + triton_swizzled_contiguous_element(feature_element, token))
        * kElementBytes;
}


__host__ __device__ constexpr int triton_v_shared_byte_offset(
        int token, int value_element) {
    return kVSharedBase + (
        token * kValueHeadDim
        + triton_swizzled_contiguous_element(value_element, token))
        * kElementBytes;
}


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
    std::uint64_t checksum;
    std::uint64_t reserved;
};


static_assert(sizeof(BlockResult) == 32, "result must occupy one sector");


struct alignas(16) ReuseResult {
    std::uint32_t checksum;
    std::uint32_t source_index;
    std::uint32_t head_group;
    std::uint32_t kv_split;
};


static_assert(sizeof(ReuseResult) == 16, "reuse result geometry changed");


__device__ __forceinline__ void cp_async_16(
        void* shared_address, const void* global_address) {
    const unsigned shared = __cvta_generic_to_shared(shared_address);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :
        : "r"(shared), "l"(global_address)
        : "memory");
}


__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}


__device__ __forceinline__ void wait_group_zero() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}


__device__ __forceinline__ void issue_k_nope(
        unsigned char* shared_stage, const unsigned char* global_tile,
        int thread_index) {
    #pragma unroll
    for (int copy = 0; copy < kKNopeCopiesPerThread; ++copy) {
        const int linear_copy = copy * kBlockThreads + thread_index;
        const int row = linear_copy / 64;
        const int row_copy = linear_copy % 64;
        const int feature_element = row_copy * kSwizzleVectorElements;
        const int shared_offset = triton_k_shared_byte_offset(
            kKNopeSharedBase, feature_element, row, kValueHeadDim);
        const int global_offset = (
            row * kQkHeadDim * kElementBytes
            + row_copy * kCopyBytesPerThread);
        cp_async_16(
            shared_stage + shared_offset, global_tile + global_offset);
    }
    commit_group();
}


__device__ __forceinline__ void issue_k_rope(
        unsigned char* shared_stage, const unsigned char* global_tile,
        int thread_index) {
    #pragma unroll
    for (int copy = 0; copy < kKRopeCopiesPerThread; ++copy) {
        const int linear_copy = copy * kBlockThreads + thread_index;
        const int row = linear_copy / 8;
        const int row_copy = linear_copy % 8;
        const int feature_element = row_copy * kSwizzleVectorElements;
        const int shared_offset = triton_k_shared_byte_offset(
            kKRopeSharedBase, feature_element, row,
            kQkHeadDim - kValueHeadDim);
        const int global_offset = (
            row * kQkHeadDim * kElementBytes
            + kValueHeadDim * kElementBytes
            + row_copy * kCopyBytesPerThread);
        cp_async_16(
            shared_stage + shared_offset, global_tile + global_offset);
    }
    commit_group();
}


__device__ __forceinline__ void issue_v(
        unsigned char* shared_stage, const unsigned char* global_tile,
        int thread_index) {
    #pragma unroll
    for (int copy = 0; copy < kVCopiesPerThread; ++copy) {
        const int linear_copy = copy * kBlockThreads + thread_index;
        const int row = linear_copy / 64;
        const int row_copy = linear_copy % 64;
        const int value_element = row_copy * kSwizzleVectorElements;
        const int shared_offset = triton_v_shared_byte_offset(
            row, value_element);
        const int global_offset = (
            row * kQkHeadDim * kElementBytes
            + row_copy * kCopyBytesPerThread);
        cp_async_16(
            shared_stage + shared_offset, global_tile + global_offset);
    }
    commit_group();
}


__device__ __forceinline__ void issue_stage(
        unsigned char* shared_stage, const unsigned char* global_tile,
        int thread_index) {
    issue_k_nope(shared_stage, global_tile, thread_index);
    issue_k_rope(shared_stage, global_tile, thread_index);
    issue_v(shared_stage, global_tile, thread_index);
}


__device__ __forceinline__ std::uint32_t consume_stage(
        const unsigned char* shared_stage, int thread_index, int tile,
        std::uint32_t checksum) {
    const int k_linear =
        (tile % kKNopeCopiesPerThread) * kBlockThreads + thread_index;
    const int k_token = k_linear / 64;
    const int k_feature = (k_linear % 64) * kSwizzleVectorElements;
    const int rope_linear =
        (tile % kKRopeCopiesPerThread) * kBlockThreads + thread_index;
    const int rope_token = rope_linear / 8;
    const int rope_feature = (rope_linear % 8) * kSwizzleVectorElements;
    const int v_linear =
        (tile % kVCopiesPerThread) * kBlockThreads + thread_index;
    const int v_token = v_linear / 64;
    const int v_element = (v_linear % 64) * kSwizzleVectorElements;
    const volatile std::uint32_t* k_word =
        reinterpret_cast<const volatile std::uint32_t*>(shared_stage
            + triton_k_shared_byte_offset(
                kKNopeSharedBase, k_feature, k_token, kValueHeadDim));
    const volatile std::uint32_t* rope_word =
        reinterpret_cast<const volatile std::uint32_t*>(shared_stage
            + triton_k_shared_byte_offset(
                kKRopeSharedBase, rope_feature, rope_token,
                kQkHeadDim - kValueHeadDim));
    const volatile std::uint32_t* v_word =
        reinterpret_cast<const volatile std::uint32_t*>(shared_stage
            + triton_v_shared_byte_offset(v_token, v_element));
    checksum = checksum * 1664525U + *k_word;
    checksum = checksum * 1013904223U + *rope_word;
    return checksum * 22695477U + *v_word
        + static_cast<std::uint32_t>(tile + 1);
}


extern "C" __global__ __launch_bounds__(kBlockThreads, 1)
void two_stage_ldgsts_loader(
        const unsigned char* __restrict__ source,
        BlockResult* __restrict__ results, int tiles) {
    extern __shared__ unsigned char shared[];
    volatile std::uint64_t* warp_start_cycles =
        reinterpret_cast<volatile std::uint64_t*>(shared + kTimerScratchBase);
    volatile std::uint64_t* warp_end_cycles =
        warp_start_cycles + kWarpsPerCta;

    const int thread_index = threadIdx.x;
    const std::size_t cta_source_bytes =
        static_cast<std::size_t>(kPipelineStages) * kKPanelBytes;
    const unsigned char* cta_source =
        source + static_cast<std::size_t>(blockIdx.x) * cta_source_bytes;
    std::uint32_t checksum = 0;
    std::uint64_t elapsed_cycles = 0;

    // This is the source/compiler two-stage schedule: the first tile is
    // prefetched in a prologue; each loop iteration waits for that tile,
    // consumes it, and issues the following tile.  The compiler reuses one
    // shared panel because the next copy starts after the current consume.
    const std::uint64_t local_start_cycle = clock64();
    issue_stage(shared, cta_source, thread_index);
    #pragma unroll 1
    for (int tile = 0; tile < tiles; ++tile) {
        wait_group_zero();
        if (tile + 1 < tiles) {
            const int next_slot = (tile + 1) & 1;
            issue_stage(
                shared, cta_source + next_slot * kKPanelBytes,
                thread_index);
        }
    }
    const std::uint64_t local_end_cycle = clock64();
    if ((thread_index % kWarpSize) == 0) {
        const int warp = thread_index / kWarpSize;
        warp_start_cycles[warp] = local_start_cycle;
        warp_end_cycles[warp] = local_end_cycle;
    }
    // This barrier is after both clock samples and therefore outside the
    // measured interval. Thread 0 reconstructs the CTA envelope from the
    // earliest warp start and latest warp completion on the same SM clock.
    __syncthreads();
    if (thread_index == 0) {
        std::uint64_t earliest = warp_start_cycles[0];
        std::uint64_t latest = warp_end_cycles[0];
        #pragma unroll
        for (int warp = 1; warp < kWarpsPerCta; ++warp) {
            earliest = min(earliest, warp_start_cycles[warp]);
            latest = max(latest, warp_end_cycles[warp]);
        }
        elapsed_cycles = latest - earliest;
    }
    __syncthreads();

    // Validate all lanes and all three streams after the timed loader-only
    // interval. The final tile selects different vector groups as tile count
    // changes, while never adding checksum LDS/arithmetic to the service rate.
    checksum = consume_stage(shared, thread_index, tiles - 1, checksum);
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        checksum ^= __shfl_xor_sync(0xffffffffU, checksum, offset);
    }
    std::uint32_t* warp_checksums = reinterpret_cast<std::uint32_t*>(
        shared + kChecksumScratchBase);
    if ((thread_index % kWarpSize) == 0) {
        warp_checksums[thread_index / kWarpSize] = checksum;
    }
    __syncthreads();
    if (thread_index == 0) {
        checksum = 0;
        #pragma unroll
        for (int warp = 0; warp < kWarpsPerCta; ++warp) {
            checksum ^= warp_checksums[warp];
        }
    }
    __syncthreads();

    if (thread_index == 0) {
        std::uint32_t sm_id;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
        BlockResult* result = results + blockIdx.x;
        result->cycles = elapsed_cycles;
        result->sm_id = sm_id;
        result->checksum = checksum;
        result->reserved = 0;
    }
}


extern "C" __global__ __launch_bounds__(kBlockThreads, 1)
void grouped_mla_kv_reuse_loader(
        const unsigned char* __restrict__ source,
        ReuseResult* __restrict__ results, int active_sources,
        int tiles_per_split) {
    extern __shared__ unsigned char shared[];
    const int source_index = static_cast<int>(blockIdx.x);
    const int head_group = static_cast<int>(blockIdx.y);
    const int kv_split = static_cast<int>(blockIdx.z);
    const int logical_cta = (
        (kv_split * kHeadGroupCount + head_group) * active_sources
        + source_index);
    const std::size_t segment_index =
        static_cast<std::size_t>(kv_split) * active_sources + source_index;
    const unsigned char* segment_source = source
        + segment_index * tiles_per_split * kKPanelBytes;
    const int thread_index = threadIdx.x;

    // CUDA linearizes grid.x fastest.  The production-shaped 3-D launch freezes
    // only the logical index mapping
    //   split -> head-group -> batch/source,
    // while four head groups replay exactly the same (split, source) segment.
    issue_stage(shared, segment_source, thread_index);
    #pragma unroll 1
    for (int tile = 0; tile < tiles_per_split; ++tile) {
        wait_group_zero();
        if (tile + 1 < tiles_per_split) {
            issue_stage(
                shared, segment_source
                    + static_cast<std::size_t>(tile + 1) * kKPanelBytes,
                thread_index);
        }
    }

    std::uint32_t checksum = consume_stage(
        shared, thread_index, tiles_per_split - 1, 0);
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        checksum ^= __shfl_xor_sync(0xffffffffU, checksum, offset);
    }
    std::uint32_t* warp_checksums = reinterpret_cast<std::uint32_t*>(
        shared + kChecksumScratchBase);
    if ((thread_index % kWarpSize) == 0) {
        warp_checksums[thread_index / kWarpSize] = checksum;
    }
    __syncthreads();
    if (thread_index == 0) {
        checksum = 0;
        #pragma unroll
        for (int warp = 0; warp < kWarpsPerCta; ++warp) {
            checksum ^= warp_checksums[warp];
        }
        checksum ^= static_cast<std::uint32_t>(head_group + 1) * 0x9e3779b9U;
        checksum ^= static_cast<std::uint32_t>(kv_split + 1) * 0x85ebca6bU;
        results[logical_cta] = {
            checksum, static_cast<std::uint32_t>(source_index),
            static_cast<std::uint32_t>(head_group),
            static_cast<std::uint32_t>(kv_split)};
    }
}


int parse_positive_integer(const char* text, const char* label) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value <= 0 || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a positive integer: %s\n", label, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}


bool contains(const int* begin, const int* end, int value) {
    return std::find(begin, end, value) != end;
}


struct Options {
    std::string mode = "full_matrix";
    int trials = kDefaultTrials;
    int profile_grid = 0;
    int profile_tiles = 0;
    int active_sources = 0;
    int tiles_per_split = 0;
};


Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--mode") {
            if (++index == argc) {
                std::fprintf(stderr, "--mode requires a value\n");
                std::exit(EXIT_FAILURE);
            }
            options.mode = argv[index];
        } else if (argument == "--trials") {
            if (++index == argc) {
                std::fprintf(stderr, "--trials requires a value\n");
                std::exit(EXIT_FAILURE);
            }
            options.trials = parse_positive_integer(argv[index], "trials");
        } else if (argument == "--grid") {
            if (++index == argc) {
                std::fprintf(stderr, "--grid requires a value\n");
                std::exit(EXIT_FAILURE);
            }
            options.profile_grid = parse_positive_integer(argv[index], "grid");
        } else if (argument == "--tiles") {
            if (++index == argc) {
                std::fprintf(stderr, "--tiles requires a value\n");
                std::exit(EXIT_FAILURE);
            }
            options.profile_tiles = parse_positive_integer(argv[index], "tiles");
        } else if (argument == "--active-sources") {
            if (++index == argc) {
                std::fprintf(stderr, "--active-sources requires a value\n");
                std::exit(EXIT_FAILURE);
            }
            options.active_sources = parse_positive_integer(
                argv[index], "active sources");
        } else if (argument == "--tiles-per-split") {
            if (++index == argc) {
                std::fprintf(stderr, "--tiles-per-split requires a value\n");
                std::exit(EXIT_FAILURE);
            }
            options.tiles_per_split = parse_positive_integer(
                argv[index], "tiles per split");
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argument.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    if (options.mode != "full_matrix" && options.mode != "smoke_cell"
            && options.mode != "profile_cell"
            && options.mode != "reuse_profile_case") {
        std::fprintf(stderr, "unknown mode: %s\n", options.mode.c_str());
        std::exit(EXIT_FAILURE);
    }
    if (options.mode == "reuse_profile_case") {
        if (options.active_sources != kReuseActiveSources
                || !contains(
                    std::begin(kReuseTilePoints), std::end(kReuseTilePoints),
                    options.tiles_per_split)
                || options.profile_grid != 0 || options.profile_tiles != 0) {
            std::fprintf(stderr,
                         "reuse_profile_case is outside the frozen "
                         "synthetic address-footprint sweep\n");
            std::exit(EXIT_FAILURE);
        }
        options.trials = 1;
    } else if (options.mode == "profile_cell") {
        if (!contains(std::begin(kGridPoints), std::end(kGridPoints),
                      options.profile_grid)
                || !contains(std::begin(kTilePoints), std::end(kTilePoints),
                             options.profile_tiles)) {
            std::fprintf(stderr, "profile cell is outside the frozen matrix\n");
            std::exit(EXIT_FAILURE);
        }
        options.trials = 1;
    } else if (options.profile_grid != 0 || options.profile_tiles != 0
               || options.active_sources != 0
               || options.tiles_per_split != 0) {
        std::fprintf(stderr, "--grid/--tiles require --mode profile_cell\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.mode == "full_matrix"
            && options.trials < kMinimumCandidateTrials) {
        std::fprintf(stderr,
                     "full_matrix requires at least seven trials per cell\n");
        std::exit(EXIT_FAILURE);
    }
    return options;
}


void run_reuse_profile_case(const Options& options) {
    const int logical_ctas = options.active_sources
        * kHeadGroupCount * kKvSplitCount;
    const std::size_t segment_tiles =
        static_cast<std::size_t>(options.active_sources)
        * kKvSplitCount * options.tiles_per_split;
    if (logical_ctas <= 0
            || segment_tiles > std::numeric_limits<std::size_t>::max()
                / kKPanelBytes) {
        std::fprintf(stderr, "reuse profile geometry overflows allocation\n");
        std::exit(EXIT_FAILURE);
    }
    const std::size_t source_bytes = segment_tiles * kKPanelBytes;
    unsigned char* source = nullptr;
    ReuseResult* device_results = nullptr;
    CHECK_CUDA(cudaMalloc(&source, source_bytes));
    std::vector<unsigned char> host_source(source_bytes);
    std::uint32_t random_state = 0x9e3779b9U;
    for (unsigned char& value : host_source) {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 17;
        random_state ^= random_state << 5;
        value = static_cast<unsigned char>(random_state >> 24);
    }
    CHECK_CUDA(cudaMemcpy(
        source, host_source.data(), source_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(
        &device_results,
        static_cast<std::size_t>(logical_ctas) * sizeof(ReuseResult)));
    const dim3 grid(
        options.active_sources, kHeadGroupCount, kKvSplitCount);
    grouped_mla_kv_reuse_loader<<<
        grid, kBlockThreads, kDynamicSharedBytes>>>(
            source, device_results, options.active_sources,
            options.tiles_per_split);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<ReuseResult> host_results(logical_ctas);
    CHECK_CUDA(cudaMemcpy(
        host_results.data(), device_results,
        static_cast<std::size_t>(logical_ctas) * sizeof(ReuseResult),
        cudaMemcpyDeviceToHost));
    std::uint32_t aggregate = 0;
    auto source_word = [&host_source](std::size_t offset) {
        return static_cast<std::uint32_t>(host_source[offset])
            | (static_cast<std::uint32_t>(host_source[offset + 1]) << 8)
            | (static_cast<std::uint32_t>(host_source[offset + 2]) << 16)
            | (static_cast<std::uint32_t>(host_source[offset + 3]) << 24);
    };
    for (int logical_cta = 0; logical_cta < logical_ctas; ++logical_cta) {
        const int source_index = logical_cta % options.active_sources;
        const int head_group =
            (logical_cta / options.active_sources) % kHeadGroupCount;
        const int kv_split =
            logical_cta / (options.active_sources * kHeadGroupCount);
        const ReuseResult& result = host_results[logical_cta];
        const std::size_t segment_index =
            static_cast<std::size_t>(kv_split) * options.active_sources
            + source_index;
        const std::size_t tile_base = (
            segment_index * options.tiles_per_split
            + options.tiles_per_split - 1) * kKPanelBytes;
        std::uint32_t expected = 0;
        for (int thread = 0; thread < kBlockThreads; ++thread) {
            const int tile = options.tiles_per_split - 1;
            const int k_linear =
                (tile % kKNopeCopiesPerThread) * kBlockThreads + thread;
            const int rope_linear =
                (tile % kKRopeCopiesPerThread) * kBlockThreads + thread;
            const int v_linear =
                (tile % kVCopiesPerThread) * kBlockThreads + thread;
            std::uint32_t lane = source_word(
                tile_base + (k_linear / 64) * kQkHeadDim * kElementBytes
                + (k_linear % 64) * kCopyBytesPerThread);
            lane = lane * 1013904223U + source_word(
                tile_base + (rope_linear / 8) * kQkHeadDim * kElementBytes
                + kValueHeadDim * kElementBytes
                + (rope_linear % 8) * kCopyBytesPerThread);
            lane = lane * 22695477U + source_word(
                tile_base + (v_linear / 64) * kQkHeadDim * kElementBytes
                + (v_linear % 64) * kCopyBytesPerThread)
                + static_cast<std::uint32_t>(tile + 1);
            expected ^= lane;
        }
        expected ^= static_cast<std::uint32_t>(head_group + 1) * 0x9e3779b9U;
        expected ^= static_cast<std::uint32_t>(kv_split + 1) * 0x85ebca6bU;
        if (result.source_index != static_cast<std::uint32_t>(source_index)
                || result.head_group != static_cast<std::uint32_t>(head_group)
                || result.kv_split != static_cast<std::uint32_t>(kv_split)
                || result.checksum != expected) {
            std::fprintf(stderr,
                         "reuse CTA mapping or independent checksum changed\n");
            std::exit(EXIT_FAILURE);
        }
        aggregate ^= result.checksum;
    }
    std::printf("meta,schema,nvidia_a800_grouped_mla_kv_reuse_native_v1\n");
    std::printf("meta,block_h,16\n");
    std::printf("meta,block_n,%d\n", kKvTile);
    std::printf("meta,head_group_count,%d\n", kHeadGroupCount);
    std::printf("meta,num_kv_splits,%d\n", kKvSplitCount);
    std::printf("meta,logical_order,kv_split_head_group_active_source\n");
    std::printf("meta,active_sources,%d\n", options.active_sources);
    std::printf("meta,tiles_per_split,%d\n", options.tiles_per_split);
    std::printf("meta,k_nope_k_rope_source_bytes_per_tile,%d\n", kKPanelBytes);
    std::printf("meta,v_source_bytes_per_tile,%d\n", kVPanelBytes);
    std::printf("meta,requested_source_bytes_per_tile,%d\n", kStageBytes);
    std::printf("meta,unique_aliased_backing_bytes_per_tile,%d\n", kKPanelBytes);
    std::printf("meta,production_or_evaluation_duration_consumed,false\n");
    std::printf("reuse_summary,%d,%zu,%u\n", logical_ctas, source_bytes,
                aggregate);
    CHECK_CUDA(cudaFree(device_results));
    CHECK_CUDA(cudaFree(source));
}


std::uint32_t first_word(const std::vector<unsigned char>& source,
                         std::size_t offset) {
    return static_cast<std::uint32_t>(source[offset])
        | (static_cast<std::uint32_t>(source[offset + 1]) << 8)
        | (static_cast<std::uint32_t>(source[offset + 2]) << 16)
        | (static_cast<std::uint32_t>(source[offset + 3]) << 24);
}


std::uint32_t expected_checksum(
        const std::vector<unsigned char>& source, int block, int tiles) {
    const std::size_t cta_base =
        static_cast<std::size_t>(block) * kPipelineStages * kKPanelBytes;
    std::uint32_t block_checksum = 0;
    for (int thread = 0; thread < kBlockThreads; ++thread) {
        std::uint32_t checksum = 0;
        const int tile = tiles - 1;
        const std::size_t tile_base = cta_base
            + static_cast<std::size_t>(tile & 1) * kKPanelBytes;
        const int k_linear =
            (tile % kKNopeCopiesPerThread) * kBlockThreads + thread;
        const int k_token = k_linear / 64;
        const int k_chunk = k_linear % 64;
        const int rope_linear =
            (tile % kKRopeCopiesPerThread) * kBlockThreads + thread;
        const int rope_token = rope_linear / 8;
        const int rope_chunk = rope_linear % 8;
        const int v_linear =
            (tile % kVCopiesPerThread) * kBlockThreads + thread;
        const int v_token = v_linear / 64;
        const int v_chunk = v_linear % 64;
        const std::uint32_t k_word = first_word(
            source, tile_base
                + k_token * kQkHeadDim * kElementBytes
                + k_chunk * kCopyBytesPerThread);
        const std::uint32_t rope_word = first_word(
            source, tile_base
                + rope_token * kQkHeadDim * kElementBytes
                + kValueHeadDim * kElementBytes
                + rope_chunk * kCopyBytesPerThread);
        const std::uint32_t v_word = first_word(
            source, tile_base
                + v_token * kQkHeadDim * kElementBytes
                + v_chunk * kCopyBytesPerThread);
        checksum = checksum * 1664525U + k_word;
        checksum = checksum * 1013904223U + rope_word;
        checksum = checksum * 22695477U + v_word
            + static_cast<std::uint32_t>(tile + 1);
        block_checksum ^= checksum;
    }
    return block_checksum;
}


void validate_swizzle_contract() {
    std::vector<bool> occupied(kDynamicSharedBytes / kCopyBytesPerThread, false);
    auto record = [&occupied](int offset, int begin, int end) {
        if (offset < begin || offset + kCopyBytesPerThread > end
                || offset % kCopyBytesPerThread != 0
                || occupied[offset / kCopyBytesPerThread]) {
            std::fprintf(stderr, "invalid or aliased Triton shared mapping\n");
            std::exit(EXIT_FAILURE);
        }
        occupied[offset / kCopyBytesPerThread] = true;
    };
    for (int token = 0; token < kKvTile; ++token) {
        for (int feature = 0; feature < kValueHeadDim;
             feature += kSwizzleVectorElements) {
            record(triton_k_shared_byte_offset(
                       kKNopeSharedBase, feature, token, kValueHeadDim),
                   kKNopeSharedBase, kVSharedBase);
            record(triton_v_shared_byte_offset(token, feature),
                   kVSharedBase, kVSharedBase + kVPanelBytes);
        }
        for (int feature = 0; feature < kQkHeadDim - kValueHeadDim;
             feature += kSwizzleVectorElements) {
            record(triton_k_shared_byte_offset(
                       kKRopeSharedBase, feature, token,
                       kQkHeadDim - kValueHeadDim),
                   kKRopeSharedBase, kChecksumScratchBase);
        }
    }
    if (triton_k_shared_byte_offset(
            kKNopeSharedBase, 0, 1, kValueHeadDim) != 1040
            || triton_v_shared_byte_offset(1, 0) != kVSharedBase + 1040
            || triton_k_shared_byte_offset(
                kKRopeSharedBase, 0, 1,
                kQkHeadDim - kValueHeadDim) != kKRopeSharedBase + 144) {
        std::fprintf(stderr, "Triton shared swizzle anchor changed\n");
        std::exit(EXIT_FAILURE);
    }
}


void print_meta(const cudaDeviceProp& property, int resident_ctas,
                const char* uuid, const char* pci, const Options& options) {
    std::printf("meta,schema,nvidia_a800_two_stage_ldgsts_native_v3\n");
    std::printf("meta,device_index,0\n");
    std::printf("meta,device_name,%s\n", property.name);
    std::printf("meta,device_uuid,%s\n", uuid);
    std::printf("meta,pci_bus_id,%s\n", pci);
    std::printf("meta,compute_capability,sm_%d%d\n", property.major, property.minor);
    std::printf("meta,sm_count,%d\n", property.multiProcessorCount);
    std::printf("meta,l2_bytes,%d\n", property.l2CacheSize);
    std::printf("meta,block_threads,%d\n", kBlockThreads);
    std::printf("meta,resident_ctas_per_sm,%d\n", resident_ctas);
    std::printf("meta,pipeline_stages,%d\n", kPipelineStages);
    std::printf("meta,dynamic_shared_bytes_per_cta,%d\n", kDynamicSharedBytes);
    std::printf("meta,k_nope_shared_base,%d\n", kKNopeSharedBase);
    std::printf("meta,v_shared_base,%d\n", kVSharedBase);
    std::printf("meta,k_rope_shared_base,%d\n", kKRopeSharedBase);
    std::printf("meta,k_krope_shared_layout,vec8_perPhase1_maxPhase8_order01\n");
    std::printf("meta,v_shared_layout,vec8_perPhase1_maxPhase8_order10\n");
    std::printf("meta,qk_head_dim,%d\n", kQkHeadDim);
    std::printf("meta,value_head_dim,%d\n", kValueHeadDim);
    std::printf("meta,kv_tile,%d\n", kKvTile);
    std::printf("meta,element_bytes,%d\n", kElementBytes);
    std::printf("meta,k_panel_bytes,%d\n", kKPanelBytes);
    std::printf("meta,v_panel_bytes,%d\n", kVPanelBytes);
    std::printf("meta,stage_payload_bytes,%d\n", kStageBytes);
    std::printf("meta,cp_async_bytes_per_thread,%d\n", kCopyBytesPerThread);
    std::printf("meta,copies_per_thread_per_stage,%d\n", kCopiesPerThreadPerStage);
    std::printf("meta,ldgsts_warp_instructions_per_stage,%d\n",
                kLdgstsWarpInstructionsPerStage);
    std::printf("meta,warmup_launches,%d\n", kWarmupLaunches);
    std::printf("meta,grid_points,1|2|8|32|64|108\n");
    std::printf("meta,tile_points,2|4|16|64\n");
    std::printf("meta,capture_mode,%s\n", options.mode.c_str());
    std::printf("meta,timer_domain,clock64_sm_local_warp_envelope_cycles\n");
    std::printf("meta,timed_barriers_included,false\n");
    std::printf("meta,timer_start,per_warp_before_prologue_copy\n");
    std::printf("meta,timer_end,per_warp_after_final_wait_group_zero\n");
    std::printf("meta,cross_sm_clocks_compared,false\n");
    std::printf("meta,production_or_evaluation_duration_consumed,false\n");
    std::printf("meta,production_parameter_published,false\n");
}


void run_cell(const Options& options, int grid, int tiles,
              const unsigned char* source, BlockResult* device_results,
              const std::vector<unsigned char>& host_source,
              std::vector<BlockResult>* host_results) {
    for (int warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        two_stage_ldgsts_loader<<<grid, kBlockThreads, kDynamicSharedBytes>>>(
            source, device_results, tiles);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> rates;
    for (int trial = 0; trial < options.trials; ++trial) {
        two_stage_ldgsts_loader<<<grid, kBlockThreads, kDynamicSharedBytes>>>(
            source, device_results, tiles);
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
            const std::uint32_t checksum = expected_checksum(
                host_source, block, tiles);
            if (result.cycles == 0 || result.sm_id >= kExpectedSmCount
                    || result.checksum != checksum || result.reserved != 0) {
                std::fprintf(stderr, "invalid block result for grid=%d tiles=%d\n",
                             grid, tiles);
                std::exit(EXIT_FAILURE);
            }
            sm_ids.insert(result.sm_id);
            sum_cycles += result.cycles;
            minimum_cycles = std::min(minimum_cycles, result.cycles);
            maximum_cycles = std::max(maximum_cycles, result.cycles);
        }
        if (static_cast<int>(sm_ids.size()) != grid) {
            std::fprintf(stderr, "grid=%d did not occupy distinct SMs\n", grid);
            std::exit(EXIT_FAILURE);
        }
        const std::uint64_t warp_instructions =
            static_cast<std::uint64_t>(grid) * tiles
            * kLdgstsWarpInstructionsPerStage;
        const std::uint64_t payload_bytes =
            static_cast<std::uint64_t>(grid) * tiles * kStageBytes;
        // The Xbar mapping is validated independently by NCU.  This source
        // count is printed as an expectation, never accepted without it.
        const std::uint64_t expected_xbar_bytes =
            warp_instructions * 1024ULL;
        const std::uint64_t expected_transactions =
            expected_xbar_bytes / 32ULL;
        const double rate = static_cast<double>(expected_transactions)
            / static_cast<double>(sum_cycles);
        rates.push_back(rate);
        std::printf(
            "raw,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%.12f\n",
            grid, tiles, trial,
            static_cast<unsigned long long>(warp_instructions),
            static_cast<unsigned long long>(payload_bytes),
            static_cast<unsigned long long>(sum_cycles),
            static_cast<unsigned long long>(minimum_cycles),
            static_cast<unsigned long long>(maximum_cycles), rate);
    }
    std::sort(rates.begin(), rates.end());
    std::printf("summary,%d,%d,%.12f\n", grid, tiles,
                rates[rates.size() / 2]);
}


}  // namespace


int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    validate_swizzle_contract();
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    if (device != 0) {
        std::fprintf(stderr, "CUDA-visible device ordinal must be zero\n");
        return EXIT_FAILURE;
    }
    cudaDeviceProp property{};
    CHECK_CUDA(cudaGetDeviceProperties(&property, device));
    if (std::strcmp(property.name, "NVIDIA A800-SXM4-80GB") != 0
            || property.major != 8 || property.minor != 0
            || property.multiProcessorCount != kExpectedSmCount
            || property.l2CacheSize != kExpectedL2Bytes
            || property.sharedMemPerBlockOptin < kDynamicSharedBytes) {
        std::fprintf(stderr, "device does not satisfy the frozen A800 contract\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        two_stage_ldgsts_loader,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    CHECK_CUDA(cudaFuncSetAttribute(
        grouped_mla_kv_reuse_loader,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int resident_ctas = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, two_stage_ldgsts_loader, kBlockThreads,
        kDynamicSharedBytes));
    if (resident_ctas != 1) {
        std::fprintf(stderr, "Triton shared envelope must enforce one CTA per SM\n");
        return EXIT_FAILURE;
    }
    int reuse_resident_ctas = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &reuse_resident_ctas, grouped_mla_kv_reuse_loader, kBlockThreads,
        kDynamicSharedBytes));
    if (reuse_resident_ctas != 1) {
        std::fprintf(stderr, "reuse probe must preserve one CTA per SM\n");
        return EXIT_FAILURE;
    }

    if (options.mode == "reuse_profile_case") {
        run_reuse_profile_case(options);
        return EXIT_SUCCESS;
    }

    char pci[32];
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
    char uuid[80];
    std::snprintf(
        uuid, sizeof(uuid),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
        "%02x%02x%02x%02x%02x%02x",
        static_cast<unsigned char>(property.uuid.bytes[0]),
        static_cast<unsigned char>(property.uuid.bytes[1]),
        static_cast<unsigned char>(property.uuid.bytes[2]),
        static_cast<unsigned char>(property.uuid.bytes[3]),
        static_cast<unsigned char>(property.uuid.bytes[4]),
        static_cast<unsigned char>(property.uuid.bytes[5]),
        static_cast<unsigned char>(property.uuid.bytes[6]),
        static_cast<unsigned char>(property.uuid.bytes[7]),
        static_cast<unsigned char>(property.uuid.bytes[8]),
        static_cast<unsigned char>(property.uuid.bytes[9]),
        static_cast<unsigned char>(property.uuid.bytes[10]),
        static_cast<unsigned char>(property.uuid.bytes[11]),
        static_cast<unsigned char>(property.uuid.bytes[12]),
        static_cast<unsigned char>(property.uuid.bytes[13]),
        static_cast<unsigned char>(property.uuid.bytes[14]),
        static_cast<unsigned char>(property.uuid.bytes[15]));
    print_meta(property, resident_ctas, uuid, pci, options);
    std::printf(
        "expected,grid,tiles,ldgsts_warp_instructions,payload_bytes,"
        "expected_xbar_bytes,expected_xbar_32b_transactions\n");
    std::printf(
        "raw,grid,tiles,trial,ldgsts_warp_instructions,payload_bytes,"
        "sum_cta_sm_cycles,min_cta_sm_cycles,max_cta_sm_cycles,"
        "xbar_32b_transactions_per_active_sm_cycle\n");

    int maximum_grid = 1;
    if (options.mode == "full_matrix") {
        maximum_grid = kGridPoints[sizeof(kGridPoints) / sizeof(kGridPoints[0]) - 1];
    } else if (options.mode == "smoke_cell") {
        maximum_grid = 1;
    } else {
        maximum_grid = options.profile_grid;
    }
    const std::size_t source_bytes =
        static_cast<std::size_t>(maximum_grid) * kPipelineStages * kKPanelBytes;
    unsigned char* source = nullptr;
    BlockResult* device_results = nullptr;
    CHECK_CUDA(cudaMalloc(&source, source_bytes));
    std::vector<unsigned char> host_source(source_bytes);
    std::uint32_t random_state = 0x9e3779b9U;
    for (unsigned char& value : host_source) {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 17;
        random_state ^= random_state << 5;
        value = static_cast<unsigned char>(random_state >> 24);
    }
    CHECK_CUDA(cudaMemcpy(
        source, host_source.data(), source_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(
        &device_results,
        static_cast<std::size_t>(maximum_grid) * sizeof(BlockResult)));
    std::vector<BlockResult> host_results(maximum_grid);

    auto print_expected = [](int grid, int tiles) {
        const std::uint64_t instructions =
            static_cast<std::uint64_t>(grid) * tiles
            * kLdgstsWarpInstructionsPerStage;
        const std::uint64_t payload =
            static_cast<std::uint64_t>(grid) * tiles * kStageBytes;
        const std::uint64_t xbar = instructions * 1024ULL;
        std::printf("expected,%d,%d,%llu,%llu,%llu,%llu\n",
                    grid, tiles,
                    static_cast<unsigned long long>(instructions),
                    static_cast<unsigned long long>(payload),
                    static_cast<unsigned long long>(xbar),
                    static_cast<unsigned long long>(xbar / 32ULL));
    };

    if (options.mode == "full_matrix") {
        for (int grid : kGridPoints) {
            for (int tiles : kTilePoints) {
                print_expected(grid, tiles);
                run_cell(
                    options, grid, tiles, source, device_results,
                    host_source, &host_results);
            }
        }
    } else if (options.mode == "smoke_cell") {
        print_expected(1, 4);
        run_cell(
            options, 1, 4, source, device_results,
            host_source, &host_results);
    } else {
        print_expected(options.profile_grid, options.profile_tiles);
        run_cell(options, options.profile_grid, options.profile_tiles,
                 source, device_results, host_source, &host_results);
    }

    CHECK_CUDA(cudaFree(device_results));
    CHECK_CUDA(cudaFree(source));
    return EXIT_SUCCESS;
}
