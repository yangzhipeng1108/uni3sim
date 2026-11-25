/* NVIDIA H200 owner copy of
 * microbench/nvidia_a800/operators/flash_attn/fa2_queued_ldgsts_source_period.cu.
 * Kernel body (causal (1,1,2,2,...) source period of 64-KiB LDGSTS rounds,
 * %globaltimer / %smid stamps), 76-KiB dynamic reservation, 10 warm-up
 * launches and the per-sample output are unchanged.  The launch grid is a
 * command-line argument (the source program's queued CTA count, not an SM
 * multiple).  Hardware coupling changes only:
 *   - kSmCount 108 -> 132 and the device contract becomes compute
 *     capability 9.0 with 132 SMs;
 *   - build for sm_90.
 * The 76-KiB dynamic reservation still admits exactly two resident 128-thread
 * CTAs per 228-KiB H200 SM (3 x (77,824 + 1,024 reserved) B > 233,472 B); the
 * probe still fails closed unless the occupancy query reports two.
 */
/*
 * Independent heterogeneous queued-cohort LDGSTS probe for SM80.
 *
 * This is not an attention kernel: it has no MMA, softmax, mask,
 * accumulation, output conversion, or production tensor shape.  Exactly two
 * 128-thread CTAs can reside on each A800 SM.  Adjacent pairs execute one
 * through eight identical 64-KiB LDGSTS rounds.  Launches larger than the
 * 216-CTA resident cohort expose how CUDA dispatches a repeated heterogeneous
 * source period from its queued suffix; no CTA-to-SM identity is assumed.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kSmCount = 132;
constexpr int kDynamicSharedBytes = 76 * 1024;
constexpr int kCopiesPerThread = 32;
constexpr int kCopyBytes = 16;
constexpr int kCopyStrideBytes = 2048;
constexpr int kThreadStrideBytes = 65536;
constexpr int kCtaStrideBytes = 2 * kThreadStrideBytes;
constexpr int kPanelBytes = kThreads * kCopiesPerThread * kCopyBytes;

static_assert(kPanelBytes == 65536, "one round must carry one 64-KiB panel");

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda((call), #call)

__device__ __forceinline__ void cp_async_16(
        void* shared_address, const void* global_address) {
    const unsigned shared = __cvta_generic_to_shared(shared_address);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        : : "r"(shared), "l"(global_address) : "memory");
}

__device__ __forceinline__ unsigned long long global_nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ unsigned sm_id() {
    unsigned value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__global__ void fa2_queued_ldgsts_source_period(
        const unsigned char* input, std::uint32_t* checksum,
        unsigned* block_sm,
        unsigned long long* begin_ns, unsigned long long* end_ns,
        int maximum_repeats, int period_offset) {
    extern __shared__ unsigned char shared[];
    if (threadIdx.x == 0) {
        block_sm[blockIdx.x] = sm_id();
        begin_ns[blockIdx.x] = global_nanoseconds();
    }
    const int thread_offset = threadIdx.x * kCopyBytes;
    const int period_mask = 2 * maximum_repeats - 1;
    const int repeats = (((blockIdx.x + period_offset) & period_mask) >> 1) + 1;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const int panel_offset = (repeat & 1) * kThreadStrideBytes;
        const unsigned char* block_input =
            input + blockIdx.x * kCtaStrideBytes + panel_offset;
#pragma unroll
        for (int copy = 0; copy < kCopiesPerThread; ++copy) {
            const int offset = thread_offset + copy * kCopyStrideBytes;
            cp_async_16(shared + offset, block_input + offset);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        checksum[blockIdx.x] = shared[(blockIdx.x * 131) & 0xffff];
        end_ns[blockIdx.x] = global_nanoseconds();
    }
}

int positive(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 1'000'000) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int non_negative(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 1'000'000) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(
            stderr,
            "usage: %s SAMPLES GRID MAXIMUM_REPEATS PERIOD_OFFSET\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int samples = positive(argv[1], "samples");
    const int grid = positive(argv[2], "grid");
    if (grid > 4096) {
        std::fprintf(stderr, "grid must not exceed 4096 CTAs\n");
        return EXIT_FAILURE;
    }
    const int maximum_repeats = positive(argv[3], "maximum_repeats");
    if (maximum_repeats > 32
            || (maximum_repeats & (maximum_repeats - 1)) != 0) {
        std::fprintf(stderr,
                     "maximum_repeats must be a power of two at most 32\n");
        return EXIT_FAILURE;
    }
    const int period_offset = non_negative(argv[4], "period_offset");
    if (period_offset < 0 || period_offset >= 2 * maximum_repeats) {
        std::fprintf(stderr, "period_offset is outside its source period\n");
        return EXIT_FAILURE;
    }
    int device = 0;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr,
                     "probe requires a 132-SM compute-capability 9.0 GPU\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        fa2_queued_ldgsts_source_period,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int occupancy = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, fa2_queued_ldgsts_source_period,
        kThreads, kDynamicSharedBytes));
    if (occupancy != 2) {
        std::fprintf(stderr, "expected two resident CTAs per SM, got %d\n",
                     occupancy);
        return EXIT_FAILURE;
    }
    const std::size_t input_bytes =
        static_cast<std::size_t>(grid) * kCtaStrideBytes;
    unsigned char* input = nullptr;
    std::uint32_t* checksum = nullptr;
    unsigned* block_sm = nullptr;
    unsigned long long* begin_ns = nullptr;
    unsigned long long* end_ns = nullptr;
    CHECK_CUDA(cudaMalloc(&input, input_bytes));
    CHECK_CUDA(cudaMalloc(&checksum, grid * sizeof(*checksum)));
    CHECK_CUDA(cudaMalloc(&block_sm, grid * sizeof(*block_sm)));
    CHECK_CUDA(cudaMalloc(&begin_ns, grid * sizeof(*begin_ns)));
    CHECK_CUDA(cudaMalloc(&end_ns, grid * sizeof(*end_ns)));
    CHECK_CUDA(cudaMemset(input, 1, input_bytes));
    for (int warmup = 0; warmup < 10; ++warmup) {
        fa2_queued_ldgsts_source_period<<<
            grid, kThreads, kDynamicSharedBytes>>>(
                input, checksum, block_sm, begin_ns, end_ns,
                maximum_repeats, period_offset);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<std::uint32_t> host_checksum(grid);
    std::vector<unsigned> host_block_sm(grid);
    std::vector<unsigned long long> host_begin_ns(grid);
    std::vector<unsigned long long> host_end_ns(grid);
    for (int sample = 0; sample < samples; ++sample) {
        fa2_queued_ldgsts_source_period<<<
            grid, kThreads, kDynamicSharedBytes>>>(
                input, checksum, block_sm, begin_ns, end_ns,
                maximum_repeats, period_offset);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(
            host_checksum.data(), checksum,
            grid * sizeof(*checksum), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            host_block_sm.data(), block_sm,
            grid * sizeof(*block_sm), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            host_begin_ns.data(), begin_ns,
            grid * sizeof(*begin_ns), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            host_end_ns.data(), end_ns,
            grid * sizeof(*end_ns), cudaMemcpyDeviceToHost));
        std::uint64_t total = 0;
        for (std::uint32_t value : host_checksum) {
            total += value;
        }
        std::printf(
            "sample=%d grid=%d resident=%d min_repeats=1 max_repeats=%d "
            "period_ctas=%d period_offset=%d panel_bytes=%d checksum=%llu\n",
            sample, grid, occupancy, maximum_repeats,
            2 * maximum_repeats, period_offset, kPanelBytes,
            static_cast<unsigned long long>(total));
        const unsigned long long origin = *std::min_element(
            host_begin_ns.begin(), host_begin_ns.end());
        for (int block = 0; block < grid; ++block) {
            std::printf(
                "block sample=%d id=%d sm=%u repeats=%d begin_ns=%llu end_ns=%llu "
                "elapsed_ns=%llu\n",
                sample, block, host_block_sm[block],
                (((block + period_offset)
                  & (2 * maximum_repeats - 1)) >> 1) + 1,
                host_begin_ns[block] - origin,
                host_end_ns[block] - origin,
                host_end_ns[block] - host_begin_ns[block]);
        }
    }
    CHECK_CUDA(cudaFree(end_ns));
    CHECK_CUDA(cudaFree(begin_ns));
    CHECK_CUDA(cudaFree(block_sm));
    CHECK_CUDA(cudaFree(checksum));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
