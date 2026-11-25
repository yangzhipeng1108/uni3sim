/* NVIDIA H200 owner copy of
 * microbench/nvidia_a800/operators/flash_attn/fa2_chunked_one_resident_ingress.cu.
 * Kernel body (static skew / uniform Q8192-S4096 source-profile CTA order,
 * 64-KiB K/V LDGSTS rounds, %globaltimer makespan), wrapper-selected dynamic
 * reservation, 10 warm-up launches and the 2048/4096 source grids are
 * unchanged.  Hardware coupling changes only:
 *   - kSmCount 108 -> 132 and the device contract becomes compute
 *     capability 9.0 with 132 SMs;
 *   - build for sm_90.
 * Residency is still selected by the wrapper and checked at runtime.
 */
/*
 * Independent SM80 resource-selected K/V ingress probe for FA2 chunked
 * prefill.  It deliberately contains no Q operand, MMA, score, softmax,
 * mask, output conversion, or production Attention duration.  The sole
 * timed program is the source-selected 64-KiB K/V LDGSTS round.
 *
 * A wrapper fixes a static shared-memory resource class and its resulting
 * occupancy before compilation; it must not infer a source rate from a
 * different resident-CTA topology.  The two profiles
 * encode only static source CTA order and physical early exits for the skew
 * and uniform Q8192/S4096 source periods; no profiler timing is consumed.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef FA2_INGRESS_DYNAMIC_SHARED_BYTES
#define FA2_INGRESS_DYNAMIC_SHARED_BYTES (163 * 1024)
#endif

#ifndef FA2_INGRESS_EXPECTED_RESIDENT_CTAS
#define FA2_INGRESS_EXPECTED_RESIDENT_CTAS 1
#endif

#ifndef FA2_INGRESS_SHARED_BUFFER_COUNT
#define FA2_INGRESS_SHARED_BUFFER_COUNT 2
#endif

namespace {

constexpr int kThreads = 128;
constexpr int kSmCount = 132;
constexpr int kDynamicSharedBytes = FA2_INGRESS_DYNAMIC_SHARED_BYTES;
constexpr int kCopiesPerThread = 32;
constexpr int kCopyBytes = 16;
constexpr int kCopyStrideBytes = 2048;
constexpr int kPanelBytes = kThreads * kCopiesPerThread * kCopyBytes;
constexpr int kBufferBytes = 64 * 1024;
constexpr int kPeriodCtas = 128;

static_assert(kPanelBytes == kBufferBytes, "one round is one 64-KiB panel");
static_assert(FA2_INGRESS_SHARED_BUFFER_COUNT >= 1,
              "the ingress probe needs at least one shared buffer");
static_assert(kDynamicSharedBytes >=
                  FA2_INGRESS_SHARED_BUFFER_COUNT * kBufferBytes,
              "dynamic shared storage cannot hold the selected ingress buffers");

enum class Profile : int {
    kSkewQ1024Q4096 = 1,
    kUniformQ256To1019 = 2,
};

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

__host__ __device__ __forceinline__ int ceil_div_128(int value) {
    return (value + 127) / 128;
}

__host__ __device__ __forceinline__ int minimum(int lhs, int rhs) {
    return lhs < rhs ? lhs : rhs;
}

__host__ __device__ __forceinline__ int profile_repeats(
        Profile profile, int block) {
    const int within_period = block % kPeriodCtas;
    if (profile == Profile::kSkewQ1024Q4096) {
        const int query_tile = within_period % 64;
        const int query_rows = within_period / 64 == 0 ? 1024 : 4096;
        if (query_tile * 64 >= query_rows) {
            return 0;
        }
        return ceil_div_128(minimum((query_tile + 1) * 64, query_rows));
    }
    if (profile == Profile::kUniformQ256To1019) {
        const int query_tile = within_period % 16;
        const int request = within_period / 16;
        const int query_rows = 256 + 109 * request;
        const int key_value_rows = 1024 + 438 * request;
        if (query_tile * 64 >= query_rows) {
            return 0;
        }
        const int end = minimum((query_tile + 1) * 64, query_rows);
        return ceil_div_128(key_value_rows - query_rows + end);
    }
    return -1;
}

__global__ void fa2_chunked_one_resident_ingress(
        const unsigned char* input, std::uint32_t* checksum,
        unsigned* block_sm, unsigned long long* begin_ns,
        unsigned long long* end_ns, Profile profile) {
    extern __shared__ unsigned char shared[];
    if (threadIdx.x == 0) {
        block_sm[blockIdx.x] = sm_id();
        begin_ns[blockIdx.x] = global_nanoseconds();
    }
    const int repeats = profile_repeats(profile, blockIdx.x);
    if (repeats < 0 || repeats > 32) {
        asm volatile("trap;");
    }
    const int thread_offset = threadIdx.x * kCopyBytes;
    const unsigned char* block_input = input + blockIdx.x * 2 * kBufferBytes;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const int global_panel_offset = (repeat & 1) * kBufferBytes;
        const int shared_panel_offset =
            (repeat % FA2_INGRESS_SHARED_BUFFER_COUNT) * kBufferBytes;
#pragma unroll
        for (int copy = 0; copy < kCopiesPerThread; ++copy) {
            const int offset = thread_offset + copy * kCopyStrideBytes;
            cp_async_16(shared + shared_panel_offset + offset,
                        block_input + global_panel_offset + offset);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    // Every participating thread consumes its own cp.async destination and
    // writes it back.  This prevents the compiler from eliding the 32 source
    // LDGSTS instructions while preserving an ingress-only program.
    const int checksum_index = blockIdx.x * kThreads + threadIdx.x;
    if (repeats > 0) {
        const int panel_offset =
            ((repeats - 1) % FA2_INGRESS_SHARED_BUFFER_COUNT) * kBufferBytes;
        checksum[checksum_index] = shared[panel_offset + thread_offset];
    } else {
        checksum[checksum_index] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
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

Profile profile_from_text(const char* text) {
    const int value = positive(text, "profile");
    if (value == static_cast<int>(Profile::kSkewQ1024Q4096)) {
        return Profile::kSkewQ1024Q4096;
    }
    if (value == static_cast<int>(Profile::kUniformQ256To1019)) {
        return Profile::kUniformQ256To1019;
    }
    std::fprintf(stderr, "profile must be 1 (skew) or 2 (uniform)\n");
    std::exit(EXIT_FAILURE);
}

const char* profile_name(Profile profile) {
    if (profile == Profile::kSkewQ1024Q4096) {
        return "skew_q1024_q4096";
    }
    if (profile == Profile::kUniformQ256To1019) {
        return "uniform_q256_to_q1019";
    }
    std::abort();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr,
                     "usage: %s SAMPLES GRID PROFILE [OUTPUT]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int samples = positive(argv[1], "samples");
    const int grid = positive(argv[2], "grid");
    const Profile profile = profile_from_text(argv[3]);
    FILE* output = stdout;
    if (argc == 5) {
        output = std::fopen(argv[4], "w");
        if (output == nullptr) {
            std::perror("fopen output");
            return EXIT_FAILURE;
        }
    }
    if ((grid != 2048 && grid != 4096) || grid % kPeriodCtas != 0) {
        std::fprintf(stderr, "grid must be 2048 or 4096 whole source CTAs\n");
        return EXIT_FAILURE;
    }
    cudaDeviceProp properties{};
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "probe requires a 132-SM compute-capability 9.0 GPU\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        fa2_chunked_one_resident_ingress,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedBytes));
    int occupancy = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, fa2_chunked_one_resident_ingress,
        kThreads, kDynamicSharedBytes));
    if (occupancy != FA2_INGRESS_EXPECTED_RESIDENT_CTAS) {
        std::fprintf(stderr,
                     "expected %d resident CTAs per SM, got %d\n",
                     FA2_INGRESS_EXPECTED_RESIDENT_CTAS, occupancy);
        return EXIT_FAILURE;
    }
    const std::size_t input_bytes = static_cast<std::size_t>(grid)
        * 2 * kBufferBytes;
    unsigned char* input = nullptr;
    std::uint32_t* checksum = nullptr;
    unsigned* block_sm = nullptr;
    unsigned long long* begin_ns = nullptr;
    unsigned long long* end_ns = nullptr;
    CHECK_CUDA(cudaMalloc(&input, input_bytes));
    CHECK_CUDA(cudaMalloc(&checksum, grid * kThreads * sizeof(*checksum)));
    CHECK_CUDA(cudaMalloc(&block_sm, grid * sizeof(*block_sm)));
    CHECK_CUDA(cudaMalloc(&begin_ns, grid * sizeof(*begin_ns)));
    CHECK_CUDA(cudaMalloc(&end_ns, grid * sizeof(*end_ns)));
    CHECK_CUDA(cudaMemset(input, 1, input_bytes));
    for (int warmup = 0; warmup < 10; ++warmup) {
        fa2_chunked_one_resident_ingress<<<grid, kThreads, kDynamicSharedBytes>>>(
            input, checksum, block_sm, begin_ns, end_ns, profile);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<std::uint32_t> host_checksum(grid * kThreads);
    std::vector<unsigned> host_block_sm(grid);
    std::vector<unsigned long long> host_begin_ns(grid);
    std::vector<unsigned long long> host_end_ns(grid);
    for (int sample = 0; sample < samples; ++sample) {
        fa2_chunked_one_resident_ingress<<<grid, kThreads, kDynamicSharedBytes>>>(
            input, checksum, block_sm, begin_ns, end_ns, profile);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(host_checksum.data(), checksum,
            grid * kThreads * sizeof(*checksum), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_block_sm.data(), block_sm,
            grid * sizeof(*block_sm), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_begin_ns.data(), begin_ns,
            grid * sizeof(*begin_ns), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_end_ns.data(), end_ns,
            grid * sizeof(*end_ns), cudaMemcpyDeviceToHost));
        std::uint64_t checksum_sum = 0;
        std::uint64_t total_rounds = 0;
        for (std::uint32_t value : host_checksum) {
            checksum_sum += value;
        }
        for (int block = 0; block < grid; ++block) {
            total_rounds += profile_repeats(profile, block);
        }
        const unsigned long long origin = *std::min_element(
            host_begin_ns.begin(), host_begin_ns.end());
        const unsigned long long completion = *std::max_element(
            host_end_ns.begin(), host_end_ns.end());
        std::fprintf(
            output,
            "sample=%d profile=%s grid=%d resident=%d source_shared_bytes=%d "
            "panel_bytes=%d total_rounds=%llu checksum=%llu makespan_ns=%llu\n",
            sample, profile_name(profile), grid, occupancy,
            kDynamicSharedBytes, kPanelBytes,
            static_cast<unsigned long long>(total_rounds),
            static_cast<unsigned long long>(checksum_sum), completion - origin);
    }
    CHECK_CUDA(cudaFree(end_ns));
    CHECK_CUDA(cudaFree(begin_ns));
    CHECK_CUDA(cudaFree(block_sm));
    CHECK_CUDA(cudaFree(checksum));
    CHECK_CUDA(cudaFree(input));
    if (output != stdout) {
        std::fclose(output);
    }
    return EXIT_SUCCESS;
}
