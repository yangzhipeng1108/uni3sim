// Isolated 256-thread SM80 CTA-barrier envelope service.
//
// The timed region contains only clock64, compiler barriers, and optionally
// __syncthreads(). Dynamic shared memory controls residency but is never read
// or written. Results are emitted only after the timed region.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr std::size_t kTotalSharedBytes = 147456;
constexpr std::size_t kStaticSharedBytes = 2 * kThreads * sizeof(std::uint64_t);
constexpr std::size_t kDynamicSharedBytes = kTotalSharedBytes - kStaticSharedBytes;

struct Result {
    std::uint64_t cycles;
    std::uint32_t sm_id;
};

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda((call), #call)

__device__ __forceinline__ std::uint32_t sm_id() {
    std::uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

__device__ __forceinline__ void compiler_barrier() {
    asm volatile("" ::: "memory");
}

template <bool WithBarrier>
__global__ __launch_bounds__(kThreads, 1)
void barrier_service_kernel(Result* results, int repeats) {
    extern __shared__ unsigned char residency_only[];
    __shared__ std::uint64_t begin_ticks[kThreads];
    __shared__ std::uint64_t end_ticks[kThreads];
    const int lane = threadIdx.x;
    // Keep the requested dynamic allocation live without adding timed traffic.
    if (residency_only == nullptr) {
        return;
    }
    __syncthreads();
    const std::uint64_t begin = clock64();
    #pragma unroll 1
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if constexpr (WithBarrier) {
            __syncthreads();
        } else {
            compiler_barrier();
        }
    }
    const std::uint64_t end = clock64();
    begin_ticks[lane] = begin;
    end_ticks[lane] = end;
    __syncthreads();
    if (lane == 0) {
        std::uint64_t first = begin_ticks[0];
        std::uint64_t last = end_ticks[0];
        for (int thread = 1; thread < kThreads; ++thread) {
            if (begin_ticks[thread] < first) {
                first = begin_ticks[thread];
            }
            if (end_ticks[thread] > last) {
                last = end_ticks[thread];
            }
        }
        results[blockIdx.x] = Result{last - first, sm_id()};
    }
}

int parse_positive(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODE GRID REPEATS SAMPLES\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int mode = parse_positive(argv[1], "MODE");
    const int grid = parse_positive(argv[2], "GRID");
    const int repeats = parse_positive(argv[3], "REPEATS");
    const int samples = parse_positive(argv[4], "SAMPLES");
    if (mode != 1 && mode != 2) {
        std::fprintf(stderr, "MODE must be one (baseline) or two (barrier)\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        barrier_service_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynamicSharedBytes)));
    CHECK_CUDA(cudaFuncSetAttribute(
        barrier_service_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynamicSharedBytes)));
    std::vector<Result> host_results(grid);
    Result* device_results = nullptr;
    CHECK_CUDA(cudaMalloc(&device_results, sizeof(Result) * grid));
    std::printf("sample,mode,grid,repeats,block,dynamic_shared_bytes,sm_id,cycles\n");
    for (int sample = 0; sample < samples; ++sample) {
        if (mode == 1) {
            barrier_service_kernel<false><<<grid, kThreads, kDynamicSharedBytes>>>(
                device_results, repeats);
        } else {
            barrier_service_kernel<true><<<grid, kThreads, kDynamicSharedBytes>>>(
                device_results, repeats);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(host_results.data(), device_results,
                              sizeof(Result) * grid,
                              cudaMemcpyDeviceToHost));
        for (int block = 0; block < grid; ++block) {
            std::printf("%d,%d,%d,%d,%d,%zu,%u,%llu\n", sample, mode, grid,
                        repeats, block, kDynamicSharedBytes,
                        host_results[block].sm_id,
                        static_cast<unsigned long long>(host_results[block].cycles));
        }
    }
    CHECK_CUDA(cudaFree(device_results));
    return EXIT_SUCCESS;
}
