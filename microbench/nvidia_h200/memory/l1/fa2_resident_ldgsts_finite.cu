/* NVIDIA H200 owner copy of
 * microbench/nvidia_a800/memory/l1/fa2_resident_ldgsts_finite.cu.
 * Kernel body (one 64-KiB LDGSTS panel per CTA recurrence, 32 x 16-B
 * cp.async.cg per thread, wait_group 0), 76-KiB dynamic reservation,
 * 10 warm-up launches and the per-sample launch loop are unchanged.  The
 * resident cohort stays two CTAs on every SM, so kResidentGrid = 2 x kSmCount
 * becomes 264.  Hardware coupling changes only:
 *   - kSmCount 108 -> 132 and the device contract becomes compute
 *     capability 9.0 with 132 SMs;
 *   - build for sm_90.
 * The 76-KiB dynamic reservation still admits exactly two resident 128-thread
 * CTAs per 228-KiB H200 SM (3 x (77,824 + 1,024 reserved) B > 233,472 B); the
 * probe still fails closed unless the occupancy query reports two.
 */
/*
 * Independent SM80 finite-stream LDGSTS service probe.
 *
 * This is not an attention kernel: it contains no MMA, softmax, mask,
 * accumulation, output conversion, or production workload dimensions.  It
 * isolates one 64-KiB K/V-sized LDGSTS round on each 128-thread CTA while a
 * 76-KiB dynamic shared-memory reservation fixes exactly two resident CTAs
 * per A800 SM.  The repeat count exposes fill, steady, and drain behavior
 * without consuming a production-kernel duration.
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kSmCount = 132;
constexpr int kResidentGrid = 2 * kSmCount;
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

__global__ void fa2_resident_ldgsts_finite(
        const unsigned char* input, std::uint32_t* checksum, int repeats) {
    extern __shared__ unsigned char shared[];
    const int thread_offset = threadIdx.x * kCopyBytes;
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s REPEATS SAMPLES\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int repeats = positive(argv[1], "repeats");
    const int samples = positive(argv[2], "samples");
    int device = 0;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "probe requires a 132-SM compute-capability 9.0 GPU\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        fa2_resident_ldgsts_finite,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int occupancy = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, fa2_resident_ldgsts_finite,
        kThreads, kDynamicSharedBytes));
    if (occupancy != 2) {
        std::fprintf(stderr, "expected two resident CTAs per SM, got %d\n",
                     occupancy);
        return EXIT_FAILURE;
    }
    const std::size_t input_bytes =
        static_cast<std::size_t>(kResidentGrid) * kCtaStrideBytes;
    unsigned char* input = nullptr;
    std::uint32_t* checksum = nullptr;
    CHECK_CUDA(cudaMalloc(&input, input_bytes));
    CHECK_CUDA(cudaMalloc(&checksum, kResidentGrid * sizeof(*checksum)));
    CHECK_CUDA(cudaMemset(input, 1, input_bytes));
    for (int warmup = 0; warmup < 10; ++warmup) {
        fa2_resident_ldgsts_finite<<<kResidentGrid, kThreads,
                                     kDynamicSharedBytes>>>(
            input, checksum, repeats);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<std::uint32_t> host_checksum(kResidentGrid);
    for (int sample = 0; sample < samples; ++sample) {
        fa2_resident_ldgsts_finite<<<kResidentGrid, kThreads,
                                     kDynamicSharedBytes>>>(
            input, checksum, repeats);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(
            host_checksum.data(), checksum,
            kResidentGrid * sizeof(*checksum), cudaMemcpyDeviceToHost));
        std::uint64_t total = 0;
        for (std::uint32_t value : host_checksum) {
            total += value;
        }
        std::printf("sample=%d grid=%d repeats=%d resident=%d "
                    "panel_bytes=%d checksum=%llu\n",
                    sample, kResidentGrid, repeats, occupancy, kPanelBytes,
                    static_cast<unsigned long long>(total));
    }
    CHECK_CUDA(cudaFree(checksum));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
