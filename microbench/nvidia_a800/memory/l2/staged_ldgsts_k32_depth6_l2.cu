/* Independent SM80 K32 depth-6 staged LDGSTS L2 service probe.
 *
 * This is not a GEMM.  Two CTAs per SM copy a circular 32-MiB backing set
 * through six 12-KiB cp.async groups.  The 74,752-byte shared allocation
 * matches the operand-buffer footprint of the source-visible block-128
 * 64x128x32 depth-6 GEMM program without reproducing its matrix workload.
 */
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CHECK_CUDA(call) do {                                                \
    const cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

constexpr int kThreads = 128;
constexpr int kGrid = 216;
constexpr int kRounds = 4096;
constexpr int kPanelBytes = 12 * 1024;
constexpr int kStages = 6;
constexpr int kSharedBytes = 74'752;
constexpr int kBackingBytes = 32 * 1024 * 1024;
constexpr int kBackingPanels = kBackingBytes / kPanelBytes;

int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < std::numeric_limits<int>::min()
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

__device__ __forceinline__ void cp_async_16(
        void* shared_destination, const void* global_source) {
    const std::uint32_t shared_address = static_cast<std::uint32_t>(
        __cvta_generic_to_shared(shared_destination));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(shared_address), "l"(global_source) : "memory");
}

__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

__device__ __forceinline__ void wait_group_5() {
    asm volatile("cp.async.wait_group 5;\n" :: : "memory");
}

__device__ __forceinline__ void wait_all() {
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
}

__device__ __forceinline__ void issue_panel(
        std::uint8_t* destination, const std::uint8_t* source) {
    constexpr int kVectors = kPanelBytes / 16;
    for (int vector = threadIdx.x; vector < kVectors;
         vector += blockDim.x) {
        cp_async_16(destination + vector * 16, source + vector * 16);
    }
    commit_group();
}

__global__ __launch_bounds__(kThreads, 2)
void staged_ldgsts_k32_depth6_l2(
        const std::uint8_t* __restrict__ input,
        std::uint32_t* __restrict__ checksums) {
    extern __shared__ __align__(16) std::uint8_t shared[];
    auto panel = [&](int round) {
        const std::size_t index = (
            static_cast<std::size_t>(blockIdx.x) * kRounds + round)
            % kBackingPanels;
        return input + index * kPanelBytes;
    };
    for (int round = 0; round < kStages; ++round) {
        issue_panel(shared + round * kPanelBytes, panel(round));
    }
    std::uint32_t checksum = 0;
    for (int round = 0; round < kRounds; ++round) {
        wait_group_5();
        __syncthreads();
        const int slot = round % kStages;
        checksum ^= reinterpret_cast<volatile std::uint32_t*>(
            shared + slot * kPanelBytes)[threadIdx.x];
        __syncthreads();
        const int next = round + kStages;
        if (next < kRounds) {
            issue_panel(shared + slot * kPanelBytes, panel(next));
        }
    }
    wait_all();
    if (threadIdx.x == 0) {
        checksums[blockIdx.x] = checksum;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = parse_int(argv[1], "device");
    const int trials = parse_int(argv[2], "trials");
    if (device < 0 || trials < 3) {
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaFuncSetAttribute(
        staged_ldgsts_k32_depth6_l2,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes));
    int resident = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, staged_ldgsts_k32_depth6_l2, kThreads, kSharedBytes));
    if (resident != 2) {
        std::fprintf(stderr, "expected two resident CTAs per SM, got %d\n", resident);
        return EXIT_FAILURE;
    }
    std::uint8_t* input = nullptr;
    std::uint32_t* checksums = nullptr;
    CHECK_CUDA(cudaMalloc(&input, kBackingBytes));
    CHECK_CUDA(cudaMalloc(&checksums, kGrid * sizeof(*checksums)));
    CHECK_CUDA(cudaMemset(input, 0x5a, kBackingBytes));
    CHECK_CUDA(cudaMemset(checksums, 0, kGrid * sizeof(*checksums)));
    for (int warmup = 0; warmup < 3; ++warmup) {
        staged_ldgsts_k32_depth6_l2<<<kGrid, kThreads, kSharedBytes>>>(
            input, checksums);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::printf("trial,duration_ms,logical_bytes,resident_ctas_per_sm\n");
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        staged_ldgsts_k32_depth6_l2<<<kGrid, kThreads, kSharedBytes>>>(
            input, checksums);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float duration_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start, stop));
        const long long logical_bytes = (
            static_cast<long long>(kGrid) * kRounds * kPanelBytes);
        std::printf("%d,%.9f,%lld,%d\n", trial, duration_ms,
                    logical_bytes, resident);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    CHECK_CUDA(cudaFree(checksums));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
