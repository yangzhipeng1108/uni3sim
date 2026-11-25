/*
 * Workload-independent scalar-half DRAM read source probe (H200 owner copy).
 *
 * Owner copy of microbench/nvidia_a800/memory/lsu/scalar_half_dram_stream.cu.
 * The kernel, the load encodings, the sweep and the output are unchanged.
 * Hardware coupling changed for H200 only:
 *   - build with -arch=sm_90 (A800: sm_80);
 *   - the device check requires compute capability 9.0 and 132 SMs
 *     (A800: 8.0, 108 SMs); grids stay SM-count multiples (1x, 2x, 3x);
 *   - dynamic shared memory 48 KiB -> 64 KiB.  The mechanism fixes residency
 *     at three CTAs (twelve warps) per SM through shared memory; with the
 *     228 KiB H200 SM (1 KiB reserved per CTA) 48 KiB would admit four CTAs,
 *     while 64 KiB admits exactly three (3 x 65 <= 228 < 4 x 65), which the
 *     unchanged occupancy check enforces.
 * The comment below describes the A800 original.
 *
 * Every 128-thread CTA contains four producer warps.  Forty-eight KiB of
 * dynamic shared memory fixes residency at three CTAs (twelve warps) per SM.
 * Each lane issues two independent, coalesced U16 streams in groups of 64,
 * then consumes all values.  The first stream repeatedly reads a shared 16 KiB
 * region while the second visits a disjoint DRAM-sized region once per trial.
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>


#define CUDA_CHECK(call) do {                                               \
    cudaError_t error = (call);                                             \
    if (error != cudaSuccess) {                                             \
        std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                     cudaGetErrorString(error));                             \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)


namespace {

constexpr int kThreads = 128;
constexpr int kLoadsPerStreamPerGroup = 64;
constexpr int kCachedFirstElements = 8192;
constexpr int kDynamicSharedBytes = 64 * 1024;
constexpr int kTrials = 9;
constexpr std::size_t kBytesPerStream = 512ULL * 1024ULL * 1024ULL;


__device__ __forceinline__ std::uint16_t load_default(
        const std::uint16_t* address) {
    std::uint16_t value;
    asm volatile("ld.global.u16 %0, [%1];"
                 : "=h"(value) : "l"(address) : "memory");
    return value;
}


__device__ __forceinline__ std::uint16_t load_evict_first(
        const std::uint16_t* address) {
    std::uint16_t value;
    asm volatile("ld.global.L1::evict_first.u16 %0, [%1];"
                 : "=h"(value) : "l"(address) : "memory");
    return value;
}


__global__ __launch_bounds__(kThreads, 3) void scalar_half_dram_stream(
        const std::uint16_t* first, const std::uint16_t* second,
        std::uint32_t* checksum, std::uint32_t* sm_ids,
        std::size_t groups_per_cta, std::size_t global_element_offset) {
    extern __shared__ unsigned char residency_limit[];
    if (threadIdx.x == 0) {
        unsigned sm_id;
        asm("mov.u32 %0, %%smid;" : "=r"(sm_id));
        sm_ids[blockIdx.x] = sm_id;
        residency_limit[0] = static_cast<unsigned char>(sm_id);
    }
    __syncthreads();

    std::uint32_t sum = residency_limit[0];
    const std::size_t cta_elements = (
        groups_per_cta * kLoadsPerStreamPerGroup * kThreads);
    const std::size_t cta_base = (
        global_element_offset + blockIdx.x * cta_elements);
    const int lane = threadIdx.x % 32;
    for (std::size_t group = 0; group < groups_per_cta; ++group) {
        std::uint16_t first_values[kLoadsPerStreamPerGroup];
        std::uint16_t second_values[kLoadsPerStreamPerGroup];
        const std::size_t group_base = (
            cta_base + group * kLoadsPerStreamPerGroup * kThreads
            + threadIdx.x);
#pragma unroll
        for (int index = 0; index < kLoadsPerStreamPerGroup; ++index) {
            const std::size_t offset = group_base + index * kThreads;
            const std::size_t cached_offset = (
                (group * kLoadsPerStreamPerGroup * 32 + index * 32 + lane)
                % kCachedFirstElements);
            first_values[index] = load_default(first + cached_offset);
            second_values[index] = load_evict_first(second + offset);
        }
#pragma unroll
        for (int index = 0; index < kLoadsPerStreamPerGroup; ++index) {
            sum += static_cast<std::uint32_t>(first_values[index]);
            sum ^= static_cast<std::uint32_t>(second_values[index]);
        }
    }
    checksum[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}


__global__ void initialize(std::uint16_t* values, std::size_t count) {
    const std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    const std::size_t stride = (
        static_cast<std::size_t>(gridDim.x) * blockDim.x);
    for (std::size_t index = thread; index < count; index += stride) {
        std::uint32_t value = static_cast<std::uint32_t>(index);
        value ^= value >> 16;
        value *= 0x7feb352dU;
        value ^= value >> 15;
        values[index] = static_cast<std::uint16_t>(value);
    }
}


double run_trial(
        const std::uint16_t* first, const std::uint16_t* second,
        std::uint32_t* checksum, std::uint32_t* sm_ids,
        int grid, std::size_t groups_per_cta,
        std::size_t global_element_offset) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    scalar_half_dram_stream<<<grid, kThreads, kDynamicSharedBytes>>>(
        first, second, checksum, sm_ids, groups_per_cta,
        global_element_offset);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms);
}

}  // namespace


int main() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != 132) {
        std::fprintf(stderr, "probe requires a 132-SM compute-capability-9.0 GPU\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        scalar_half_dram_stream,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSharedBytes));
    int resident_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks, scalar_half_dram_stream,
        kThreads, kDynamicSharedBytes));
    if (resident_blocks != 3) {
        std::fprintf(stderr, "expected exactly 3 resident CTAs/SM, got %d\n",
                     resident_blocks);
        return EXIT_FAILURE;
    }

    const std::size_t elements = kBytesPerStream / sizeof(std::uint16_t);
    std::uint16_t* first = nullptr;
    std::uint16_t* second = nullptr;
    std::uint32_t* checksum = nullptr;
    std::uint32_t* sm_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&first, kBytesPerStream));
    CUDA_CHECK(cudaMalloc(&second, kBytesPerStream));
    CUDA_CHECK(cudaMalloc(
        &checksum, properties.multiProcessorCount * 3 * kThreads
        * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(
        &sm_ids, properties.multiProcessorCount * 3 * sizeof(std::uint32_t)));
    initialize<<<properties.multiProcessorCount * 4, 256>>>(first, elements);
    initialize<<<properties.multiProcessorCount * 4, 256>>>(second, elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("schema,a800_scalar_half_dram_stream_v1\n");
    std::printf("meta,device,%s\n", properties.name);
    std::printf("meta,sm_count,%d\n", properties.multiProcessorCount);
    std::printf("meta,block_threads,%d\n", kThreads);
    std::printf("meta,resident_ctas_per_sm,%d\n", resident_blocks);
    std::printf("meta,producer_warps_per_cta,%d\n", kThreads / 32);
    std::printf("meta,loads_per_stream_per_group,%d\n",
                kLoadsPerStreamPerGroup);
    std::printf("meta,cached_first_stream_bytes,%d\n",
                kCachedFirstElements * static_cast<int>(sizeof(std::uint16_t)));
    std::printf("raw,grid,groups_per_cta,trial,bytes,elapsed_ms,gb_per_s,sectors_per_active_sm_s\n");
    const std::size_t short_group_points[] = {1, 2, 3, 4, 8};
    const int resident_grid = properties.multiProcessorCount * 3;
    for (const std::size_t groups_per_cta : short_group_points) {
        const std::size_t elements_per_trial = (
            static_cast<std::size_t>(resident_grid) * kThreads
            * kLoadsPerStreamPerGroup * groups_per_cta);
        const std::size_t bytes = elements_per_trial * sizeof(std::uint16_t);
        if (elements_per_trial * kTrials > elements) {
            std::fprintf(stderr, "short-stream sweep exceeds allocation\n");
            return EXIT_FAILURE;
        }
        for (int trial = 0; trial < kTrials; ++trial) {
            const double elapsed_ms = run_trial(
                first, second, checksum, sm_ids, resident_grid,
                groups_per_cta, trial * elements_per_trial);
            const double gb_per_s = bytes / (elapsed_ms * 1.0e6);
            const double sectors_per_active_sm_s = (
                bytes / 32.0 / (elapsed_ms * 1.0e-3)
                / properties.multiProcessorCount);
            std::printf("raw,%d,%zu,%d,%zu,%.9f,%.9f,%.9f\n",
                        resident_grid, groups_per_cta, trial, bytes,
                        elapsed_ms, gb_per_s, sectors_per_active_sm_s);
        }
    }
    for (int ctas_per_sm = 1; ctas_per_sm <= 3; ++ctas_per_sm) {
        const int grid = properties.multiProcessorCount * ctas_per_sm;
        const std::size_t elements_per_group = (
            static_cast<std::size_t>(grid) * kThreads
            * kLoadsPerStreamPerGroup);
        const std::size_t groups_per_cta = elements / elements_per_group;
        const std::size_t bytes = (
            static_cast<std::size_t>(grid) * groups_per_cta * kThreads
            * kLoadsPerStreamPerGroup * sizeof(std::uint16_t));
        for (int trial = 0; trial < kTrials; ++trial) {
            const double elapsed_ms = run_trial(
                first, second, checksum, sm_ids, grid, groups_per_cta, 0);
            const double gb_per_s = bytes / (elapsed_ms * 1.0e6);
            const double sectors_per_active_sm_s = (
                bytes / 32.0 / (elapsed_ms * 1.0e-3)
                / properties.multiProcessorCount);
            std::printf("raw,%d,%zu,%d,%zu,%.9f,%.9f,%.9f\n",
                        grid, groups_per_cta, trial, bytes, elapsed_ms, gb_per_s,
                        sectors_per_active_sm_s);
        }
    }
    CUDA_CHECK(cudaFree(first));
    CUDA_CHECK(cudaFree(second));
    CUDA_CHECK(cudaFree(checksum));
    CUDA_CHECK(cudaFree(sm_ids));
    return EXIT_SUCCESS;
}
