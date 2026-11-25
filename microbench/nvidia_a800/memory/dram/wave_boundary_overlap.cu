/* A800 launch-time growth across a CTA wave boundary.
 *
 * A launch whose workgroup count is not a multiple of the resident population
 * ends in a partial wave.  How much that partial wave costs depends on how far
 * the hardware overlaps it with the wave before it: CTAs are dispatched as slots
 * free, so waves are not lockstep, but they are not free either.  This probe
 * measures the overlap directly, as the growth of total launch time against CTA
 * count across a wave boundary, at a controlled ratio of per-CTA streaming to
 * per-CTA independent compute.
 *
 * Lockstep waves would double the time at the first CTA past a boundary and stay
 * flat until the next one.  Perfect overlap would grow by one CTA's share
 * throughout.  The measured shape between those two is the mechanism.
 *
 * Each CTA reads its own contiguous slice with 16-byte loads and issues
 * PROBE_FMA independent fused multiply-adds per loaded vector on register
 * values, so the compute neither depends on the load nor is removable.  The
 * probe contains no operator, no matrix instruction, no shared-memory staging
 * and no production shape, and reports only a duration at a CTA count.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -DPROBE_FMA=64 -DPROBE_RESIDENT=1 \
 *        -o wave_boundary_overlap wave_boundary_overlap.cu
 */
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call) do {                                                \
    const cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

#ifndef PROBE_FMA
#define PROBE_FMA 64
#endif
#ifndef PROBE_RESIDENT
#define PROBE_RESIDENT 1
#endif
constexpr int kThreads = 256;
constexpr int kFma = PROBE_FMA;
constexpr int kResident = PROBE_RESIDENT;
// 164 KiB of shared memory per SM: more than half admits one CTA, a quarter
// admits two, and so on, which pins the resident population the wave boundary
// is measured against.
constexpr int kResidencyShmemBytes =
    (kResident == 1) ? (100 * 1024)
    : ((kResident == 2) ? (56 * 1024) : ((kResident == 4) ? (28 * 1024) : 0));
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

__device__ __forceinline__ uint4 load_cg(const uint4* address) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address));
    return value;
}

__global__ __launch_bounds__(kThreads)
void streaming_compute_cta(const uint4* __restrict__ input,
                           long long vectors_per_cta,
                           std::uint32_t* __restrict__ sink) {
    extern __shared__ std::uint8_t residency_block[];
    (void)residency_block;
    const long long base =
        static_cast<long long>(blockIdx.x) * vectors_per_cta;
    unsigned int checksum = 0;
    float accumulator = 1.0f + static_cast<float>(threadIdx.x);
    float step = 1.0000001f;
    for (long long offset = threadIdx.x; offset < vectors_per_cta;
            offset += kThreads) {
        const uint4 value = load_cg(input + base + offset);
        checksum ^= value.x ^ value.y ^ value.z ^ value.w;
#pragma unroll
        for (int operation = 0; operation < kFma; ++operation) {
            accumulator = __fmaf_rn(accumulator, step, 1.0e-7f);
        }
    }
    if (checksum == 0xffffffffu && accumulator == 0.0f) {
        sink[blockIdx.x] = checksum;
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s DEVICE TRIALS GRID KIB_PER_CTA\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::atoi(argv[1]);
    const int trials = std::atoi(argv[2]);
    const int grid = std::atoi(argv[3]);
    const long long kib = std::atoll(argv[4]);
    if (device < 0 || trials < 3 || trials > 15 || grid <= 0 || kib <= 0) {
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::strcmp(properties.name, kExpectedDevice) != 0
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108) {
        std::fprintf(stderr, "unexpected A800 device contract\n");
        return EXIT_FAILURE;
    }
    if (kResidencyShmemBytes > 0) {
        CHECK_CUDA(cudaFuncSetAttribute(
            streaming_compute_cta,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kResidencyShmemBytes));
    }
    int resident = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, streaming_compute_cta, kThreads, kResidencyShmemBytes));
    if (resident != kResident) {
        std::fprintf(stderr, "expected %d resident CTAs per SM, got %d\n",
                     kResident, resident);
        return EXIT_FAILURE;
    }

    const long long bytes_per_cta = kib * 1024;
    const long long vectors_per_cta = bytes_per_cta / 16;
    uint4* input = nullptr;
    std::uint32_t* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&input, bytes_per_cta * grid));
    CHECK_CUDA(cudaMalloc(&sink, grid * sizeof(*sink)));
    CHECK_CUDA(cudaMemset(input, 0x1b, bytes_per_cta * grid));
    CHECK_CUDA(cudaMemset(sink, 0, grid * sizeof(*sink)));
    for (int warmup = 0; warmup < 3; ++warmup) {
        streaming_compute_cta<<<grid, kThreads, kResidencyShmemBytes>>>(
            input, vectors_per_cta, sink);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::printf("grid,threads,fma_per_vector,resident_ctas_per_sm,kib_per_cta,"
                "trial,duration_ms\n");
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        streaming_compute_cta<<<grid, kThreads, kResidencyShmemBytes>>>(
            input, vectors_per_cta, sink);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        std::printf("%d,%d,%d,%d,%lld,%d,%.9f\n",
                    grid, kThreads, kFma, kResident, kib, trial, milliseconds);
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
    }
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
