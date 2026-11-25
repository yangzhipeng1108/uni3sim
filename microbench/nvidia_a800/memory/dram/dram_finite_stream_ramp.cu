/* A800 finite DRAM read-stream bandwidth ramp.
 *
 * A finite DRAM read stream does not reach its asymptotic rate immediately: the
 * achieved bandwidth rises with how many bytes each CTA streams.  This probe
 * measures that ramp directly -- achieved aggregate GB/s against per-CTA stream
 * length, at a fixed CTA grid and a controlled number of outstanding requests
 * per CTA.  It is a bandwidth mechanism, contains no operator, no matrix
 * instruction, no shared-memory staging and no production shape, and reports
 * only bytes per second at a stream length.
 *
 * Each CTA reads its own contiguous slice with ld.global.cg 16-byte vector
 * loads, issuing PROBE_BATCH of them before consuming any, so the bytes in
 * flight per CTA are PROBE_THREADS * PROBE_BATCH * 16 and can be set to match
 * the request depth of the source class whose stream is being described.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -DPROBE_GRID=64 -DPROBE_BATCH=32 \
 *        -o dram_finite_stream_ramp dram_finite_stream_ramp.cu
 */
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK_CUDA(call) do {                                                \
    const cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

#ifndef PROBE_GRID
#define PROBE_GRID 64
#endif
#ifndef PROBE_BATCH
#define PROBE_BATCH 32
#endif
constexpr int kThreads = 128;
constexpr int kGrid = PROBE_GRID;
constexpr int kBatch = PROBE_BATCH;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

__device__ __forceinline__ uint4 load_cg(const uint4* address) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address));
    return value;
}

/* One finite read stream per CTA over its own contiguous slice.  The batch of
 * loads is issued before any of it is consumed, so the in-flight bytes per CTA
 * stay at kThreads * kBatch * 16 for the whole stream. */
__global__ __launch_bounds__(kThreads, 1)
void finite_read_stream(const uint4* __restrict__ input,
                        long long vectors_per_cta,
                        std::uint32_t* __restrict__ checksums) {
    const long long base =
        static_cast<long long>(blockIdx.x) * vectors_per_cta;
    unsigned int checksum = 0;
    const long long stride = static_cast<long long>(kThreads) * kBatch;
    for (long long offset = 0; offset < vectors_per_cta; offset += stride) {
        uint4 batch[kBatch];
#pragma unroll
        for (int slot = 0; slot < kBatch; ++slot) {
            const long long index =
                offset + static_cast<long long>(slot) * kThreads + threadIdx.x;
            batch[slot] = load_cg(
                input + base + (index < vectors_per_cta ? index : 0));
        }
#pragma unroll
        for (int slot = 0; slot < kBatch; ++slot) {
            checksum ^= batch[slot].x ^ batch[slot].y
                        ^ batch[slot].z ^ batch[slot].w;
        }
    }
    if (threadIdx.x == 0) checksums[blockIdx.x] = checksum;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s DEVICE TRIALS KIB_PER_CTA_CSV\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::atoi(argv[1]);
    const int trials = std::atoi(argv[2]);
    if (device < 0 || trials < 3 || trials > 15) return EXIT_FAILURE;
    std::vector<long long> lengths_kib;
    for (const char* cursor = argv[3]; *cursor != '\0';) {
        char* end = nullptr;
        const long long value = std::strtoll(cursor, &end, 10);
        if (end == cursor || value <= 0) {
            std::fprintf(stderr, "stream lengths must be positive KiB\n");
            return EXIT_FAILURE;
        }
        lengths_kib.push_back(value);
        cursor = (*end == ',') ? end + 1 : end;
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

    const long long maximum_bytes =
        static_cast<long long>(kGrid) * lengths_kib.back() * 1024;
    uint4* input = nullptr;
    std::uint32_t* checksums = nullptr;
    CHECK_CUDA(cudaMalloc(&input, maximum_bytes));
    CHECK_CUDA(cudaMalloc(&checksums, kGrid * sizeof(*checksums)));
    CHECK_CUDA(cudaMemset(input, 0x3c, maximum_bytes));
    CHECK_CUDA(cudaMemset(checksums, 0, kGrid * sizeof(*checksums)));

    std::printf("grid,threads,batch,in_flight_bytes_per_cta,kib_per_cta,"
                "trial,duration_ms,logical_bytes\n");
    for (const long long kib : lengths_kib) {
        const long long bytes_per_cta = kib * 1024;
        const long long vectors_per_cta = bytes_per_cta / 16;
        // Warm the launch path without leaving the slice resident: the stream
        // is far larger than L2 at every measured length.
        for (int warmup = 0; warmup < 3; ++warmup) {
            finite_read_stream<<<kGrid, kThreads>>>(
                input, vectors_per_cta, checksums);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        for (int trial = 0; trial < trials; ++trial) {
            cudaEvent_t start;
            cudaEvent_t stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaEventRecord(start));
            finite_read_stream<<<kGrid, kThreads>>>(
                input, vectors_per_cta, checksums);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float milliseconds = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
            std::printf("%d,%d,%d,%lld,%lld,%d,%.9f,%lld\n",
                        kGrid, kThreads, kBatch,
                        static_cast<long long>(kThreads) * kBatch * 16, kib,
                        trial, milliseconds,
                        static_cast<long long>(kGrid) * bytes_per_cta);
            CHECK_CUDA(cudaEventDestroy(stop));
            CHECK_CUDA(cudaEventDestroy(start));
        }
    }
    CHECK_CUDA(cudaFree(checksums));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
