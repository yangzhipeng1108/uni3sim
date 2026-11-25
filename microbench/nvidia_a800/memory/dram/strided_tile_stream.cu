/* A800 achieved DRAM read rate of a two-dimensional tile stream.
 *
 * A staged GEMM does not read its weight operand as a contiguous stream.  Each
 * CTA reads a tile: tile_n columns by tile_k rows of a row-major matrix, so the
 * stream is a sequence of short contiguous runs separated by the matrix row
 * pitch.  A contiguous-stream probe therefore measures a rate no GEMM operand
 * stream reaches.  This probe measures the tile stream directly: achieved
 * aggregate bytes per second against the contiguous run length and the stride
 * between runs, at a fixed grid and a footprint far beyond L2.
 *
 * It is a bandwidth mechanism.  It contains no operator, no matrix instruction,
 * no shared-memory staging, no tile schedule and no production duration; run
 * length and stride are the only inputs and bytes per second the only output.
 *
 * Forward consumes nothing from this probe, and the reason is its result.  It
 * was written because a staged GEMM's weight stream is 256-byte runs at the
 * matrix row pitch, and the three production cases whose efficiency needs
 * explaining reach only 0.70 to 0.785 of DRAM peak on their compulsory traffic.
 * At grid 224 and 2 MiB per CTA the measured fractions are 1.175, 1.134, 1.109,
 * 1.081 and 0.928 for strides 8192, 16384, 22016, 28672 and 57344, against 0.925
 * for a contiguous 1 MiB run.  The tile shape is not a slowdown: it matches or
 * beats a contiguous stream.
 *
 * The fractions above one are real and worth reading: adjacent CTAs read
 * adjacent column bands of the same rows, so those rows are shared and part of
 * the traffic is served by L2, exactly as in a GEMM whose N tiles share K rows.
 * The probe's reuse structure is therefore not identical to a production tile
 * schedule's, so this excludes the access shape as the mechanism without
 * measuring production's own reuse.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -DPROBE_RUN_BYTES=256 -DPROBE_STRIDE_BYTES=57344 \
 *        -o strided_tile_stream strided_tile_stream.cu
 *   ./strided_tile_stream 0 9 <grid> <runs>
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

#ifndef PROBE_RUN_BYTES
#define PROBE_RUN_BYTES 256
#endif
#ifndef PROBE_STRIDE_BYTES
#define PROBE_STRIDE_BYTES 57344
#endif
constexpr int kThreads = 128;
constexpr long long kRunBytes = PROBE_RUN_BYTES;
constexpr long long kStrideBytes = PROBE_STRIDE_BYTES;
constexpr long long kVectorsPerRun = kRunBytes / 16;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

__device__ __forceinline__ uint4 load_cg(const uint4* address) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address));
    return value;
}

/* Every CTA walks its own column band: ``runs`` contiguous runs of kRunBytes,
 * each separated by kStrideBytes, exactly the shape of a tile_n-wide column
 * band read down tile_k rows of a row-major matrix. */
__global__ __launch_bounds__(kThreads)
void strided_tile_stream(const std::uint8_t* __restrict__ input,
                         long long band_offset_bytes,
                         int runs,
                         std::uint32_t* __restrict__ sink) {
    const long long base =
        static_cast<long long>(blockIdx.x) * band_offset_bytes;
    unsigned int checksum = 0;
    // A GEMM loads a whole tile cooperatively: the CTA's threads cover one run,
    // wrap to the next, and so on, so every thread is active even when a run is
    // narrower than the CTA.  Walking the tile's linear vector index and
    // decomposing it into (run, offset-in-run) reproduces exactly that.
    const long long total_vectors =
        static_cast<long long>(runs) * kVectorsPerRun;
    for (long long index = threadIdx.x; index < total_vectors;
            index += kThreads) {
        const long long run = index / kVectorsPerRun;
        const long long offset = index - run * kVectorsPerRun;
        const std::uint8_t* row = input + base + run * kStrideBytes;
        const uint4 value = load_cg(
            reinterpret_cast<const uint4*>(row) + offset);
        checksum ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    if (checksum == 0xffffffffu) sink[blockIdx.x] = checksum;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS GRID RUNS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::atoi(argv[1]);
    const int trials = std::atoi(argv[2]);
    const int grid = std::atoi(argv[3]);
    const int runs = std::atoi(argv[4]);
    if (device < 0 || trials < 3 || trials > 15 || grid <= 0 || runs < 1) {
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

    // Bands are laid side by side, so consecutive CTAs read adjacent column
    // bands of the same rows, which is how a GEMM's N tiles are distributed.
    const long long band_offset = kRunBytes;
    const long long footprint =
        static_cast<long long>(grid) * band_offset
        + static_cast<long long>(runs) * kStrideBytes;
    if (footprint <= 64LL * 1024 * 1024) {
        std::fprintf(stderr, "footprint must exceed L2 by a wide margin\n");
        return EXIT_FAILURE;
    }
    std::uint8_t* input = nullptr;
    std::uint32_t* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&input, footprint));
    CHECK_CUDA(cudaMalloc(&sink, grid * sizeof(*sink)));
    CHECK_CUDA(cudaMemset(input, 0x63, footprint));
    CHECK_CUDA(cudaMemset(sink, 0, grid * sizeof(*sink)));
    for (int warmup = 0; warmup < 3; ++warmup) {
        strided_tile_stream<<<grid, kThreads>>>(
            input, band_offset, runs, sink);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    const long long logical_bytes =
        static_cast<long long>(grid) * runs * kRunBytes;
    std::printf("grid,threads,run_bytes,stride_bytes,runs,footprint_bytes,"
                "trial,duration_ms,logical_bytes\n");
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        strided_tile_stream<<<grid, kThreads>>>(
            input, band_offset, runs, sink);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        std::printf("%d,%d,%lld,%lld,%d,%lld,%d,%.9f,%lld\n",
                    grid, kThreads, kRunBytes, kStrideBytes, runs,
                    footprint, trial, milliseconds, logical_bytes);
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
    }
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
