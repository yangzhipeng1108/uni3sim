/* A800 per-tile cost of a staged loop against its length.
 *
 * A software-pipelined loop pays a fill before its steady state and a drain
 * after it, so its cost per tile falls as it lengthens.  This probe measures
 * that amortisation directly: per-tile nanoseconds against the number of tiles,
 * at a fixed per-tile byte count and a fixed per-tile compute, on a single full
 * wave so no launch imbalance is mixed in.
 *
 * Each iteration loads one tile with 16-byte vector loads and issues PROBE_FMA
 * independent fused multiply-adds per loaded vector.  The loads of an iteration
 * do not depend on the previous iteration's compute, so the hardware is free to
 * pipeline them within a loop.
 *
 * The sweep holds the total tile count, and therefore the footprint and the
 * total work, fixed: a loop of L tiles is run TOTAL/L times over the same
 * buffer, and consecutive loops are separated by a dependency on the running
 * accumulator so each one pays its own fill and drain.  Without that the sweep
 * would confound loop length with residency -- a 16-tile loop over 108 CTAs
 * fits the 40 MiB L2 while a 128-tile one does not, which is why an earlier
 * version of this probe showed a minimum at 16 tiles rather than a monotone
 * amortisation.
 *
 * It is a latency/bandwidth mechanism.  It contains no operator, no matrix
 * instruction, no shared-memory staging, no production shape, and consumes no
 * production duration.  It reports only a duration at a tile count.
 *
 * Forward consumes nothing from this probe, and the reason is its result.  It
 * was written to price the per-CTA fill and drain that a short staged K loop was
 * thought to pay, because the production cases whose loops are 21 and 64 tiles
 * long need more time than an imbalance charge explains while a 128-tile one
 * needs less.  At a fixed total tile count, footprint and work, the per-tile cost
 * is flat: 3209 ns for 448 separate one-tile loops against 3175 ns for a single
 * 448-tile loop, a 1.1 percent spread, with every length in between inside it.
 * Loop length costs nothing measurable here, so the hypothesis is excluded and
 * the residual is not fill and drain.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -DPROBE_FMA=64 -DPROBE_TILE_KIB=16 \
 *        -o k_loop_fill_drain k_loop_fill_drain.cu
 *   ./k_loop_fill_drain 0 9 <inner_tiles> <total_tiles>
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
#ifndef PROBE_TILE_KIB
#define PROBE_TILE_KIB 16
#endif
constexpr int kThreads = 128;
constexpr int kFma = PROBE_FMA;
constexpr long long kTileBytes = PROBE_TILE_KIB * 1024LL;
constexpr long long kVectorsPerTile = kTileBytes / 16;
constexpr int kGrid = 108;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

__device__ __forceinline__ uint4 load_cg(const uint4* address) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address));
    return value;
}

__global__ __launch_bounds__(kThreads)
void staged_tile_loop(const uint4* __restrict__ input,
                      long long vectors_per_cta,
                      int inner_tiles, int outer_loops,
                      std::uint32_t* __restrict__ sink) {
    const long long base =
        static_cast<long long>(blockIdx.x) * vectors_per_cta;
    unsigned int checksum = 0;
    float accumulator = 1.0f + static_cast<float>(threadIdx.x);
    int cursor = 0;
    for (int loop = 0; loop < outer_loops; ++loop) {
        // Depend on the previous loop's result so each inner loop pays its own
        // fill and drain instead of being fused with its neighbour.
        cursor += (accumulator > 0.0f) ? inner_tiles : 0;
        for (int tile = 0; tile < inner_tiles; ++tile) {
            const long long tile_base =
                base + ((cursor - inner_tiles + tile) % (vectors_per_cta
                        / kVectorsPerTile)) * kVectorsPerTile;
            for (long long offset = threadIdx.x; offset < kVectorsPerTile;
                    offset += kThreads) {
                const uint4 value = load_cg(input + tile_base + offset);
                checksum ^= value.x ^ value.y ^ value.z ^ value.w;
#pragma unroll
                for (int operation = 0; operation < kFma; ++operation) {
                    accumulator = __fmaf_rn(
                        accumulator, 1.0000001f, 1.0e-7f);
                }
            }
        }
    }
    if (checksum == 0xffffffffu && accumulator == 0.0f) {
        sink[blockIdx.x] = checksum;
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s DEVICE TRIALS INNER_TILES TOTAL_TILES\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::atoi(argv[1]);
    const int trials = std::atoi(argv[2]);
    const int inner_tiles = std::atoi(argv[3]);
    const int total_tiles = std::atoi(argv[4]);
    if (device < 0 || trials < 3 || trials > 15 || inner_tiles < 1
            || total_tiles < inner_tiles
            || total_tiles % inner_tiles != 0) {
        std::fprintf(stderr, "total tiles must be a multiple of inner\n");
        return EXIT_FAILURE;
    }
    const int outer_loops = total_tiles / inner_tiles;
    const int tiles = total_tiles;

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::strcmp(properties.name, kExpectedDevice) != 0
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kGrid) {
        std::fprintf(stderr, "unexpected A800 device contract\n");
        return EXIT_FAILURE;
    }

    const long long vectors_per_cta = kVectorsPerTile * tiles;
    const long long bytes = vectors_per_cta * 16 * kGrid;
    uint4* input = nullptr;
    std::uint32_t* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&input, bytes));
    CHECK_CUDA(cudaMalloc(&sink, kGrid * sizeof(*sink)));
    CHECK_CUDA(cudaMemset(input, 0x4d, bytes));
    CHECK_CUDA(cudaMemset(sink, 0, kGrid * sizeof(*sink)));
    for (int warmup = 0; warmup < 3; ++warmup) {
        staged_tile_loop<<<kGrid, kThreads>>>(
            input, vectors_per_cta, inner_tiles, outer_loops, sink);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::printf("grid,threads,fma_per_vector,tile_bytes,inner_tiles,"
                "total_tiles,trial,duration_ms,logical_bytes\n");
    for (int trial = 0; trial < trials; ++trial) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        staged_tile_loop<<<kGrid, kThreads>>>(
            input, vectors_per_cta, inner_tiles, outer_loops, sink);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        std::printf("%d,%d,%d,%lld,%d,%d,%d,%.9f,%lld\n",
                    kGrid, kThreads, kFma, kTileBytes, inner_tiles,
                    total_tiles, trial, milliseconds, bytes);
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(start));
    }
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(input));
    return EXIT_SUCCESS;
}
