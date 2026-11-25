/*
 * NVIDIA A800 grouped-program operand-sharing mechanism benchmark.
 *
 * A grouped matrix program assigns its programs to output tiles through a
 * group-of-M swizzle: with group width G, the G programs that share one N
 * coordinate are dispatched consecutively, so at any K step those G programs
 * reference the same B tile.  A traffic model that treats concurrently
 * resident programs as advancing through K together therefore credits that B
 * tile with G-1 hits.  This benchmark measures whether the hardware actually
 * delivers that sharing.
 *
 * The probe keeps every byte, every instruction and the total logical request
 * volume fixed and varies only the group width.  A and B footprints both far
 * exceed the 40 MiB L2, so the only reuse available is between
 * programs that reference the same tile at the same K step.  If that sharing
 * happens, a wider group finishes proportionally sooner; if concurrent
 * programs drift apart in K, the group width does not change the time.
 *
 * This measures whether same-K-step operand sharing occurs between resident
 * programs.  It does not measure a production operator duration, an L2
 * capacity, or a replacement policy.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o grouped_program_operand_sharing \\
 *       grouped_program_operand_sharing.cu
 */

#include <cuda_runtime.h>

#define gpuDeviceProp          cudaDeviceProp
#define gpuDeviceSynchronize   cudaDeviceSynchronize
#define gpuError_t             cudaError_t
#define gpuEventCreate         cudaEventCreate
#define gpuEventDestroy        cudaEventDestroy
#define gpuEventElapsedTime    cudaEventElapsedTime
#define gpuEventRecord         cudaEventRecord
#define gpuEvent_t             cudaEvent_t
#define gpuFree                cudaFree
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuGetErrorString      cudaGetErrorString
#define gpuMalloc              cudaMalloc
#define gpuMemset              cudaMemset
#define gpuSuccess             cudaSuccess

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    gpuError_t error = call;                                               \
    if (error != gpuSuccess) {                                             \
        std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,             \
                     gpuGetErrorString(error));                            \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

/* One operand tile is 64x64 half-precision elements, the tile the retained
 * C500 grouped-MoE launch reports. */
static const int kTileBytes = 64 * 64 * 2;
static const int kBlockThreads = 256;
static const int kVectorBytes = 16;
static const int kTileVectors = kTileBytes / kVectorBytes;

__global__ __launch_bounds__(kBlockThreads) void grouped_operand_stream(
    const uint4* __restrict__ a,
    const uint4* __restrict__ b,
    uint32_t* __restrict__ out,
    int m_tiles,
    int n_tiles,
    int k_steps,
    int group_width
) {
    const int program = blockIdx.x;
    const int programs_per_group = group_width * n_tiles;
    const int group = program / programs_per_group;
    const int first_m = group * group_width;
    int width = m_tiles - first_m;
    if (width > group_width) width = group_width;
    const int remainder = program % programs_per_group;
    const int tile_m = first_m + remainder % width;
    const int tile_n = remainder / width;
    if (tile_m >= m_tiles || tile_n >= n_tiles) return;

    const int lane = threadIdx.x;
    uint32_t checksum = 0u;
    for (int step = 0; step < k_steps; step++) {
        const size_t a_tile =
            (static_cast<size_t>(tile_m) * k_steps + step) * kTileVectors;
        const size_t b_tile =
            (static_cast<size_t>(tile_n) * k_steps + step) * kTileVectors;
        for (int index = lane; index < kTileVectors; index += kBlockThreads) {
            const uint4 left = a[a_tile + index];
            const uint4 right = b[b_tile + index];
            checksum += left.x ^ right.x;
            checksum += left.w ^ right.w;
        }
    }
    out[program * kBlockThreads + lane] = checksum;
}

int main(int argc, char** argv) {
    int m_tiles = 264;
    int n_tiles = 64;
    int k_steps = 448;
    int trials = 5;
    if (argc > 1) m_tiles = std::atoi(argv[1]);
    if (argc > 2) n_tiles = std::atoi(argv[2]);
    if (argc > 3) k_steps = std::atoi(argv[3]);
    if (argc > 4) trials = std::atoi(argv[4]);
    if (m_tiles <= 0 || n_tiles <= 0 || k_steps <= 0 || trials <= 0) {
        std::fprintf(stderr, "geometry and trial count must be positive\n");
        return 1;
    }

    gpuDeviceProp properties;
    CHECK(gpuGetDeviceProperties(&properties, 0));

    const size_t a_bytes =
        static_cast<size_t>(m_tiles) * k_steps * kTileBytes;
    const size_t b_bytes =
        static_cast<size_t>(n_tiles) * k_steps * kTileBytes;
    const size_t programs = static_cast<size_t>(m_tiles) * n_tiles;
    const size_t out_bytes = programs * kBlockThreads * sizeof(uint32_t);

    uint4* a = nullptr;
    uint4* b = nullptr;
    uint32_t* out = nullptr;
    CHECK(gpuMalloc(&a, a_bytes));
    CHECK(gpuMalloc(&b, b_bytes));
    CHECK(gpuMalloc(&out, out_bytes));
    CHECK(gpuMemset(a, 0x5a, a_bytes));
    CHECK(gpuMemset(b, 0xa5, b_bytes));
    CHECK(gpuMemset(out, 0, out_bytes));

    gpuEvent_t start;
    gpuEvent_t stop;
    CHECK(gpuEventCreate(&start));
    CHECK(gpuEventCreate(&stop));

    const double logical_bytes =
        static_cast<double>(programs) * k_steps * 2.0 * kTileBytes;

    std::printf("{\"schema\":\"a800_grouped_program_operand_sharing_v1\",");
    std::printf("\"device\":\"%s\",", properties.name);
    std::printf("\"tile_bytes\":%d,\"block_threads\":%d,", kTileBytes,
                kBlockThreads);
    std::printf("\"m_tiles\":%d,\"n_tiles\":%d,\"k_steps\":%d,", m_tiles,
                n_tiles, k_steps);
    std::printf("\"a_footprint_bytes\":%zu,\"b_footprint_bytes\":%zu,",
                a_bytes, b_bytes);
    std::printf("\"logical_request_bytes\":%.0f,\"trials\":%d,",
                logical_bytes, trials);
    std::printf("\"measurements\":[");

    const int widths[] = {1, 2, 4, 8, 16};
    bool first_record = true;
    for (int width_index = 0;
         width_index < static_cast<int>(sizeof(widths) / sizeof(widths[0]));
         width_index++) {
        const int group_width = widths[width_index];
        if (group_width > m_tiles) continue;
        /* Warm the instruction and page state without timing it. */
        grouped_operand_stream<<<programs, kBlockThreads>>>(
            a, b, out, m_tiles, n_tiles, k_steps, group_width);
        CHECK(gpuDeviceSynchronize());

        std::vector<double> milliseconds;
        for (int trial = 0; trial < trials; trial++) {
            CHECK(gpuEventRecord(start, 0));
            grouped_operand_stream<<<programs, kBlockThreads>>>(
                a, b, out, m_tiles, n_tiles, k_steps, group_width);
            CHECK(gpuEventRecord(stop, 0));
            CHECK(gpuDeviceSynchronize());
            float elapsed = 0.0f;
            CHECK(gpuEventElapsedTime(&elapsed, start, stop));
            milliseconds.push_back(static_cast<double>(elapsed));
        }
        std::sort(milliseconds.begin(), milliseconds.end());
        const double median = milliseconds[milliseconds.size() / 2];
        const double spread =
            (milliseconds.back() - milliseconds.front()) / median;
        if (!first_record) std::printf(",");
        first_record = false;
        std::printf("{\"group_width\":%d,\"median_ms\":%.6f,",
                    group_width, median);
        std::printf("\"relative_spread\":%.6f,", spread);
        std::printf("\"logical_byte_rate_per_s\":%.6f}",
                    logical_bytes / (median * 1e-3));
    }
    std::printf("]}\n");

    CHECK(gpuEventDestroy(start));
    CHECK(gpuEventDestroy(stop));
    CHECK(gpuFree(a));
    CHECK(gpuFree(b));
    CHECK(gpuFree(out));
    return 0;
}
