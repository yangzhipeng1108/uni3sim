/*
 * MetaX C500 grouped-program L2 service-knee benchmark.
 *
 * A companion to grouped_program_operand_sharing, which showed that this
 * target's L2 carries far fewer of a grouped program's requests than a model
 * of same-K-step sharing predicts.  That result has two incompatible
 * explanations and this probe separates them.
 *
 *   capacity form  The programs resident together do share, but they drift
 *                  apart in K, so the bytes that must stay live to serve that
 *                  sharing are the per-step slice multiplied by the drift.
 *                  Sharing then survives while that product fits the cache
 *                  and collapses past a knee, and the knee's position states
 *                  the drift.
 *   bypass form    Some access class does not occupy the cache at all, from a
 *                  streaming or non-temporal policy or a sector/line
 *                  granularity that wastes the fetch.  Then the served
 *                  fraction is low even at a working set far inside the
 *                  cache, and there is no knee.
 *
 * The sharing structure is held fixed -- one group width, one N tile count,
 * so the same number of programs reference the same object at each K step --
 * and only the object size moves, which moves the resident working set
 * across the documented 8 MiB L2.  Total logical request volume is held
 * roughly constant by scaling the K depth against the object size.
 *
 * This measures the served fraction of a grouped operand stream against its
 * resident working set.  It does not measure a production operator duration,
 * a replacement policy, or an L2 capacity in isolation.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o grouped_program_l2_service_knee \
 *       grouped_program_l2_service_knee.cpp
 */

#include <mc_runtime.h>

#define gpuDeviceProp          mcDeviceProp_t
#define gpuDeviceSynchronize   mcDeviceSynchronize
#define gpuError_t             mcError_t
#define gpuEventCreate         mcEventCreate
#define gpuEventDestroy        mcEventDestroy
#define gpuEventElapsedTime    mcEventElapsedTime
#define gpuEventRecord         mcEventRecord
#define gpuEvent_t             mcEvent_t
#define gpuFree                mcFree
#define gpuGetDeviceProperties mcGetDeviceProperties
#define gpuGetErrorString      mcGetErrorString
#define gpuMalloc              mcMalloc
#define gpuMemset              mcMemset
#define gpuSuccess             mcSuccess

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

static const int kBlockThreads = 256;
static const int kVectorBytes = 16;

__global__ __launch_bounds__(kBlockThreads) void grouped_operand_stream(
    const uint4* __restrict__ a,
    const uint4* __restrict__ b,
    uint32_t* __restrict__ out,
    int m_tiles,
    int n_tiles,
    int k_steps,
    int tile_vectors
) {
    const int program = blockIdx.x;
    /* Group width one: consecutive programs sweep N for one fixed M, so one
     * A object is referenced by every program of the sweep and each B object
     * by one.  This is the widest sharing the grouped map can present. */
    const int tile_m = program / n_tiles;
    const int tile_n = program % n_tiles;
    if (tile_m >= m_tiles) return;

    const int lane = threadIdx.x;
    uint32_t checksum = 0u;
    for (int step = 0; step < k_steps; step++) {
        const size_t a_tile =
            (static_cast<size_t>(tile_m) * k_steps + step)
            * static_cast<size_t>(tile_vectors);
        const size_t b_tile =
            (static_cast<size_t>(tile_n) * k_steps + step)
            * static_cast<size_t>(tile_vectors);
        for (int index = lane; index < tile_vectors; index += kBlockThreads) {
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
    int trials = 5;
    long long target_logical_bytes = 120LL * 1024 * 1024 * 1024;
    if (argc > 1) m_tiles = std::atoi(argv[1]);
    if (argc > 2) n_tiles = std::atoi(argv[2]);
    if (argc > 3) trials = std::atoi(argv[3]);
    int single_tile_kib = 0;
    if (argc > 4) single_tile_kib = std::atoi(argv[4]);
    if (m_tiles <= 0 || n_tiles <= 0 || trials <= 0) {
        std::fprintf(stderr, "geometry and trial count must be positive\n");
        return 1;
    }

    gpuDeviceProp properties;
    CHECK(gpuGetDeviceProperties(&properties, 0));

    const size_t programs = static_cast<size_t>(m_tiles) * n_tiles;
    const size_t out_bytes = programs * kBlockThreads * sizeof(uint32_t);
    uint32_t* out = nullptr;
    CHECK(gpuMalloc(&out, out_bytes));
    CHECK(gpuMemset(out, 0, out_bytes));

    gpuEvent_t start;
    gpuEvent_t stop;
    CHECK(gpuEventCreate(&start));
    CHECK(gpuEventCreate(&stop));

    std::printf("{\"schema\":\"c500_grouped_program_l2_service_knee_v1\",");
    std::printf("\"device\":\"%s\",", properties.name);
    std::printf("\"block_threads\":%d,\"group_width\":1,", kBlockThreads);
    std::printf("\"m_tiles\":%d,\"n_tiles\":%d,\"trials\":%d,", m_tiles,
                n_tiles, trials);
    std::printf("\"measurements\":[");

    const int full_sweep[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    const int single[] = {single_tile_kib};
    const int* tile_kib = single_tile_kib > 0 ? single : full_sweep;
    const int tile_count = single_tile_kib > 0
        ? 1 : static_cast<int>(sizeof(full_sweep) / sizeof(full_sweep[0]));
    bool first_record = true;
    for (int index = 0;
         index < tile_count;
         index++) {
        const int tile_bytes = tile_kib[index] * 1024;
        const int tile_vectors = tile_bytes / kVectorBytes;
        /* Hold the logical request volume roughly fixed across object sizes. */
        long long k_steps_64 = target_logical_bytes
            / (static_cast<long long>(programs) * 2 * tile_bytes);
        if (k_steps_64 < 16) k_steps_64 = 16;
        const int k_steps = static_cast<int>(k_steps_64);

        const size_t a_bytes =
            static_cast<size_t>(m_tiles) * k_steps * tile_bytes;
        const size_t b_bytes =
            static_cast<size_t>(n_tiles) * k_steps * tile_bytes;
        uint4* a = nullptr;
        uint4* b = nullptr;
        CHECK(gpuMalloc(&a, a_bytes));
        CHECK(gpuMalloc(&b, b_bytes));
        CHECK(gpuMemset(a, 0x5a, a_bytes));
        CHECK(gpuMemset(b, 0xa5, b_bytes));

        grouped_operand_stream<<<programs, kBlockThreads>>>(
            a, b, out, m_tiles, n_tiles, k_steps, tile_vectors);
        CHECK(gpuDeviceSynchronize());

        std::vector<double> milliseconds;
        for (int trial = 0; trial < trials; trial++) {
            CHECK(gpuEventRecord(start, 0));
            grouped_operand_stream<<<programs, kBlockThreads>>>(
                a, b, out, m_tiles, n_tiles, k_steps, tile_vectors);
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
        const double logical_bytes =
            static_cast<double>(programs) * k_steps * 2.0 * tile_bytes;

        if (!first_record) std::printf(",");
        first_record = false;
        std::printf("{\"tile_bytes\":%d,\"k_steps\":%d,", tile_bytes, k_steps);
        std::printf("\"a_footprint_bytes\":%zu,\"b_footprint_bytes\":%zu,",
                    a_bytes, b_bytes);
        std::printf("\"logical_request_bytes\":%.0f,", logical_bytes);
        std::printf("\"median_ms\":%.6f,\"relative_spread\":%.6f,",
                    median, spread);
        std::printf("\"logical_byte_rate_per_s\":%.6f}",
                    logical_bytes / (median * 1e-3));

        CHECK(gpuFree(a));
        CHECK(gpuFree(b));
    }
    std::printf("]}\n");

    CHECK(gpuEventDestroy(start));
    CHECK(gpuEventDestroy(stop));
    CHECK(gpuFree(out));
    return 0;
}
