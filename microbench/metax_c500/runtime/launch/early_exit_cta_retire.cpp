/*
 * MetaX C500 early-exit CTA retire-rate benchmark.
 *
 * The Q64 prefill FlashAttention launch schedules one CTA per
 * (request, query tile, query head) using the batch's maximum query tile
 * count, so a request shorter than that maximum contributes CTAs whose tile
 * lies beyond its own query length.  The source rejects those tiles before
 * the Q/K/V path.  The Forward model prices them at exactly zero on the
 * pipeline critical path, which is only correct if an AP can retire them for
 * free.  This benchmark measures whether it can.
 *
 * The kernel body is the rejection itself and nothing else: each CTA reads
 * the two ints that describe its tile, compares them, and returns.  There is
 * no production operator,
 * no attention arithmetic, no matrix instruction and no shared memory: the
 * reported quantity is a launch duration against a CTA count at a fixed
 * 256-thread block, and the derived quantity is its slope in nanoseconds per
 * CTA per AP.
 *
 * The grid is swept well past the resident window so the measured slope is
 * the steady-state retire rate of a queue that must refill, which is the
 * regime the padded tiles occupy.
 *
 * Two controls make the slope independent of the device's clock ramp, which
 * a first version of this probe was defeated by: a sustained warm-up that
 * launches continuously for a wall-clock budget before anything is timed,
 * and trial-outer/grid-inner ordering so every grid is sampled once per
 * round and therefore sees the same clock state.  A sweep that visits one
 * grid to exhaustion before moving on measures the ramp, not the retire
 * rate, and its slope depends on sweep direction.
 *
 * Build:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 early_exit_cta_retire.cpp \
 *       -o early_exit_cta_retire
 */

#include <mc_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

constexpr int kThreads = 256;
/* The pinned D128 Q64 FlashAttention trait launches 256 threads with 32 KiB
 * of dynamic shared memory, which is what holds it to two resident CTAs per
 * AP.  A padded tile is retired out of that same two-deep window, so the
 * probe must request the identical footprint: without it the probe keeps far
 * more CTAs in flight and measures a retire rate the real launch never
 * reaches.  The same constant pins the residency probe in
 * ``cta_admission.cpp``. */
constexpr int kDynamicSharedBytes = 32 * 1024;

/* Exactly the rejection a padded query tile performs: read this CTA's tile
 * index and its request's tile count, and return when the tile is beyond the
 * request.  ``retired`` proves every planned CTA actually ran. */
__global__ void early_exit_cta(const int* __restrict__ tile_of_cta,
                               const int* __restrict__ tiles_of_request,
                               unsigned int* __restrict__ retired,
                               unsigned int* __restrict__ sink) {
    /* Touched so the allocation cannot be elided; the rejection below still
     * happens before any Q/K/V work, exactly as in the source. */
    extern __shared__ unsigned char dynamic_shared[];
    dynamic_shared[threadIdx.x] = static_cast<unsigned char>(threadIdx.x);
    const int cta = blockIdx.x;
    const int tile = tile_of_cta[cta];
    const int limit = tiles_of_request[cta];
    if (tile >= limit) {
        if (threadIdx.x == 0) {
            atomicAdd(retired, 1U);
        }
        return;
    }
    /* Never taken in this benchmark's configuration; present so the compiler
     * cannot delete the predicate or the loads that feed it. */
    if (threadIdx.x == 0) {
        atomicAdd(sink, static_cast<unsigned int>(tile + limit));
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s <trials> <warmup_seconds> <grid_csv>\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    const int trials = std::atoi(argv[1]);
    if (trials < 1 || trials > 64) {
        std::fprintf(stderr, "trials must be within 1..64\n");
        return EXIT_FAILURE;
    }
    const int warmup_seconds = std::atoi(argv[2]);
    if (warmup_seconds < 0 || warmup_seconds > 600) {
        std::fprintf(stderr, "warmup_seconds must be within 0..600\n");
        return EXIT_FAILURE;
    }
    std::vector<int> grids;
    for (const char* cursor = argv[3]; *cursor != '\0';) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value <= 0 || value > 2000000) {
            std::fprintf(stderr, "grids must be within 1..2000000\n");
            return EXIT_FAILURE;
        }
        grids.push_back(static_cast<int>(value));
        cursor = (*end == ',') ? end + 1 : end;
    }

    mcDeviceProp_t properties{};
    CHECK(mcGetDeviceProperties(&properties, 0));
    std::printf("device,%s\n", properties.name);
    std::printf("multiprocessors,%d\n", properties.multiProcessorCount);
    std::printf("dynamic_shared_bytes,%d\n", kDynamicSharedBytes);
    std::printf("grid_ctas,threads_per_cta,trial_index,retired,duration_ms\n");

    const int max_grid = *std::max_element(grids.begin(), grids.end());
    std::vector<int> host_tile(max_grid);
    std::vector<int> host_limit(max_grid);
    for (int index = 0; index < max_grid; ++index) {
        /* Every CTA is a padded tile: its index is at or beyond its
         * request's tile count, so every CTA takes the rejection path. */
        host_tile[index] = 8;
        host_limit[index] = 1;
    }
    int* tile_of_cta = nullptr;
    int* tiles_of_request = nullptr;
    unsigned int* retired = nullptr;
    unsigned int* sink = nullptr;
    CHECK(mcMalloc(&tile_of_cta, sizeof(int) * max_grid));
    CHECK(mcMalloc(&tiles_of_request, sizeof(int) * max_grid));
    CHECK(mcMalloc(&retired, sizeof(unsigned int)));
    CHECK(mcMalloc(&sink, sizeof(unsigned int)));
    CHECK(mcMemcpy(tile_of_cta, host_tile.data(), sizeof(int) * max_grid,
                   mcMemcpyHostToDevice));
    CHECK(mcMemcpy(tiles_of_request, host_limit.data(), sizeof(int) * max_grid,
                   mcMemcpyHostToDevice));

    mcEvent_t start;
    mcEvent_t stop;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&stop));
    /* Sustained warm-up: launch continuously for a wall-clock budget so the
     * device reaches its steady clock before anything is timed. */
    const auto warmup_deadline =
        std::chrono::steady_clock::now()
        + std::chrono::seconds(warmup_seconds);
    while (std::chrono::steady_clock::now() < warmup_deadline) {
        for (int burst = 0; burst < 64; ++burst) {
            early_exit_cta<<<max_grid, kThreads, kDynamicSharedBytes>>>(
                tile_of_cta, tiles_of_request, retired, sink);
        }
        CHECK(mcDeviceSynchronize());
    }
    /* Trial-outer / grid-inner: every grid is visited once per round, so a
     * residual drift affects all grids alike instead of tilting the slope. */
    for (int trial = 0; trial < trials; ++trial) {
        for (const int grid : grids) {
            CHECK(mcMemset(retired, 0, sizeof(unsigned int)));
            CHECK(mcDeviceSynchronize());
            CHECK(mcEventRecord(start, 0));
            early_exit_cta<<<grid, kThreads, kDynamicSharedBytes>>>(
                tile_of_cta, tiles_of_request, retired, sink);
            CHECK(mcEventRecord(stop, 0));
            CHECK(mcEventSynchronize(stop));
            float milliseconds = 0.0f;
            CHECK(mcEventElapsedTime(&milliseconds, start, stop));
            unsigned int host_retired = 0U;
            CHECK(mcMemcpy(&host_retired, retired, sizeof(unsigned int),
                           mcMemcpyDeviceToHost));
            if (static_cast<int>(host_retired) != grid) {
                std::fprintf(stderr,
                             "grid %d retired %u CTAs\n", grid, host_retired);
                return EXIT_FAILURE;
            }
            std::printf("%d,%d,%d,%u,%.6f\n", grid, kThreads, trial,
                        host_retired, static_cast<double>(milliseconds));
        }
    }
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(stop));
    CHECK(mcFree(tile_of_cta));
    CHECK(mcFree(tiles_of_request));
    CHECK(mcFree(retired));
    CHECK(mcFree(sink));
    return EXIT_SUCCESS;
}
