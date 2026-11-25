/*
 * MetaX C500 operand double-read phase residency benchmark.
 *
 * The canonical MetaX BM64 fused-MoE launch observation records an L2-ingress
 * stream multiplicity of two for both operands (`operand_replay_a`,
 * `operand_replay_b`): every operand byte crosses the L2 source interface
 * twice in one invocation.  That observation is a counter identity and its
 * own docstring states it is not a cross-tile replay law -- it does not say
 * *where* in the k loop the second crossing happens, and the two possibilities
 * have opposite DRAM consequences:
 *
 *   adjacent   the second read follows the first inside the same k step, so
 *              it is served by the line the first read just allocated and
 *              adds no DRAM traffic at any footprint;
 *   separated  the second read happens a full k sweep later, so it doubles
 *              the reuse distance and can miss once the swept footprint
 *              exceeds L2.
 *
 * This probe holds the grid, the block, the footprint and the total logical
 * read bytes exactly fixed and moves only that phase, across a footprint
 * sweep that brackets the L2 size.  It is not a fused-MoE surrogate: there is
 * no routing, no expert blocking, no MMA and no epilogue, only the double
 * read whose phase is in question.
 *
 * Traffic counters only.  No duration here may price a Forward path, and the
 * phase-to-traffic relation must not be fitted or interpolated.
 *
 * Every configuration runs inside one mcProfiler marker window, so the
 * layer-1 per-kernel pass reports each as its own ordinal.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o operand_replay_phase_residency \
 *       operand_replay_phase_residency.cpp
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = call;                                                \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,             \
                     mcGetErrorString(error));                             \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

static const int kBlockThreads = 256;
/* One "tile" is 16 uint4 per thread-block pass = 4 KiB, the BK x BN operand
 * slab width the BM64 launch issues per k step. */
static const int kVectorsPerTile = 256;

/* `separated` selects the phase: 0 reads each tile twice back to back, 1
 * sweeps every tile once and then sweeps them all a second time.  Both emit
 * exactly 2 * tiles reads of the same tiles in the same addresses. */
__global__ __launch_bounds__(kBlockThreads) void operand_replay_phase(
    const uint4* __restrict__ src,
    uint32_t* __restrict__ out,
    int tiles,
    int separated
) {
    const int worker = blockIdx.x;
    const int lane = threadIdx.x;
    const size_t base = (size_t)worker * tiles * kVectorsPerTile;

    uint32_t checksum = 0u;
    if (separated) {
        for (int pass = 0; pass < 2; pass++) {
            for (int t = 0; t < tiles; t++) {
                const size_t off = base + (size_t)t * kVectorsPerTile;
                for (int i = lane; i < kVectorsPerTile; i += kBlockThreads) {
                    const uint4 value = src[off + i];
                    checksum += value.x ^ value.w;
                }
            }
        }
    } else {
        for (int t = 0; t < tiles; t++) {
            const size_t off = base + (size_t)t * kVectorsPerTile;
            for (int pass = 0; pass < 2; pass++) {
                for (int i = lane; i < kVectorsPerTile; i += kBlockThreads) {
                    const uint4 value = src[off + i];
                    checksum += value.x ^ value.w;
                }
            }
        }
    }
    out[worker] = checksum;
}

struct Config { int workers, tiles, separated; const char* axis; };

int main() {
    /* Footprint = workers * tiles * 4 KiB.  The sweep brackets the 8 MiB L2:
     * 2, 8, 32 and 128 MiB, each at both phases. */
    std::vector<Config> configs;
    const int worker_counts[] = {104, 104, 104, 104};
    const int tile_counts[] = {5, 20, 80, 320};
    const char* axes[] = {"fp2mib", "fp8mib", "fp32mib", "fp128mib"};
    for (int index = 0; index < 4; index++) {
        for (int separated = 0; separated < 2; separated++) {
            configs.push_back({worker_counts[index], tile_counts[index],
                               separated, axes[index]});
        }
    }
    /* Repeatability on the two 32 MiB points. */
    configs.push_back({104, 80, 0, "repeat"});
    configs.push_back({104, 80, 1, "repeat"});

    size_t max_vectors = 0;
    for (const Config& c : configs) {
        max_vectors = std::max(
            max_vectors, (size_t)c.workers * c.tiles * kVectorsPerTile);
    }
    uint4* src = nullptr;
    uint32_t* out = nullptr;
    CHECK(mcMalloc(&src, max_vectors * sizeof(uint4)));
    CHECK(mcMalloc(&out, 1024 * sizeof(uint32_t)));
    CHECK(mcMemset(src, 0x5a, max_vectors * sizeof(uint4)));
    CHECK(mcMemset(out, 0, 1024 * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"operand_replay_phase_residency_v1\",");
    std::printf("\"block_threads\":%d,\"tile_bytes\":%d,\"ordinals\":[",
                kBlockThreads, kVectorsPerTile * 16);
    for (size_t index = 0; index < configs.size(); index++) {
        const Config& c = configs[index];
        const double footprint =
            (double)c.workers * c.tiles * kVectorsPerTile * 16;
        if (index != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"workers\":%d,"
                    "\"tiles\":%d,\"separated\":%d,\"footprint_bytes\":%.0f,"
                    "\"logical_read_bytes\":%.0f}",
                    index, c.axis, c.workers, c.tiles, c.separated,
                    footprint, footprint * 2.0);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        operand_replay_phase<<<c.workers, kBlockThreads>>>(
            src, out, c.tiles, c.separated);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());

    CHECK(mcFree(src));
    CHECK(mcFree(out));
    return 0;
}
