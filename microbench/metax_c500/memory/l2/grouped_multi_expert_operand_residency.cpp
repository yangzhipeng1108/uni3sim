/*
 * MetaX C500 multi-expert grouped-program operand residency benchmark.
 *
 * Identical configuration list to the A800 probe of the same name, so the two
 * cards can be compared ordinal by ordinal.  The single-expert probe validated
 * the shared miss core on this target, but only in the regime where the
 * distinct footprint dwarfs L2 and compulsory traffic dominates.  This probe
 * starts at that anchor and moves one axis at a time toward the production
 * grouped-MoE corner: distinct B matrices, the partition of M tiles among
 * them, the N extent, the K depth, and tile raggedness.
 *
 * Program p maps to an output tile through the same global group-of-M swizzle
 * the production kernel uses, the expert of an M tile is that tile index
 * divided by the tiles each expert owns, and at each K step a program reads
 * one A object of rows*BK elements and one B object of BK*BN elements.
 * Object identity is the subject, so the kernel only reduces what it loads.
 *
 * The aliased-B configurations point every expert at one B buffer, holding the
 * reference order exactly fixed while changing only the distinct footprint,
 * which separates a capacity effect from an ordering effect.
 *
 * Every configuration runs inside one mcProfiler marker window, so the layer-1
 * per-kernel pass reports each as its own ordinal.  The quantities of interest
 * are traffic counters, never the instrumented duration, which layer-1 replay
 * inflates and which must not reach a Forward price.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o grouped_multi_expert_operand_residency \
 *       grouped_multi_expert_operand_residency.cpp
 */

#include <mc_runtime.h>


#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = call;                                              \
    if (error != mcSuccess) {                                            \
        std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,             \
                     mcGetErrorString(error));                           \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

static const int kBlockThreads = 256;
static const int kVectorBytes = 16;

/* Fixed operand tile: the production BM64/BN64/BK32 signature in half
 * precision.  BM, BN and BK were already moved independently by the
 * single-expert probe and are held here so the new axes move alone. */
static const int kBlockM = 64;
static const int kBlockN = 64;
static const int kBlockK = 32;
static const int kElementBytes = 2;
static const int kAStrideVectors = kBlockM * kBlockK * kElementBytes / kVectorBytes;
static const int kBVectors = kBlockK * kBlockN * kElementBytes / kVectorBytes;

__global__ __launch_bounds__(kBlockThreads) void grouped_multi_expert_residency(
    const uint4* __restrict__ a,
    const uint4* __restrict__ b,
    uint32_t* __restrict__ out,
    int m_tiles,
    int n_tiles,
    int k_steps,
    int group_width,
    int tiles_per_expert,
    int alias_experts,
    int a_vectors
) {
    const int program = blockIdx.x;
    const int programs_per_group = group_width * n_tiles;
    const int group = program / programs_per_group;
    const int first_m = group * group_width;
    int width = m_tiles - first_m;
    if (width > group_width) width = group_width;
    if (width <= 0) return;
    const int remainder = program % programs_per_group;
    const int tile_m = first_m + remainder % width;
    const int tile_n = remainder / width;
    if (tile_m >= m_tiles || tile_n >= n_tiles) return;

    const int expert = alias_experts ? 0 : (tile_m / tiles_per_expert);

    const int lane = threadIdx.x;
    uint32_t checksum = 0u;
    for (int step = 0; step < k_steps; step++) {
        const size_t a_base =
            (static_cast<size_t>(tile_m) * k_steps + step)
            * static_cast<size_t>(kAStrideVectors);
        const size_t b_base =
            ((static_cast<size_t>(expert) * n_tiles + tile_n)
             * static_cast<size_t>(k_steps) + step)
            * static_cast<size_t>(kBVectors);
        for (int i = lane; i < a_vectors; i += kBlockThreads) {
            const uint4 v = a[a_base + i];
            checksum += v.x ^ v.w;
        }
        for (int i = lane; i < kBVectors; i += kBlockThreads) {
            const uint4 v = b[b_base + i];
            checksum += v.y ^ v.z;
        }
    }
    out[static_cast<size_t>(program) * kBlockThreads + lane] = checksum;
}

struct Config {
    int experts;
    int tiles_per_expert;
    int n_tiles;
    int k_steps;
    int gsm;
    int rows;          /* rows the M tile actually carries, <= kBlockM */
    int alias;         /* 1 = every expert shares one B buffer */
    const char* axis;
};

int main() {
    std::vector<Config> configs;
    /* Ordinal 0 is the validated single-expert anchor, repeated at the end so
     * any drift across the campaign is visible. */
    configs.push_back({1, 128, 64, 64, 8, 64, 0, "anchor"});

    /* A: iso-grid partition.  Total M tiles pinned at 128; only the split
     * between experts and tiles-per-expert moves. */
    const int iso[][2] = {{1,128},{2,64},{4,32},{8,16},{16,8},{32,4},{64,2},{128,1}};
    for (const auto& pair : iso)
        configs.push_back({pair[0], pair[1], 64, 64, 8, 64, 0, "iso_grid_partition"});

    /* B: experts at fixed tiles-per-expert. */
    for (int e : {1, 2, 4, 8, 16})
        configs.push_back({e, 16, 64, 64, 8, 64, 0, "experts"});

    /* C: tiles-per-expert at fixed expert count. */
    for (int t : {1, 2, 4, 8, 16, 32})
        configs.push_back({8, t, 64, 64, 8, 64, 0, "tiles_per_expert"});

    /* D: group width inside a multi-expert grid. */
    for (int g : {1, 2, 4, 8, 16})
        configs.push_back({8, 16, 64, 64, g, 64, 0, "group_width"});

    /* E: N extent -- bytes between two references to one B object. */
    for (int n : {16, 64, 256, 1024})
        configs.push_back({8, 16, n, 64, 8, 64, 0, "n_extent"});

    /* F: K depth. */
    for (int k : {32, 64, 128, 224})
        configs.push_back({8, 16, 64, k, 8, 64, 0, "k_depth"});

    /* G: raggedness at the Region B geometry (one ragged tile per expert
     * across many experts), which references every B object exactly once. */
    for (int r : {64, 48, 32, 26})
        configs.push_back({128, 1, 48, 128, 8, r, 0, "raggedness"});

    /* H: aliased-B control at both failing geometries. */
    configs.push_back({8, 16, 64, 64, 8, 64, 0, "alias_control_region_a"});
    configs.push_back({8, 16, 64, 64, 8, 64, 1, "alias_control_region_a"});
    configs.push_back({128, 1, 48, 128, 8, 26, 0, "alias_control_region_b"});
    configs.push_back({128, 1, 48, 128, 8, 26, 1, "alias_control_region_b"});

    configs.push_back({1, 128, 64, 64, 8, 64, 0, "anchor_repeat"});

    size_t a_capacity = 0, b_capacity = 0, programs_max = 0;
    for (const Config& c : configs) {
        const size_t m_tiles = static_cast<size_t>(c.experts) * c.tiles_per_expert;
        a_capacity = std::max(a_capacity,
            m_tiles * c.k_steps * kAStrideVectors * kVectorBytes);
        const size_t experts_live = c.alias ? 1 : static_cast<size_t>(c.experts);
        b_capacity = std::max(b_capacity,
            experts_live * c.n_tiles * c.k_steps * kBVectors * kVectorBytes);
        programs_max = std::max(programs_max, m_tiles * c.n_tiles);
    }

    uint4* a = nullptr;
    uint4* b = nullptr;
    uint32_t* out = nullptr;
    CHECK(mcMalloc(&a, a_capacity));
    CHECK(mcMalloc(&b, b_capacity));
    CHECK(mcMalloc(&out, programs_max * kBlockThreads * sizeof(uint32_t)));
    CHECK(mcMemset(a, 0x5a, a_capacity));
    CHECK(mcMemset(b, 0xa5, b_capacity));
    CHECK(mcMemset(out, 0, programs_max * kBlockThreads * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"c500_grouped_multi_expert_operand_residency_v1\",");
    std::printf("\"block_m\":%d,\"block_n\":%d,\"block_k\":%d,", kBlockM, kBlockN, kBlockK);
    std::printf("\"block_threads\":%d,\"element_bytes\":%d,", kBlockThreads, kElementBytes);
    std::printf("\"a_bytes\":%zu,\"b_bytes\":%zu,", a_capacity, b_capacity);
    std::printf("\"ordinals\":[");
    for (size_t index = 0; index < configs.size(); index++) {
        const Config& c = configs[index];
        const int a_vectors = c.rows * kBlockK * kElementBytes / kVectorBytes;
        const size_t m_tiles = static_cast<size_t>(c.experts) * c.tiles_per_expert;
        const size_t programs = m_tiles * c.n_tiles;
        if (index != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"experts\":%d,"
                    "\"tiles_per_expert\":%d,\"m_tiles\":%zu,\"n_tiles\":%d,"
                    "\"k_steps\":%d,\"gsm\":%d,\"rows\":%d,\"alias\":%d,"
                    "\"programs\":%zu,\"a_object_bytes\":%d,\"b_object_bytes\":%d,"
                    "\"logical_request_bytes\":%.0f}",
                    index, c.axis, c.experts, c.tiles_per_expert, m_tiles,
                    c.n_tiles, c.k_steps, c.gsm, c.rows, c.alias, programs,
                    a_vectors * kVectorBytes, kBVectors * kVectorBytes,
                    static_cast<double>(programs) * c.k_steps
                        * (a_vectors + kBVectors) * kVectorBytes);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    /* One marker window holds every configuration, so the layer-1 pass emits
     * one numbered invocation per ordinal in this exact order. */
    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        const int a_vectors = c.rows * kBlockK * kElementBytes / kVectorBytes;
        const size_t m_tiles = static_cast<size_t>(c.experts) * c.tiles_per_expert;
        const size_t programs = m_tiles * c.n_tiles;
        grouped_multi_expert_residency<<<programs, kBlockThreads>>>(
            a, b, out, static_cast<int>(m_tiles), c.n_tiles, c.k_steps, c.gsm,
            c.tiles_per_expert, c.alias, a_vectors);
        CHECK(mcDeviceSynchronize());
    }

    CHECK(mcProfilerStop());

    CHECK(mcFree(a));
    CHECK(mcFree(b));
    CHECK(mcFree(out));
    return 0;
}
