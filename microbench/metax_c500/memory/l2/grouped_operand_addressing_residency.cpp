/*
 * MetaX C500 grouped-program operand addressing residency benchmark.
 *
 * The multi-expert residency scan swept every structural axis of a grouped
 * matrix program -- distinct weight matrices, the partition of M tiles among
 * them, N extent, K depth, raggedness and footprint -- and none of them
 * reproduced the production grouped-MoE shortfall at the production group
 * width.  The one remaining difference between that probe and the production
 * kernel is how the operands are addressed, not how many bytes they are.
 *
 * The probe read each operand tile as one contiguous object.  The production
 * kernel does not: weights live in an [E, N, K] layout, so a BK x BN weight
 * tile is BN separate BK-element segments strided by K; and activations are
 * gathered through sorted_token_ids, so a BM x BK activation tile is BM
 * separate BK-element segments at scattered row offsets.  Byte counts and
 * object identity are unchanged -- only the addresses move.
 *
 * Each configuration therefore appears in four addressing modes over an
 * otherwise identical geometry and an identical buffer footprint:
 * contiguous tiles, strided weights, gathered activations, and both.  A mode
 * that changes the measured downstream read while holding every byte count
 * fixed isolates addressing as the mechanism; a mode that does not closes the
 * last structural difference.
 *
 * The gather permutation is deterministic and stated: row i of the packed
 * routing order maps to source row (i * 2654435761u) % rows, an odd-multiplier
 * permutation, so an M tile's rows are scattered across the activation buffer
 * exactly as a routing histogram scatters them, without any data-dependent
 * choice.
 *
 * Traffic counters only.  No duration here may price a Forward path.
 *
 * Every configuration runs inside one mcProfiler marker window, so the layer-1
 * per-kernel pass reports each as its own ordinal.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o grouped_operand_addressing_residency \
 *       grouped_operand_addressing_residency.cpp
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
static const int kBlockM = 64;
static const int kBlockN = 64;
static const int kBlockK = 32;          /* elements per segment */
static const int kElementBytes = 2;
/* A BK-element half-precision segment is 64 B, which is four uint4 loads. */
static const int kVectorsPerSegment = kBlockK * kElementBytes / 16;
static const int kThreadsPerSegment = kVectorsPerSegment;
static const int kSegmentsPerPass = kBlockThreads / kThreadsPerSegment;

__global__ __launch_bounds__(kBlockThreads) void grouped_operand_addressing(
    const uint4* __restrict__ a,
    const uint4* __restrict__ b,
    const int* __restrict__ gather,
    uint32_t* __restrict__ out,
    int m_tiles,
    int n_tiles,
    int k_steps,
    int group_width,
    int tiles_per_expert,
    int rows,
    int strided_weights,
    int gathered_activations
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

    const int expert = tile_m / tiles_per_expert;
    const int lane = threadIdx.x;
    const int segment = lane / kThreadsPerSegment;
    const int sub = lane % kThreadsPerSegment;

    /* Row pitch of the [E, N, K] weight buffer and of the activation buffer,
     * both in uint4 units. */
    const size_t k_pitch = static_cast<size_t>(k_steps) * kVectorsPerSegment;
    const size_t weight_rows = static_cast<size_t>(n_tiles) * kBlockN;

    uint32_t checksum = 0u;
    for (int step = 0; step < k_steps; step++) {
        const size_t step_offset =
            static_cast<size_t>(step) * kVectorsPerSegment + sub;

        for (int r = segment; r < rows; r += kSegmentsPerPass) {
            size_t row;
            if (gathered_activations) {
                row = static_cast<size_t>(
                    gather[static_cast<size_t>(tile_m) * kBlockM + r]);
            } else {
                row = static_cast<size_t>(tile_m) * kBlockM + r;
            }
            const uint4 v = a[row * k_pitch + step_offset];
            checksum += v.x ^ v.w;
        }

        for (int r = segment; r < kBlockN; r += kSegmentsPerPass) {
            size_t row = static_cast<size_t>(expert) * weight_rows
                + static_cast<size_t>(tile_n) * kBlockN + r;
            size_t index;
            if (strided_weights) {
                index = row * k_pitch + step_offset;
            } else {
                /* Contiguous tile: the same bytes, laid out so one BK x BN
                 * tile occupies one run.  Same object, same volume. */
                index = ((static_cast<size_t>(expert) * n_tiles
                          + tile_n) * k_steps + step)
                        * (static_cast<size_t>(kBlockN) * kVectorsPerSegment)
                        + static_cast<size_t>(r) * kVectorsPerSegment + sub;
            }
            const uint4 v = b[index];
            checksum += v.y ^ v.z;
        }
    }
    out[static_cast<size_t>(program) * kBlockThreads + lane] = checksum;
}

struct Config {
    int experts, tiles_per_expert, n_tiles, k_steps, gsm, rows;
    int strided_weights, gathered_activations;
    const char* axis;
};

int main() {
    std::vector<Config> configs;
    /* Region A (many M tiles per expert) and Region B (one ragged tile across
     * many experts), each in all four addressing modes. */
    for (int s = 0; s < 2; s++) {
        for (int g = 0; g < 2; g++) {
            configs.push_back({8, 16, 64, 64, 8, 64, s, g, "region_a_addressing"});
        }
    }
    for (int s = 0; s < 2; s++) {
        for (int g = 0; g < 2; g++) {
            configs.push_back({128, 1, 48, 128, 8, 26, s, g, "region_b_addressing"});
        }
    }
    /* Group width under full production addressing, to see whether addressing
     * changes the proximity response the contiguous scan measured. */
    for (int w : {1, 2, 4, 8, 16}) {
        configs.push_back({8, 16, 64, 64, w, 64, 1, 1, "group_width_addressed"});
    }

    size_t a_capacity = 0, b_capacity = 0, programs_max = 0, gather_max = 0;
    for (const Config& c : configs) {
        const size_t m_tiles = static_cast<size_t>(c.experts) * c.tiles_per_expert;
        const size_t k_pitch = static_cast<size_t>(c.k_steps) * kVectorsPerSegment;
        a_capacity = std::max(a_capacity, m_tiles * kBlockM * k_pitch * 16);
        b_capacity = std::max(b_capacity,
            static_cast<size_t>(c.experts) * c.n_tiles * kBlockN * k_pitch * 16);
        programs_max = std::max(programs_max, m_tiles * c.n_tiles);
        gather_max = std::max(gather_max, m_tiles * kBlockM);
    }

    std::vector<int> gather(gather_max);
    for (size_t i = 0; i < gather_max; i++) {
        gather[i] = static_cast<int>(
            (i * 2654435761u) % static_cast<uint32_t>(gather_max));
    }

    uint4* a = nullptr;
    uint4* b = nullptr;
    int* gather_device = nullptr;
    uint32_t* out = nullptr;
    CHECK(mcMalloc(&a, a_capacity));
    CHECK(mcMalloc(&b, b_capacity));
    CHECK(mcMalloc(&gather_device, gather_max * sizeof(int)));
    CHECK(mcMalloc(&out, programs_max * kBlockThreads * sizeof(uint32_t)));
    CHECK(mcMemset(a, 0x5a, a_capacity));
    CHECK(mcMemset(b, 0xa5, b_capacity));
    CHECK(mcMemcpy(gather_device, gather.data(), gather_max * sizeof(int),
                     mcMemcpyHostToDevice));
    CHECK(mcMemset(out, 0, programs_max * kBlockThreads * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"grouped_operand_addressing_residency_v1\",");
    std::printf("\"block_m\":%d,\"block_n\":%d,\"block_k\":%d,", kBlockM, kBlockN, kBlockK);
    std::printf("\"block_threads\":%d,\"element_bytes\":%d,", kBlockThreads, kElementBytes);
    std::printf("\"a_bytes\":%zu,\"b_bytes\":%zu,", a_capacity, b_capacity);
    std::printf("\"gather_permutation\":\"(i*2654435761)%%rows\",");
    std::printf("\"ordinals\":[");
    for (size_t index = 0; index < configs.size(); index++) {
        const Config& c = configs[index];
        const size_t m_tiles = static_cast<size_t>(c.experts) * c.tiles_per_expert;
        const size_t programs = m_tiles * c.n_tiles;
        const double logical = static_cast<double>(programs) * c.k_steps
            * (c.rows + kBlockN) * kBlockK * kElementBytes;
        if (index != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"experts\":%d,"
                    "\"tiles_per_expert\":%d,\"m_tiles\":%zu,\"n_tiles\":%d,"
                    "\"k_steps\":%d,\"gsm\":%d,\"rows\":%d,"
                    "\"strided_weights\":%d,\"gathered_activations\":%d,"
                    "\"programs\":%zu,\"logical_request_bytes\":%.0f}",
                    index, c.axis, c.experts, c.tiles_per_expert, m_tiles,
                    c.n_tiles, c.k_steps, c.gsm, c.rows,
                    c.strided_weights, c.gathered_activations,
                    programs, logical);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        const size_t m_tiles = static_cast<size_t>(c.experts) * c.tiles_per_expert;
        const size_t programs = m_tiles * c.n_tiles;
        grouped_operand_addressing<<<programs, kBlockThreads>>>(
            a, b, gather_device, out, static_cast<int>(m_tiles), c.n_tiles,
            c.k_steps, c.gsm, c.tiles_per_expert, c.rows,
            c.strided_weights, c.gathered_activations);
        CHECK(mcDeviceSynchronize());
    }

    CHECK(mcProfilerStop());

    CHECK(mcFree(a));
    CHECK(mcFree(b));
    CHECK(mcFree(gather_device));
    CHECK(mcFree(out));
    return 0;
}
