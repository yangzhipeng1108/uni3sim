/*
 * MetaX C500 production-stream phase sharing law benchmark.
 *
 * Round one of this mechanism (concurrent_phase_sharing_law.cpp) swept the
 * dispatch-physical axes -- k depth, spacing, sharer count, resident CTA
 * count, per-step bytes, dispatch width -- around the production sharing
 * structure with synthetic contiguous slices, and the hardware realised
 * rA ~= 5.6 at the production point everywhere, against the production value
 * of 2.0-2.7.  The synthetic-stream family therefore over-realises the
 * activation sharing it set out to discount, at every value of every axis
 * that is expressible without the production stream geometry.
 *
 * What the production stream has and no probe so far carried:
 *   - the activation operand is a gather of BLOCK_SIZE_M token rows picked
 *     by sorted_token_ids out of the original [TOK, K] activation buffer, so
 *     each row's k stream is a 64 B half-line per step and the rows of one
 *     tile are scattered over the whole buffer, with TOPK rows read once per
 *     routing (each token row is shared across TOPK experts);
 *   - the weight operand is a column block of an [E, K, N] weight tensor, so
 *     each k step reads BK rows of 128 B spaced N*2 bytes apart;
 *   - the kernel writes its C tile (8 KiB per CTA) through the same L2;
 *   - the operand balance is A:B ~= 1:6 by unique bytes, not 1:1;
 *   - the grid is expert-major, 46080 CTAs for the controlled shape.
 *
 * This probe carries all of that at the production point of the controlled
 * shape FUSEDMOE_TOK8192_E32_TOPK4_H2880_INT2880 and toggles each difference
 * off alone, so whichever toggle moves the measured ratio toward the
 * production value carries the mechanism.  The decisive comparison is the
 * GSM curve: production measures 0.573 / 0.439 / 0.312 / 0.446 at
 * GSM = 1 / 2 / 4 / 8 / 16 (with the GSM=16 uptick when spacing x per-program
 * bytes crosses the L2 replacement capacity).
 *
 * It carries no routing, no expert arithmetic, no MMA and no duration
 * pricing; it is a geometry-faithful load/store stream, not a replication of
 * the production operator.
 *
 * Traffic counters only.  No duration here may price a Forward path, and the
 * toggle-to-traffic relation must not be fitted or interpolated.
 *
 * mcProfiler stops emitting per-kernel reports after about fifty kernels in
 * one marker window, so the sweep is split by MIDLAYER_PROBE_PART (0, 1, 2, 3):
 * part 0 the GSM curve and the structure toggles, part 1 the depth, resident
 * count and work axes, part 2 the stream-resolved decomposition.  Parts 0 and
 * 1 repeat the base configuration.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o production_stream_phase_law \
 *       production_stream_phase_law.cpp
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
/* Production BM64/BN64/BK32 tile signature of the controlled shape. */
static const int kBM = 64;          /* activation rows per program */
static const int kBN = 64;          /* weight columns per program */
static const int kBK = 32;          /* k rows per step */
static const int kExperts = 32;     /* E of the controlled shape */
static const int kMTilesPerExpert = 16;   /* 1024 routed rows / BM */
static const int kNTiles = 90;      /* N / BN, W1 with fused gate+up */
static const int kMaxK = 5760;      /* depth axis bound, in elements */
static const int kProductionK = 2880;
static const int kARowTotal = 32768;   /* TOK x TOPK slot rows */
static const int kMaxBPad = 8;         /* weight row-stride pad bound, lines */
static const int kWorkers = kExperts * kMTilesPerExpert * kNTiles;
static const int kAObjects = kExperts * kMTilesPerExpert;   /* 512 */
static const int kAMapStride = kAObjects * kBM;
static const int kAVec = 4 * kBM;   /* 256: A uint4 per k step */
static const int kBVec = 8 * kBK;   /* 256: B uint4 per k step */
/* a_map selector: 0 gather with TOPK duplication (production), 1 permutation
 * without duplication, 2 contiguous identity (no lookup). */
static const int kMapCount = 2;

__global__ __launch_bounds__(kBlockThreads) void production_stream(
    const uint4* __restrict__ buf_a,   /* [kARowTotal, kMaxK/8] */
    const uint4* __restrict__ buf_b,   /* [kExperts, kMaxK, kNTiles*kBN/8] */
    uint4* __restrict__ buf_c,         /* [kWorkers, 512] */
    const int32_t* __restrict__ row_maps,  /* [kMapCount, kAObjects, kBM] */
    uint32_t* __restrict__ out,
    int steps,            /* K / kBK */
    int k_extent,         /* K / 8, row stride of buf_a in uint4 */
    int gsm,
    int a_map,
    int b_contig,         /* 0 column block of [E,K,N], 1 per-object slab */
    int c_write,          /* 0 skip, 1 write the C tile */
    int work,             /* dependent FMAs per thread per k step */
    int stages,
    int a_skip,           /* 0 load the activation stream, 1 skip it entirely */
    int b_skip,           /* 0 load the weight stream, 1 skip it entirely */
    int b_pad             /* extra lines added to the weight row stride */
) {
    extern __shared__ uint4 stage[];
    const int pid = blockIdx.x;
    const int lane = threadIdx.x;
    /* Expert-major dispatch, pid_m fastest inside a GROUP_SIZE_M block,
     * which is the launch fact the miss core replays: the weight tile is
     * shared by GSM adjacent programs, the activation tile by kNTiles
     * programs spaced GSM apart. */
    const int expert_span = kMTilesPerExpert * kNTiles;
    const int e = pid / expert_span;
    const int j = pid % expert_span;
    const int block = gsm * kNTiles;
    const int g = j / block;
    const int r = j % block;
    const int m_local = g * gsm + (r % gsm);
    const int n = r / gsm;
    const int a_obj = e * kMTilesPerExpert + m_local;
    const int b_obj = e * kNTiles + n;
    const int row_base = a_obj * kBM;
    const int b_col = n * (kBN / 8);
    const int b_row_stride = kNTiles * (kBN / 8);

    uint32_t checksum = 0u;
    for (int s = 0; s < steps + stages; s++) {
        if (s < steps) {
            const int slot = (s % stages) * (kAVec + kBVec);
            if (!a_skip && lane < kBM * 4) {
                /* Four threads per activation row, one uint4 (64 B per row
                 * per k step -- the BK x bf16 half-line). */
                const int rr = lane >> 2;
                const int c = lane & 3;
                int row;
                if (a_map == 2) {
                    row = row_base + rr;
                } else {
                    row = row_maps[(size_t)a_map * kAMapStride + row_base
                                   + rr];
                }
                stage[slot + lane] =
                    buf_a[(size_t)row * k_extent + s * 4 + c];
            }
            if (!b_skip && lane < kBK * 8) {
                /* Eight threads per weight k row, one uint4 (128 B). */
                const int kr = lane >> 3;
                const int c8 = lane & 7;
                size_t idx;
                if (b_contig) {
                    /* Per-object contiguous slab [steps*kBK, kBN]. */
                    idx = (size_t)b_obj * ((size_t)steps * kBK * (kBN / 8))
                        + (size_t)(s * kBK + kr) * (kBN / 8) + c8;
                } else {
                    /* Column block of the [E, K, N] tensor: rows spaced
                     * N*2 bytes apart, the production weight layout.  The
                     * pad widens that stride by whole lines, breaking the
                     * stride's line periodicity without touching bytes,
                     * order or footprint. */
                    idx = (size_t)(e * kMaxK + s * kBK + kr)
                        * (b_row_stride + b_pad)
                        + b_col + c8;
                }
                stage[slot + kAVec + lane] = buf_b[idx];
            }
        }
        const int c_step = s - (stages - 1);
        if (c_step >= 0 && c_step < steps) {
            __syncthreads();
            const int slot = (c_step % stages) * (kAVec + kBVec);
            for (int i = lane; i < kAVec + kBVec; i += kBlockThreads) {
                const uint4 v = stage[slot + i];
                checksum += v.x ^ v.w;
            }
            float acc = (float)(checksum & 0xffff) + 1.0f;
            for (int i = 0; i < work; i++) {
                acc = acc * 1.0000001f + 0.5f;
            }
            checksum += (uint32_t)acc;
        }
    }
    /* C epilogue: BM x BN x 2 B = 8 KiB per program, written once through
     * the same L2 the operand streams live in. */
    if (c_write) {
        for (int i = lane; i < 512; i += kBlockThreads) {
            buf_c[(size_t)pid * 512 + i] =
                make_uint4(checksum, checksum, checksum, checksum);
        }
    }
    out[pid] = checksum;
}

struct Config {
    int k_extent_el;   /* K in elements; steps = K / kBK */
    int gsm, a_map, b_contig, c_write, work, stages;
    const char* axis;
    int a_skip, b_skip;   /* stream isolation controls; absent means zero */
    int b_pad;            /* extra lines on the weight row stride; 0 default */
};

static int steps_of(const Config& c) { return c.k_extent_el / kBK; }

int main() {
    const char* part_text = std::getenv("MIDLAYER_PROBE_PART");
    const int part = part_text ? std::atoi(part_text) : 0;

    /* The production point of the controlled shape: gather-dup activations,
     * column-block weights, C written, BM64xBN64xBK32 MMA-equivalent work,
     * GSM = 8, K = 2880, stages = 2 (16 KiB shared, 4 blocks per AP). */
    const Config base{kProductionK, 8, 0, 0, 1, 512, 2, "base"};

    std::vector<Config> configs;
    for (int rep = 0; rep < 3; rep++) configs.push_back(base);

    if (part == 0) {
        /* Axis GSM -- the decisive curve.  Production measures 0.573 /
         * 0.439 / 0.311 / 0.312 / 0.446 at GSM 1 / 2 / 4 / 8 / 16.  GSM = 8
         * is the base carried in every part. */
        for (int gsm : {1, 2, 4, 16}) {
            configs.push_back({kProductionK, gsm, 0, 0, 1, 512, 2, "gsm"});
        }
        /* Toggle: activation layout.  gather-dup (production) against a
         * row permutation without duplication and the contiguous identity,
         * same bytes and same order inside each CTA. */
        configs.push_back({kProductionK, 8, 1, 0, 1, 512, 2, "a_scatter_nodup"});
        configs.push_back({kProductionK, 8, 2, 0, 1, 512, 2, "a_contiguous"});
        /* Toggle: the C write stream. */
        configs.push_back({kProductionK, 8, 0, 0, 0, 512, 2, "no_c_write"});
        /* Toggle: the weight layout. */
        configs.push_back({kProductionK, 8, 0, 1, 1, 512, 2, "b_contig"});
        /* Toggle: per-k-step arithmetic. */
        configs.push_back({kProductionK, 8, 0, 0, 1, 0, 2, "work0"});
    } else if (part == 1) {
        /* Axis K -- the k depth, at the faithful geometry. */
        configs.push_back({1440, 8, 0, 0, 1, 512, 2, "k_depth"});
        configs.push_back({5760, 8, 0, 0, 1, 512, 2, "k_depth"});
        /* Axis R -- resident CTA count through the shared-memory footprint:
         * stages 1/2/3 = 8/16/24 KiB = 8/4/2 blocks per AP. */
        for (int stg : {1, 3}) {
            configs.push_back({kProductionK, 8, 0, 0, 1, 512, stg, "stages"});
        }
        /* Axis D -- work swept past and below the production point. */
        configs.push_back({kProductionK, 8, 0, 0, 1, 64, 2, "work"});
        configs.push_back({kProductionK, 8, 0, 0, 1, 2048, 2, "work"});
        /* The GSM=16 corner with the deep k extent, where the production
         * spacing x per-program bytes argument crosses capacity by the
         * largest margin. */
        configs.push_back({5760, 16, 0, 0, 1, 512, 2, "gsm16_deep"});
    } else if (part == 2) {
        /* Part 2: stream-resolved decomposition.  The part-1 total ratio moves
         * non-monotonically with K and collapses under stages 3 and work 2048,
         * which no per-stream statement explains yet.  The skip flags load one
         * stream entirely, with the C write off so a skipped run reads exactly
         * one stream, and the per-stream DRAM passes fall out differenced
         * instead of solved. */
        for (int rep = 0; rep < 2; rep++) {
            configs.push_back({kProductionK, 8, 0, 0, 1, 512, 2, "base"});
        }
        configs.push_back({kProductionK, 8, 0, 0, 0, 512, 2, "a_only",
                           0, 1});
        configs.push_back({kProductionK, 8, 0, 0, 0, 512, 2, "b_only",
                           1, 0});
        for (int k : {1440, 5760}) {
            configs.push_back({k, 8, 0, 0, 1, 512, 2, "k_depth"});
            configs.push_back({k, 8, 0, 0, 0, 512, 2, "k_depth_a_only",
                               0, 1});
            configs.push_back({k, 8, 0, 0, 0, 512, 2, "k_depth_b_only",
                               1, 0});
        }
        for (int gsm : {1, 16}) {
            configs.push_back({kProductionK, gsm, 0, 0, 0, 512, 2,
                               "gsm_a_only", 0, 1});
            configs.push_back({kProductionK, gsm, 0, 0, 0, 512, 2,
                               "gsm_b_only", 1, 0});
        }
        /* The two regimes the total ratio collapsed in, decomposed. */
        configs.push_back({kProductionK, 8, 0, 0, 0, 512, 3,
                           "stages3_a_only", 0, 1});
        configs.push_back({kProductionK, 8, 0, 0, 0, 512, 3,
                           "stages3_b_only", 1, 0});
        configs.push_back({kProductionK, 8, 0, 0, 0, 2048, 2,
                           "work2048_a_only", 0, 1});
        configs.push_back({kProductionK, 8, 0, 0, 0, 2048, 2,
                           "work2048_b_only", 1, 0});
        /* Layout toggles decomposed: whether each stream's deficit rides on
         * its own layout or on the sharing structure. */
        configs.push_back({kProductionK, 8, 0, 1, 0, 512, 2,
                           "b_contig_b_only", 1, 0});
        configs.push_back({kProductionK, 8, 2, 0, 0, 512, 2,
                           "a_contig_a_only", 0, 1});
    } else if (part == 3) {
        /* Part 3: stride-periodicity resonance.  The production weight layout
         * spaces its k rows exactly N*2 bytes apart, so a column block's 32
         * rows hit addresses whose line offsets advance by one fixed stride
         * (kNTiles * kBN / 16 = 90 lines) every step.  Every other candidate
         * carrier of the co-resident interference has now been differenced
         * out: each single stream realises its ideal law (B exactly 2 passes,
         * A the distance law), any LRU interleave replays to the no-
         * interference total, and the interference itself scales with the
         * resident count.  What remains is the mapping of these strides onto
         * the cache's index geometry, which only a stride change can probe.
         * The pad widens the weight row stride by whole lines: the bytes, the
         * order, the footprint, the sharer structure and the per-CTA stream
         * all stay identical, only the line periodicity of the stride moves.
         * If the interference collapses under any nonzero pad it is a
         * mapping resonance, not a capacity effect. */
        for (int rep = 0; rep < 2; rep++) {
            configs.push_back({kProductionK, 8, 0, 0, 1, 512, 2, "base"});
        }
        for (int pad : {1, 2, 4, 8}) {
            configs.push_back({kProductionK, 8, 0, 0, 1, 512, 2,
                               "b_pad", 0, 0, pad});
        }
        /* The deepest k extent carries the widest stride and the strongest
         * interference per part-1, so it discriminates the periodicity most. */
        configs.push_back({5760, 8, 0, 0, 1, 512, 2, "b_pad", 0, 0, 1});
        configs.push_back({5760, 8, 0, 0, 1, 512, 2, "b_pad", 0, 0, 8});
        /* Single-stream invariance under pad: the weight stream alone must
         * keep its exact 2 passes, otherwise the pad leaks into capacity. */
        configs.push_back({kProductionK, 8, 0, 0, 0, 512, 2,
                           "b_pad_b_only", 1, 0, 8});
        /* Anchor the b_contig full-run point next to its padded peers. */
        configs.push_back({kProductionK, 8, 0, 1, 1, 512, 2, "b_contig"});
    }

    /* Two row maps, uploaded before the profiler window opens: the routed
     * map (each of TOK=8192 token rows appears TOPK=4 times, scattered over
     * the 32768 slots) and a permutation of all 32768 slots with no
     * duplication.  The contiguous map needs no table. */
    std::vector<int32_t> maps((size_t)kMapCount * kAMapStride, -1);
    {
        /* Map 0: routed gather.  Slots of one A object hold token rows; the
         * slot-to-token assignment is a shuffle of TOK x TOPK slots over TOK
         * rows with each row exactly TOPK times. */
        std::vector<int32_t> slots((size_t)kARowTotal);
        for (int i = 0; i < kARowTotal; i++) slots[i] = i % (kARowTotal / 4);
        uint64_t state = 0x9e3779b97f4a7c15ull;
        for (size_t i = slots.size(); i > 1; i--) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            std::swap(slots[i - 1], slots[state % i]);
        }
        for (size_t i = 0; i < slots.size(); i++) maps[i] = slots[i];
        /* Map 1: permutation of all slots, no duplication. */
        for (int i = 0; i < kARowTotal; i++)
            maps[(size_t)kAMapStride + i] = i;
        state = 0x6a09e667f3bcc909ull;
        for (size_t i = (size_t)kARowTotal; i > 1; i--) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            std::swap(maps[(size_t)kAMapStride + i - 1],
                      maps[(size_t)kAMapStride + state % i]);
        }
    }

    int32_t* row_maps = nullptr;
    uint4 *buf_a = nullptr, *buf_b = nullptr, *buf_c = nullptr;
    uint32_t* out = nullptr;
    /* The weight buffer carries the widest row stride plus the pad headroom,
     * so a padded run stays inside the allocation. */
    const size_t b_vecs = (size_t)kExperts * kMaxK
        * (kNTiles * kBN / 8 + kMaxBPad);
    CHECK(mcMalloc(&row_maps, maps.size() * sizeof(int32_t)));
    CHECK(mcMalloc(&buf_a,
                   (size_t)kARowTotal * (kMaxK / 8) * sizeof(uint4)));
    CHECK(mcMalloc(&buf_b, b_vecs * sizeof(uint4)));
    CHECK(mcMalloc(&buf_c, (size_t)kWorkers * 512 * sizeof(uint4)));
    CHECK(mcMalloc(&out, (size_t)kWorkers * sizeof(uint32_t)));
    CHECK(mcMemcpy(row_maps, maps.data(), maps.size() * sizeof(int32_t),
                   mcMemcpyHostToDevice));
    CHECK(mcMemset(buf_a, 0x5a,
                   (size_t)kARowTotal * (kMaxK / 8) * sizeof(uint4)));
    CHECK(mcMemset(buf_b, 0xa5, b_vecs * sizeof(uint4)));
    CHECK(mcMemset(buf_c, 0, (size_t)kWorkers * 512 * sizeof(uint4)));
    CHECK(mcMemset(out, 0, (size_t)kWorkers * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"production_stream_phase_law_v1\",");
    std::printf("\"part\":%d,\"workers\":%d,\"block_threads\":%d,",
                part, kWorkers, kBlockThreads);
    std::printf("\"bm\":%d,\"bn\":%d,\"bk\":%d,", kBM, kBN, kBK);
    std::printf("\"experts\":%d,\"m_tiles_per_expert\":%d,\"n_tiles\":%d,",
                kExperts, kMTilesPerExpert, kNTiles);
    std::printf("\"l2_replacement_capacity_bytes\":8388608,");
    std::printf("\"production_gsm_curve\":{\"1\":0.573,\"2\":0.439,"
                "\"4\":0.311,\"8\":0.312,\"16\":0.446},");
    std::printf("\"ordinals\":[");
    for (size_t i = 0; i < configs.size(); i++) {
        const Config& c = configs[i];
        const int steps = steps_of(c);
        const double logical_a = c.a_skip
            ? 0.0 : (double)kWorkers * steps * (kAVec * 16);
        const double logical_b = c.b_skip
            ? 0.0 : (double)kWorkers * steps * (kBVec * 16);
        /* Unique activation bytes: the routed gather touches every token
         * row TOPK times (TOK x K), the other maps touch every slot row
         * once (TOK x TOPK x K).  Unique weight bytes: the whole weight
         * footprint of the experts.  A skipped stream contributes none. */
        const double unique_a = c.a_skip
            ? 0.0
            : (c.a_map == 0)
                ? (double)(kARowTotal / 4) * c.k_extent_el * 2
                : (double)kARowTotal * c.k_extent_el * 2;
        const double unique_b = c.b_skip
            ? 0.0
            : (double)kExperts * c.k_extent_el * kNTiles * kBN * 2;
        const double per_cta = (double)steps * (kAVec + kBVec) * 16;
        if (i != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"k_elements\":%d,"
                    "\"steps\":%d,\"gsm\":%d,\"a_map\":%d,"
                    "\"a_map_name\":\"%s\","
                    "\"b_contig\":%d,\"c_write\":%d,\"work\":%d,"
                    "\"stages\":%d,\"a_skip\":%d,\"b_skip\":%d,\"b_pad\":%d,"
                    "\"smem_bytes\":%d,"
                    "\"per_cta_stream_bytes\":%.0f,"
                    "\"a_sharer_count\":%d,\"a_sharer_spacing\":%d,"
                    "\"b_sharer_count\":%d,\"b_sharer_spacing\":%d,"
                    "\"a_reuse_distance_bytes\":%.0f,"
                    "\"logical_a_bytes\":%.0f,\"logical_b_bytes\":%.0f,"
                    "\"unique_a_bytes\":%.0f,\"unique_b_bytes\":%.0f,"
                    "\"logical_total_bytes\":%.0f,"
                    "\"c_bytes\":%.0f}",
                    i, c.axis, c.k_extent_el, steps, c.gsm, c.a_map,
                    c.a_map == 0 ? "gather_dup"
                        : (c.a_map == 1 ? "scatter_nodup" : "contiguous"),
                    c.b_contig, c.c_write, c.work, c.stages,
                    c.a_skip, c.b_skip, c.b_pad,
                    c.stages * (kAVec + kBVec) * 16, per_cta,
                    kNTiles, c.gsm == 1 ? 1 : c.gsm, c.gsm,
                    c.gsm == 1 ? kNTiles : 1,
                    (double)(c.gsm == 1 ? 1 : c.gsm) * per_cta,
                    logical_a, logical_b, unique_a, unique_b,
                    logical_a + logical_b,
                    c.c_write ? (double)kWorkers * 8192.0 : 0.0);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        const size_t smem =
            (size_t)c.stages * (kAVec + kBVec) * sizeof(uint4);
        production_stream<<<kWorkers, kBlockThreads, smem>>>(
            buf_a, buf_b, buf_c, row_maps, out, steps_of(c),
            c.k_extent_el / 8, c.gsm, c.a_map, c.b_contig, c.c_write,
            c.work, c.stages, c.a_skip, c.b_skip, c.b_pad);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());

    CHECK(mcFree(row_maps));
    CHECK(mcFree(buf_a));
    CHECK(mcFree(buf_b));
    CHECK(mcFree(buf_c));
    CHECK(mcFree(out));
    return 0;
}
