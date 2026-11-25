/*
 * MetaX C500 weight-sharing phase-spread stamps.
 *
 * Timing-instrumented twin of stream_sharing_interference.cpp: identical
 * configurations and address streams, plus one global-realtime stamp per CTA
 * at k steps 0, S/4, S/2, 3S/4 and S.  The stamps order the access stream as
 * the hardware issued it, so the order can be replayed offline through a
 * replacement model; the spread of one weight tile's GSM sharers at one step,
 * in steps, is the phase quantity the traffic depends on.
 *
 * The stamps are mechanism evidence about execution order inside this
 * microbenchmark.  No duration here may price a Forward path, and no stamp
 * statistic may be fitted or interpolated into a predictor.  The added stamp
 * store changes code shape, so this binary's DRAM counts are not substituted
 * for the counter binary's.
 *
 * Build and run inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o stream_sharing_phase_stamps \
 *       stream_sharing_phase_stamps.cpp
 *   MIDLAYER_PROBE_PART=<0|1> ./stream_sharing_phase_stamps   # writes stamps_<axis>.bin
 */

#include <mc_runtime.h>

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
static const int kBM = 64;
static const int kBN = 64;
static const int kBK = 32;
static const int kExperts = 32;
static const int kMTilesPerExpert = 16;
static const int kNTiles = 90;
static const int kK = 2880;
static const int kSteps = kK / kBK;
static const int kARowTotal = 32768;
static const int kWorkers = kExperts * kMTilesPerExpert * kNTiles;
static const int kAObjects = kExperts * kMTilesPerExpert;
static const int kAVec = 4 * kBM;
static const int kBVec = 8 * kBK;
static const int kStages = 2;
/* 1 MiB hot region: 182 rows of the K=2880 activation row. */
static const int kHotRows = (1 << 20) / (kK * 2);
/* 2 GiB cold ring, in 128-B lines. */
static const long long kRingLines = (2LL << 30) / 128;
/* Routed-gather maps (a_mode 0): see main(). */
static const int kMaps = 5;

__global__ __launch_bounds__(kBlockThreads) void sharing_interference(
    const uint4* __restrict__ buf_a,
    const uint4* __restrict__ buf_b,
    const uint4* __restrict__ ring,
    uint4* __restrict__ buf_c,
    const int32_t* __restrict__ row_map,
    uint32_t* __restrict__ out,
    int gsm, int a_mode, int a_skip, int b_skip, int c_write, int map_sel,
    unsigned long long* stamps
) {
    extern __shared__ uint4 stage[];
    const int pid = blockIdx.x;
    const int lane = threadIdx.x;
    const int expert_span = kMTilesPerExpert * kNTiles;
    const int e = pid / expert_span;
    const int j = pid % expert_span;
    const int block = gsm * kNTiles;
    const int g = j / block;
    const int r = j % block;
    const int m_local = g * gsm + (r % gsm);
    const int n = r / gsm;
    const int a_obj = e * kMTilesPerExpert + m_local;
    const int row_base = a_obj * kBM;
    const int b_col = n * (kBN / 8);
    const int b_row_stride = kNTiles * (kBN / 8);
    const int k_extent = kK / 8;
    const int a_rows = a_mode == 3 ? kBM / 4 : kBM;

    uint32_t checksum = 0u;
    for (int s = 0; s < kSteps + kStages; s++) {
        if (lane == 0 && (s == 0 || s == kSteps/4 || s == kSteps/2 || s == 3*kSteps/4 || s == kSteps)) {
            const int k = s == 0 ? 0 : s == kSteps/4 ? 1 : s == kSteps/2 ? 2 : s == 3*kSteps/4 ? 3 : 4;
            stamps[(size_t)pid * 5 + k] = __builtin_mxc_get_realtime();
        }
        if (s < kSteps) {
            const int slot = (s % kStages) * (kAVec + kBVec);
            if (!a_skip && lane < a_rows * 4) {
                const int rr = lane >> 2;
                const int c = lane & 3;
                uint4 v;
                if (a_mode == 0) {
                    const int row = row_map[(size_t)map_sel * kARowTotal + row_base + rr];
                    v = buf_a[(size_t)row * k_extent + s * 4 + c];
                } else if (a_mode == 1) {
                    const int row = (row_base + rr) % kHotRows;
                    v = buf_a[(size_t)row * k_extent + s * 4 + c];
                } else {
                    /* One never-reused line per (program, step, row); the
                     * first 64 B of it, i.e. the same half-line request. */
                    const long long line =
                        (((long long)pid * kSteps + s) * kBM + rr)
                        % kRingLines;
                    v = ring[line * 8 + c];
                }
                stage[slot + lane] = v;
            }
            if (!b_skip && lane < kBK * 8) {
                const int kr = lane >> 3;
                const int c8 = lane & 7;
                const size_t idx = (size_t)(e * kK + s * kBK + kr)
                    * b_row_stride + b_col + c8;
                stage[slot + kAVec + lane] = buf_b[idx];
            }
        }
        const int c_step = s - (kStages - 1);
        if (c_step >= 0 && c_step < kSteps) {
            __syncthreads();
            const int slot = (c_step % kStages) * (kAVec + kBVec);
            for (int i = lane; i < kAVec + kBVec; i += kBlockThreads) {
                const uint4 v = stage[slot + i];
                checksum += v.x ^ v.w;
            }
            float acc = (float)(checksum & 0xffff) + 1.0f;
            for (int i = 0; i < 512; i++) acc = acc * 1.0000001f + 0.5f;
            checksum += (uint32_t)acc;
        }
    }
    if (c_write) {
        for (int i = lane; i < 512; i += kBlockThreads) {
            buf_c[(size_t)pid * 512 + i] =
                make_uint4(checksum, checksum, checksum, checksum);
        }
    }
    if (checksum == 0x9e3779b9u) out[pid] = checksum;
}

struct Config { const char* axis; int gsm, a_mode, a_skip, b_skip, c_write, map_sel; };

int main() {
    const char* part_text = std::getenv("MIDLAYER_PROBE_PART");
    const int part = part_text ? std::atoi(part_text) : 0;
    std::vector<Config> configs;
    if (part == 0) {
        configs = {
            {"base",            8, 0, 0, 0, 1, 0},
            {"b_only",          8, 0, 1, 0, 0, 0},
            {"hot_a_with_b",    8, 1, 0, 0, 0, 0},
            {"hot_a_only",      8, 1, 0, 1, 0, 0},
            {"cold_a_with_b",   8, 2, 0, 0, 0, 0},
            {"cold_a_only",     8, 2, 0, 1, 0, 0},
            {"cold_q_with_b",   8, 3, 0, 0, 0, 0},
            {"cold_q_only",     8, 3, 0, 1, 0, 0},
            {"gather_with_b",   8, 0, 0, 0, 0, 0},
            {"gather_only",     8, 0, 0, 1, 0, 0},
            {"base_repeat",     8, 0, 0, 0, 1, 0},
        };
    } else {
        /* Part 1 separates the two things the routed map changes at once:
         * how often a token row repeats and how large the address range
         * holding the rows is. */
        configs = {
            {"dup4_47m_with_b",   8, 0, 0, 0, 0, 0},
            {"dup4_47m_only",     8, 0, 0, 1, 0, 0},
            {"nodup_189m_with_b", 8, 0, 0, 0, 0, 1},
            {"nodup_189m_only",   8, 0, 0, 1, 0, 1},
            {"dup4_189m_with_b",  8, 0, 0, 0, 0, 2},
            {"dup4_189m_only",    8, 0, 0, 1, 0, 2},
            {"dup2_94m_with_b",   8, 0, 0, 0, 0, 3},
            {"dup2_94m_only",     8, 0, 0, 1, 0, 3},
            {"dup16_12m_with_b",  8, 0, 0, 0, 0, 4},
            {"dup16_12m_only",    8, 0, 0, 1, 0, 4},
            {"b_only",            8, 0, 1, 0, 0, 0},
            {"dup4_47m_with_b_r", 8, 0, 0, 0, 0, 0},
        };
    }

    /* Routed-gather maps over the TOK x TOPK slots, each a shuffle with the
     * phase-law probe's generator:
     *   0 dup4, rows [0, 8192) each 4 times        (47 MiB range, production)
     *   1 no duplication, rows [0, 32768)          (189 MiB range)
     *   2 dup4, rows 4*[0, 8192)                   (189 MiB range, same dup)
     *   3 dup2, rows [0, 16384) each twice         (94 MiB range)
     *   4 dup16, rows [0, 2048) each 16 times      (12 MiB range) */
    std::vector<int32_t> map((size_t)kMaps * kARowTotal);
    const int distinct[kMaps] = {kARowTotal / 4, kARowTotal, kARowTotal / 4,
                                 kARowTotal / 2, kARowTotal / 16};
    const int spread[kMaps] = {1, 1, 4, 1, 1};
    for (int m = 0; m < kMaps; m++) {
        int32_t* slots = map.data() + (size_t)m * kARowTotal;
        for (int i = 0; i < kARowTotal; i++)
            slots[i] = (i % distinct[m]) * spread[m];
        uint64_t state = 0x9e3779b97f4a7c15ull + (uint64_t)m;
        for (size_t i = kARowTotal; i > 1; i--) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            std::swap(slots[i - 1], slots[state % i]);
        }
    }

    int32_t* row_map = nullptr;
    uint4 *buf_a = nullptr, *buf_b = nullptr, *ring = nullptr, *buf_c = nullptr;
    uint32_t* out = nullptr;
    const size_t a_vecs = (size_t)kARowTotal * (kK / 8);
    const size_t b_vecs = (size_t)kExperts * kK * (kNTiles * kBN / 8);
    CHECK(mcMalloc(&row_map, map.size() * sizeof(int32_t)));
    CHECK(mcMalloc(&buf_a, a_vecs * sizeof(uint4)));
    CHECK(mcMalloc(&buf_b, b_vecs * sizeof(uint4)));
    CHECK(mcMalloc(&ring, (size_t)kRingLines * 128));
    CHECK(mcMalloc(&buf_c, (size_t)kWorkers * 512 * sizeof(uint4)));
    CHECK(mcMalloc(&out, (size_t)kWorkers * sizeof(uint32_t)));
    CHECK(mcMemcpy(row_map, map.data(), map.size() * sizeof(int32_t),
                   mcMemcpyHostToDevice));
    CHECK(mcMemset(buf_a, 0x5a, a_vecs * sizeof(uint4)));
    CHECK(mcMemset(buf_b, 0xa5, b_vecs * sizeof(uint4)));
    CHECK(mcMemset(ring, 0x3c, (size_t)kRingLines * 128));
    CHECK(mcMemset(buf_c, 0, (size_t)kWorkers * 512 * sizeof(uint4)));
    CHECK(mcMemset(out, 0, (size_t)kWorkers * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"stream_sharing_interference_v1\",\"part\":%d,"
                "\"workers\":%d,\"k\":%d,\"steps\":%d,\"stages\":%d,"
                "\"hot_rows\":%d,\"ring_lines\":%lld,\"ordinals\":[",
                part, kWorkers, kK, kSteps, kStages, kHotRows, kRingLines);
    for (size_t i = 0; i < configs.size(); i++) {
        const Config& c = configs[i];
        const int a_rows = c.a_mode == 3 ? kBM / 4 : kBM;
        std::printf("%s{\"ordinal\":%zu,\"axis\":\"%s\",\"gsm\":%d,"
                    "\"a_mode\":%d,\"a_skip\":%d,\"b_skip\":%d,"
                    "\"c_write\":%d,\"map_sel\":%d,\"a_requests\":%.0f,"
                    "\"b_requests\":%.0f}",
                    i ? "," : "", i, c.axis, c.gsm, c.a_mode, c.a_skip,
                    c.b_skip, c.c_write, c.map_sel,
                    c.a_skip ? 0.0 : (double)kWorkers * kSteps * a_rows,
                    c.b_skip ? 0.0 : (double)kWorkers * kSteps * kBK);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    const size_t smem = (size_t)kStages * (kAVec + kBVec) * sizeof(uint4);
    unsigned long long* stamps = nullptr;
    CHECK(mcMalloc(&stamps, (size_t)kWorkers * 5 * 8));
    std::vector<unsigned long long> h((size_t)kWorkers * 5);
    for (size_t ci = 0; ci < configs.size(); ci++) {
        const Config& c = configs[ci];
        for (int rep = 0; rep < 2; rep++) {
            sharing_interference<<<kWorkers, kBlockThreads, smem>>>(
                buf_a, buf_b, ring, buf_c, row_map, out,
                c.gsm, c.a_mode, c.a_skip, c.b_skip, c.c_write, c.map_sel, stamps);
            CHECK(mcGetLastError());
            CHECK(mcDeviceSynchronize());
        }
        CHECK(mcMemcpy(h.data(), stamps, h.size() * 8, mcMemcpyDeviceToHost));
        char fn[256]; std::snprintf(fn, sizeof fn, "stamps_%s.bin", c.axis);
        FILE* f = std::fopen(fn, "wb"); std::fwrite(h.data(), 8, h.size(), f); std::fclose(f);
    }
    CHECK(mcFree(row_map));
    CHECK(mcFree(buf_a));
    CHECK(mcFree(buf_b));
    CHECK(mcFree(ring));
    CHECK(mcFree(buf_c));
    CHECK(mcFree(out));
    return 0;
}
