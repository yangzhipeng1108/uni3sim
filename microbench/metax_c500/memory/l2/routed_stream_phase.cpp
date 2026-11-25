/*
 * MetaX C500 production-address grouped-MoE stream probe.
 *
 * stream_sharing_interference.cpp located the weight-sharing loss in a
 * phase-locked versus desynchronised regime of CTA progress, but its
 * activation rows were random shuffles and its weights a [E, K, N] column
 * block.  The canonical dataset runner builds neither: it derives top-k ids
 * with a deterministic heap from the case's tok_per_expert, moe_align_block_size
 * sorts every expert's routes in ascending flat index, and the weights are
 * [E, N, K] with K contiguous, so a weight tile is BN rows of BK*2-byte half
 * lines.
 *
 * This probe reads a routing description written offline from that runner
 * source (sorted_token_ids, per-block expert ids, padded row count) and
 * issues exactly the kernel's address stream: the Triton GROUP_SIZE_M
 * program swizzle, early return past num_tokens_post_padded, token-masked
 * activation half lines (row = route / top_k), [E, N, K] weight half lines
 * with the modulo-N tail, and the masked C store.  No MMA, no routing
 * arithmetic, a fixed dependent FMA body per k step; it is an address
 * replica, not an operator replica.
 *
 * Built twice: without STAMPS for the mcProfiler counter window, and with
 * -DSTAMPS=1 so every CTA records the global realtime at each k step (the
 * execution order).  No duration here may price a Forward path and no stamp
 * statistic may be fitted or interpolated into a predictor.
 *
 * Usage:  routed_stream_phase <routing.bin> <projection 1|2> [stamps_prefix]
 * Build:  /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *           --maca-path=/opt/maca -O3 [-DSTAMPS=1] -o <bin> routed_stream_phase.cpp
 */

#include <mc_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef STAMPS
#define STAMPS 0
#endif

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
static const int kGSM = 8;
static const int kStages = 2;
static const int kAVec = kBM * 4;   /* 64 rows x 64 B */
static const int kBVec = kBN * 4;   /* 64 rows x 64 B */

__global__ __launch_bounds__(kBlockThreads) void routed_stream(
    const uint4* __restrict__ act, const uint4* __restrict__ wgt,
    uint4* __restrict__ out_c, const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ block_expert, uint32_t* __restrict__ sink,
    int num_pid_m, int num_pid_n, int ntpp, int valid, int top_k,
    int n_extent, int k_extent, int steps, int a_skip, int b_skip,
    int c_write, unsigned long long* __restrict__ stamps
) {
    extern __shared__ uint4 stage[];
    const int pid = blockIdx.x;
    const int lane = threadIdx.x;
    const int in_group = kGSM * num_pid_n;
    const int group = pid / in_group;
    const int first_m = group * kGSM;
    const int size_m = min(num_pid_m - first_m, kGSM);
    const int pid_m = first_m + (pid % in_group) % size_m;
    const int pid_n = (pid % in_group) / size_m;
    if (pid_m * kBM >= ntpp) return;
    const int expert = block_expert[pid_m];
    const int row = lane >> 2;
    const int part = lane & 3;
    const int route = sorted_ids[pid_m * kBM + row];
    const bool valid_row = route < valid;
    const size_t a_row = (size_t)(route / top_k);
    const int n_row = (pid_n * kBN + row) % n_extent;
    const size_t b_row = (size_t)expert * n_extent + n_row;
    const size_t row_vecs = (size_t)k_extent / 8;
    uint32_t checksum = 0u;
    for (int s = 0; s < steps + kStages; s++) {
#if STAMPS
        if (lane == 0 && s <= steps)
            stamps[(size_t)pid * (steps + 1) + s] =
                __builtin_mxc_get_realtime();
#endif
        if (s < steps) {
            const int slot = (s % kStages) * (kAVec + kBVec);
            if (!a_skip) {
                uint4 v = make_uint4(0, 0, 0, 0);
                if (valid_row) v = act[a_row * row_vecs + s * 4 + part];
                stage[slot + lane] = v;
            }
            if (!b_skip) {
                stage[slot + kAVec + lane] =
                    wgt[b_row * row_vecs + s * 4 + part];
            }
        }
        const int c_step = s - (kStages - 1);
        if (c_step >= 0 && c_step < steps) {
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
        /* BM x BN output tile: row = route, 128 B per row (8 uint4). */
        for (int i = lane; i < kBM * 8; i += kBlockThreads) {
            const int r = i >> 3;
            const int c = i & 7;
            const int rt = sorted_ids[pid_m * kBM + r];
            const int col = pid_n * kBN;
            if (rt < valid && col < n_extent)
                out_c[((size_t)rt * n_extent + col) / 8 + c] =
                    make_uint4(checksum, checksum, checksum, checksum);
        }
    }
    if (checksum == 0x9e3779b9u) sink[pid] = checksum;
}

struct Config { const char* axis; int a_skip, b_skip, c_write; };

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s routing.bin projection [prefix]\n",
                     argv[0]);
        return 2;
    }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror(argv[1]); return 2; }
    long long hd[9];
    if (std::fread(hd, 8, 9, f) != 9) return 2;
    const long long T = hd[0], E = hd[1], TOPK = hd[2], H = hd[3], I = hd[4];
    const long long BM = hd[5], ntpp = hd[6], EM = hd[7], nb = hd[8];
    if (BM != kBM) { std::fprintf(stderr, "BM64 only\n"); return 2; }
    std::vector<int32_t> sid((size_t)ntpp), eob((size_t)nb);
    if (std::fread(sid.data(), 4, sid.size(), f) != sid.size()) return 2;
    if (std::fread(eob.data(), 4, eob.size(), f) != eob.size()) return 2;
    std::fclose(f);
    const int projection = std::atoi(argv[2]);
    const long long N = projection == 2 ? H : 2 * I;
    const long long K = projection == 2 ? I : H;
    const int top_k = projection == 2 ? 1 : (int)TOPK;
    const long long a_rows = projection == 2 ? T * TOPK : T;
    const int steps = (int)((K + kBK - 1) / kBK);
    const int num_pid_m = (int)((EM + kBM - 1) / kBM);
    const int num_pid_n = (int)((N + kBN - 1) / kBN);
    const int grid = num_pid_m * num_pid_n;
    /* Padding blocks past ntpp are never read: size the id array to EM. */
    sid.resize((size_t)EM, (int32_t)(T * TOPK));
    eob.resize((size_t)num_pid_m, 0);

    std::vector<Config> configs = {
        {"routed_full", 0, 0, 1},
        {"routed_no_c", 0, 0, 0},
        {"routed_a_only", 0, 1, 0},
        {"routed_b_only", 1, 0, 0},
        {"routed_full_repeat", 0, 0, 1},
    };

    int32_t *d_sid = nullptr, *d_eob = nullptr;
    uint4 *act = nullptr, *wgt = nullptr, *out_c = nullptr;
    uint32_t* sink = nullptr;
    unsigned long long* stamps = nullptr;
    CHECK(mcMalloc(&d_sid, sid.size() * 4));
    CHECK(mcMalloc(&d_eob, eob.size() * 4));
    CHECK(mcMemcpy(d_sid, sid.data(), sid.size() * 4, mcMemcpyHostToDevice));
    CHECK(mcMemcpy(d_eob, eob.data(), eob.size() * 4, mcMemcpyHostToDevice));
    CHECK(mcMalloc(&act, (size_t)a_rows * K * 2));
    CHECK(mcMalloc(&wgt, (size_t)E * N * K * 2));
    CHECK(mcMalloc(&out_c, (size_t)T * TOPK * N * 2));
    CHECK(mcMalloc(&sink, (size_t)grid * 4));
    CHECK(mcMemset(act, 0x5a, (size_t)a_rows * K * 2));
    CHECK(mcMemset(wgt, 0xa5, (size_t)E * N * K * 2));
    CHECK(mcMemset(out_c, 0, (size_t)T * TOPK * N * 2));
#if STAMPS
    CHECK(mcMalloc(&stamps, (size_t)grid * (steps + 1) * 8));
    CHECK(mcMemset(stamps, 0, (size_t)grid * (steps + 1) * 8));
#endif
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"routed_stream_phase_v1\",\"routing\":\"%s\","
                "\"projection\":%d,\"T\":%lld,\"E\":%lld,\"topk\":%lld,"
                "\"N\":%lld,\"K\":%lld,\"steps\":%d,\"grid\":%d,"
                "\"ntpp\":%lld,\"stamps\":%d,\"ordinals\":[",
                argv[1], projection, T, E, TOPK, N, K, steps, grid, ntpp,
                STAMPS);
    for (size_t i = 0; i < configs.size(); i++)
        std::printf("%s{\"ordinal\":%zu,\"axis\":\"%s\",\"a_skip\":%d,"
                    "\"b_skip\":%d,\"c_write\":%d}", i ? "," : "", i,
                    configs[i].axis, configs[i].a_skip, configs[i].b_skip,
                    configs[i].c_write);
    std::printf("]}\n");
    std::fflush(stdout);

    const size_t smem = (size_t)kStages * (kAVec + kBVec) * sizeof(uint4);
    std::vector<unsigned long long> host;
    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        routed_stream<<<grid, kBlockThreads, smem>>>(
            act, wgt, out_c, d_sid, d_eob, sink, num_pid_m, num_pid_n,
            (int)ntpp, (int)(T * TOPK), top_k, (int)N, (int)K, steps,
            c.a_skip, c.b_skip, c.c_write, stamps);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
#if STAMPS
        if (argc > 3) {
            host.resize((size_t)grid * (steps + 1));
            CHECK(mcMemcpy(host.data(), stamps, host.size() * 8,
                           mcMemcpyDeviceToHost));
            char fn[512];
            std::snprintf(fn, sizeof fn, "%s_%s.bin", argv[3], c.axis);
            FILE* o = std::fopen(fn, "wb");
            std::fwrite(host.data(), 8, host.size(), o);
            std::fclose(o);
        }
#endif
    }
    CHECK(mcProfilerStop());
    CHECK(mcFree(d_sid)); CHECK(mcFree(d_eob)); CHECK(mcFree(act));
    CHECK(mcFree(wgt)); CHECK(mcFree(out_c)); CHECK(mcFree(sink));
#if STAMPS
    CHECK(mcFree(stamps));
#endif
    return 0;
}
