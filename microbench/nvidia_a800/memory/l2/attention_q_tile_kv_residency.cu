/*
 * NVIDIA A800 attention Q-tile K/V residency benchmark.
 *
 * Two production prefill variants of one identical shape are launched with
 * different Q tiles -- prefill_q64_full_tile on 8192 CTAs and
 * prefill_q128_full_tile on 4096 -- and measure 450.9 MiB and 803.4 MiB of
 * downstream read respectively, while the traffic model emits the same
 * compulsory 322 MiB for both.  The wider Q tile issues half as many CTAs, so
 * at any moment far fewer programs are consuming the same K/V tile.  This
 * probe tests whether the realised K/V reuse follows that concurrency.
 *
 * Production couples the two candidate causes: a wider Q tile is also fewer
 * CTAs.  The probe separates them.  The first axis moves the Q tile with one
 * CTA per tile, so tile width and CTA count move together as in production.
 * The second holds the Q tile at 128 and splits each tile's K/V range across
 * several CTAs, which raises the CTA count while leaving the tile width and
 * every byte of the operand footprint alone.  If the downstream read follows
 * CTA count in both, the mechanism is concurrency; if it follows tile width,
 * it is not.
 *
 * The K/V footprint is identical in every configuration, and the CTA order is
 * the launch order the repository already retains: grid (q_blocks, batch,
 * heads) dispatched x before y before z.
 *
 * Traffic counters only.  No duration here may price a Forward path.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o attention_q_tile_kv_residency \
 *       attention_q_tile_kv_residency.cu
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,             \
                     cudaGetErrorString(error));                           \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

static const int kBlockThreads = 256;
static const int kBatch = 8;
static const int kSeq = 2048;
static const int kQueryHeads = 32;
static const int kKvHeads = 8;
static const int kHeadDim = 128;
static const int kElementBytes = 2;
static const int kBlockKv = 64;
/* One K/V row is head_dim halves = 256 B = 16 uint4. */
static const int kVectorsPerRow = kHeadDim * kElementBytes / 16;

__global__ __launch_bounds__(kBlockThreads) void attention_q_tile_kv(
    const uint4* __restrict__ q,
    const uint4* __restrict__ k,
    const uint4* __restrict__ v,
    uint32_t* __restrict__ out,
    int tile_q,
    int splits
) {
    const int q_block = blockIdx.x / splits;
    const int split = blockIdx.x % splits;
    const int entry = blockIdx.y;
    const int head = blockIdx.z;
    const int kv_head = head / (kQueryHeads / kKvHeads);
    const int lane = threadIdx.x;

    /* Bottom-right causal extent for this Q tile, in K/V blocks. */
    const int attended = (q_block + 1) * tile_q;
    const int kv_blocks = (attended + kBlockKv - 1) / kBlockKv;
    /* The split partitions this tile's K/V range; every byte is still read. */
    const int begin = (int)((long long)split * kv_blocks / splits);
    const int stop = (int)((long long)(split + 1) * kv_blocks / splits);

    uint32_t checksum = 0u;

    /* Q tile, read once per CTA. */
    const size_t q_rows = (size_t)tile_q;
    const size_t q_base =
        ((size_t)(entry * kQueryHeads + head) * kSeq
         + (size_t)q_block * tile_q) * kVectorsPerRow;
    for (size_t i = lane; i < q_rows * kVectorsPerRow; i += kBlockThreads) {
        const uint4 value = q[q_base + i];
        checksum += value.x ^ value.w;
    }

    for (int block = begin; block < stop; block++) {
        const size_t base =
            ((size_t)(entry * kKvHeads + kv_head) * kSeq
             + (size_t)block * kBlockKv) * kVectorsPerRow;
        const size_t span = (size_t)kBlockKv * kVectorsPerRow;
        for (size_t i = lane; i < span; i += kBlockThreads) {
            const uint4 kv = k[base + i];
            checksum += kv.y ^ kv.z;
        }
        for (size_t i = lane; i < span; i += kBlockThreads) {
            const uint4 vv = v[base + i];
            checksum += vv.x ^ vv.z;
        }
    }
    out[(size_t)blockIdx.z * gridDim.y * gridDim.x
        + (size_t)blockIdx.y * gridDim.x + blockIdx.x] = checksum;
}

struct Config { int tile_q, splits; const char* axis; };

int main() {
    std::vector<Config> configs;
    /* Axis 1: Q tile and CTA count move together, as production couples them. */
    for (int t : {32, 64, 128, 256})
        configs.push_back({t, 1, "q_tile"});
    /* Axis 2: Q tile pinned at 128, CTA count raised by splitting the K/V
     * range.  Splits 2 and 4 reproduce the CTA counts of q64 and q32. */
    for (int s : {1, 2, 4})
        configs.push_back({128, s, "cta_count_at_fixed_tile"});
    configs.push_back({64, 1, "q_tile_repeat"});

    const size_t kv_rows = (size_t)kBatch * kKvHeads * kSeq;
    const size_t kv_bytes = kv_rows * kVectorsPerRow * 16;
    const size_t q_bytes =
        (size_t)kBatch * kQueryHeads * kSeq * kVectorsPerRow * 16;
    size_t out_max = 0;
    for (const Config& c : configs) {
        const size_t blocks =
            (size_t)((kSeq + c.tile_q - 1) / c.tile_q) * c.splits
            * kBatch * kQueryHeads;
        out_max = std::max(out_max, blocks);
    }

    uint4 *q = nullptr, *k = nullptr, *v = nullptr;
    uint32_t* out = nullptr;
    CHECK(cudaMalloc(&q, q_bytes));
    CHECK(cudaMalloc(&k, kv_bytes));
    CHECK(cudaMalloc(&v, kv_bytes));
    CHECK(cudaMalloc(&out, out_max * sizeof(uint32_t)));
    CHECK(cudaMemset(q, 0x5a, q_bytes));
    CHECK(cudaMemset(k, 0xa5, kv_bytes));
    CHECK(cudaMemset(v, 0x3c, kv_bytes));
    CHECK(cudaMemset(out, 0, out_max * sizeof(uint32_t)));
    CHECK(cudaDeviceSynchronize());

    std::printf("{\"schema\":\"attention_q_tile_kv_residency_v1\",");
    std::printf("\"batch\":%d,\"seq\":%d,\"query_heads\":%d,\"kv_heads\":%d,",
                kBatch, kSeq, kQueryHeads, kKvHeads);
    std::printf("\"head_dim\":%d,\"block_kv\":%d,\"element_bytes\":%d,",
                kHeadDim, kBlockKv, kElementBytes);
    std::printf("\"kv_unique_bytes\":%zu,\"ordinals\":[", 2 * kv_bytes);
    for (size_t index = 0; index < configs.size(); index++) {
        const Config& c = configs[index];
        const int q_blocks = (kSeq + c.tile_q - 1) / c.tile_q;
        const size_t ctas =
            (size_t)q_blocks * c.splits * kBatch * kQueryHeads;
        /* Causal logical K/V volume, independent of how it is split. */
        double kv_logical = 0.0;
        for (int b = 0; b < q_blocks; b++) {
            const int attended = (b + 1) * c.tile_q;
            kv_logical += (double)((attended + kBlockKv - 1) / kBlockKv);
        }
        kv_logical *= (double)kBatch * kQueryHeads * kBlockKv
            * kHeadDim * kElementBytes * 2.0;
        if (index != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"tile_q\":%d,"
                    "\"splits\":%d,\"q_blocks\":%d,\"ctas\":%zu,"
                    "\"kv_logical_bytes\":%.0f}",
                    index, c.axis, c.tile_q, c.splits, q_blocks, ctas,
                    kv_logical);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    for (const Config& c : configs) {
        const int q_blocks = (kSeq + c.tile_q - 1) / c.tile_q;
        dim3 grid(q_blocks * c.splits, kBatch, kQueryHeads);
        attention_q_tile_kv<<<grid, kBlockThreads>>>(
            q, k, v, out, c.tile_q, c.splits);
        CHECK(cudaDeviceSynchronize());
    }

    CHECK(cudaFree(q));
    CHECK(cudaFree(k));
    CHECK(cudaFree(v));
    CHECK(cudaFree(out));
    return 0;
}
