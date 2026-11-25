/*
 * NVIDIA A800 attention K/V paging residency benchmark.
 *
 * Two production prefill variants of one identical shape -- identical batch,
 * sequence, heads, head dimension and K/V footprint, differing only in
 * kv_layout -- measure 1.78x apart on C500 and 1.21x apart on A800, and the
 * traffic model emits the same compulsory floor for both.  The Q tile and the
 * CTA concurrency were falsified as the cause by attention_q_tile_kv_residency.
 * What remains is the addressing: a paged cache reaches its K/V blocks through
 * a block table, a contiguous one walks them in order.
 *
 * This probe holds the geometry, the footprint and the reference order exactly
 * fixed and moves only that indirection, crossed with the two production Q
 * tiles.  The block table is a stated deterministic permutation, so the paged
 * form touches the same blocks in the same order at different addresses.
 *
 * The companion C500 build of this file answers the same question on the
 * target where the production difference is largest; this target is the
 * control where it is nearly absent.
 *
 * Traffic counters only.  No duration here may price a Forward path.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o attention_kv_paging_residency \
 *       attention_kv_paging_residency.cu
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
static const int kVectorsPerRow = kHeadDim * kElementBytes / 16;
static const int kBlocksPerSeq = kSeq / kBlockKv;

__global__ __launch_bounds__(kBlockThreads) void attention_kv_paging(
    const uint4* __restrict__ k,
    const uint4* __restrict__ v,
    const int* __restrict__ block_table,
    uint32_t* __restrict__ out,
    int tile_q,
    int paged
) {
    const int q_block = blockIdx.x;
    const int entry = blockIdx.y;
    const int head = blockIdx.z;
    const int kv_head = head / (kQueryHeads / kKvHeads);
    const int lane = threadIdx.x;

    const int attended = (q_block + 1) * tile_q;
    int kv_blocks = (attended + kBlockKv - 1) / kBlockKv;
    if (kv_blocks > kBlocksPerSeq) kv_blocks = kBlocksPerSeq;

    uint32_t checksum = 0u;
    for (int block = 0; block < kv_blocks; block++) {
        /* Same logical block, same order; only its address moves. */
        int physical;
        if (paged) {
            physical = block_table[
                (entry * kKvHeads + kv_head) * kBlocksPerSeq + block];
        } else {
            physical = (entry * kKvHeads + kv_head) * kBlocksPerSeq + block;
        }
        const size_t base = (size_t)physical * kBlockKv * kVectorsPerRow;
        const size_t span = (size_t)kBlockKv * kVectorsPerRow;
        for (size_t i = lane; i < span; i += kBlockThreads) {
            const uint4 kk = k[base + i];
            checksum += kk.y ^ kk.z;
        }
        for (size_t i = lane; i < span; i += kBlockThreads) {
            const uint4 vv = v[base + i];
            checksum += vv.x ^ vv.z;
        }
    }
    out[(size_t)blockIdx.z * gridDim.y * gridDim.x
        + (size_t)blockIdx.y * gridDim.x + blockIdx.x] = checksum;
}

struct Config { int tile_q, paged; const char* axis; };

int main() {
    std::vector<Config> configs = {
        {128, 0, "production_pair"}, {128, 1, "production_pair"},
        {64, 0, "production_pair"},  {64, 1, "production_pair"},
        {128, 0, "repeat"},          {128, 1, "repeat"},
    };

    const size_t total_blocks =
        (size_t)kBatch * kKvHeads * kBlocksPerSeq;
    const size_t kv_bytes = total_blocks * kBlockKv * kVectorsPerRow * 16;

    /* Block table: an odd-multiplier permutation of the physical block pool,
     * so every logical block maps to a distinct physical block and the
     * footprint is unchanged -- only the addresses are scattered. */
    std::vector<int> table(total_blocks);
    for (size_t i = 0; i < total_blocks; i++) {
        table[i] = (int)((i * 2654435761u) % (uint32_t)total_blocks);
    }

    uint4 *k = nullptr, *v = nullptr;
    int* table_device = nullptr;
    uint32_t* out = nullptr;
    size_t out_max = 0;
    for (const Config& c : configs) {
        out_max = std::max(out_max,
            (size_t)((kSeq + c.tile_q - 1) / c.tile_q) * kBatch * kQueryHeads);
    }
    CHECK(cudaMalloc(&k, kv_bytes));
    CHECK(cudaMalloc(&v, kv_bytes));
    CHECK(cudaMalloc(&table_device, total_blocks * sizeof(int)));
    CHECK(cudaMalloc(&out, out_max * sizeof(uint32_t)));
    CHECK(cudaMemset(k, 0xa5, kv_bytes));
    CHECK(cudaMemset(v, 0x3c, kv_bytes));
    CHECK(cudaMemcpy(table_device, table.data(), total_blocks * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out, 0, out_max * sizeof(uint32_t)));
    CHECK(cudaDeviceSynchronize());

    std::printf("{\"schema\":\"attention_kv_paging_residency_v1\",");
    std::printf("\"batch\":%d,\"seq\":%d,\"query_heads\":%d,\"kv_heads\":%d,",
                kBatch, kSeq, kQueryHeads, kKvHeads);
    std::printf("\"head_dim\":%d,\"block_kv\":%d,", kHeadDim, kBlockKv);
    std::printf("\"kv_unique_bytes\":%zu,", 2 * kv_bytes);
    std::printf("\"block_table\":\"(i*2654435761)%%blocks\",\"ordinals\":[");
    for (size_t index = 0; index < configs.size(); index++) {
        const Config& c = configs[index];
        const int q_blocks = (kSeq + c.tile_q - 1) / c.tile_q;
        double logical = 0.0;
        for (int b = 0; b < q_blocks; b++) {
            int n = ((b + 1) * c.tile_q + kBlockKv - 1) / kBlockKv;
            if (n > kBlocksPerSeq) n = kBlocksPerSeq;
            logical += n;
        }
        logical *= (double)kBatch * kQueryHeads * kBlockKv * kHeadDim
            * kElementBytes * 2.0;
        if (index != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"tile_q\":%d,"
                    "\"paged\":%d,\"ctas\":%zu,\"kv_logical_bytes\":%.0f}",
                    index, c.axis, c.tile_q, c.paged,
                    (size_t)q_blocks * kBatch * kQueryHeads, logical);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    for (const Config& c : configs) {
        const int q_blocks = (kSeq + c.tile_q - 1) / c.tile_q;
        dim3 grid(q_blocks, kBatch, kQueryHeads);
        attention_kv_paging<<<grid, kBlockThreads>>>(
            k, v, table_device, out, c.tile_q, c.paged);
        CHECK(cudaDeviceSynchronize());
    }

    CHECK(cudaFree(k));
    CHECK(cudaFree(v));
    CHECK(cudaFree(table_device));
    CHECK(cudaFree(out));
    return 0;
}
