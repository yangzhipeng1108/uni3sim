/*
 * MetaX C500 attention K/V panel-width residency benchmark.
 *
 * Two production prefill programs of one identical shape measure 8.55x and
 * 3.05x the unique K/V footprint on C500.  Three independent controls already
 * land at ~3x and so falsify the candidates they moved: addressing
 * (attention_kv_paging_residency, 2.85x contiguous against 2.77x paged), and
 * Q tile width together with CTA concurrency (attention_q_tile_kv_residency on
 * A800, 192.7-202.1 MiB across tile 32-256 and splits 1-4).  Every one of
 * those controls held `block_kv` at 64.
 *
 * The two production programs differ on exactly that held axis: the non-split
 * `flash_fwd_kernel` carries Flash_fwd_kernel_traits<128, 128, ...> and the
 * split one carries <128, 64, ...>.  This probe moves only the K/V panel
 * width, holding the batch, sequence, heads, head dimension, footprint,
 * addressing, Q tile and reference order fixed, and crosses it with the two
 * production Q tiles so the panel axis can be read independently of the tile.
 *
 * A wider panel changes how many contiguous K/V bytes one CTA consumes before
 * it advances, which is a reference-distance property of the launch, not a
 * per-shape correction.  The measured quantity is realised reuse: downstream
 * global read bytes over the unique K/V footprint.
 *
 * Traffic counters only.  No duration here may price a Forward path, and the
 * panel-to-traffic relation must not be fitted or interpolated.
 *
 * Every configuration runs inside one mcProfiler marker window, so the
 * layer-1 per-kernel pass reports each as its own ordinal.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o attention_kv_panel_width_residency \
 *       attention_kv_panel_width_residency.cpp
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
static const int kBatch = 8;
static const int kSeq = 2048;
static const int kQueryHeads = 32;
static const int kKvHeads = 8;
static const int kHeadDim = 128;
static const int kElementBytes = 2;
static const int kVectorsPerRow = kHeadDim * kElementBytes / 16;

/* One K/V row of one kv_head is kVectorsPerRow uint4.  A panel of `panel`
 * rows is therefore panel * kVectorsPerRow vectors, contiguous, exactly as
 * the production traits' BlockN describes. */
__global__ __launch_bounds__(kBlockThreads) void attention_kv_panel(
    const uint4* __restrict__ k,
    const uint4* __restrict__ v,
    uint32_t* __restrict__ out,
    int tile_q,
    int panel
) {
    const int q_block = blockIdx.x;
    const int entry = blockIdx.y;
    const int head = blockIdx.z;
    const int kv_head = head / (kQueryHeads / kKvHeads);
    const int lane = threadIdx.x;

    const int panels_per_seq = (kSeq + panel - 1) / panel;
    const int attended = (q_block + 1) * tile_q;
    int kv_panels = (attended + panel - 1) / panel;
    if (kv_panels > panels_per_seq) kv_panels = panels_per_seq;

    /* Row-major [entry][kv_head][row], so a panel is contiguous and the
     * footprint is identical for every panel width. */
    const size_t seq_base =
        (size_t)(entry * kKvHeads + kv_head) * kSeq * kVectorsPerRow;

    uint32_t checksum = 0u;
    for (int p = 0; p < kv_panels; p++) {
        const size_t base = seq_base + (size_t)p * panel * kVectorsPerRow;
        size_t span = (size_t)panel * kVectorsPerRow;
        const size_t remaining =
            (size_t)(kSeq - p * panel) * kVectorsPerRow;
        if (span > remaining) span = remaining;
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

struct Config { int tile_q, panel; const char* axis; };

int main() {
    std::vector<Config> configs = {
        /* The production pair's own panels at the production Q tile. */
        {128, 128, "panel_at_tile128"}, {128, 64, "panel_at_tile128"},
        {128, 256, "panel_at_tile128"}, {128, 32, "panel_at_tile128"},
        /* Same panel axis at the other production Q tile, so the panel can
         * be read independently of the tile. */
        {64, 128, "panel_at_tile64"},   {64, 64, "panel_at_tile64"},
        /* Repeatability on the two production points. */
        {128, 128, "repeat"},           {128, 64, "repeat"},
    };

    const size_t rows = (size_t)kBatch * kKvHeads * kSeq;
    const size_t kv_bytes = rows * kVectorsPerRow * 16;

    uint4 *k = nullptr, *v = nullptr;
    uint32_t* out = nullptr;
    size_t out_max = 0;
    for (const Config& c : configs) {
        out_max = std::max(out_max,
            (size_t)((kSeq + c.tile_q - 1) / c.tile_q) * kBatch * kQueryHeads);
    }
    CHECK(mcMalloc(&k, kv_bytes));
    CHECK(mcMalloc(&v, kv_bytes));
    CHECK(mcMalloc(&out, out_max * sizeof(uint32_t)));
    CHECK(mcMemset(k, 0xa5, kv_bytes));
    CHECK(mcMemset(v, 0x3c, kv_bytes));
    CHECK(mcMemset(out, 0, out_max * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"attention_kv_panel_width_residency_v1\",");
    std::printf("\"batch\":%d,\"seq\":%d,\"query_heads\":%d,\"kv_heads\":%d,",
                kBatch, kSeq, kQueryHeads, kKvHeads);
    std::printf("\"head_dim\":%d,\"element_bytes\":%d,",
                kHeadDim, kElementBytes);
    std::printf("\"kv_unique_bytes\":%zu,\"addressing\":\"contiguous\",",
                2 * kv_bytes);
    std::printf("\"ordinals\":[");
    for (size_t index = 0; index < configs.size(); index++) {
        const Config& c = configs[index];
        const int q_blocks = (kSeq + c.tile_q - 1) / c.tile_q;
        const int panels_per_seq = (kSeq + c.panel - 1) / c.panel;
        double logical_rows = 0.0;
        for (int b = 0; b < q_blocks; b++) {
            int n = ((b + 1) * c.tile_q + c.panel - 1) / c.panel;
            if (n > panels_per_seq) n = panels_per_seq;
            /* Tail panel carries only the rows that exist. */
            logical_rows += std::min((double)n * c.panel, (double)kSeq);
        }
        const double logical = logical_rows
            * (double)kBatch * kQueryHeads * kHeadDim * kElementBytes * 2.0;
        if (index != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"tile_q\":%d,"
                    "\"panel\":%d,\"ctas\":%zu,\"kv_logical_bytes\":%.0f}",
                    index, c.axis, c.tile_q, c.panel,
                    (size_t)q_blocks * kBatch * kQueryHeads, logical);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        const int q_blocks = (kSeq + c.tile_q - 1) / c.tile_q;
        dim3 grid(q_blocks, kBatch, kQueryHeads);
        attention_kv_panel<<<grid, kBlockThreads>>>(
            k, v, out, c.tile_q, c.panel);
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());

    CHECK(mcFree(k));
    CHECK(mcFree(v));
    CHECK(mcFree(out));
    return 0;
}
