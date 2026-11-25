/*
 * MetaX C500 grouped-program operand residency benchmark (traffic counters).
 *
 * The production fused-MoE launches this target uses come in exactly two tile
 * signatures, BM16/BN32/BK64/GSM1 and BM64/BN64/BK32/GSM8, so BM, BK and GSM
 * are perfectly collinear in that data and it cannot say which of them drives
 * the residency shortfall.  This probe breaks the collinearity by moving one
 * of them at a time around a common centre.
 *
 * It states the operand stream of a grouped matrix program directly: program
 * p maps to an output tile through the GSM group swizzle, and at each K step
 * it reads one A object of BM*BK elements and one B object of BK*BN elements.
 * Object identity, not arithmetic, is the subject, so the kernel only reduces
 * what it loads.
 *
 * Every configuration runs inside one mcProfiler marker window, so the layer-1
 * per-kernel pass reports each as its own ordinal.  The quantities of interest
 * are traffic counters -- downstream global bytes against the L2 source
 * interface -- never the instrumented duration, which layer-1 replay inflates
 * and which must not reach a Forward price.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o grouped_program_operand_residency \
 *       grouped_program_operand_residency.cpp
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
static const int kVectorBytes = 16;

__global__ __launch_bounds__(kBlockThreads) void grouped_operand_residency(
    const uint4* __restrict__ a,
    const uint4* __restrict__ b,
    uint32_t* __restrict__ out,
    int m_tiles,
    int n_tiles,
    int k_steps,
    int group_width,
    int a_vectors,
    int b_vectors
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

    const int lane = threadIdx.x;
    uint32_t checksum = 0u;
    for (int step = 0; step < k_steps; step++) {
        const size_t a_base =
            (static_cast<size_t>(tile_m) * k_steps + step)
            * static_cast<size_t>(a_vectors);
        const size_t b_base =
            (static_cast<size_t>(tile_n) * k_steps + step)
            * static_cast<size_t>(b_vectors);
        for (int i = lane; i < a_vectors; i += kBlockThreads) {
            const uint4 v = a[a_base + i];
            checksum += v.x ^ v.w;
        }
        for (int i = lane; i < b_vectors; i += kBlockThreads) {
            const uint4 v = b[b_base + i];
            checksum += v.y ^ v.z;
        }
    }
    out[program * kBlockThreads + lane] = checksum;
}

struct Config { int bm, bn, bk, gsm; const char* axis; };

int main() {
    /* Centre point is the production BM64/BN64/BK32/GSM8 signature.  Each
     * block below moves exactly one of BM, BN, BK, GSM away from it; the last
     * two rows are the two production corners themselves. */
    const std::vector<Config> configs = {
        {64, 64, 32,  1, "gsm"}, {64, 64, 32,  2, "gsm"},
        {64, 64, 32,  4, "gsm"}, {64, 64, 32,  8, "gsm"},
        {64, 64, 32, 16, "gsm"},
        {64, 64, 16,  8, "bk"},  {64, 64, 64,  8, "bk"},
        {64, 64,128,  8, "bk"},
        {16, 64, 32,  8, "bm"},  {32, 64, 32,  8, "bm"},
        {128,64, 32,  8, "bm"},
        {64, 32, 32,  8, "bn"},  {64,128, 32,  8, "bn"},
        {16, 32, 64,  1, "production_bm16"},
        {64, 64, 32,  8, "production_bm64"},
    };
    const int m_tiles = 128;
    const int n_tiles = 64;
    const int k_steps = 64;
    const int element_bytes = 2;

    size_t a_capacity = 0;
    size_t b_capacity = 0;
    for (const Config& c : configs) {
        a_capacity = std::max(a_capacity,
            static_cast<size_t>(m_tiles) * k_steps * c.bm * c.bk * element_bytes);
        b_capacity = std::max(b_capacity,
            static_cast<size_t>(n_tiles) * k_steps * c.bk * c.bn * element_bytes);
    }
    const size_t programs = static_cast<size_t>(m_tiles) * n_tiles;
    uint4* a = nullptr;
    uint4* b = nullptr;
    uint32_t* out = nullptr;
    CHECK(mcMalloc(&a, a_capacity));
    CHECK(mcMalloc(&b, b_capacity));
    CHECK(mcMalloc(&out, programs * kBlockThreads * sizeof(uint32_t)));
    CHECK(mcMemset(a, 0x5a, a_capacity));
    CHECK(mcMemset(b, 0xa5, b_capacity));
    CHECK(mcMemset(out, 0, programs * kBlockThreads * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    std::printf("{\"schema\":\"c500_grouped_program_operand_residency_v1\",");
    std::printf("\"m_tiles\":%d,\"n_tiles\":%d,\"k_steps\":%d,", m_tiles,
                n_tiles, k_steps);
    std::printf("\"block_threads\":%d,\"element_bytes\":%d,", kBlockThreads,
                element_bytes);
    std::printf("\"ordinals\":[");
    for (size_t index = 0; index < configs.size(); index++) {
        const Config& c = configs[index];
        const int a_vectors = c.bm * c.bk * element_bytes / kVectorBytes;
        const int b_vectors = c.bk * c.bn * element_bytes / kVectorBytes;
        if (index != 0) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"bm\":%d,\"bn\":%d,"
                    "\"bk\":%d,\"gsm\":%d,\"a_object_bytes\":%d,"
                    "\"b_object_bytes\":%d,\"logical_request_bytes\":%.0f}",
                    index, c.axis, c.bm, c.bn, c.bk, c.gsm,
                    a_vectors * kVectorBytes, b_vectors * kVectorBytes,
                    static_cast<double>(programs) * k_steps
                        * (a_vectors + b_vectors) * kVectorBytes);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    /* One marker window holds every configuration, so the layer-1 pass emits
     * one numbered invocation per ordinal in this exact order. */
    CHECK(mcProfilerStart());
    for (const Config& c : configs) {
        const int a_vectors = c.bm * c.bk * element_bytes / kVectorBytes;
        const int b_vectors = c.bk * c.bn * element_bytes / kVectorBytes;
        grouped_operand_residency<<<programs, kBlockThreads>>>(
            a, b, out, m_tiles, n_tiles, k_steps, c.gsm, a_vectors, b_vectors);
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());

    CHECK(mcFree(a));
    CHECK(mcFree(b));
    CHECK(mcFree(out));
    return 0;
}
