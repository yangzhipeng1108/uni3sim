/*
 * MetaX C500 grouped-program weight stride residency benchmark.
 *
 * The addressing probe showed that laying a weight tile out as production does
 * -- BN segments of BK elements strided by K -- costs 7.6x the downstream read
 * of the same bytes laid out contiguously, on this target only.  That says the
 * effect is address-indexed, but not which hardware parameter indexes it.
 *
 * This probe separates the two candidates by holding the touched bytes exactly
 * fixed and moving only the row stride.  Every configuration reads the same
 * BN segments of BK elements per tile and the same number of tiles, so the
 * requested volume and the distinct touched volume are identical; only the
 * addresses spread.  A stride sweep that is flat in magnitude but recovers
 * sharply when the stride is offset off a power-of-two boundary indicts set
 * aliasing, which is indexed by line size and associativity.  A cost that
 * instead grows smoothly with the address span indicts something spanning
 * pages rather than sets, which those two parameters would not express.
 *
 * Strides below the 4096 B a row actually carries are excluded, because they
 * would overlap rows and change the distinct footprint rather than its layout.
 *
 * Traffic counters only.  The layer-1 pass replays kernels, so its reported
 * time is not execution time and must not price a Forward path.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o grouped_weight_stride_residency \
 *       grouped_weight_stride_residency.cpp
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
static const int kBlockM = 64, kBlockN = 64, kBlockK = 32;
static const int kElementBytes = 2;
static const int kVectorsPerSegment = kBlockK * kElementBytes / 16;   /* 4 */
static const int kSegmentsPerPass = kBlockThreads / kVectorsPerSegment;

__global__ __launch_bounds__(kBlockThreads) void grouped_weight_stride(
    const uint4* __restrict__ a,
    const uint4* __restrict__ b,
    uint32_t* __restrict__ out,
    int m_tiles, int n_tiles, int k_steps, int group_width,
    int tiles_per_expert, int row_pitch_vectors
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
    const int segment = lane / kVectorsPerSegment;
    const int sub = lane % kVectorsPerSegment;
    const size_t pitch = static_cast<size_t>(row_pitch_vectors);
    const size_t weight_rows = static_cast<size_t>(n_tiles) * kBlockN;
    /* Activations keep the contiguous layout throughout, so the weight stride
     * is the only thing moving. */
    const size_t a_pitch = static_cast<size_t>(k_steps) * kVectorsPerSegment;

    uint32_t checksum = 0u;
    for (int step = 0; step < k_steps; step++) {
        const size_t step_offset =
            static_cast<size_t>(step) * kVectorsPerSegment + sub;
        for (int r = segment; r < kBlockM; r += kSegmentsPerPass) {
            const size_t row = static_cast<size_t>(tile_m) * kBlockM + r;
            const uint4 v = a[row * a_pitch + step_offset];
            checksum += v.x ^ v.w;
        }
        for (int r = segment; r < kBlockN; r += kSegmentsPerPass) {
            const size_t row = static_cast<size_t>(expert) * weight_rows
                + static_cast<size_t>(tile_n) * kBlockN + r;
            const uint4 v = b[row * pitch + step_offset];
            checksum += v.y ^ v.z;
        }
    }
    out[static_cast<size_t>(program) * kBlockThreads + lane] = checksum;
}

int main() {
    /* Region A geometry, the bulk of the production tail. */
    const int experts = 8, tiles_per_expert = 16, n_tiles = 64;
    const int k_steps = 64, gsm = 8;
    const int m_tiles = experts * tiles_per_expert;
    const size_t programs = static_cast<size_t>(m_tiles) * n_tiles;
    /* A row carries k_steps * BK * 2 = 4096 B.  Strides at and above that keep
     * the touched bytes disjoint and identical; the offsets probe alignment. */
    /* The list opens, middles and closes on the same two strides, so
     * within-run repeatability is measured rather than assumed. */
    const std::vector<int> pitch_bytes = {
        4096, 8192, 4096 + 64, 4096 + 128, 4096 + 256, 5120, 6144,
        8192, 8192 + 128, 12288, 16384, 16384 + 128, 32768, 4096, 8192,
    };

    const size_t weight_rows =
        static_cast<size_t>(experts) * n_tiles * kBlockN;
    size_t b_capacity = 0;
    for (int p : pitch_bytes)
        b_capacity = std::max(b_capacity, weight_rows * static_cast<size_t>(p));
    const size_t a_capacity =
        static_cast<size_t>(m_tiles) * kBlockM * k_steps * kBlockK * kElementBytes;

    uint4 *a = nullptr, *b = nullptr;
    uint32_t* out = nullptr;
    CHECK(mcMalloc(&a, a_capacity));
    CHECK(mcMalloc(&b, b_capacity));
    CHECK(mcMalloc(&out, programs * kBlockThreads * sizeof(uint32_t)));
    CHECK(mcMemset(a, 0x5a, a_capacity));
    CHECK(mcMemset(b, 0xa5, b_capacity));
    CHECK(mcMemset(out, 0, programs * kBlockThreads * sizeof(uint32_t)));
    CHECK(mcDeviceSynchronize());

    const double logical = static_cast<double>(programs) * k_steps
        * (kBlockM + kBlockN) * kBlockK * kElementBytes;
    std::printf("{\"schema\":\"grouped_weight_stride_residency_v1\",");
    std::printf("\"experts\":%d,\"tiles_per_expert\":%d,\"n_tiles\":%d,",
                experts, tiles_per_expert, n_tiles);
    std::printf("\"k_steps\":%d,\"gsm\":%d,\"programs\":%zu,",
                k_steps, gsm, programs);
    std::printf("\"row_bytes_touched\":%d,\"a_bytes\":%zu,\"b_bytes\":%zu,",
                k_steps * kBlockK * kElementBytes, a_capacity, b_capacity);
    std::printf("\"logical_request_bytes\":%.0f,\"ordinals\":[", logical);
    for (size_t i = 0; i < pitch_bytes.size(); i++) {
        if (i) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"row_pitch_bytes\":%d}", i, pitch_bytes[i]);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(mcProfilerStart());
    for (int p : pitch_bytes) {
        grouped_weight_stride<<<programs, kBlockThreads>>>(
            a, b, out, m_tiles, n_tiles, k_steps, gsm, tiles_per_expert,
            p / 16);
        CHECK(mcDeviceSynchronize());
    }
    CHECK(mcProfilerStop());

    CHECK(mcFree(a));
    CHECK(mcFree(b));
    CHECK(mcFree(out));
    return 0;
}
