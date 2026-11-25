/*
 * NVIDIA A800 grouped-program operand run-length residency benchmark.
 *
 * The addressing probe established that a production-shaped weight tile --
 * BN segments of BK elements strided by K -- costs many times the contiguous
 * read of the identical bytes on this target and ~0 more on A800.  It held
 * BK at 32 elements throughout, so every strided point it measured has a
 * 64 B contiguous run, and its only other point is a fully contiguous 4096 B
 * tile.  Between those two it measured nothing.
 *
 * That gap matters because two different mechanisms fit both points.  The
 * cost may be indexed by the L2 request granularity, which is a 32 B sector
 * on this target and an unsectored 128 B line on C500; or it may be a smooth
 * function of the address span a tile covers.  The first is a hardware number
 * both HWSpecs already carry.  The second is a property of the allocation,
 * which the stride sweep already showed is not a launch fact.  They license
 * opposite conclusions, and the two production signatures differ in exactly
 * this variable: the accurate population reads 128 B runs and the failing
 * population reads 64 B runs.
 *
 * The sweep trades BK against BN at a fixed tile byte count, which is the
 * production variation itself.  Because BN * BK is held and the weight
 * layout stays [E, N, K], the following are all invariant across every
 * configuration:
 *
 *   row pitch            K * element_bytes, the real [E, N, K] pitch
 *   distinct footprint   E * N * K * element_bytes
 *   total logical bytes  programs * k_steps * tile_bytes, since
 *                        n_tiles * k_steps = N * K / (BN * BK) is constant
 *
 * Only the contiguous run length per row moves, against a grid that trades
 * width for k-loop depth at constant total work.  A second pass repeats the
 * sweep with the row pitch doubled, which moves the address span while
 * holding the run, so span and run are separated rather than confounded.
 *
 * Realised weight reuse is reported as the measured downstream read over the
 * distinct weight footprint, both counted.  Traffic counters only.  The
 * layer-1 pass replays kernels, so its reported time is not execution time
 * and must not price a Forward path.
 *
 * This is the control for the C500 probe of the same name.  A800's L2 is
 * sectored at 32 B against C500's unsectored 128 B line, so if sub-line
 * request granularity is what indexes the cost, a knee the C500 sweep shows
 * at the line size must be absent or much weaker here.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o grouped_operand_run_length_residency \
 *       grouped_operand_run_length_residency.cu
 */

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                   \
    cudaError_t error = call;                                                \
    if (error != cudaSuccess) {                                              \
        std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,             \
                     cudaGetErrorString(error));                             \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

static const int kBlockThreads = 256;
static const int kBlockM = 64;
static const int kElementBytes = 2;
static const int kVectorBytes = 16;
static const int kTileBytes = 4096;        /* BN * BK * element_bytes */
static const int kExperts = 8;
static const int kGroupWidth = 8;
/* Geometry scale.  The defaults are the production Region A corner the
 * addressing probe used.  argv may shrink them together for a lighter
 * layer-1 pass; the sweep's logic, invariants and reading are unchanged,
 * and the weight footprint must stay above L2 for reuse to be at stake. */
static int kTilesPerExpert = 16;
static int kExtentN = 4096;                /* elements */
static int kExtentK = 2048;                /* elements */
/* The activation stream keeps a 64 B run at every configuration, so its row
 * must carry the deepest k loop in the sweep: 2048/16 steps of 64 B. */
static int kActivationPitchBytes = 0;

/* One program reads its BM x BK activation tile and its BN x BK weight tile
 * at every k step.  Both operands are addressed exactly as production
 * addresses them: a run of `run_vectors` uint4 per row, rows `pitch_vectors`
 * apart, the run advancing along K with the step. */
__global__ __launch_bounds__(kBlockThreads) void grouped_operand_run_length(
    const uint4* __restrict__ a,
    const uint4* __restrict__ b,
    uint32_t* __restrict__ out,
    int m_tiles, int n_tiles, int k_steps, int group_width,
    int tiles_per_expert, int block_n, int run_vectors,
    int a_pitch_vectors, int b_pitch_vectors, int extent_n
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
    const size_t expert_rows = static_cast<size_t>(extent_n);

    uint32_t checksum = 0u;
    for (int step = 0; step < k_steps; step++) {
        const size_t step_offset =
            static_cast<size_t>(step) * run_vectors;

        /* The activation run is held at 64 B for every configuration, so the
         * weight run length is the only variable that moves.  Its total
         * volume is then constant too, because programs * k_steps is. */
        const int a_run_vectors = 4;
        const int a_vectors = kBlockM * a_run_vectors;
        for (int index = lane; index < a_vectors; index += kBlockThreads) {
            const int r = index / a_run_vectors;
            const int off = index % a_run_vectors;
            const size_t row = static_cast<size_t>(tile_m) * kBlockM + r;
            const uint4 v = a[row * a_pitch_vectors
                              + static_cast<size_t>(step) * a_run_vectors
                              + off];
            checksum += v.x ^ v.w;
        }

        const int b_vectors = block_n * run_vectors;
        for (int index = lane; index < b_vectors; index += kBlockThreads) {
            const int r = index / run_vectors;
            const int off = index % run_vectors;
            const size_t row = static_cast<size_t>(expert) * expert_rows
                + static_cast<size_t>(tile_n) * block_n + r;
            const uint4 v = b[row * b_pitch_vectors + step_offset + off];
            checksum += v.y ^ v.z;
        }
    }
    out[static_cast<size_t>(program) * kBlockThreads + lane] = checksum;
}

struct Config {
    int block_k;           /* elements; the run is block_k * 2 bytes */
    int pitch_multiplier;  /* 1 = the real [E, N, K] pitch */
    const char* axis;
};

int main(int argc, char** argv) {
    if (argc == 4) {
        kTilesPerExpert = std::atoi(argv[1]);
        kExtentN = std::atoi(argv[2]);
        kExtentK = std::atoi(argv[3]);
        if (kTilesPerExpert <= 0 || kExtentN <= 0 || kExtentK <= 0) {
            std::fprintf(stderr, "geometry scale must be positive\n");
            return 1;
        }
    } else if (argc != 1) {
        std::fprintf(stderr,
                     "usage: %s [tiles_per_expert extent_n extent_k]\n",
                     argv[0]);
        return 1;
    }
    kActivationPitchBytes = (kExtentK / 16) * 64;
    std::vector<Config> configs;
    /* Axis A: run length at the real row pitch.  block_k 32 and 64 are the
     * two production signatures; the rest bracket them. */
    for (int bk : {16, 32, 64, 128, 256})
        configs.push_back({bk, 1, "run_at_source_pitch"});
    /* Axis B: the same run lengths with the span doubled, so span moves
     * while the run is held. */
    for (int bk : {16, 32, 64, 128, 256})
        configs.push_back({bk, 2, "run_at_doubled_pitch"});
    /* Repeatability: the two production points again, last. */
    configs.push_back({32, 1, "repeat"});
    configs.push_back({64, 1, "repeat"});

    const int m_tiles = kExperts * kTilesPerExpert;
    size_t a_capacity = 0, b_capacity = 0, programs_max = 0;
    for (const Config& c : configs) {
        const size_t pitch_bytes = static_cast<size_t>(kExtentK)
            * kElementBytes * c.pitch_multiplier;
        (void)pitch_bytes;
        a_capacity = std::max(a_capacity,
            static_cast<size_t>(m_tiles) * kBlockM * kActivationPitchBytes);
        b_capacity = std::max(b_capacity,
            static_cast<size_t>(kExperts) * kExtentN * pitch_bytes);
        const int block_n = kTileBytes / (c.block_k * kElementBytes);
        programs_max = std::max(programs_max,
            static_cast<size_t>(m_tiles) * (kExtentN / block_n));
    }

    uint4 *a = nullptr, *b = nullptr;
    uint32_t* out = nullptr;
    CHECK(cudaMalloc(&a, a_capacity));
    CHECK(cudaMalloc(&b, b_capacity));
    CHECK(cudaMalloc(&out, programs_max * kBlockThreads * sizeof(uint32_t)));
    CHECK(cudaMemset(a, 0x5a, a_capacity));
    CHECK(cudaMemset(b, 0xa5, b_capacity));
    CHECK(cudaMemset(out, 0, programs_max * kBlockThreads * sizeof(uint32_t)));
    CHECK(cudaDeviceSynchronize());

    const double weight_footprint = static_cast<double>(kExperts)
        * kExtentN * kExtentK * kElementBytes;
    const double activation_footprint = static_cast<double>(m_tiles)
        * kBlockM * kExtentK * kElementBytes;
    std::printf("{\"schema\":\"grouped_operand_run_length_residency_v1\",");
    std::printf("\"experts\":%d,\"tiles_per_expert\":%d,\"m_tiles\":%d,",
                kExperts, kTilesPerExpert, m_tiles);
    std::printf("\"extent_n\":%d,\"extent_k\":%d,\"gsm\":%d,",
                kExtentN, kExtentK, kGroupWidth);
    std::printf("\"tile_bytes\":%d,\"block_m\":%d,\"element_bytes\":%d,",
                kTileBytes, kBlockM, kElementBytes);
    std::printf("\"a_bytes\":%zu,\"b_bytes\":%zu,", a_capacity, b_capacity);
    std::printf("\"weight_footprint_bytes\":%.0f,", weight_footprint);
    std::printf("\"activation_footprint_bytes\":%.0f,", activation_footprint);
    std::printf("\"ordinals\":[");
    for (size_t i = 0; i < configs.size(); i++) {
        const Config& c = configs[i];
        const int block_n = kTileBytes / (c.block_k * kElementBytes);
        const int n_tiles = kExtentN / block_n;
        const int k_steps = kExtentK / c.block_k;
        const size_t programs = static_cast<size_t>(m_tiles) * n_tiles;
        const double logical = static_cast<double>(programs) * k_steps
            * (kBlockM * 64 + block_n * c.block_k * kElementBytes);
        if (i) std::printf(",");
        std::printf("{\"ordinal\":%zu,\"axis\":\"%s\",\"block_k\":%d,"
                    "\"block_n\":%d,\"run_bytes\":%d,\"pitch_bytes\":%d,"
                    "\"n_tiles\":%d,\"k_steps\":%d,\"programs\":%zu,"
                    "\"logical_request_bytes\":%.0f}",
                    i, c.axis, c.block_k, block_n,
                    c.block_k * kElementBytes,
                    kExtentK * kElementBytes * c.pitch_multiplier,
                    n_tiles, k_steps, programs, logical);
    }
    std::printf("]}\n");
    std::fflush(stdout);

    CHECK(cudaProfilerStart());
    for (const Config& c : configs) {
        const int block_n = kTileBytes / (c.block_k * kElementBytes);
        const int n_tiles = kExtentN / block_n;
        const int k_steps = kExtentK / c.block_k;
        const size_t programs = static_cast<size_t>(m_tiles) * n_tiles;
        const int pitch_vectors = kExtentK * kElementBytes
            * c.pitch_multiplier / kVectorBytes;
        grouped_operand_run_length<<<programs, kBlockThreads>>>(
            a, b, out, m_tiles, n_tiles, k_steps, kGroupWidth,
            kTilesPerExpert, block_n,
            c.block_k * kElementBytes / kVectorBytes,
            kActivationPitchBytes / kVectorBytes, pitch_vectors,
            kExtentN);
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaProfilerStop());

    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(out));
    return 0;
}
