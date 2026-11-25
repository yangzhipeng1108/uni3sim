// Host for mte2_tile microbench: sweep tile size × mode, calibrate SYS_CNT.
#include <acl/acl.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

#define CLOCK_CHECK(call) do {                                            \
    if ((call) != 0) {                                                    \
        std::perror(#call);                                               \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_mte2_tile(uint32_t blockDim, void* stream,
    void* gm_src, void* out, int64_t tile_bytes, int64_t n_tiles,
    int64_t n_iters, int64_t mode);

static int64_t parse_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

__attribute__((constructor(101))) static void early_init() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

struct Cell {
    int64_t tile_bytes;
    int64_t mode;
    int64_t n_iters;
    int64_t n_tiles;
    size_t buffer_bytes;
};

int main(int argc, char** argv) {
    if (argc < 6 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 3) % 3 != 0) {
        fprintf(stderr,
                "usage: %s --batch footprint_MB "
                "tile_bytes mode n_iters [tile_bytes mode n_iters ...]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    int64_t footprint_mb = parse_integer(argv[2], "footprint_MB");
    if (footprint_mb < 384 || footprint_mb > 4096) {
        fprintf(stderr,
                "footprint_MB must be in [384,4096]\n");
        return EXIT_FAILURE;
    }
    const int cell_count = (argc - 3) / 3;
    if (cell_count > 64) {
        fprintf(stderr, "a batch cannot contain more than 64 cells\n");
        return EXIT_FAILURE;
    }
    size_t requested_bytes = static_cast<size_t>(footprint_mb) * 1024 * 1024;
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    for (int argument = 3; argument < argc; argument += 3) {
        const int64_t tile_bytes = parse_integer(
            argv[argument], "tile_bytes");
        const int64_t mode = parse_integer(argv[argument + 1], "mode");
        const int64_t n_iters = parse_integer(
            argv[argument + 2], "n_iters");
        if (tile_bytes <= 0 || tile_bytes > 96 * 1024
                || tile_bytes % 32 != 0) {
            fprintf(stderr,
                    "tile_bytes must be a multiple of 32 in [32,98304]\n");
            return EXIT_FAILURE;
        }
        if (mode != 0 && mode != 1 && mode != 2) {
            fprintf(stderr, "mode must be 0, 1, or 2\n");
            return EXIT_FAILURE;
        }
        if (n_iters <= 0
                || n_iters > std::numeric_limits<int64_t>::max() / 4) {
            fprintf(stderr, "n_iters must be positive and safe for 4x\n");
            return EXIT_FAILURE;
        }
        const int64_t n_tiles = static_cast<int64_t>(
            requested_bytes / tile_bytes);
        if (n_tiles <= 0 || n_iters < n_tiles) {
            fprintf(stderr,
                    "n_iters=%ld must cover n_tiles=%ld for tile=%ld\n",
                    n_iters, n_tiles, tile_bytes);
            return EXIT_FAILURE;
        }
        cells.push_back(Cell{
            tile_bytes,
            mode,
            n_iters,
            n_tiles,
            static_cast<size_t>(n_tiles) * tile_bytes,
        });
    }

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void *d_src, *d_out;
    ACL_CHECK(aclrtMalloc(
        &d_src, requested_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_out, 64, ACL_MEM_MALLOC_HUGE_FIRST));
    constexpr int kSourcePattern = 0x5a;
    ACL_CHECK(aclrtMemset(
        d_src, requested_bytes, kSourcePattern, requested_bytes));

    auto run = [&](const Cell& cell, int64_t iters,
                   int64_t& cycles, int64_t& wall_ns) {
        struct timespec a, b;
        int64_t sentinel = -1;
        ACL_CHECK(aclrtMemcpy(
            d_out, 64, &sentinel, sizeof(sentinel),
            ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtlaunch_mte2_tile(
            1, stream, d_src, d_out, cell.tile_bytes, cell.n_tiles,
            iters, cell.mode));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &a));
        ACL_CHECK(aclrtlaunch_mte2_tile(
            1, stream, d_src, d_out, cell.tile_bytes, cell.n_tiles,
            iters, cell.mode));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &b));
        wall_ns = static_cast<int64_t>(b.tv_sec - a.tv_sec) * 1000000000LL
            + static_cast<int64_t>(b.tv_nsec - a.tv_nsec);
        int64_t h[8]={0};
        ACL_CHECK(aclrtMemcpy(h, 64, d_out, 64, ACL_MEMCPY_DEVICE_TO_HOST));
        if (h[0] <= 0) {
            fprintf(stderr, "MTE2 kernel produced invalid cycle count: %ld\n", h[0]);
            std::exit(EXIT_FAILURE);
        }
        if (h[1] != kSourcePattern) {
            fprintf(stderr,
                    "MTE2 data validation failed: got %ld expected %d\n",
                    h[1], kSourcePattern);
            std::exit(EXIT_FAILURE);
        }
        cycles = h[0];
    };
    for (const Cell& cell : cells) {
        int64_t short_cycles = 0;
        int64_t short_wall_ns = 0;
        int64_t long_cycles = 0;
        int64_t long_wall_ns = 0;
        run(cell, cell.n_iters, short_cycles, short_wall_ns);
        run(cell, cell.n_iters * 4, long_cycles, long_wall_ns);
        if (long_cycles <= short_cycles || long_wall_ns <= short_wall_ns) {
            fprintf(stderr,
                    "MTE2 calibration must increase: cycles %ld -> %ld, "
                    "wall_ns %ld -> %ld\n",
                    short_cycles, long_cycles,
                    short_wall_ns, long_wall_ns);
            return EXIT_FAILURE;
        }
        const double delta_cycles = static_cast<double>(
            long_cycles - short_cycles);
        const double tick_ns = static_cast<double>(
            long_wall_ns - short_wall_ns) / delta_cycles;
        const double per_iter_cycles = delta_cycles
            / static_cast<double>(cell.n_iters * 3);
        const double per_iter_ns = per_iter_cycles * tick_ns;
        const double bw = cell.tile_bytes / per_iter_ns;
        if (!std::isfinite(tick_ns) || tick_ns <= 0.0
                || !std::isfinite(per_iter_cycles)
                || per_iter_cycles <= 0.0
                || !std::isfinite(per_iter_ns) || per_iter_ns <= 0.0
                || !std::isfinite(bw) || bw <= 0.0) {
            fprintf(stderr,
                    "MTE2 derived measurement is not positive and finite\n");
            return EXIT_FAILURE;
        }
        const char* mode_name = (
            cell.mode == 0 ? "hard_event_completion"
            : cell.mode == 1 ? "pipeline"
            : "dynamic_params_completion");
        printf(
            "%ld,%zu,%s,%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,%.9f,%.9f,pass\n",
            cell.tile_bytes, cell.buffer_bytes, mode_name, cell.n_iters,
            short_cycles, short_wall_ns, long_cycles, long_wall_ns,
            tick_ns, per_iter_cycles, per_iter_ns, bw);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtFree(d_src));
    ACL_CHECK(aclrtFree(d_out));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
