// ACL host for the independent AIC GM-to-L1 ND2NZ benchmark.
#include <acl/acl.h>
#include <algorithm>
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

extern "C" uint32_t aclrtlaunch_aic_mte2_nd2nz(
    uint32_t blockDim, void* stream, void* gm_source, void* gm_output,
    int64_t rows, int64_t columns, int64_t source_tiles,
    int64_t iterations, int64_t mode);

namespace {

constexpr int64_t kElementBytes = 2;
constexpr int64_t kCycleSlotBytes = 64;
constexpr double kSystemCounterTickNs = 20.0;

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

struct Cell {
    int64_t rows;
    int64_t columns;
    int64_t mode;
    int64_t block_dim;
    int64_t iterations;
    int64_t tile_bytes;
    int64_t source_tiles;
};

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 8 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 3) % 5 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch footprint_MB "
            "rows columns mode block_dim iterations [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t footprint_mb = ParseInteger(argv[2], "footprint_MB");
    if (footprint_mb < 16 || footprint_mb > 1024) {
        std::fprintf(stderr, "footprint_MB must be in [16,1024]\n");
        return EXIT_FAILURE;
    }
    const int cell_count = (argc - 3) / 5;
    if (cell_count > 128) {
        std::fprintf(stderr, "a batch cannot contain more than 128 cells\n");
        return EXIT_FAILURE;
    }
    const size_t source_bytes = static_cast<size_t>(
        footprint_mb) * 1024 * 1024;
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    int64_t maximum_block_dim = 0;
    for (int argument = 3; argument < argc; argument += 5) {
        const int64_t rows = ParseInteger(argv[argument], "rows");
        const int64_t columns = ParseInteger(argv[argument + 1], "columns");
        const int64_t mode = ParseInteger(argv[argument + 2], "mode");
        const int64_t block_dim = ParseInteger(
            argv[argument + 3], "block_dim");
        const int64_t iterations = ParseInteger(
            argv[argument + 4], "iterations");
        const int64_t tile_bytes = rows * columns * kElementBytes;
        const int64_t source_tiles = static_cast<int64_t>(
            source_bytes / static_cast<size_t>(tile_bytes));
        if (rows <= 0 || rows > 512 || columns < 16 || columns > 256
                || columns % 16 != 0 || mode < 0 || mode > 4
                || block_dim <= 0 || block_dim > 24 || iterations <= 0
                || iterations > std::numeric_limits<int64_t>::max() / 4
                || ((rows + 15) / 16 * 16) * columns * kElementBytes
                    > 256 * 1024
                || source_tiles <= 0
                || ((mode == 3 || mode == 4)
                    && ((rows != 32 && rows != 128) || columns != 64
                        || (64 * 256) % tile_bytes != 0
                        || source_tiles
                            / (1 + (64 * 256) / tile_bytes) < block_dim
                        || (mode == 4 && iterations % 4 != 0)))) {
            std::fprintf(stderr, "invalid AIC MTE2 cell at argument %d\n", argument);
            return EXIT_FAILURE;
        }
        cells.push_back(Cell{
            rows, columns, mode, block_dim, iterations,
            tile_bytes, source_tiles});
        maximum_block_dim = std::max(maximum_block_dim, block_dim);
    }

    void* device_source = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output,
        static_cast<size_t>(maximum_block_dim * kCycleSlotBytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_source, source_bytes, 0, source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto run = [&](const Cell& cell, int64_t iterations,
                   int64_t& cycles, int64_t& wall_ns) {
        const size_t output_bytes = static_cast<size_t>(
            cell.block_dim * kCycleSlotBytes);
        ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_aic_mte2_nd2nz(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_source, device_output, cell.rows, cell.columns,
            cell.source_tiles, iterations, cell.mode));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        struct timespec begin;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &begin));
        ACL_CHECK(aclrtlaunch_aic_mte2_nd2nz(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_source, device_output, cell.rows, cell.columns,
            cell.source_tiles, iterations, cell.mode));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        wall_ns = static_cast<int64_t>(end.tv_sec - begin.tv_sec) * 1000000000LL
            + static_cast<int64_t>(end.tv_nsec - begin.tv_nsec);
        std::vector<int64_t> output(
            output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            output.data(), output_bytes, device_output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        cycles = 0;
        for (int64_t block = 0; block < cell.block_dim; ++block) {
            const int64_t block_cycles = output[static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t))];
            const int64_t validation = output[static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t) + 1)];
            if (block_cycles <= 0 || validation != 1) {
                std::fprintf(stderr, "AIC MTE2 block %ld failed validation\n", block);
                std::exit(EXIT_FAILURE);
            }
            cycles = std::max(cycles, block_cycles);
        }
    };

    for (const Cell& cell : cells) {
        int64_t short_cycles = 0;
        int64_t short_wall_ns = 0;
        int64_t long_cycles = 0;
        int64_t long_wall_ns = 0;
        run(cell, cell.iterations, short_cycles, short_wall_ns);
        run(cell, cell.iterations * 4, long_cycles, long_wall_ns);
        if (long_cycles <= short_cycles || long_wall_ns <= short_wall_ns) {
            std::fprintf(stderr, "AIC MTE2 differential interval did not increase\n");
            return EXIT_FAILURE;
        }
        const double delta_cycles = static_cast<double>(
            long_cycles - short_cycles);
        const double tick_ns = kSystemCounterTickNs;
        const double cycles_per_copy = delta_cycles
            / static_cast<double>(cell.iterations * 3);
        const double ns_per_copy = cycles_per_copy * tick_ns;
        const int64_t logical_bytes_per_copy = (
            (cell.mode == 3 || cell.mode == 4)
            ? cell.tile_bytes + 64 * 256 * kElementBytes
            : cell.tile_bytes);
        const double gb_per_s_per_aic = (
            logical_bytes_per_copy / ns_per_copy);
        if (!std::isfinite(ns_per_copy) || ns_per_copy <= 0.0
                || !std::isfinite(gb_per_s_per_aic)
                || gb_per_s_per_aic <= 0.0) {
            std::fprintf(stderr, "AIC MTE2 derived result is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,%.9f,%.9f,pass\n",
            cell.rows, cell.columns, cell.tile_bytes, cell.mode, cell.block_dim,
            cell.iterations, short_cycles, short_wall_ns, long_cycles,
            long_wall_ns,
            tick_ns, cycles_per_copy, ns_per_copy, gb_per_s_per_aic);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
