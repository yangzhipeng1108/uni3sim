// ACL host for the independent grouped-MatMul A/B MTE2 pair probe.
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

extern "C" uint32_t aclrtlaunch_aic_mte2_grouped_pair(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t a_rows, int64_t b_columns,
    int64_t a_source_tiles, int64_t b_source_tiles, int64_t iterations,
    int64_t drain_interval);

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
    int64_t a_rows;
    int64_t b_columns;
    int64_t a_source_tiles;
    int64_t b_source_tiles;
    int64_t drain_interval;
    int64_t block_dim;
    int64_t iterations;
};

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 9 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 7 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch "
            "a_rows b_columns a_source_tiles b_source_tiles drain_interval "
            "block_dim iterations [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int cell_count = (argc - 2) / 7;
    if (cell_count > 64) {
        std::fprintf(stderr, "a batch cannot contain more than 64 cells\n");
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    int64_t maximum_block_dim = 0;
    int64_t maximum_a_bytes = 0;
    int64_t maximum_b_bytes = 0;
    for (int argument = 2; argument < argc; argument += 7) {
        const int64_t a_rows = ParseInteger(argv[argument], "a_rows");
        const int64_t b_columns = ParseInteger(
            argv[argument + 1], "b_columns");
        const int64_t a_source_tiles = ParseInteger(
            argv[argument + 2], "a_source_tiles");
        const int64_t b_source_tiles = ParseInteger(
            argv[argument + 3], "b_source_tiles");
        const int64_t drain_interval = ParseInteger(
            argv[argument + 4], "drain_interval");
        const int64_t block_dim = ParseInteger(
            argv[argument + 5], "block_dim");
        const int64_t iterations = ParseInteger(
            argv[argument + 6], "iterations");
        if ((a_rows != 16 && a_rows != 32 && a_rows != 128)
                || (b_columns != 240 && b_columns != 256)
                || a_source_tiles <= 0 || b_source_tiles <= 0
                || (drain_interval != 0 && drain_interval != 4)
                || block_dim <= 0 || block_dim > 24
                || a_source_tiles < block_dim || b_source_tiles < block_dim
                || iterations <= 0
                || iterations > std::numeric_limits<int64_t>::max() / 4
                || (drain_interval == 4 && iterations % 4 != 0)) {
            std::fprintf(stderr, "invalid grouped MTE2 cell at argument %d\n", argument);
            return EXIT_FAILURE;
        }
        cells.push_back(Cell{
            a_rows, b_columns, a_source_tiles, b_source_tiles, drain_interval,
            block_dim, iterations});
        maximum_block_dim = std::max(maximum_block_dim, block_dim);
        maximum_a_bytes = std::max(
            maximum_a_bytes, a_rows * 64 * kElementBytes * a_source_tiles);
        maximum_b_bytes = std::max(
            maximum_b_bytes,
            64 * b_columns * kElementBytes * b_source_tiles);
    }

    void* device_a_source = nullptr;
    void* device_b_source = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_a_source, static_cast<size_t>(maximum_a_bytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_b_source, static_cast<size_t>(maximum_b_bytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output,
        static_cast<size_t>(maximum_block_dim * kCycleSlotBytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(
        device_a_source, static_cast<size_t>(maximum_a_bytes), 0,
        static_cast<size_t>(maximum_a_bytes)));
    ACL_CHECK(aclrtMemset(
        device_b_source, static_cast<size_t>(maximum_b_bytes), 0,
        static_cast<size_t>(maximum_b_bytes)));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto run = [&](const Cell& cell, int64_t iterations,
                   int64_t& cycles, int64_t& wall_ns) {
        const size_t output_bytes = static_cast<size_t>(
            cell.block_dim * kCycleSlotBytes);
        ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_aic_mte2_grouped_pair(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_a_source, device_b_source, device_output, cell.a_rows,
            cell.b_columns, cell.a_source_tiles, cell.b_source_tiles, iterations,
            cell.drain_interval));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        struct timespec begin;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &begin));
        ACL_CHECK(aclrtlaunch_aic_mte2_grouped_pair(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_a_source, device_b_source, device_output, cell.a_rows,
            cell.b_columns, cell.a_source_tiles, cell.b_source_tiles, iterations,
            cell.drain_interval));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        wall_ns = static_cast<int64_t>(end.tv_sec - begin.tv_sec) * 1000000000LL
            + static_cast<int64_t>(end.tv_nsec - begin.tv_nsec);
        std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
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
                std::fprintf(stderr, "grouped MTE2 block %ld failed validation\n", block);
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
            std::fprintf(stderr, "grouped MTE2 differential interval did not increase\n");
            return EXIT_FAILURE;
        }
        const double cycles_per_pair = static_cast<double>(
            long_cycles - short_cycles) / static_cast<double>(
                cell.iterations * 3);
        const double ns_per_pair = cycles_per_pair * kSystemCounterTickNs;
        const int64_t a_tile_bytes = cell.a_rows * 64 * kElementBytes;
        const int64_t b_tile_bytes = 64 * cell.b_columns * kElementBytes;
        const double gb_per_s_per_aic = (
            (a_tile_bytes + b_tile_bytes) / ns_per_pair);
        if (!std::isfinite(ns_per_pair) || ns_per_pair <= 0.0
                || !std::isfinite(gb_per_s_per_aic)
                || gb_per_s_per_aic <= 0.0) {
            std::fprintf(stderr, "grouped MTE2 derived result is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,"
            "%.9f,%.9f,%.9f,%.9f,pass\n",
            cell.a_rows, cell.b_columns, a_tile_bytes, b_tile_bytes,
            cell.a_source_tiles, cell.b_source_tiles, cell.drain_interval,
            cell.block_dim, cell.iterations, short_cycles, short_wall_ns,
            long_cycles, long_wall_ns, kSystemCounterTickNs, cycles_per_pair,
            ns_per_pair, gb_per_s_per_aic);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtFree(device_a_source));
    ACL_CHECK(aclrtFree(device_b_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
