// ACL host for the source-shaped MatMulV2 A1/B1 cold-pair MTE2 probe.
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

extern "C" uint32_t aclrtlaunch_aic_mte2_legacy_l1_pair(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t a_rows, int64_t a_columns,
    int64_t a_source_row_stride, int64_t a_source_tile_stride,
    int64_t a_source_tiles, int64_t b_rows, int64_t b_columns,
    int64_t b_source_row_stride, int64_t b_source_tile_stride,
    int64_t b_source_tiles, int64_t pairs_per_program, int64_t programs,
    int64_t drain_after_each_pair);

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
    int64_t geometry_id;
    int64_t drain_after_each_pair;
    int64_t a_rows;
    int64_t a_columns;
    int64_t a_source_row_stride;
    int64_t b_rows;
    int64_t b_columns;
    int64_t b_source_row_stride;
    int64_t pairs_per_program;
    int64_t block_dim;
};

struct Timing {
    int64_t cycles;
    int64_t wall_ns;
};

Cell MakeCell(int64_t geometry_id, int64_t drain_after_each_pair) {
    if (drain_after_each_pair != 0 && drain_after_each_pair != 1) {
        std::fprintf(stderr, "drain_after_each_pair must be zero or one\n");
        std::exit(EXIT_FAILURE);
    }
    if (geometry_id == 0) {
        return Cell{
            geometry_id, drain_after_each_pair,
            128, 256, 14336, 96, 256, 14336, 112, 22};
    }
    if (geometry_id == 1) {
        return Cell{
            geometry_id, drain_after_each_pair,
            256, 256, 5120, 112, 256, 5120, 80, 24};
    }
    if (geometry_id >= 2 && geometry_id <= 5) {
        return Cell{
            geometry_id, drain_after_each_pair,
            16 * (geometry_id - 1), 256, 14336,
            96, 256, 14336, 112, 22};
    }
    // Descriptor-coalescing controls retain total A/B bytes and source order
    // while replacing every two 256-column copies with one 512-column copy.
    // They isolate the L1 source-program effect from Cube and FixPipe work.
    if (geometry_id == 6 || geometry_id == 7) {
        const int64_t columns = geometry_id == 6 ? 256 : 512;
        return Cell{
            geometry_id, drain_after_each_pair,
            80, columns, 14336, 128, columns, 14336,
            geometry_id == 6 ? 168 : 84, 22};
    }
    if (geometry_id == 8 || geometry_id == 9) {
        const int64_t columns = geometry_id == 8 ? 256 : 512;
        return Cell{
            geometry_id, drain_after_each_pair,
            112, columns, 2048, 96, columns, 2048,
            geometry_id == 8 ? 16 : 8, 22};
    }
    if (geometry_id == 10 || geometry_id == 11) {
        const int64_t columns = geometry_id == 10 ? 256 : 512;
        return Cell{
            geometry_id, drain_after_each_pair,
            128, columns, 12800, 112, columns, 12800,
            geometry_id == 10 ? 200 : 100, 23};
    }
    std::fprintf(stderr, "geometry_id must be in [0,11]\n");
    std::exit(EXIT_FAILURE);
}

int64_t TileStride(const Cell& cell, bool operand_a) {
    const int64_t rows = operand_a ? cell.a_rows : cell.b_rows;
    const int64_t columns = operand_a ? cell.a_columns : cell.b_columns;
    const int64_t row_stride = operand_a
        ? cell.a_source_row_stride : cell.b_source_row_stride;
    return (rows - 1) * row_stride + columns;
}

int64_t LocalBytes(const Cell& cell, bool operand_a) {
    const int64_t rows = operand_a ? cell.a_rows : cell.b_rows;
    const int64_t columns = operand_a ? cell.a_columns : cell.b_columns;
    return ((rows + 15) / 16 * 16) * columns * kElementBytes;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 7 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 5) % 2 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch footprint_MB short_programs long_programs "
            "geometry_id drain_after_each_pair [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t footprint_mb = ParseInteger(argv[2], "footprint_MB");
    const int64_t short_programs = ParseInteger(argv[3], "short_programs");
    const int64_t long_programs = ParseInteger(argv[4], "long_programs");
    if (footprint_mb != 16384) {
        std::fprintf(stderr, "footprint_MB must be exactly 16384 per operand\n");
        return EXIT_FAILURE;
    }
    if (short_programs <= 0 || long_programs <= short_programs) {
        std::fprintf(stderr, "program counts must be positive and increasing\n");
        return EXIT_FAILURE;
    }
    const int cell_count = (argc - 5) / 2;
    if (cell_count <= 0 || cell_count > 12) {
        std::fprintf(stderr, "a batch must contain one to twelve cells\n");
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    int64_t maximum_block_dim = 0;
    for (int argument = 5; argument < argc; argument += 2) {
        const Cell cell = MakeCell(
            ParseInteger(argv[argument], "geometry_id"),
            ParseInteger(argv[argument + 1], "drain_after_each_pair"));
        const int64_t l1_bytes = 2 * (
            LocalBytes(cell, true) + LocalBytes(cell, false));
        if (l1_bytes > 512 * 1024) {
            std::fprintf(stderr, "pair geometry exceeds the 910B AIC L1\n");
            return EXIT_FAILURE;
        }
        cells.push_back(cell);
        maximum_block_dim = std::max(maximum_block_dim, cell.block_dim);
    }

    const size_t source_bytes = static_cast<size_t>(footprint_mb)
        * 1024 * 1024;
    const int64_t source_elements = static_cast<int64_t>(
        source_bytes / kElementBytes);
    void* device_a_source = nullptr;
    void* device_b_source = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_a_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_b_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output,
        static_cast<size_t>(maximum_block_dim * kCycleSlotBytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_a_source, source_bytes, 0, source_bytes));
    ACL_CHECK(aclrtMemset(device_b_source, source_bytes, 0, source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto run = [&](const Cell& cell, int64_t programs) -> Timing {
        const int64_t a_source_tile_stride = TileStride(cell, true);
        const int64_t b_source_tile_stride = TileStride(cell, false);
        const int64_t a_source_tiles = source_elements
            / (a_source_tile_stride * cell.block_dim);
        const int64_t b_source_tiles = source_elements
            / (b_source_tile_stride * cell.block_dim);
        if (a_source_tiles < cell.pairs_per_program
                || b_source_tiles < cell.pairs_per_program) {
            std::fprintf(stderr, "footprint cannot hold one disjoint program\n");
            std::exit(EXIT_FAILURE);
        }
        const size_t output_bytes = static_cast<size_t>(
            cell.block_dim * kCycleSlotBytes);
        ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_aic_mte2_legacy_l1_pair(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_a_source, device_b_source, device_output,
            cell.a_rows, cell.a_columns, cell.a_source_row_stride,
            a_source_tile_stride, a_source_tiles,
            cell.b_rows, cell.b_columns, cell.b_source_row_stride,
            b_source_tile_stride, b_source_tiles, cell.pairs_per_program,
            1, cell.drain_after_each_pair));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        struct timespec begin;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &begin));
        ACL_CHECK(aclrtlaunch_aic_mte2_legacy_l1_pair(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_a_source, device_b_source, device_output,
            cell.a_rows, cell.a_columns, cell.a_source_row_stride,
            a_source_tile_stride, a_source_tiles,
            cell.b_rows, cell.b_columns, cell.b_source_row_stride,
            b_source_tile_stride, b_source_tiles, cell.pairs_per_program,
            programs, cell.drain_after_each_pair));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            output.data(), output_bytes, device_output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t cycles = 0;
        for (int64_t block = 0; block < cell.block_dim; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            if (output[offset] <= 0 || output[offset + 1] != 1) {
                std::fprintf(
                    stderr, "legacy L1 pair block %ld failed validation\n",
                    block);
                std::exit(EXIT_FAILURE);
            }
            cycles = std::max(cycles, output[offset]);
        }
        return Timing{
            cycles,
            static_cast<int64_t>(end.tv_sec - begin.tv_sec) * 1000000000LL
                + static_cast<int64_t>(end.tv_nsec - begin.tv_nsec),
        };
    };

    for (const Cell& cell : cells) {
        const Timing short_timing = run(cell, short_programs);
        const Timing long_timing = run(cell, long_programs);
        if (long_timing.cycles <= short_timing.cycles
                || long_timing.wall_ns <= short_timing.wall_ns) {
            std::fprintf(
                stderr, "legacy L1 pair differential interval did not increase\n");
            return EXIT_FAILURE;
        }
        const double cycles_per_program = static_cast<double>(
            long_timing.cycles - short_timing.cycles)
            / static_cast<double>(long_programs - short_programs);
        const double ns_per_program = cycles_per_program
            * kSystemCounterTickNs;
        const double ns_per_pair = ns_per_program
            / static_cast<double>(cell.pairs_per_program);
        const int64_t pair_bytes = (
            cell.a_rows * cell.a_columns
            + cell.b_rows * cell.b_columns) * kElementBytes;
        const double gb_per_s_per_aic = pair_bytes / ns_per_pair;
        if (!std::isfinite(ns_per_program) || ns_per_program <= 0.0
                || !std::isfinite(ns_per_pair) || ns_per_pair <= 0.0
                || !std::isfinite(gb_per_s_per_aic)
                || gb_per_s_per_aic <= 0.0) {
            std::fprintf(stderr, "legacy L1 pair result is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,"
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,%.9f,%.9f,pass\n",
            cell.geometry_id, cell.drain_after_each_pair, footprint_mb,
            cell.block_dim, cell.a_rows, cell.a_columns,
            cell.a_source_row_stride, cell.b_rows, cell.b_columns,
            cell.b_source_row_stride, cell.pairs_per_program,
            LocalBytes(cell, true), LocalBytes(cell, false),
            pair_bytes, short_programs, long_programs,
            short_timing.cycles, short_timing.wall_ns,
            long_timing.cycles, long_timing.wall_ns,
            static_cast<int64_t>(kSystemCounterTickNs), cycles_per_program,
            ns_per_program, ns_per_pair, gb_per_s_per_aic);
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
