// ACL host for the independent full-A/B-credit MTE2 probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_full_a_b_credit(
    uint32_t block_dim, void* stream, void* gm_a, void* gm_b,
    void* gm_output, int64_t a_rows, int64_t a_columns, int64_t b_rows,
    int64_t b_columns, int64_t b_source_stride, int64_t b_tile_span,
    int64_t b_source_tiles, int64_t first_tile, int64_t waves,
    int64_t n_slabs, int64_t mode);

namespace {

constexpr int64_t kElementBytes = 2;
constexpr int64_t kMaxBlocks = 24;
constexpr int64_t kMaxWaves = 24;
constexpr int64_t kOutputSlotsPerBlock = 32;
constexpr int64_t kL1BudgetBytes = 512 * 1024;
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

int64_t Align16(int64_t value) {
    return (value + 15) / 16 * 16;
}

struct Cell {
    int64_t a_rows;
    int64_t a_columns;
    int64_t b_rows;
    int64_t b_columns;
    int64_t b_source_stride;
    int64_t n_slabs;
    int64_t waves;
    int64_t block_dim;
    int64_t b_tile_span;
    int64_t b_source_tiles;
    int64_t next_tile;
};

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 6 || std::strcmp(argv[1], "--trials") != 0
            || std::strcmp(argv[3], "--footprint-mb") != 0
            || (argc - 5) % 8 != 0) {
        std::fprintf(stderr,
            "usage: %s --trials COUNT --footprint-mb MB "
            "[a_rows a_columns b_rows b_columns b_source_stride n_slabs "
            "waves block_dim ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[2], "trials");
    const int64_t footprint_mb = ParseInteger(argv[4], "footprint-mb");
    const int64_t cell_count = (argc - 5) / 8;
    if (trials < 3 || trials > 24 || footprint_mb < 16
            || footprint_mb > 57344 || cell_count <= 0 || cell_count > 128) {
        std::fprintf(stderr, "invalid full-A B-credit probe sweep\n");
        return EXIT_FAILURE;
    }
    const size_t b_source_bytes = static_cast<size_t>(footprint_mb) * 1024
        * 1024;
    const int64_t b_source_elements = static_cast<int64_t>(
        b_source_bytes / kElementBytes);

    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    size_t a_source_bytes = 0;
    for (int argument = 5; argument < argc; argument += 8) {
        Cell cell;
        cell.a_rows = ParseInteger(argv[argument], "a_rows");
        cell.a_columns = ParseInteger(argv[argument + 1], "a_columns");
        cell.b_rows = ParseInteger(argv[argument + 2], "b_rows");
        cell.b_columns = ParseInteger(argv[argument + 3], "b_columns");
        cell.b_source_stride = ParseInteger(
            argv[argument + 4], "b_source_stride");
        cell.n_slabs = ParseInteger(argv[argument + 5], "n_slabs");
        cell.waves = ParseInteger(argv[argument + 6], "waves");
        cell.block_dim = ParseInteger(argv[argument + 7], "block_dim");
        cell.b_tile_span = (
            (cell.b_rows - 1) * cell.b_source_stride + cell.b_columns);
        cell.b_source_tiles = b_source_elements
            / (cell.b_tile_span * cell.block_dim);
        cell.next_tile = 0;
        const int64_t a_l1_bytes = (
            Align16(cell.a_rows) * cell.a_columns * kElementBytes);
        const int64_t b_l1_bytes = (
            Align16(cell.b_rows) * cell.b_columns * kElementBytes);
        if (cell.a_rows != 16
                || cell.a_columns < 64 || cell.a_columns > 16384
                || cell.a_columns % 64 != 0
                || cell.b_rows < 64 || cell.b_rows > 512
                || cell.b_rows % 64 != 0
                || cell.b_columns < 16 || cell.b_columns > 8192
                || cell.b_columns % 16 != 0
                || cell.b_source_stride < cell.b_columns
                || cell.b_source_stride > 65535
                || cell.n_slabs <= 0 || cell.n_slabs > 64
                || cell.waves <= 0 || cell.waves > kMaxWaves
                || cell.block_dim <= 0 || cell.block_dim > kMaxBlocks
                || a_l1_bytes + 2 * b_l1_bytes > kL1BudgetBytes
                || cell.b_source_tiles < 2) {
            std::fprintf(stderr,
                "invalid full-A B-credit cell at argument %d\n", argument);
            return EXIT_FAILURE;
        }
        a_source_bytes = std::max(a_source_bytes, static_cast<size_t>(
            cell.block_dim * cell.a_rows * cell.a_columns * kElementBytes));
        cells.push_back(cell);
    }

    void* device_a = nullptr;
    void* device_b = nullptr;
    void* device_output = nullptr;
    const size_t output_bytes = static_cast<size_t>(
        kMaxBlocks * kOutputSlotsPerBlock) * sizeof(int64_t);
    ACL_CHECK(aclrtMalloc(&device_a, a_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_b, b_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_a, a_source_bytes, 0, a_source_bytes));
    ACL_CHECK(aclrtMemset(device_b, b_source_bytes, 0, b_source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
    // launch returns per-wave max cycles across AICs into wave_max.
    auto launch = [&](Cell& cell, int64_t mode,
                      std::vector<int64_t>& wave_max,
                      int64_t& total_max, int64_t& a_load_max) {
        ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
        const auto wall_start = std::chrono::steady_clock::now();
        ACL_CHECK(aclrtlaunch_aic_mte2_full_a_b_credit(
            static_cast<uint32_t>(cell.block_dim), stream, device_a,
            device_b, device_output, cell.a_rows, cell.a_columns,
            cell.b_rows, cell.b_columns, cell.b_source_stride,
            cell.b_tile_span, cell.b_source_tiles, cell.next_tile,
            cell.waves, cell.n_slabs, mode));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        const auto wall_end = std::chrono::steady_clock::now();
        if (std::getenv("PROBE_DEBUG_DUMP") != nullptr) {
            std::fprintf(stderr, "launch wall_us %.1f mode %ld\n",
                std::chrono::duration<double, std::micro>(
                    wall_end - wall_start).count(), mode);
        }
        ACL_CHECK(aclrtMemcpy(output.data(), output_bytes, device_output,
                              output_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        const int64_t expected_slabs = cell.waves * cell.n_slabs;
        wave_max.assign(static_cast<size_t>(cell.waves), 0);
        total_max = 0;
        a_load_max = 0;
        for (int64_t block = 0; block < cell.block_dim; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kOutputSlotsPerBlock);
            if (std::getenv("PROBE_DEBUG_DUMP") != nullptr) {
                std::fprintf(stderr, "block %ld slots %ld %ld %ld %ld %ld\n",
                    block, output[offset], output[offset + 1],
                    output[offset + 2], output[offset + 3],
                    output[offset + 4]);
            }
            if (output[offset] <= 0 || output[offset + 1] <= 0
                    || output[offset + 2] != expected_slabs
                    || output[offset + 3] != mode) {
                std::fprintf(stderr,
                    "full-A B-credit probe validation failed on AIC %ld:"
                    " slots %ld %ld %ld %ld %ld (expected_slabs %ld mode"
                    " %ld)\n",
                    block, output[offset], output[offset + 1],
                    output[offset + 2], output[offset + 3],
                    output[offset + 4], expected_slabs, mode);
                if (std::getenv("PROBE_DEBUG_DUMP") != nullptr) {
                    continue;
                }
                std::exit(EXIT_FAILURE);
            }
            total_max = std::max(total_max, output[offset]);
            a_load_max = std::max(a_load_max, output[offset + 1]);
            for (int64_t wave = 0; wave < cell.waves; ++wave) {
                const int64_t wave_cycles = output[
                    offset + 4 + static_cast<size_t>(wave)];
                if (wave_cycles <= 0) {
                    std::fprintf(stderr,
                        "full-A B-credit probe wave validation failed on "
                        "AIC %ld wave %ld\n", block, wave);
                    std::exit(EXIT_FAILURE);
                }
                wave_max[static_cast<size_t>(wave)] = std::max(
                    wave_max[static_cast<size_t>(wave)], wave_cycles);
            }
        }
        // Advance the cyclic source so back-to-back launches do not replay
        // the same L2 region.  Skip the 32 wake-up tiles as well: they are
        // streamed right before each timed section and would otherwise be
        // re-read as the next launch's timed tiles.
        cell.next_tile = (cell.next_tile + expected_slabs + 32)
            % cell.b_source_tiles;
    };

    std::puts("trial,cell_index,mode,wave_index,waves,n_slabs,"
              "wave_max_cycles,wave_ns,wave_bytes_per_aic,gbps_per_aic,"
              "a_rows,a_columns,b_rows,b_columns,b_source_stride,block_dim,"
              "total_max_cycles,a_load_max_cycles,validation");
    std::vector<int64_t> wave_max;
    for (size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
        Cell& cell = cells[cell_index];
        // Warm both modes once per cell before the timed trials.
        for (int64_t mode = 0; mode < 2; ++mode) {
            int64_t total_max = 0;
            int64_t a_load_max = 0;
            launch(cell, mode, wave_max, total_max, a_load_max);
        }
        for (int64_t trial = 0; trial < trials; ++trial) {
            for (int64_t position = 0; position < 2; ++position) {
                const int64_t mode = (trial + position) & 1;
                int64_t total_max = 0;
                int64_t a_load_max = 0;
                launch(cell, mode, wave_max, total_max, a_load_max);
                const int64_t wave_bytes = cell.n_slabs * cell.b_rows
                    * cell.b_columns * kElementBytes;
                for (int64_t wave = 0; wave < cell.waves; ++wave) {
                    const int64_t cycles = wave_max[
                        static_cast<size_t>(wave)];
                    const double wave_ns = cycles * kSystemCounterTickNs;
                    const double gbps = static_cast<double>(wave_bytes)
                        / wave_ns;
                    std::printf("%ld,%zu,%s,%ld,%ld,%ld,%ld,%.9f,%ld,%.9f,"
                        "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,pass\n",
                        trial, cell_index,
                        mode == 0 ? "mte2_descriptor_drain"
                                  : "mte2_credit_preserved",
                        wave, cell.waves, cell.n_slabs, cycles, wave_ns,
                        wave_bytes, gbps, cell.a_rows, cell.a_columns,
                        cell.b_rows, cell.b_columns, cell.b_source_stride,
                        cell.block_dim, total_max, a_load_max);
                }
                std::fflush(stdout);
            }
        }
    }

    ACL_CHECK(aclrtFree(device_a));
    ACL_CHECK(aclrtFree(device_b));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
