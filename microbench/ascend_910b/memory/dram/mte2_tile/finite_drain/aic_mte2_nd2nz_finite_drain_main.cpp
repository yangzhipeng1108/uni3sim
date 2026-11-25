// ACL host for the independent finite drained AIC ND2NZ program.
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

extern "C" uint32_t aclrtlaunch_aic_mte2_nd2nz_finite_drain(
    uint32_t blockDim, void* stream, void* gm_source, void* gm_output,
    int64_t rows, int64_t columns, int64_t source_row_stride,
    int64_t source_tile_span, int64_t base_tile,
    int64_t descriptor_count);

namespace {

constexpr int64_t kElementBytes = 2;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kMaximumDescriptorCount = 160;
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
    int64_t source_row_stride;
    int64_t block_dim;
    int64_t descriptor_count;
    int64_t source_tile_span;
};

constexpr int64_t kDisjointSource = 0;
constexpr int64_t kReusedSource = 1;

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 10 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 5) % 5 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch footprint_MB trials source_mode rows columns "
            "source_row_stride block_dim descriptor_count [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t footprint_mb = ParseInteger(argv[2], "footprint_MB");
    const int64_t trials = ParseInteger(argv[3], "trials");
    const int64_t source_mode = ParseInteger(argv[4], "source_mode");
    if (footprint_mb < 512 || footprint_mb > 49152
            || trials <= 0 || trials > 16
            || (source_mode != kDisjointSource
                && source_mode != kReusedSource)) {
        std::fprintf(stderr, "invalid footprint, trial count, or source mode\n");
        return EXIT_FAILURE;
    }

    std::vector<Cell> cells;
    int64_t maximum_block_dim = 0;
    int64_t required_elements = 0;
    for (int argument = 5; argument < argc; argument += 5) {
        const int64_t rows = ParseInteger(argv[argument], "rows");
        const int64_t columns = ParseInteger(argv[argument + 1], "columns");
        const int64_t source_row_stride = ParseInteger(
            argv[argument + 2], "source_row_stride");
        const int64_t block_dim = ParseInteger(
            argv[argument + 3], "block_dim");
        const int64_t descriptor_count = ParseInteger(
            argv[argument + 4], "descriptor_count");
        const int64_t source_tile_span = (
            (rows - 1) * source_row_stride + columns);
        if (rows <= 0 || rows > 256 || columns < 16 || columns > 4096
                || columns % 16 != 0 || source_row_stride < columns
                || source_row_stride > 65535
                || block_dim <= 0 || block_dim > 24
                || descriptor_count <= 0
                || descriptor_count > kMaximumDescriptorCount
                || ((rows + 15) / 16 * 16) * columns * kElementBytes
                    > 256 * 1024) {
            std::fprintf(stderr, "invalid finite-drain cell\n");
            return EXIT_FAILURE;
        }
        cells.push_back(Cell{
            rows, columns, source_row_stride, block_dim,
            descriptor_count, source_tile_span});
        maximum_block_dim = std::max(maximum_block_dim, block_dim);
        const int64_t source_programs = (
            source_mode == kDisjointSource ? trials + 1 : 1);
        required_elements += (
            source_programs * block_dim * descriptor_count
            * source_tile_span);
    }
    const size_t source_bytes = static_cast<size_t>(
        footprint_mb) * 1024 * 1024;
    if (required_elements > static_cast<int64_t>(
            source_bytes / kElementBytes)) {
        std::fprintf(stderr, "finite-drain cells exceed the source footprint\n");
        return EXIT_FAILURE;
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

    std::printf(
        "rows,columns,source_row_stride,active_aic_groups,"
        "descriptor_count,source_mode,trial,max_cycles,duration_ns,"
        "validation\n");
    int64_t base_tile = 0;
    for (const Cell& cell : cells) {
        for (int64_t trial = 0; trial <= trials; ++trial) {
            const size_t output_bytes = static_cast<size_t>(
                cell.block_dim * kCycleSlotBytes);
            ACL_CHECK(aclrtMemset(
                device_output, output_bytes, 0, output_bytes));
            ACL_CHECK(aclrtlaunch_aic_mte2_nd2nz_finite_drain(
                static_cast<uint32_t>(cell.block_dim), stream,
                device_source, device_output, cell.rows, cell.columns,
                cell.source_row_stride, cell.source_tile_span,
                base_tile, cell.descriptor_count));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<int64_t> output(
                output_bytes / sizeof(int64_t), 0);
            ACL_CHECK(aclrtMemcpy(
                output.data(), output_bytes, device_output, output_bytes,
                ACL_MEMCPY_DEVICE_TO_HOST));
            int64_t maximum_cycles = 0;
            for (int64_t block = 0; block < cell.block_dim; ++block) {
                const int64_t offset = (
                    block * kCycleSlotBytes / sizeof(int64_t));
                const int64_t cycles = output[static_cast<size_t>(offset)];
                const int64_t validation = output[
                    static_cast<size_t>(offset + 1)];
                if (cycles <= 0 || validation != cell.descriptor_count) {
                    std::fprintf(stderr, "finite-drain validation failed\n");
                    return EXIT_FAILURE;
                }
                maximum_cycles = std::max(maximum_cycles, cycles);
            }
            if (trial > 0) {
                std::printf(
                    "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,pass\n",
                    cell.rows, cell.columns, cell.source_row_stride,
                    cell.block_dim, cell.descriptor_count, source_mode, trial,
                    maximum_cycles,
                    maximum_cycles * kSystemCounterTickNs);
                std::fflush(stdout);
            }
            if (source_mode == kDisjointSource) {
                base_tile += (
                    cell.block_dim * cell.descriptor_count
                    * cell.source_tile_span);
            }
        }
        if (source_mode == kReusedSource) {
            base_tile += (
                cell.block_dim * cell.descriptor_count
                * cell.source_tile_span);
        }
    }

    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
