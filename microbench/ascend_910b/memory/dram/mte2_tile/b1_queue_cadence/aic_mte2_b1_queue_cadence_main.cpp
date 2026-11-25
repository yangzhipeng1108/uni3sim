// ACL host for the independent GroupedMatmul B1 queue-cadence probe.
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
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",     \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_b1_queue_cadence(
    uint32_t blockDim, void* stream, void* gm_source, void* gm_output,
    int64_t source_row_stride, int64_t source_group_span,
    int64_t source_groups, int64_t iterations, int64_t source_mode);

namespace {

constexpr int64_t kElementBytes = 2;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kGroupRows = 256;
constexpr int64_t kGroupColumns = 256;
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
    int64_t source_row_stride;
    int64_t source_mode;
    int64_t block_dim;
    int64_t iterations;
    int64_t source_group_span;
    int64_t source_groups;
};

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 7 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 3) % 4 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch footprint_MB source_row_stride source_mode "
            "block_dim iterations [cell ...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t footprint_mb = ParseInteger(argv[2], "footprint_MB");
    if (footprint_mb < 64 || footprint_mb > 49152) {
        std::fprintf(stderr, "footprint_MB must be in [64,49152]\n");
        return EXIT_FAILURE;
    }
    const size_t source_bytes = static_cast<size_t>(footprint_mb) * 1024 * 1024;
    const int64_t source_elements = static_cast<int64_t>(source_bytes / kElementBytes);
    std::vector<Cell> cells;
    int64_t maximum_block_dim = 0;
    for (int argument = 3; argument < argc; argument += 4) {
        const int64_t source_row_stride = ParseInteger(
            argv[argument], "source_row_stride");
        const int64_t source_mode = ParseInteger(argv[argument + 1], "source_mode");
        const int64_t block_dim = ParseInteger(argv[argument + 2], "block_dim");
        const int64_t iterations = ParseInteger(argv[argument + 3], "iterations");
        const int64_t source_group_span = (
            (kGroupRows - 1) * source_row_stride + kGroupColumns);
        const int64_t source_groups = source_elements /
            (block_dim * source_group_span);
        if (source_row_stride < kGroupColumns || source_row_stride > 65535
                || source_mode < 0 || source_mode > 3
                || block_dim <= 0 || block_dim > 24 || iterations <= 0
                || iterations > std::numeric_limits<int64_t>::max() / 4
                || source_groups <= 0) {
            std::fprintf(stderr, "invalid B1 queue cell\n");
            return EXIT_FAILURE;
        }
        cells.push_back(Cell{source_row_stride, source_mode, block_dim, iterations,
                             source_group_span, source_groups});
        maximum_block_dim = std::max(maximum_block_dim, block_dim);
    }

    void* device_source = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(&device_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, static_cast<size_t>(maximum_block_dim * kCycleSlotBytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_source, source_bytes, 0, source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::puts(
        "source_row_stride,source_mode,active_aic_groups,iterations,source_groups,"
        "short_max_cycles,long_max_cycles,cycles_per_group,ns_per_group,"
        "b1_bytes_per_group,source_span_bytes,validation");
    for (const Cell& cell : cells) {
        auto run = [&](int64_t iterations) {
            const size_t output_bytes = static_cast<size_t>(
                cell.block_dim * kCycleSlotBytes);
            ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
            ACL_CHECK(aclrtlaunch_aic_mte2_b1_queue_cadence(
                static_cast<uint32_t>(cell.block_dim), stream, device_source,
                device_output, cell.source_row_stride, cell.source_group_span,
                cell.source_groups, iterations, cell.source_mode));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
            ACL_CHECK(aclrtMemcpy(output.data(), output_bytes, device_output,
                                  output_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
            int64_t maximum_cycles = 0;
            for (int64_t block = 0; block < cell.block_dim; ++block) {
                const size_t offset = static_cast<size_t>(
                    block * kCycleSlotBytes / sizeof(int64_t));
                if (output[offset] <= 0 || output[offset + 1] != iterations) {
                    std::fprintf(stderr, "B1 queue validation failed\n");
                    std::exit(EXIT_FAILURE);
                }
                maximum_cycles = std::max(maximum_cycles, output[offset]);
            }
            return maximum_cycles;
        };
        run(cell.iterations);
        const int64_t short_cycles = run(cell.iterations);
        const int64_t long_cycles = run(cell.iterations * 4);
        if (long_cycles <= short_cycles) {
            std::fprintf(stderr, "B1 queue differential interval did not increase\n");
            return EXIT_FAILURE;
        }
        const double cycles_per_group = static_cast<double>(
            long_cycles - short_cycles) / (3.0 * cell.iterations);
        const double ns_per_group = cycles_per_group * kSystemCounterTickNs;
        if (!std::isfinite(ns_per_group) || ns_per_group <= 0.0) {
            std::fprintf(stderr, "B1 queue duration is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,%ld,%ld,pass\n",
            cell.source_row_stride, cell.source_mode, cell.block_dim, cell.iterations,
            cell.source_groups, short_cycles, long_cycles, cycles_per_group,
            ns_per_group, kGroupRows * kGroupColumns * kElementBytes,
            cell.source_group_span * kElementBytes);
        std::fflush(stdout);
    }
    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
