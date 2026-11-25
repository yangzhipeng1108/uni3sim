// ACL host for the independent GroupedMatmul L1-to-L0 pair benchmark.
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

extern "C" uint32_t aclrtlaunch_grouped_matmul_mte1_pair(
    uint32_t blockDim, void* stream, void* gm_source, void* gm_cycles,
    int64_t base_m, int64_t base_n, int64_t b_step_k, int64_t iterations);

namespace {

constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kSourceElements = 128 * 64 + 256 * 256;
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
    int64_t base_m;
    int64_t base_n;
    int64_t b_step_k;
    int64_t block_dim;
    int64_t iterations;
};

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    const bool legacy_batch = argc >= 5 && std::strcmp(argv[1], "--batch") == 0;
    const bool geometry_batch = (
        argc >= 7 && std::strcmp(argv[1], "--batch-geometry") == 0);
    const int fields_per_cell = legacy_batch ? 3 : 5;
    if ((!legacy_batch && !geometry_batch)
            || (argc - 2) % fields_per_cell != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch base_m block_dim iterations [cell ...]\n"
            "   or: %s --batch-geometry base_m base_n b_step_k block_dim "
            "iterations [cell ...]\n",
            argv[0],
            argv[0]);
        return EXIT_FAILURE;
    }
    const int cell_count = (argc - 2) / fields_per_cell;
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    int64_t maximum_block_dim = 0;
    for (int argument = 2; argument < argc; argument += fields_per_cell) {
        const int64_t base_m = ParseInteger(argv[argument], "base_m");
        const int64_t base_n = legacy_batch
            ? 256 : ParseInteger(argv[argument + 1], "base_n");
        const int64_t b_step_k = legacy_batch
            ? 256 : ParseInteger(argv[argument + 2], "b_step_k");
        const int block_offset = legacy_batch ? 1 : 3;
        const int iteration_offset = legacy_batch ? 2 : 4;
        Cell cell{
            base_m,
            base_n,
            b_step_k,
            ParseInteger(argv[argument + block_offset], "block_dim"),
            ParseInteger(argv[argument + iteration_offset], "iterations"),
        };
        if (cell.base_m < 16 || cell.base_m > 128 || cell.base_m % 16 != 0
                || cell.base_n < 16 || cell.base_n > 256
                || cell.base_n % 16 != 0
                || cell.b_step_k < 64 || cell.b_step_k > 256
                || cell.b_step_k % 64 != 0
                || cell.block_dim <= 0 || cell.block_dim > 24
                || cell.iterations <= 0
                || cell.iterations > std::numeric_limits<int64_t>::max() / 4) {
            std::fprintf(stderr, "invalid MTE1 cell at argument %d\n", argument);
            return EXIT_FAILURE;
        }
        cells.push_back(cell);
        maximum_block_dim = std::max(maximum_block_dim, cell.block_dim);
    }

    void* device_source = nullptr;
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_source, kSourceElements * sizeof(uint16_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles,
        static_cast<size_t>(maximum_block_dim * kCycleSlotBytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(
        device_source, kSourceElements * sizeof(uint16_t), 0,
        kSourceElements * sizeof(uint16_t)));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto run = [&](const Cell& cell, int64_t iterations) {
        const size_t output_bytes = static_cast<size_t>(
            cell.block_dim * kCycleSlotBytes);
        ACL_CHECK(aclrtMemset(device_cycles, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_grouped_matmul_mte1_pair(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_source, device_cycles, cell.base_m, cell.base_n,
            cell.b_step_k, iterations));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            output.data(), output_bytes, device_cycles, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t minimum_cycles = std::numeric_limits<int64_t>::max();
        int64_t maximum_cycles = 0;
        for (int64_t block = 0; block < cell.block_dim; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            if (output[offset] <= 0 || output[offset + 1] != 1) {
                std::fprintf(stderr, "MTE1 block %ld failed validation\n", block);
                std::exit(EXIT_FAILURE);
            }
            minimum_cycles = std::min(minimum_cycles, output[offset]);
            maximum_cycles = std::max(maximum_cycles, output[offset]);
        }
        return std::pair<int64_t, int64_t>{minimum_cycles, maximum_cycles};
    };

    std::puts(
        "base_m,base_n,b_step_k,block_dim,iterations,short_min_cycles,short_max_cycles,"
        "long_min_cycles,long_max_cycles,cycles_per_pair,ns_per_pair,validation");
    for (const Cell& cell : cells) {
        run(cell, cell.iterations);
        const auto short_cycles = run(cell, cell.iterations);
        const auto long_cycles = run(cell, cell.iterations * 4);
        if (long_cycles.second <= short_cycles.second) {
            std::fprintf(stderr, "MTE1 differential interval did not increase\n");
            return EXIT_FAILURE;
        }
        const double cycles_per_pair = static_cast<double>(
            long_cycles.second - short_cycles.second)
            / (3.0 * cell.iterations);
        const double ns_per_pair = cycles_per_pair * kSystemCounterTickNs;
        if (!std::isfinite(ns_per_pair) || ns_per_pair <= 0.0) {
            std::fprintf(stderr, "MTE1 derived result is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,pass\n",
            cell.base_m, cell.base_n, cell.b_step_k, cell.block_dim,
            cell.iterations,
            short_cycles.first, short_cycles.second,
            long_cycles.first, long_cycles.second,
            cycles_per_pair, ns_per_pair);
    }

    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
