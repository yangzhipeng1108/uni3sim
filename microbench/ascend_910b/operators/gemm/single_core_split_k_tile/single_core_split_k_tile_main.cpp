// ACL host for the resident MatMulV3 key-65632 base-tile service probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
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

extern "C" uint32_t aclrtlaunch_single_core_split_k_tile(
    uint32_t block_dim, void* stream, void* gm_a, void* gm_b,
    void* gm_workspace, void* gm_cycles, int64_t tile_k,
    int64_t iterations);

namespace {

constexpr int64_t kTileM = 128;
constexpr int64_t kTileN = 128;
constexpr int64_t kMaximumTileK = 384;
constexpr int64_t kWorkerCount = 24;
constexpr int64_t kCycleSlotElements = 8;
constexpr double kSystemCounterTickNs = 20.0;

int64_t ParsePositiveInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "%s must be positive, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 2 != 0) {
        std::fprintf(
            stderr, "usage: %s --batch tile_k iterations [...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    struct Cell {
        int64_t tile_k;
        int64_t iterations;
    };
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>((argc - 2) / 2));
    for (int argument = 2; argument < argc; argument += 2) {
        const int64_t tile_k = ParsePositiveInteger(argv[argument], "tile_k");
        if (tile_k != 256 && tile_k != kMaximumTileK) {
            std::fprintf(stderr, "tile_k must be 256 or 384\n");
            return EXIT_FAILURE;
        }
        cells.push_back({
            tile_k,
            ParsePositiveInteger(argv[argument + 1], "iterations"),
        });
    }

    constexpr size_t a_bytes =
        static_cast<size_t>(kTileM * kMaximumTileK * sizeof(uint16_t));
    constexpr size_t b_bytes =
        static_cast<size_t>(kMaximumTileK * kTileN * sizeof(uint16_t));
    constexpr size_t cycle_bytes = static_cast<size_t>(
        kWorkerCount * kCycleSlotElements * sizeof(int64_t));
    constexpr size_t workspace_bytes = static_cast<size_t>(
        kWorkerCount * kTileM * kTileN * sizeof(float));
    void* device_a = nullptr;
    void* device_b = nullptr;
    void* device_workspace = nullptr;
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(&device_a, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_b, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_workspace, workspace_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles, cycle_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_a, a_bytes, 0, a_bytes));
    ACL_CHECK(aclrtMemset(device_b, b_bytes, 0, b_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto run = [&](const Cell& cell) {
        ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0, cycle_bytes));
        ACL_CHECK(aclrtlaunch_single_core_split_k_tile(
            static_cast<uint32_t>(kWorkerCount), stream,
            device_a, device_b, device_workspace, device_cycles,
            cell.tile_k, cell.iterations));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> output(
            cycle_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            output.data(), cycle_bytes, device_cycles, cycle_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        std::vector<int64_t> cycles;
        cycles.reserve(static_cast<size_t>(kWorkerCount));
        for (int64_t worker = 0; worker < kWorkerCount; ++worker) {
            const size_t offset = static_cast<size_t>(
                worker * kCycleSlotElements);
            if (output[offset] <= 0
                    || output[offset + 1] != cell.iterations) {
                std::fprintf(stderr, "worker %ld failed validation\n", worker);
                std::exit(EXIT_FAILURE);
            }
            cycles.push_back(output[offset]);
        }
        return cycles;
    };

    std::puts(
        "iterations,block_dim,tile_m,tile_n,tile_k,max_cycles,"
        "max_ns,worker_cycles,validation");
    for (const Cell& cell : cells) {
        run(cell);
        const std::vector<int64_t> cycles = run(cell);
        const int64_t maximum = *std::max_element(
            cycles.begin(), cycles.end());
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%.1f,\"",
            cell.iterations, kWorkerCount, kTileM, kTileN, cell.tile_k,
            maximum, maximum * kSystemCounterTickNs);
        for (size_t index = 0; index < cycles.size(); ++index) {
            if (index != 0) {
                std::putchar(';');
            }
            std::printf("%ld", cycles[index]);
        }
        std::puts("\",exact");
    }

    ACL_CHECK(aclrtFree(device_a));
    ACL_CHECK(aclrtFree(device_b));
    ACL_CHECK(aclrtFree(device_workspace));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
