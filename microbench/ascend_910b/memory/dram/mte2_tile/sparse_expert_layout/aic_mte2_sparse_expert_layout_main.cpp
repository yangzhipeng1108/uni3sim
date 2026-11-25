// ACL host for the independent GroupedMatmul sparse-expert B-ingress probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",      \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_sparse_expert_layout(
    uint32_t blockDim, void* stream, void* gm_source, void* gm_output,
    int64_t layout, int64_t repetitions);

namespace {

constexpr int64_t kElementBytes = 2;
constexpr int64_t kActiveAic = 24;
constexpr int64_t kExperts = 8;
constexpr int64_t kNTiles = 128;
constexpr int64_t kTileN = 256;
constexpr int64_t kProjectionK = 6144;
constexpr int64_t kExpertSpan = kNTiles * kTileN * kProjectionK;
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

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 5 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 2 != 0) {
        std::fprintf(stderr, "usage: %s --batch layout repetitions [cell ...]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    struct Cell { int64_t layout; int64_t repetitions; };
    std::vector<Cell> cells;
    for (int argument = 2; argument < argc; argument += 2) {
        const int64_t layout = ParseInteger(argv[argument], "layout");
        const int64_t repetitions = ParseInteger(
            argv[argument + 1], "repetitions");
        if ((layout != 0 && layout != 1) || repetitions <= 0) {
            std::fprintf(stderr, "layout must be 0/1 and repetitions positive\n");
            return EXIT_FAILURE;
        }
        cells.push_back(Cell{layout, repetitions});
    }
    const size_t source_bytes = static_cast<size_t>(
        kExperts * kExpertSpan * kElementBytes);
    void* device_source = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(&device_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_output,
                          static_cast<size_t>(kActiveAic * kCycleSlotBytes),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_source, source_bytes, 0, source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::puts("layout,repetitions,short_max_cycles,long_max_cycles,"
              "cycles_per_pattern,ns_per_pattern,cycles_per_b1_group,"
              "ns_per_b1_group,b1_groups_per_worker,validation");
    for (const Cell& cell : cells) {
        auto run = [&](int64_t repetitions) {
            const size_t output_bytes = static_cast<size_t>(
                kActiveAic * kCycleSlotBytes);
            ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
            ACL_CHECK(aclrtlaunch_aic_mte2_sparse_expert_layout(
                kActiveAic, stream, device_source, device_output,
                cell.layout, repetitions));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
            ACL_CHECK(aclrtMemcpy(output.data(), output_bytes, device_output,
                                  output_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
            int64_t maximum_cycles = 0;
            int64_t expected_groups = -1;
            for (int64_t core = 0; core < kActiveAic; ++core) {
                const size_t offset = static_cast<size_t>(
                    core * kCycleSlotBytes / sizeof(int64_t));
                if (output[offset] <= 0 || output[offset + 1] <= 0) {
                    std::fprintf(stderr, "sparse-expert layout validation failed\n");
                    std::exit(EXIT_FAILURE);
                }
                if (expected_groups < 0) {
                    expected_groups = output[offset + 1];
                } else if (output[offset + 1] != expected_groups) {
                    std::fprintf(stderr, "workers issued unequal B1 group counts\n");
                    std::exit(EXIT_FAILURE);
                }
                maximum_cycles = std::max(maximum_cycles, output[offset]);
            }
            return std::pair<int64_t, int64_t>{maximum_cycles, expected_groups};
        };
        run(cell.repetitions);
        const auto short_run = run(cell.repetitions);
        const auto long_run = run(cell.repetitions * 4);
        if (long_run.first <= short_run.first
                || long_run.second != short_run.second * 4) {
            std::fprintf(stderr, "sparse-expert differential interval is invalid\n");
            return EXIT_FAILURE;
        }
        const double cycles_per_pattern = static_cast<double>(
            long_run.first - short_run.first) / (3.0 * cell.repetitions);
        const int64_t groups_per_pattern = short_run.second / cell.repetitions;
        const double cycles_per_group = cycles_per_pattern / groups_per_pattern;
        if (!std::isfinite(cycles_per_pattern) || cycles_per_pattern <= 0.0
                || groups_per_pattern <= 0 || !std::isfinite(cycles_per_group)
                || cycles_per_group <= 0.0) {
            std::fprintf(stderr, "sparse-expert duration is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf("%ld,%ld,%ld,%ld,%.9f,%.9f,%.9f,%.9f,%ld,pass\n",
                    cell.layout, cell.repetitions, short_run.first,
                    long_run.first, cycles_per_pattern,
                    cycles_per_pattern * kSystemCounterTickNs,
                    cycles_per_group, cycles_per_group * kSystemCounterTickNs,
                    groups_per_pattern);
        std::fflush(stdout);
    }
    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
