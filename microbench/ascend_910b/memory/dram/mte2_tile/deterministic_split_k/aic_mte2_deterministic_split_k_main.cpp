// ACL host for the independent deterministic Split-K MTE2 topology probe.
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                 \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_deterministic_split_k(
    uint32_t block_dim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t a_rows);

namespace {

constexpr int64_t kAicWorkers = 24;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kPhysicalK = 28672;
constexpr int64_t kPhysicalN = 8192;
constexpr int64_t kMaximumATailRows = 128;
constexpr int64_t kATailRows[] = {16, 32, 48, 64, 80, 96, 112, 128};
constexpr int64_t kFullDescriptorCount = 2 * 4 * 33;
constexpr int64_t kRegularDescriptorCount = 2 * 3 * 33;
constexpr int64_t kFullBasePairCount = 2 * 4 * 32 * 3;
constexpr int64_t kRegularBasePairCount = 2 * 3 * 32 * 3;
constexpr int64_t kTailBasePairCount = 2 * (2 * 32 * 3 + 32 * 2);
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

// The generated AscendC host stub registers its kernel before main().  ACL
// must therefore be initialized before C++ static constructors run, matching
// the other standalone AscendC mechanism probes in this repository.
__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc != 5 || std::strcmp(argv[1], "--trials") != 0
            || std::strcmp(argv[3], "--device-id") != 0) {
        std::fprintf(stderr, "usage: %s --trials COUNT --device-id ID\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[2], "trials");
    const int64_t device_id = ParseInteger(argv[4], "device_id");
    if (trials < 3 || trials > 32 || device_id < 0) {
        std::fprintf(stderr, "trials must be in [3,32] and device ID non-negative\n");
        return EXIT_FAILURE;
    }

    const size_t a_bytes = static_cast<size_t>(
        kMaximumATailRows * kPhysicalK * sizeof(uint16_t));
    const size_t b_bytes = static_cast<size_t>(
        kPhysicalN * kPhysicalK * sizeof(uint16_t));
    const size_t output_bytes = static_cast<size_t>(
        kAicWorkers * kCycleSlotBytes);
    void* a_source = nullptr;
    void* b_source = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclrtMalloc(&a_source, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&b_source, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(a_source, a_bytes, 0, a_bytes));
    ACL_CHECK(aclrtMemset(b_source, b_bytes, 0, b_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto launch = [&](int64_t a_rows) {
        ACL_CHECK(aclrtMemset(output, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_aic_mte2_deterministic_split_k(
            kAicWorkers, stream, a_source, b_source, output, a_rows));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> values(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            values.data(), output_bytes, output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t maximum_cycles = 0;
        for (int64_t block = 0; block < kAicWorkers; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            const int64_t expected_descriptors = (
                block < 3 ? kFullDescriptorCount : kRegularDescriptorCount);
            const int64_t expected_chunk_start = (
                block < 3 ? block * 4 : block * 3 + 3);
            const int64_t expected_chunk_count = block < 3 ? 4 : 3;
            const int64_t expected_base_pairs = block < 3
                ? kFullBasePairCount
                : (block == kAicWorkers - 1
                   ? kTailBasePairCount : kRegularBasePairCount);
            if (values[offset] <= 0
                    || values[offset + 1] != expected_descriptors
                    || values[offset + 2] != expected_chunk_start
                    || values[offset + 3] != expected_chunk_count
                    || values[offset + 4] != expected_base_pairs
                    || values[offset + 5] != a_rows) {
                std::fprintf(stderr, "invalid topology output for block=%ld\n", block);
                std::exit(EXIT_FAILURE);
            }
            maximum_cycles = std::max(maximum_cycles, values[offset]);
        }
        return maximum_cycles;
    };

    for (int64_t a_rows : kATailRows) {
        launch(a_rows);
    }
    std::printf("a_rows,trial,max_cycles,duration_ns,validation\n");
    for (int64_t a_rows : kATailRows) {
        for (int64_t trial = 0; trial < trials; ++trial) {
            const int64_t cycles = launch(a_rows);
            std::printf("%ld,%ld,%ld,%.9f,pass\n", a_rows, trial, cycles,
                        cycles * kSystemCounterTickNs);
            std::fflush(stdout);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(a_source));
    ACL_CHECK(aclrtFree(b_source));
    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtResetDevice(static_cast<int32_t>(device_id)));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
