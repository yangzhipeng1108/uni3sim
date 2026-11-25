// Host driver for the independent short-GroupedMatmul B-ingress probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                     \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

extern "C" uint32_t aclrtlaunch_grouped_matmul_short_b_ingress(
    uint32_t block_dim, void* stream, void* group_rows, void* packed_b,
    void* result, int64_t group_count);

namespace {

constexpr int32_t kLogicalDeviceId = 0;
constexpr int64_t kAicCount = 24;
constexpr int64_t kMaximumGroups = 512;
constexpr int64_t kProjectionK = 512;
constexpr int64_t kProjectionN = 2048;
constexpr int64_t kResultStrideElements = 8;
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

size_t CheckedBBytes(int64_t populated_groups) {
    const uint64_t elements = static_cast<uint64_t>(populated_groups)
        * kProjectionN * kProjectionK;
    if (elements == 0 || elements > std::numeric_limits<size_t>::max()
            / sizeof(uint16_t)) {
        std::fprintf(stderr, "packed B allocation is invalid\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(elements) * sizeof(uint16_t);
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0'
            || ParseInteger(device_text, "ASCEND_DEVICE_ID")
                != kLogicalDeviceId) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must be logical device 0 after remapping\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

int main(int argc, char** argv) {
    if (argc < 5 || std::strcmp(argv[1], "--case") != 0) {
        std::fprintf(stderr, "usage: %s --case group_count count0 [... countN]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t group_count = ParseInteger(argv[2], "group_count");
    if (group_count <= 0 || group_count > kMaximumGroups
            || argc != 3 + group_count) {
        std::fprintf(stderr, "case is outside the source-ring contract\n");
        return EXIT_FAILURE;
    }
    std::vector<int64_t> counts(kMaximumGroups, 0);
    int64_t populated_groups = 0;
    for (int64_t group = 0; group < group_count; ++group) {
        const int64_t count = ParseInteger(argv[3 + group], "group row count");
        if (count < 0) {
            std::fprintf(stderr, "group row count is invalid\n");
            return EXIT_FAILURE;
        }
        counts[group] = count;
        populated_groups += count > 0;
    }
    if (populated_groups == 0) {
        std::fprintf(stderr, "source ring has no populated groups\n");
        return EXIT_FAILURE;
    }

    void* device_counts = nullptr;
    void* device_b = nullptr;
    void* device_result = nullptr;
    const size_t counts_bytes = counts.size() * sizeof(int64_t);
    const size_t b_bytes = CheckedBBytes(populated_groups);
    const size_t result_bytes = static_cast<size_t>(
        kAicCount * kResultStrideElements * sizeof(int64_t));
    ACL_CHECK(aclrtMalloc(&device_counts, counts_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_b, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_result, result_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(device_counts, counts_bytes, counts.data(), counts_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(device_b, b_bytes, 0, b_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    // Warm a complete independent source ring before the single timed ring.
    ACL_CHECK(aclrtlaunch_grouped_matmul_short_b_ingress(
        kAicCount, stream, device_counts, device_b, device_result, group_count));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemset(device_result, result_bytes, 0, result_bytes));
    ACL_CHECK(aclrtlaunch_grouped_matmul_short_b_ingress(
        kAicCount, stream, device_counts, device_b, device_result, group_count));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> result(kAicCount * kResultStrideElements, 0);
    ACL_CHECK(aclrtMemcpy(result.data(), result_bytes, device_result, result_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t maximum_cycles = 0;
    int64_t total_b_groups = 0;
    bool valid = true;
    std::printf("worker,completion_cycles,completion_ns,issued_b_groups,"
                "ring_offset,group_count,validation\n");
    for (int64_t worker = 0; worker < kAicCount; ++worker) {
        const int64_t offset = worker * kResultStrideElements;
        const int64_t completion = result[offset];
        const int64_t issued = result[offset + 1];
        const bool row_valid = completion > 0 && issued >= 0
            && result[offset + 3] == group_count;
        std::printf("%ld,%ld,%.9f,%ld,%ld,%ld,%s\n",
                    static_cast<long>(worker), static_cast<long>(completion),
                    static_cast<double>(completion) * kSystemCounterTickNs,
                    static_cast<long>(issued), static_cast<long>(result[offset + 2]),
                    static_cast<long>(result[offset + 3]),
                    row_valid ? "pass" : "fail");
        maximum_cycles = std::max(maximum_cycles, completion);
        total_b_groups += issued;
        valid = valid && row_valid;
    }
    if (total_b_groups <= 0) {
        valid = false;
    }
    std::fprintf(stderr,
                 "max_completion_ns=%.9f total_b_groups=%ld populated_groups=%ld\n",
                 static_cast<double>(maximum_cycles) * kSystemCounterTickNs,
                 static_cast<long>(total_b_groups),
                 static_cast<long>(populated_groups));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_result));
    ACL_CHECK(aclrtFree(device_b));
    ACL_CHECK(aclrtFree(device_counts));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
