// Host driver for the independent GroupedMatmul controller-to-first-issue
// probe.  It accepts an arbitrary cumulative source group list and never
// launches a production FusedMoE or matrix workload.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                 \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_grouped_matmul_first_issue_control(
    uint32_t block_dim, void* stream, void* group_offsets,
    void* first_issue_cycles, void* completion_cycles, void* output_checksums,
    int64_t group_count, int64_t repeats, int64_t projection_k,
    int64_t projection_n, int64_t base_m, int64_t base_n);

namespace {

constexpr int32_t kLogicalDeviceId = 0;
constexpr int64_t kAicCount = 24;
constexpr int64_t kMaximumGroups = 512;
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
    if (argc < 10 || std::strcmp(argv[1], "--case") != 0) {
        std::fprintf(
            stderr,
            "usage: %s --case group_count projection_k projection_n base_m "
            "base_n repeats count0 [... countN]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t group_count = ParseInteger(argv[2], "group_count");
    const int64_t projection_k = ParseInteger(argv[3], "projection_k");
    const int64_t projection_n = ParseInteger(argv[4], "projection_n");
    const int64_t base_m = ParseInteger(argv[5], "base_m");
    const int64_t base_n = ParseInteger(argv[6], "base_n");
    const int64_t repeats = ParseInteger(argv[7], "repeats");
    if (group_count <= 0 || group_count > kMaximumGroups || projection_k <= 0
            || projection_n <= 0 || base_m <= 0 || base_n <= 0 || repeats != 1
            || argc != 8 + group_count) {
        std::fprintf(stderr, "case is outside the one-pass probe contract\n");
        return EXIT_FAILURE;
    }
    std::vector<int64_t> offsets(kMaximumGroups, 0);
    int64_t cumulative_rows = 0;
    for (int64_t group = 0; group < group_count; ++group) {
        const int64_t count = ParseInteger(argv[8 + group], "group row count");
        if (count < 0 || count > INT32_MAX - cumulative_rows) {
            std::fprintf(stderr, "group row count is invalid\n");
            return EXIT_FAILURE;
        }
        cumulative_rows += count;
        offsets[group] = cumulative_rows;
    }
    if (cumulative_rows <= 0) {
        std::fprintf(stderr, "the source program must contain work\n");
        return EXIT_FAILURE;
    }

    void* device_offsets = nullptr;
    void* device_first = nullptr;
    void* device_completion = nullptr;
    void* device_checksums = nullptr;
    ACL_CHECK(aclrtMalloc(&device_offsets, offsets.size() * sizeof(int64_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_first,
                          kAicCount * kResultStrideElements * sizeof(int64_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_completion,
                          kAicCount * kResultStrideElements * sizeof(int64_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_checksums,
                          kAicCount * kResultStrideElements * sizeof(uint64_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(device_offsets, offsets.size() * sizeof(int64_t),
                          offsets.data(), offsets.size() * sizeof(int64_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    // Warm one short source pass so measurement starts after runtime setup.
    ACL_CHECK(aclrtlaunch_grouped_matmul_first_issue_control(
        kAicCount, stream, device_offsets, device_first, device_completion,
        device_checksums, group_count, 2, projection_k, projection_n,
        base_m, base_n));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemset(device_first,
                          kAicCount * kResultStrideElements * sizeof(int64_t),
                          0, kAicCount * kResultStrideElements * sizeof(int64_t)));
    ACL_CHECK(aclrtMemset(device_completion,
                          kAicCount * kResultStrideElements * sizeof(int64_t),
                          0, kAicCount * kResultStrideElements * sizeof(int64_t)));
    ACL_CHECK(aclrtMemset(device_checksums,
                          kAicCount * kResultStrideElements * sizeof(uint64_t),
                          0, kAicCount * kResultStrideElements * sizeof(uint64_t)));
    ACL_CHECK(aclrtlaunch_grouped_matmul_first_issue_control(
        kAicCount, stream, device_offsets, device_first, device_completion,
        device_checksums, group_count, repeats, projection_k, projection_n,
        base_m, base_n));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> first(
        kAicCount * kResultStrideElements, 0);
    std::vector<int64_t> completion(
        kAicCount * kResultStrideElements, 0);
    std::vector<uint64_t> checksums(
        kAicCount * kResultStrideElements, 0);
    ACL_CHECK(aclrtMemcpy(first.data(), first.size() * sizeof(int64_t),
                          device_first, first.size() * sizeof(int64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(completion.data(), completion.size() * sizeof(int64_t),
                          device_completion,
                          completion.size() * sizeof(int64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(checksums.data(), checksums.size() * sizeof(uint64_t),
                          device_checksums,
                          checksums.size() * sizeof(uint64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    std::printf(
        "worker,first_issue_cycles,completion_cycles,first_issue_ns,"
        "completion_ns,checksum,validation\n");
    bool valid = true;
    for (int64_t worker = 0; worker < kAicCount; ++worker) {
        const int64_t slot = worker * kResultStrideElements;
        const bool row_valid = first[slot] > 0 && completion[slot] >= first[slot]
            && checksums[slot] != 0;
        std::printf("%ld,%ld,%ld,%.9f,%.9f,%lu,%s\n",
                    static_cast<long>(worker), static_cast<long>(first[slot]),
                    static_cast<long>(completion[slot]),
                    static_cast<double>(first[slot]) * kSystemCounterTickNs,
                    static_cast<double>(completion[slot]) * kSystemCounterTickNs,
                    static_cast<unsigned long>(checksums[slot]),
                    row_valid ? "pass" : "fail");
        valid = valid && row_valid;
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_checksums));
    ACL_CHECK(aclrtFree(device_completion));
    ACL_CHECK(aclrtFree(device_first));
    ACL_CHECK(aclrtFree(device_offsets));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
