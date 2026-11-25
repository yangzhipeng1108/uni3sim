// Host driver for the independent GroupedMatmul zero-expert controller probe.
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
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_grouped_matmul_zero_expert_control(
    uint32_t block_dim, void* stream, void* group_offsets,
    void* output_cycles, void* output_checksums, int64_t group_count,
    int64_t repeats, int64_t projection_k, int64_t projection_n,
    int64_t base_m, int64_t base_n);

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
    if (argc < 4 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 2 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch group_count repeats "
            "[group_count repeats ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    std::vector<int64_t> offsets(kMaximumGroups, 0);
    void* device_offsets = nullptr;
    void* device_cycles = nullptr;
    void* device_checksums = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_offsets, offsets.size() * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles,
        kAicCount * kResultStrideElements * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_checksums,
        kAicCount * kResultStrideElements * sizeof(uint64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_offsets, offsets.size() * sizeof(int64_t), offsets.data(),
        offsets.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::printf(
        "group_count,repeats,min_cycles,max_cycles,service_ns_per_scan,"
        "checksum,validation\n");
    for (int argument = 2; argument < argc; argument += 2) {
        const int64_t group_count = ParseInteger(
            argv[argument], "group_count");
        const int64_t repeats = ParseInteger(argv[argument + 1], "repeats");
        if (group_count <= 0 || group_count > kMaximumGroups
                || repeats <= 0) {
            std::fprintf(stderr, "cell is outside the frozen scan domain\n");
            return EXIT_FAILURE;
        }

        ACL_CHECK(aclrtlaunch_grouped_matmul_zero_expert_control(
            kAicCount, stream, device_offsets, device_cycles,
            device_checksums, group_count, 8, 2048, 512, 16, 256));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemset(
            device_cycles,
            kAicCount * kResultStrideElements * sizeof(int64_t), 0,
            kAicCount * kResultStrideElements * sizeof(int64_t)));
        ACL_CHECK(aclrtMemset(
            device_checksums,
            kAicCount * kResultStrideElements * sizeof(uint64_t), 0,
            kAicCount * kResultStrideElements * sizeof(uint64_t)));
        ACL_CHECK(aclrtlaunch_grouped_matmul_zero_expert_control(
            kAicCount, stream, device_offsets, device_cycles,
            device_checksums, group_count, repeats, 2048, 512, 16, 256));
        ACL_CHECK(aclrtSynchronizeStream(stream));

        std::vector<int64_t> cycles(
            kAicCount * kResultStrideElements, 0);
        std::vector<uint64_t> checksums(
            kAicCount * kResultStrideElements, 0);
        ACL_CHECK(aclrtMemcpy(
            cycles.data(), cycles.size() * sizeof(int64_t), device_cycles,
            cycles.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(
            checksums.data(), checksums.size() * sizeof(uint64_t),
            device_checksums, checksums.size() * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t minimum_cycles = cycles[0];
        int64_t maximum_cycles = cycles[0];
        uint64_t checksum = 0;
        bool valid = cycles[0] > 0;
        for (int64_t worker = 0; worker < kAicCount; ++worker) {
            const int64_t slot = worker * kResultStrideElements;
            if (cycles[slot] <= 0 || checksums[slot] == 0) {
                valid = false;
            }
            if (cycles[slot] < minimum_cycles) {
                minimum_cycles = cycles[slot];
            }
            if (cycles[slot] > maximum_cycles) {
                maximum_cycles = cycles[slot];
            }
            checksum ^= checksums[slot];
        }
        const double service_ns = (
            static_cast<double>(maximum_cycles) * kSystemCounterTickNs
            / static_cast<double>(repeats));
        std::printf(
            "%ld,%ld,%ld,%ld,%.9f,%lu,%s\n",
            static_cast<long>(group_count), static_cast<long>(repeats),
            static_cast<long>(minimum_cycles),
            static_cast<long>(maximum_cycles), service_ns,
            static_cast<unsigned long>(checksum), valid ? "pass" : "fail");
        if (!valid) {
            return EXIT_FAILURE;
        }
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_checksums));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_offsets));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
