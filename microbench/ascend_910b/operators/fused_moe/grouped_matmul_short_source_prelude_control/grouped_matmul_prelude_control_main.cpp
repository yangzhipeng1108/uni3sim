// Host driver for the independent GroupedMatmul prelude-control probe.
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
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",     \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_grouped_matmul_prelude_control(
    uint32_t block_dim, void* stream, void* output_cycles,
    void* output_checksums, int64_t tile_count, int64_t k_rounds,
    int64_t repeats);

namespace {

constexpr int32_t kLogicalDeviceId = 0;
constexpr int64_t kAicCount = 24;
constexpr int64_t kResultStrideElements = 8;
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kLogicalN = GROUPED_MATMUL_PRELUDE_LOGICAL_N;

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
    if (argc < 6 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 4 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch logical_n tile_count k_rounds repeats "
            "[logical_n tile_count k_rounds repeats ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    void* device_cycles = nullptr;
    void* device_checksums = nullptr;
    const size_t result_bytes = (
        kAicCount * kResultStrideElements * sizeof(int64_t));
    ACL_CHECK(aclrtMalloc(
        &device_cycles, result_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_checksums, result_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::printf(
        "logical_n,tile_count,k_rounds,repeats,min_cycles,max_cycles,"
        "service_ns_per_scan,checksum,validation\n");
    for (int argument = 2; argument < argc; argument += 4) {
        const int64_t logical_n = ParseInteger(
            argv[argument], "logical_n");
        const int64_t tile_count = ParseInteger(
            argv[argument + 1], "tile_count");
        const int64_t k_rounds = ParseInteger(
            argv[argument + 2], "k_rounds");
        const int64_t repeats = ParseInteger(argv[argument + 3], "repeats");
        if (logical_n != kLogicalN
                || tile_count <= 0 || tile_count > 256
                || (k_rounds != 45 && k_rounds != 96 && k_rounds != 256)
                || repeats <= 0) {
            std::fprintf(stderr, "cell is outside the frozen probe domain\n");
            return EXIT_FAILURE;
        }

        ACL_CHECK(aclrtlaunch_grouped_matmul_prelude_control(
            kAicCount, stream, device_cycles, device_checksums,
            tile_count, k_rounds, 8));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemset(device_cycles, result_bytes, 0, result_bytes));
        ACL_CHECK(aclrtMemset(
            device_checksums, result_bytes, 0, result_bytes));
        ACL_CHECK(aclrtlaunch_grouped_matmul_prelude_control(
            kAicCount, stream, device_cycles, device_checksums,
            tile_count, k_rounds, repeats));
        ACL_CHECK(aclrtSynchronizeStream(stream));

        std::vector<int64_t> cycles(
            kAicCount * kResultStrideElements, 0);
        std::vector<uint64_t> checksums(
            kAicCount * kResultStrideElements, 0);
        ACL_CHECK(aclrtMemcpy(
            cycles.data(), result_bytes, device_cycles, result_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(
            checksums.data(), result_bytes, device_checksums, result_bytes,
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
        const double service_ns = static_cast<double>(maximum_cycles)
            * kSystemCounterTickNs / static_cast<double>(repeats);
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%lu,%s\n",
            static_cast<long>(logical_n), static_cast<long>(tile_count),
            static_cast<long>(k_rounds), static_cast<long>(repeats),
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
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
