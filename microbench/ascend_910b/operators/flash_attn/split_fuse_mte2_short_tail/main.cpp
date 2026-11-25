#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define ACL_CHECK(call) do {                                                \
    int status = static_cast<int>(call);                                    \
    if (status != 0) {                                                      \
        std::fprintf(stderr, "ACL error: %s returned %d\n", #call, status); \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

extern "C" uint32_t aclrtlaunch_split_fuse_mte2_interleaved(
    uint32_t block_dim, void* stream, void* gm_source, void* gm_cycles,
    int64_t active_groups, int64_t descriptor_count, int64_t issue_mode,
    int64_t base_page);

namespace {
constexpr int64_t kMaximumGroups = 24;
constexpr int64_t kPageElements = 128 * 8 * 128;
constexpr int64_t kCycleSlotBytes = 64;
#ifndef MIDLAYER_TIMED_LAUNCHES
#define MIDLAYER_TIMED_LAUNCHES 15
#endif
constexpr int64_t kTimedLaunches = MIDLAYER_TIMED_LAUNCHES;
constexpr int64_t kPagesPerLaunch = 24;
constexpr int64_t kSourceBytes = 2048LL * 1024 * 1024;
constexpr int64_t kSourcePages = kSourceBytes
    / (kPageElements * sizeof(uint16_t));
constexpr double kSystemCounterTickNs = 20.0;

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return (n % 2 == 1) ? values[n / 2]
        : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s active_groups count issue_mode\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t groups = ParseInteger(argv[1], "active_groups");
    const int64_t count = ParseInteger(argv[2], "count");
    const int64_t mode = ParseInteger(argv[3], "issue_mode");
    if (groups != 24
            || (count != 3 && count != 4 && count != 5 && count != 7)
            || (mode != 0 && mode != 1)) {
        std::fprintf(stderr, "parameters are outside the registered cells\n");
        return EXIT_FAILURE;
    }
    const int64_t cycle_bytes = kMaximumGroups * kCycleSlotBytes;
    ACL_CHECK(aclInit(nullptr));
    const char* device_text = std::getenv("MIDLAYER_ASCEND_DEVICE");
    if (device_text == nullptr) {
        std::fprintf(stderr, "MIDLAYER_ASCEND_DEVICE is required\n");
        return EXIT_FAILURE;
    }
    const int32_t device = static_cast<int32_t>(
        ParseInteger(device_text, "MIDLAYER_ASCEND_DEVICE"));
    ACL_CHECK(aclrtSetDevice(device));
    void* device_source = nullptr;
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_source, kSourceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles, cycle_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_source, kSourceBytes, 0, kSourceBytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    std::vector<int64_t> cycles(
        static_cast<size_t>(cycle_bytes / sizeof(int64_t)), 0);
    std::vector<double> launch_medians[2];
    launch_medians[0].reserve(kTimedLaunches);
    launch_medians[1].reserve(kTimedLaunches);
    int64_t minimum[2] = {0, 0};
    int64_t maximum[2] = {0, 0};
    // Warm the kernel without touching any of the timed first-touch pages.
    ACL_CHECK(aclrtlaunch_split_fuse_mte2_interleaved(
        static_cast<uint32_t>(groups), stream, device_source,
        device_cycles, groups, count, mode,
        kSourcePages - kPagesPerLaunch));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (int64_t launch = 0; launch < kTimedLaunches; ++launch) {
        const int64_t base_page = launch * kPagesPerLaunch;
        for (int64_t pass = 0; pass < 2; ++pass) {
            // Pass 0 first-touches these pages; pass 1 immediately replays.
            ACL_CHECK(aclrtlaunch_split_fuse_mte2_interleaved(
                static_cast<uint32_t>(groups), stream, device_source,
                device_cycles, groups, count, mode, base_page));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            ACL_CHECK(aclrtMemcpy(cycles.data(), cycle_bytes, device_cycles,
                                  cycle_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
            std::vector<double> worker_cycles;
            worker_cycles.reserve(static_cast<size_t>(groups));
            for (int64_t worker = 0; worker < groups; ++worker) {
                const size_t offset = static_cast<size_t>(
                    worker * kCycleSlotBytes / sizeof(int64_t));
                if (cycles[offset] <= 0 || cycles[offset + 1] != 1) {
                    std::fprintf(stderr, "worker %ld failed validation\n",
                                 static_cast<long>(worker));
                    return EXIT_FAILURE;
                }
                worker_cycles.push_back(static_cast<double>(cycles[offset]));
                if (launch == 0 || cycles[offset] < minimum[pass]) {
                    minimum[pass] = cycles[offset];
                }
                if (cycles[offset] > maximum[pass]) {
                    maximum[pass] = cycles[offset];
                }
            }
            launch_medians[pass].push_back(Median(worker_cycles));
        }
    }
    const double cold_cycles = Median(launch_medians[0]);
    const double warm_cycles = Median(launch_medians[1]);
    std::printf("groups,count,issue_mode,rows,columns,stride_elements,"
                "cold_min_cycles,cold_median_cycles,cold_max_cycles,"
                "warm_min_cycles,warm_median_cycles,warm_max_cycles,"
                "tick_ns,cold_program_ns,warm_program_ns,validation\n");
    std::printf("%ld,%ld,%ld,128,128,1024,%ld,%.1f,%ld,%ld,%.1f,%ld,"
                "%.1f,%.1f,%.1f,pass\n",
                static_cast<long>(groups), static_cast<long>(count),
                static_cast<long>(mode), static_cast<long>(minimum[0]),
                cold_cycles, static_cast<long>(maximum[0]),
                static_cast<long>(minimum[1]), warm_cycles,
                static_cast<long>(maximum[1]), kSystemCounterTickNs,
                cold_cycles * kSystemCounterTickNs,
                warm_cycles * kSystemCounterTickNs);
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtResetDevice(device));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
