// ACL host for the independent FIA paged MM1 task-program probe.
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
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error: %s returned %d\n", #call, status); \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_fia_paged_mte2_task_program(
    uint32_t block_dim, void* stream, void* gm_q, void* gm_source,
    void* gm_block_table, void* gm_output, void* gm_cycles,
    int64_t profile, int64_t actual_m);

namespace {
constexpr int64_t kWorkers = 24;
constexpr int64_t kAlignedM = 16;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageRows = 128;
constexpr int64_t kPagesPerTask = 32;
constexpr int64_t kMaximumTasks = 43;
constexpr int64_t kPagesPerWorker = kMaximumTasks * kPagesPerTask;
constexpr int64_t kPageBytes = kPageRows * kHeadDim * sizeof(uint16_t);
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kQBytes = kWorkers * kAlignedM * kHeadDim * sizeof(uint16_t);
constexpr int64_t kOutputBytes = kWorkers * 2 * kAlignedM * 128 * sizeof(float);
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

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s profile actual_m\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t profile = ParseInteger(argv[1], "profile");
    const int64_t actual_m = ParseInteger(argv[2], "actual_m");
    if (profile < 0 || profile > 3 || (actual_m != 4 && actual_m != 8)
            || (profile == 3 && actual_m != 4)) {
        std::fprintf(stderr,
                     "profile must be 0..3, actual_m must be 4 or 8, "
                     "and profile 3 requires actual_m=4\n");
        return EXIT_FAILURE;
    }
    const int64_t source_bytes = kWorkers * kPagesPerWorker * kPageBytes;
    const int64_t table_elements = kWorkers * kPagesPerWorker;
    const int64_t table_bytes = table_elements * sizeof(int32_t);
    const int64_t cycle_bytes = kWorkers * kCycleSlotBytes;
    std::vector<int32_t> table(static_cast<size_t>(table_elements));
    for (int64_t worker = 0; worker < kWorkers; ++worker) {
        for (int64_t page = 0; page < kPagesPerWorker; ++page) {
            table[static_cast<size_t>(worker * kPagesPerWorker + page)] =
                static_cast<int32_t>(page);
        }
    }
    void* device_q = nullptr;
    void* device_source = nullptr;
    void* device_table = nullptr;
    void* device_output = nullptr;
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(&device_q, kQBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_table, table_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_output, kOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_cycles, cycle_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_q, kQBytes, 0, kQBytes));
    ACL_CHECK(aclrtMemset(device_source, source_bytes, 0, source_bytes));
    ACL_CHECK(aclrtMemcpy(device_table, table_bytes, table.data(), table_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(device_output, kOutputBytes, 0, kOutputBytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    // A fixed full-profile warmup is outside the timed interval.  It removes
    // mixed-task launch history without using a production workload.
    ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0, cycle_bytes));
    ACL_CHECK(aclrtlaunch_fia_paged_mte2_task_program(
        kWorkers, stream, device_q, device_source, device_table, device_output,
        device_cycles, 0, actual_m));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0, cycle_bytes));
    ACL_CHECK(aclrtlaunch_fia_paged_mte2_task_program(
        kWorkers, stream, device_q, device_source, device_table, device_output,
        device_cycles, profile, actual_m));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<int64_t> cycles(static_cast<size_t>(cycle_bytes / sizeof(int64_t)), 0);
    ACL_CHECK(aclrtMemcpy(cycles.data(), cycle_bytes, device_cycles, cycle_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t minimum = std::numeric_limits<int64_t>::max();
    int64_t maximum = 0;
    std::vector<int64_t> worker_cycles;
    worker_cycles.reserve(static_cast<size_t>(kWorkers));
    for (int64_t worker = 0; worker < kWorkers; ++worker) {
        const size_t offset = static_cast<size_t>(worker * kCycleSlotBytes / sizeof(int64_t));
        if (cycles[offset] <= 0 || cycles[offset + 1] != 1) {
            std::fprintf(stderr, "worker %ld failed validation\n", worker);
            return EXIT_FAILURE;
        }
        minimum = std::min(minimum, cycles[offset]);
        maximum = std::max(maximum, cycles[offset]);
        worker_cycles.push_back(cycles[offset]);
    }
    std::sort(worker_cycles.begin(), worker_cycles.end());
    const double median_cycles = (
        static_cast<double>(worker_cycles[kWorkers / 2 - 1])
        + static_cast<double>(worker_cycles[kWorkers / 2])) / 2.0;
    std::printf("profile,actual_m,workers,task_counts,min_cycles,median_cycles,max_cycles,tick_ns,min_ns,median_ns,max_ns,validation\n");
    const char* task_counts = profile == 0
        ? "43x16+42x8"
        : (profile == 1
           ? "42x24"
           : (profile == 2
              ? "43x24"
              : "2048x[36+36+36+36+36+36+37+36+15]|"
                "4096x[11+18+18+18+19+18+19+7]|"
                "8192x[6+9+9+9+9+10+9+10+9]"));
    std::printf("%ld,%ld,%ld,%s,%ld,%.1f,%ld,%.1f,%.1f,%.1f,%.1f,pass\n",
                profile, actual_m, kWorkers, task_counts, minimum,
                median_cycles, maximum, kSystemCounterTickNs,
                minimum * kSystemCounterTickNs,
                median_cycles * kSystemCounterTickNs,
                maximum * kSystemCounterTickNs);
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtFree(device_table));
    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtFree(device_q));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
