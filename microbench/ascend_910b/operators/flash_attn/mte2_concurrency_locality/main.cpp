// ACL host for the FIA drained ND2NZ descriptor concurrency/locality probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define ACL_CHECK(call) do {                                                \
    int status = static_cast<int>(call);                                    \
    if (status != 0) {                                                      \
        std::fprintf(stderr, "ACL error: %s returned %d\n", #call, status); \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

extern "C" uint32_t aclrtlaunch_fia_mte2_concurrency_locality(
    uint32_t block_dim, void* stream, void* gm_source, void* gm_cycles,
    int64_t rows, int64_t columns, int64_t stride_elements, int64_t count,
    int64_t pass_index, int64_t address_mode, int64_t launch_offset);

namespace {
constexpr int64_t kMaximumWorkers = 24;
constexpr int64_t kMaximumRows = 256;
constexpr int64_t kMaximumStrideElements = 1024;
constexpr int64_t kMaximumCount = 128;
constexpr int64_t kSkewElements = 8192;
constexpr int64_t kPackedTileElements = kMaximumRows * kMaximumStrideElements;
constexpr int64_t kDescriptorSpanElements =
    kMaximumRows * kMaximumStrideElements + kSkewElements;
constexpr int64_t kPasses = 2;
constexpr int64_t kElementsPerWorker =
    kPasses * kMaximumCount * kDescriptorSpanElements;
constexpr int64_t kCycleSlotBytes = 64;
constexpr double kSystemCounterTickNs = 20.0;
// Amendment 3: the process value is the median over this many timed launches,
// each a fresh short program on addresses no other launch touched.
constexpr int64_t kTimedLaunches = 15;

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

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: %s groups rows columns stride_elements count "
                     "address_mode\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t groups = ParseInteger(argv[1], "groups");
    const int64_t rows = ParseInteger(argv[2], "rows");
    const int64_t columns = ParseInteger(argv[3], "columns");
    const int64_t stride_elements = ParseInteger(argv[4], "stride_elements");
    const int64_t count = ParseInteger(argv[5], "count");
    const int64_t address_mode = ParseInteger(argv[6], "address_mode");
    if (groups <= 0 || groups > kMaximumWorkers || rows <= 0
            || rows > kMaximumRows || columns <= 0
            || stride_elements < columns
            || stride_elements > kMaximumStrideElements
            || count <= 0 || count > kMaximumCount
            || address_mode < 0 || address_mode > 1) {
        std::fprintf(stderr, "parameters are outside the registered domain\n");
        return EXIT_FAILURE;
    }
    ACL_CHECK(aclInit(nullptr));
    const char* device_text = std::getenv("MIDLAYER_ASCEND_DEVICE");
    const int32_t device = device_text == nullptr
        ? 0 : static_cast<int32_t>(ParseInteger(device_text, "device"));
    ACL_CHECK(aclrtSetDevice(device));

    const int64_t disjoint_elements = kMaximumWorkers * kElementsPerWorker;
    const int64_t packed_elements =
        kPasses * kMaximumCount * kMaximumWorkers * kPackedTileElements;
    const int64_t source_bytes =
        std::max(disjoint_elements, packed_elements)
        * static_cast<int64_t>(sizeof(uint16_t));
    const int64_t cycle_bytes = kMaximumWorkers * kCycleSlotBytes;
    void* device_source = nullptr;
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(&device_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_cycles, cycle_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_source, source_bytes, 0, source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    // Untimed warmup on pass 0.  It removes launch history without touching a
    // production workload, and it reads a region disjoint from the timed
    // pass, so the timed launch's source is genuinely DRAM-cold.
    const int64_t launches =
        std::min(kTimedLaunches, kMaximumCount / count);

    ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0, cycle_bytes));
    ACL_CHECK(aclrtlaunch_fia_mte2_concurrency_locality(
        static_cast<uint32_t>(groups), stream, device_source, device_cycles,
        rows, columns, stride_elements, count, 0, address_mode, 0));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> cycles(
        static_cast<size_t>(cycle_bytes / sizeof(int64_t)), 0);
    std::vector<double> launch_medians;
    launch_medians.reserve(static_cast<size_t>(launches));
    int64_t worst_min = 0;
    int64_t worst_max = 0;
    for (int64_t launch = 0; launch < launches; ++launch) {
        ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0, cycle_bytes));
        ACL_CHECK(aclrtlaunch_fia_mte2_concurrency_locality(
            static_cast<uint32_t>(groups), stream, device_source,
            device_cycles, rows, columns, stride_elements, count, 1,
            address_mode, launch));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemcpy(cycles.data(), cycle_bytes, device_cycles,
                              cycle_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        std::vector<int64_t> worker_cycles;
        worker_cycles.reserve(static_cast<size_t>(groups));
        for (int64_t worker = 0; worker < groups; ++worker) {
            const size_t offset =
                static_cast<size_t>(worker * kCycleSlotBytes / sizeof(int64_t));
            if (cycles[offset] <= 0 || cycles[offset + 1] != 1) {
                std::fprintf(stderr, "worker %ld failed validation\n",
                             static_cast<long>(worker));
                return EXIT_FAILURE;
            }
            worker_cycles.push_back(cycles[offset]);
        }
        std::sort(worker_cycles.begin(), worker_cycles.end());
        const size_t n = worker_cycles.size();
        launch_medians.push_back((n % 2 == 1)
            ? static_cast<double>(worker_cycles[n / 2])
            : (static_cast<double>(worker_cycles[n / 2 - 1])
               + static_cast<double>(worker_cycles[n / 2])) / 2.0);
        if (launch == 0 || worker_cycles.front() < worst_min) {
            worst_min = worker_cycles.front();
        }
        if (worker_cycles.back() > worst_max) {
            worst_max = worker_cycles.back();
        }
    }
    std::sort(launch_medians.begin(), launch_medians.end());
    const size_t l = launch_medians.size();
    const double median_cycles = (l % 2 == 1)
        ? launch_medians[l / 2]
        : (launch_medians[l / 2 - 1] + launch_medians[l / 2]) / 2.0;
    const double program_ns = median_cycles * kSystemCounterTickNs;
    std::printf("groups,rows,columns,stride_elements,count,address_mode,"
                "min_cycles,median_cycles,max_cycles,tick_ns,program_ns,"
                "per_descriptor_ns,validation\n");
    std::printf("%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.1f,%ld,%.1f,%.1f,%.4f,pass\n",
                static_cast<long>(groups), static_cast<long>(rows),
                static_cast<long>(columns),
                static_cast<long>(stride_elements), static_cast<long>(count),
                static_cast<long>(address_mode),
                static_cast<long>(worst_min), median_cycles,
                static_cast<long>(worst_max),
                kSystemCounterTickNs, program_ns,
                program_ns / static_cast<double>(count));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtResetDevice(device));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
