// ACL host for the FIA AIC scalar control-floor probe.
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

extern "C" uint32_t aclrtlaunch_aic_scalar_control_floor(
    uint32_t block_dim, void* stream, void* gm_cycles, void* gm_probe,
    int64_t iters, int64_t active_aic_groups, int64_t gm_reads,
    int64_t handoffs, int64_t shared);

namespace {
constexpr int64_t kMaximumGroups = 24;
constexpr int64_t kMaximumIters = 256;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kSlotCount = 3 * kMaximumGroups;  // 1 AIC + 2 AIV per group
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kMaximumGmReads = 512;
constexpr int64_t kProbeStrideElements = 4096;
constexpr int64_t kProbeElementsPerWorker =
    kMaximumGmReads * kProbeStrideElements;

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
    if (argc < 3 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s active_aic_groups iters [gm_reads] "
                     "[launches] [handoffs] [shared]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t groups = ParseInteger(argv[1], "active_aic_groups");
    const int64_t iters = ParseInteger(argv[2], "iters");
    const int64_t gm_reads = (argc >= 4)
        ? ParseInteger(argv[3], "gm_reads") : 0;
    const int64_t launches = (argc >= 5)
        ? ParseInteger(argv[4], "launches") : 1;
    const int64_t handoffs = (argc >= 6)
        ? ParseInteger(argv[5], "handoffs") : 0;
    const int64_t shared = (argc == 7)
        ? ParseInteger(argv[6], "shared") : 0;
    if (groups <= 0 || groups > kMaximumGroups
            || iters < 0 || iters > kMaximumIters
            || gm_reads < 0 || gm_reads > kMaximumGmReads
            || launches < 1 || launches > 256
            || handoffs < 0 || handoffs > 128
            || shared < 0 || shared > 1) {
        std::fprintf(stderr, "parameters are outside the registered domain\n");
        return EXIT_FAILURE;
    }
    ACL_CHECK(aclInit(nullptr));
    const char* device_text = std::getenv("MIDLAYER_ASCEND_DEVICE");
    const int32_t device = device_text == nullptr
        ? 0 : static_cast<int32_t>(ParseInteger(device_text, "device"));
    ACL_CHECK(aclrtSetDevice(device));

    const int64_t cycle_bytes = kSlotCount * kCycleSlotBytes;
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(&device_cycles, cycle_bytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    const int64_t probe_elements = kMaximumGroups * kProbeElementsPerWorker;
    const int64_t probe_bytes =
        probe_elements * static_cast<int64_t>(sizeof(int64_t));
    void* device_probe = nullptr;
    ACL_CHECK(aclrtMalloc(&device_probe, probe_bytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    // Each slot holds the next index of its own worker-disjoint chain.
    {
        std::vector<int64_t> host_probe(
            static_cast<size_t>(probe_elements), 0);
        for (int64_t worker = 0; worker < kMaximumGroups; ++worker) {
            for (int64_t step = 0; step < kMaximumGmReads; ++step) {
                const int64_t here = step * kProbeStrideElements;
                host_probe[static_cast<size_t>(
                    worker * kProbeElementsPerWorker + here)] =
                    ((step + 1) % kMaximumGmReads) * kProbeStrideElements;
            }
        }
        ACL_CHECK(aclrtMemcpy(device_probe, probe_bytes, host_probe.data(),
                              probe_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    aclrtEvent start_event;
    aclrtEvent stop_event;
    ACL_CHECK(aclrtCreateEvent(&start_event));
    ACL_CHECK(aclrtCreateEvent(&stop_event));

    // Untimed warmup removes launch history without a production workload.
    ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0, cycle_bytes));
    ACL_CHECK(aclrtlaunch_aic_scalar_control_floor(
        static_cast<uint32_t>(groups), stream, device_cycles, device_probe,
        iters, groups, gm_reads, handoffs, shared));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0, cycle_bytes));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    // Extra launches exist so an external msprof capture sees a population
    // of identical tasks; the event-timed launch below is unchanged.
    for (int64_t extra = 1; extra < launches; ++extra) {
        ACL_CHECK(aclrtlaunch_aic_scalar_control_floor(
            static_cast<uint32_t>(groups), stream, device_cycles, device_probe,
            iters, groups, gm_reads, handoffs, shared));
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtRecordEvent(start_event, stream));
    ACL_CHECK(aclrtlaunch_aic_scalar_control_floor(
        static_cast<uint32_t>(groups), stream, device_cycles, device_probe,
        iters, groups, gm_reads, handoffs, shared));
    ACL_CHECK(aclrtRecordEvent(stop_event, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    float host_ms = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&host_ms, start_event, stop_event));

    std::vector<int64_t> cycles(
        static_cast<size_t>(cycle_bytes / sizeof(int64_t)), 0);
    ACL_CHECK(aclrtMemcpy(cycles.data(), cycle_bytes, device_cycles,
                          cycle_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t aic_max = 0;
    int64_t aiv_max = 0;
    for (int64_t index = 0; index < 3 * groups; ++index) {
        const size_t offset =
            static_cast<size_t>(index * kCycleSlotBytes / sizeof(int64_t));
        const int64_t span = cycles[offset];
        const int64_t valid = cycles[offset + 1];
        const bool is_aic = index < groups;
        if (valid != (is_aic ? 2 : 1) || span < 0) {
            std::fprintf(stderr, "core %ld failed validation (valid=%ld)\n",
                         static_cast<long>(index), static_cast<long>(valid));
            return EXIT_FAILURE;
        }
        if (is_aic) {
            aic_max = std::max(aic_max, span);
        } else {
            aiv_max = std::max(aiv_max, span);
        }
    }
    const double internal_ns =
        static_cast<double>(aic_max) * kSystemCounterTickNs;
    const double host_ns = static_cast<double>(host_ms) * 1.0e6;
    std::printf("active_aic_groups,iters,gm_reads,handoffs,shared,aic_max_cycles,"
                "aiv_max_cycles,tick_ns,internal_ns,host_ns,envelope_ns,"
                "validation\n");
    std::printf("%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.1f,%.1f,%.1f,%.1f,pass\n",
                static_cast<long>(groups), static_cast<long>(iters),
                static_cast<long>(gm_reads), static_cast<long>(handoffs),
                static_cast<long>(shared),
                static_cast<long>(aic_max), static_cast<long>(aiv_max),
                kSystemCounterTickNs, internal_ns, host_ns,
                host_ns - internal_ns);

    ACL_CHECK(aclrtDestroyEvent(start_event));
    ACL_CHECK(aclrtDestroyEvent(stop_event));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_probe));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtResetDevice(device));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
