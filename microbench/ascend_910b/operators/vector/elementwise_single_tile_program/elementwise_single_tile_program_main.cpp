// ACL host for the fixed 47-worker CANN elementwise single-tile probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "elementwise_single_tile_program_contract.h"

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                         \
    }                                                                    \
} while (0)

extern "C" uint32_t aclrtlaunch_elementwise_single_tile_program(
    uint32_t block_dim, void* stream, void* source_a, void* source_b,
    void* destination, void* worker_cycles);

namespace {

using namespace elementwise_single_tile_program_contract;

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr || *text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < 0 || value > 7) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be in [0, 7]\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

constexpr size_t RegionElements() {
    return static_cast<size_t>(kAivWorkers * kFullTileElements);
}

struct LaunchCycles {
    int64_t maximum_worker_body;
    int64_t source_program_span;
};

LaunchCycles RunOne(
        aclrtStream stream, void* source_a, void* source_b,
        void* destination, void* worker_cycles, int64_t region) {
    const size_t region_bytes = RegionElements() * sizeof(uint16_t);
    const size_t region_offset = static_cast<size_t>(region) * region_bytes;
    ACL_CHECK(aclrtMemset(
        worker_cycles, kAivWorkers * kResultValuesPerWorker * sizeof(int64_t),
        0xff,
        kAivWorkers * kResultValuesPerWorker * sizeof(int64_t)));
    ACL_CHECK(aclrtlaunch_elementwise_single_tile_program(
        kAivWorkers, stream,
        static_cast<uint8_t*>(source_a) + region_offset,
        static_cast<uint8_t*>(source_b) + region_offset,
        static_cast<uint8_t*>(destination) + region_offset,
        worker_cycles));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<int64_t> padded(
        kAivWorkers * kResultValuesPerWorker, -1);
    ACL_CHECK(aclrtMemcpy(
        padded.data(), padded.size() * sizeof(int64_t), worker_cycles,
        padded.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t earliest_start = -1;
    int64_t latest_end = -1;
    int64_t maximum_worker_body = -1;
    for (int64_t worker = 0; worker < kAivWorkers; ++worker) {
        const size_t offset = worker * kResultValuesPerWorker;
        const int64_t start = padded[offset];
        const int64_t end = padded[offset + 1];
        const int64_t body = padded[offset + 2];
        if (start <= 0 || end <= start || body != end - start) {
            std::fprintf(stderr, "worker %ld emitted invalid cycles\n",
                         static_cast<long>(worker));
            std::exit(EXIT_FAILURE);
        }
        earliest_start = (
            earliest_start < 0 ? start : std::min(earliest_start, start));
        latest_end = std::max(latest_end, end);
        maximum_worker_body = std::max(maximum_worker_body, body);
    }
    return LaunchCycles{
        maximum_worker_body,
        latest_end - earliest_start,
    };
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
}

int main() {
    const size_t region_bytes = RegionElements() * sizeof(uint16_t);
    const int64_t regions = kWarmupLaunches + kMeasuredLaunches;
    const size_t allocation_bytes = region_bytes * regions;
    void* source_a = nullptr;
    void* source_b = nullptr;
    void* destination = nullptr;
    void* worker_cycles = nullptr;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(&source_a, allocation_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&source_b, allocation_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&destination, allocation_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &worker_cycles,
        kAivWorkers * kResultValuesPerWorker * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(source_a, allocation_bytes, 0x3f, allocation_bytes));
    ACL_CHECK(aclrtMemset(source_b, allocation_bytes, 0x40, allocation_bytes));
    ACL_CHECK(aclrtMemset(destination, allocation_bytes, 0, allocation_bytes));

    for (int64_t warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        RunOne(
            stream, source_a, source_b, destination, worker_cycles, warmup);
    }
    std::vector<int64_t> maximum_worker_bodies;
    std::vector<int64_t> source_program_spans;
    maximum_worker_bodies.reserve(kMeasuredLaunches);
    source_program_spans.reserve(kMeasuredLaunches);
    for (int64_t launch = 0; launch < kMeasuredLaunches; ++launch) {
        const auto cycles = RunOne(
            stream, source_a, source_b, destination, worker_cycles,
            kWarmupLaunches + launch);
        maximum_worker_bodies.push_back(cycles.maximum_worker_body);
        source_program_spans.push_back(cycles.source_program_span);
    }
    std::sort(maximum_worker_bodies.begin(), maximum_worker_bodies.end());
    std::sort(source_program_spans.begin(), source_program_spans.end());
    const double median_body_cycles =
        0.5 * static_cast<double>(
            maximum_worker_bodies[kMeasuredLaunches / 2 - 1]
            + maximum_worker_bodies[kMeasuredLaunches / 2]);
    const double median_span_cycles =
        0.5 * static_cast<double>(
            source_program_spans[kMeasuredLaunches / 2 - 1]
            + source_program_spans[kMeasuredLaunches / 2]);
    const double median_body_ns = median_body_cycles * kSystemCycleNs;
    const double median_span_ns = median_span_cycles * kSystemCycleNs;
    if (!std::isfinite(median_body_ns) || median_body_ns <= 0.0
            || !std::isfinite(median_span_ns)
            || median_span_ns < median_body_ns) {
        std::fprintf(stderr, "invalid finite-program timing\n");
        return EXIT_FAILURE;
    }
    std::printf(
        "workers,full_tile_elements,tail_tile_elements,tbuf_bytes,"
        "measured_launches,median_maximum_worker_cycles,median_body_ns,"
        "median_source_program_span_cycles,median_source_program_span_ns,"
        "validation\n"
        "%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,%.9f,%.9f,pass\n",
        static_cast<long>(kAivWorkers),
        static_cast<long>(kFullTileElements),
        static_cast<long>(kTailTileElements),
        static_cast<long>(kTbufBytes),
        static_cast<long>(kMeasuredLaunches), median_body_cycles,
        median_body_ns, median_span_cycles, median_span_ns);

    ACL_CHECK(aclrtFree(worker_cycles));
    ACL_CHECK(aclrtFree(destination));
    ACL_CHECK(aclrtFree(source_b));
    ACL_CHECK(aclrtFree(source_a));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
