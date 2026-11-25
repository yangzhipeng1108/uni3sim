// ACL host for the exact five-tile 48-AIV q4/q1 finite-program probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "aiv_mul_q4q1_five_tile_program_contract.h"

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aiv_mul_q4q1_five_tile_program(
    uint32_t block_dim, void* stream, void* source_a, void* source_b,
    void* destination, void* worker_cycles);

namespace {

using namespace aiv_mul_q4q1_five_tile_program_contract;

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
            || value < 0 || value > INT32_MAX) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be an int32 device id\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

void ValidateOutput(void* destination, size_t stream_bytes) {
    uint16_t first = 0;
    uint16_t last = 0;
    ACL_CHECK(aclrtMemcpy(
        &first, sizeof(first), destination, sizeof(first),
        ACL_MEMCPY_DEVICE_TO_HOST));
    const auto* final_address = static_cast<const uint8_t*>(destination)
        + stream_bytes - sizeof(last);
    ACL_CHECK(aclrtMemcpy(
        &last, sizeof(last), final_address, sizeof(last),
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (first != kOutputHalfOne || last != kOutputHalfOne) {
        std::fprintf(
            stderr, "output validation failed: first=0x%04x last=0x%04x\n",
            first, last);
        std::exit(EXIT_FAILURE);
    }
}

void Measure(aclrtStream stream, void* worker_cycles) {
    const size_t stream_bytes = kTotalElements * sizeof(uint16_t);
    const int64_t region_count = kWarmupLaunches + kMeasuredLaunches;
    const size_t allocation_bytes = stream_bytes * region_count;
    void* source_a = nullptr;
    void* source_b = nullptr;
    void* destination = nullptr;
    ACL_CHECK(aclrtMalloc(
        &source_a, allocation_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &source_b, allocation_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &destination, allocation_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(source_a, allocation_bytes, 0x3c, allocation_bytes));
    ACL_CHECK(aclrtMemset(source_b, allocation_bytes, 0x40, allocation_bytes));
    ACL_CHECK(aclrtMemset(destination, allocation_bytes, 0, allocation_bytes));
    for (int64_t warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        const size_t region_offset = warmup * stream_bytes;
        ACL_CHECK(aclrtlaunch_aiv_mul_q4q1_five_tile_program(
            kAivWorkers, stream,
            static_cast<uint8_t*>(source_a) + region_offset,
            static_cast<uint8_t*>(source_b) + region_offset,
            static_cast<uint8_t*>(destination) + region_offset,
            worker_cycles));
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    aclrtEvent start;
    aclrtEvent end;
    ACL_CHECK(aclrtCreateEventWithFlag(&start, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtRecordEvent(start, stream));
    for (int64_t launch = 0; launch < kMeasuredLaunches; ++launch) {
        const size_t region_offset = (
            kWarmupLaunches + launch) * stream_bytes;
        ACL_CHECK(aclrtlaunch_aiv_mul_q4q1_five_tile_program(
            kAivWorkers, stream,
            static_cast<uint8_t*>(source_a) + region_offset,
            static_cast<uint8_t*>(source_b) + region_offset,
            static_cast<uint8_t*>(destination) + region_offset,
            worker_cycles));
    }
    ACL_CHECK(aclrtRecordEvent(end, stream));
    ACL_CHECK(aclrtSynchronizeEvent(end));
    float elapsed_ms = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&elapsed_ms, start, end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
        std::fprintf(stderr, "invalid CANN event duration\n");
        std::exit(EXIT_FAILURE);
    }
    ValidateOutput(
        static_cast<uint8_t*>(destination)
            + (region_count - 1) * stream_bytes,
        stream_bytes);
    std::vector<int64_t> host_cycles(kAivWorkers);
    ACL_CHECK(aclrtMemcpy(
        host_cycles.data(), host_cycles.size() * sizeof(int64_t),
        worker_cycles, host_cycles.size() * sizeof(int64_t),
        ACL_MEMCPY_DEVICE_TO_HOST));
    const int64_t minimum_cycles = *std::min_element(
        host_cycles.begin(), host_cycles.end());
    const int64_t maximum_cycles = *std::max_element(
        host_cycles.begin(), host_cycles.end());
    if (minimum_cycles < 0 || maximum_cycles <= 0) {
        std::fprintf(stderr, "invalid worker system-counter duration\n");
        std::exit(EXIT_FAILURE);
    }
    const double per_launch_ns = (
        static_cast<double>(elapsed_ms) * 1.0e6 / kMeasuredLaunches);
    const double traffic_bytes = 3.0 * static_cast<double>(stream_bytes);
    const double bandwidth_gbs = traffic_bytes / per_launch_ns;
    std::printf(
        "full_tile_elements,full_tiles_per_worker,full_worker_tail_elements,"
        "last_worker_tail_elements,total_elements,measured_launches,"
        "minimum_worker_cycles,maximum_worker_cycles,maximum_worker_ns,"
        "per_launch_ns,total_2r1w_bandwidth_gbs,validation\n");
    std::printf(
        "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,%.9f,pass\n",
        static_cast<long>(kFullTileElements),
        static_cast<long>(kFullTilesPerWorker),
        static_cast<long>(kFullWorkerTailElements),
        static_cast<long>(kLastWorkerTailElements),
        static_cast<long>(kTotalElements),
        static_cast<long>(kMeasuredLaunches),
        static_cast<long>(minimum_cycles),
        static_cast<long>(maximum_cycles),
        maximum_cycles * kSystemCycleNs, per_launch_ns, bandwidth_gbs);
    ACL_CHECK(aclrtFree(destination));
    ACL_CHECK(aclrtFree(source_b));
    ACL_CHECK(aclrtFree(source_a));
    ACL_CHECK(aclrtDestroyEvent(end));
    ACL_CHECK(aclrtDestroyEvent(start));
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
}

int main() {
    aclrtStream stream;
    void* worker_cycles = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(
        &worker_cycles, kAivWorkers * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    Measure(stream, worker_cycles);
    ACL_CHECK(aclrtFree(worker_cycles));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
