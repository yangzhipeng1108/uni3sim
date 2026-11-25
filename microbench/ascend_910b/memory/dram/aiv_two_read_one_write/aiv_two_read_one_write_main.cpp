// ACL host for the independent full-chip AIV 2-read/1-write GM probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aiv_two_read_one_write_contract.h"

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aiv_two_read_one_write(
    uint32_t block_dim, void* stream, void* source_a, void* source_b,
    void* destination, int64_t tile_elements, int64_t tiles_per_worker);

namespace {

using namespace aiv_two_read_one_write_contract;

int64_t ParseInteger(const char* text, const char* label) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr || *text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t value = ParseInteger(text, "ASCEND_DEVICE_ID");
    if (value < 0 || value > INT32_MAX) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID is outside int32 range\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

size_t StreamBytes(int64_t tile_bytes) {
    return static_cast<size_t>(
        kAivWorkers * kTilesPerWorker * tile_bytes);
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

void Measure(
        aclrtStream stream, void* source_a, void* source_b,
        void* destination, int64_t tile_bytes) {
    if (!IsSupportedTileBytes(tile_bytes)) {
        std::fprintf(stderr, "unsupported tile_bytes=%ld\n",
                     static_cast<long>(tile_bytes));
        std::exit(EXIT_FAILURE);
    }
    const int64_t tile_elements = tile_bytes / sizeof(uint16_t);
    const size_t stream_bytes = StreamBytes(tile_bytes);
    ACL_CHECK(aclrtMemset(destination, stream_bytes, 0, stream_bytes));
    for (int64_t warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        ACL_CHECK(aclrtlaunch_aiv_two_read_one_write(
            kAivWorkers, stream, source_a, source_b, destination,
            tile_elements, kTilesPerWorker));
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    aclrtEvent start;
    aclrtEvent end;
    ACL_CHECK(aclrtCreateEventWithFlag(&start, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE));
    ACL_CHECK(aclrtRecordEvent(start, stream));
    for (int64_t launch = 0; launch < kMeasuredLaunches; ++launch) {
        ACL_CHECK(aclrtlaunch_aiv_two_read_one_write(
            kAivWorkers, stream, source_a, source_b, destination,
            tile_elements, kTilesPerWorker));
    }
    ACL_CHECK(aclrtRecordEvent(end, stream));
    ACL_CHECK(aclrtSynchronizeEvent(end));
    float elapsed_ms = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&elapsed_ms, start, end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
        std::fprintf(stderr, "invalid CANN event duration\n");
        std::exit(EXIT_FAILURE);
    }
    ValidateOutput(destination, stream_bytes);
    const double per_launch_ns = (
        static_cast<double>(elapsed_ms) * 1.0e6 / kMeasuredLaunches);
    const double traffic_bytes = 3.0 * static_cast<double>(stream_bytes);
    const double bandwidth_gbs = traffic_bytes / per_launch_ns;
    std::printf(
        "%ld,%ld,%zu,%ld,%.9f,%.9f,pass\n",
        static_cast<long>(tile_bytes),
        static_cast<long>(kTilesPerWorker), stream_bytes,
        static_cast<long>(kMeasuredLaunches), per_launch_ns, bandwidth_gbs);
    ACL_CHECK(aclrtDestroyEvent(end));
    ACL_CHECK(aclrtDestroyEvent(start));
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s TILE_BYTES [TILE_BYTES ...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<int64_t> tile_bytes;
    size_t maximum_stream_bytes = 0;
    for (int index = 1; index < argc; ++index) {
        const int64_t value = ParseInteger(argv[index], "tile_bytes");
        if (!IsSupportedTileBytes(value)) {
            std::fprintf(stderr, "unsupported tile_bytes=%ld\n",
                         static_cast<long>(value));
            return EXIT_FAILURE;
        }
        tile_bytes.push_back(value);
        maximum_stream_bytes = std::max(
            maximum_stream_bytes, StreamBytes(value));
    }

    aclrtStream stream;
    void* source_a = nullptr;
    void* source_b = nullptr;
    void* destination = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(
        &source_a, maximum_stream_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &source_b, maximum_stream_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &destination, maximum_stream_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(
        source_a, maximum_stream_bytes, 0x3c, maximum_stream_bytes));
    ACL_CHECK(aclrtMemset(
        source_b, maximum_stream_bytes, 0x40, maximum_stream_bytes));

    std::printf(
        "tile_bytes,tiles_per_worker,stream_bytes,measured_launches,"
        "per_launch_ns,total_2r1w_bandwidth_gbs,validation\n");
    for (int64_t value : tile_bytes) {
        Measure(stream, source_a, source_b, destination, value);
    }

    ACL_CHECK(aclrtFree(destination));
    ACL_CHECK(aclrtFree(source_b));
    ACL_CHECK(aclrtFree(source_a));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
