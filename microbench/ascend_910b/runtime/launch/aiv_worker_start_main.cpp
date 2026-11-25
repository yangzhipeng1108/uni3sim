// Capture per-AIV entry timestamps without a profiler or host timing model.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>


#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",     \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)


extern "C" uint32_t aclrtlaunch_aiv_worker_start(
    uint32_t block_dim, void* stream, void* timestamps);


namespace {

constexpr int64_t kMaximumAivBlocks = 48;
constexpr int64_t kValuesPerCacheLine = 8;
constexpr size_t kTimestampBytes = static_cast<size_t>(
    kMaximumAivBlocks * kValuesPerCacheLine * sizeof(int64_t));

int64_t ParseNonnegativeInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        std::fprintf(
            stderr, "%s must be a non-negative integer, got %s\n",
            name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int64_t ParsePositiveInteger(const char* text, const char* name) {
    const int64_t value = ParseNonnegativeInteger(text, name);
    if (value == 0) {
        std::fprintf(stderr, "%s must be positive, got 0\n", name);
        std::exit(EXIT_FAILURE);
    }
    return value;
}

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must explicitly select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(
        ParseNonnegativeInteger(text, "ASCEND_DEVICE_ID"));
}

void Capture(
        aclrtStream stream, void* device_timestamps, int64_t block_dim,
        int64_t trial) {
    ACL_CHECK(aclrtMemset(
        device_timestamps, kTimestampBytes, 0, kTimestampBytes));
    ACL_CHECK(aclrtlaunch_aiv_worker_start(
        static_cast<uint32_t>(block_dim), stream, device_timestamps));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<int64_t> timestamps(
        kMaximumAivBlocks * kValuesPerCacheLine, 0);
    ACL_CHECK(aclrtMemcpy(
        timestamps.data(), kTimestampBytes,
        device_timestamps, kTimestampBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t first_start = timestamps[0];
    for (int64_t block = 0; block < block_dim; ++block) {
        const int64_t offset = block * kValuesPerCacheLine;
        if (timestamps[offset + 2] != block + 1) {
            std::fprintf(
                stderr, "validation failed for block %ld\n",
                static_cast<long>(block));
            std::exit(EXIT_FAILURE);
        }
        first_start = std::min(first_start, timestamps[offset]);
    }
    for (int64_t block = 0; block < block_dim; ++block) {
        const int64_t offset = block * kValuesPerCacheLine;
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld\n",
            static_cast<long>(trial), static_cast<long>(block_dim),
            static_cast<long>(block),
            static_cast<long>(timestamps[offset]),
            static_cast<long>(timestamps[offset] - first_start),
            static_cast<long>(timestamps[offset + 1]
                              - timestamps[offset]));
    }
}

}  // namespace


__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <block_dim> <trials>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t block_dim = ParsePositiveInteger(argv[1], "block_dim");
    const int64_t trials = ParsePositiveInteger(argv[2], "trials");
    if (block_dim > kMaximumAivBlocks) {
        std::fprintf(
            stderr, "block_dim cannot exceed 48, got %ld\n",
            static_cast<long>(block_dim));
        return EXIT_FAILURE;
    }

    aclrtStream stream;
    void* device_timestamps = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(
        &device_timestamps, kTimestampBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    for (int64_t warmup = 0; warmup < 10; ++warmup) {
        ACL_CHECK(aclrtlaunch_aiv_worker_start(
            static_cast<uint32_t>(block_dim), stream, device_timestamps));
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::printf(
        "trial,block_dim,block_index,start_cycle,relative_start_cycles,"
        "entry_probe_cycles\n");
    for (int64_t trial = 0; trial < trials; ++trial) {
        Capture(stream, device_timestamps, block_dim, trial);
    }

    ACL_CHECK(aclrtFree(device_timestamps));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
