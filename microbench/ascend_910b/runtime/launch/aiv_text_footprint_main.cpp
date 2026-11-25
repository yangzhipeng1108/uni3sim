// Host driver for the AIV instruction-text footprint sweep.
//
// One process issues `launches` launches of one kernel binary, synchronizing
// between them so every launch is a standalone device task.  The first task of
// the process runs with a cold AI-vector instruction cache; every later task
// re-executes text the core already holds.  msprof reports both the per-task
// duration and aiv_icache_miss_rate, so the PMU arbitrates which tasks were
// actually cold instead of the driver asserting it.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aiv_text_footprint(
    uint32_t block_dim, void* stream, void* source, void* destination);

namespace {

constexpr int64_t kMaximumAivBlocks = 48;
constexpr int64_t kElementsPerAiv = 64;
constexpr size_t kBufferBytes = static_cast<size_t>(
    kMaximumAivBlocks * kElementsPerAiv * sizeof(uint16_t));

int64_t ParsePositiveInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "%s must be a positive integer, got %s\n",
                     name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        std::fprintf(
            stderr, "ASCEND_DEVICE_ID must explicitly select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be non-negative\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <block_dim> <launches>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t block_dim = ParsePositiveInteger(argv[1], "block_dim");
    const int64_t launches = ParsePositiveInteger(argv[2], "launches");
    if (block_dim > kMaximumAivBlocks) {
        std::fprintf(stderr, "block_dim must not exceed %ld\n",
                     static_cast<long>(kMaximumAivBlocks));
        return EXIT_FAILURE;
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    void* device_source = nullptr;
    void* device_destination = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_source, kBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_destination, kBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<uint16_t> host_source(
        kBufferBytes / sizeof(uint16_t), 0x3c00);
    ACL_CHECK(aclrtMemcpy(
        device_source, kBufferBytes, host_source.data(), kBufferBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(device_destination, kBufferBytes, 0, kBufferBytes));

    for (int64_t index = 0; index < launches; ++index) {
        ACL_CHECK(aclrtlaunch_aiv_text_footprint(
            static_cast<uint32_t>(block_dim), stream, device_source,
            device_destination));
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    std::vector<uint16_t> output(kBufferBytes / sizeof(uint16_t), 0);
    ACL_CHECK(aclrtMemcpy(
        output.data(), kBufferBytes, device_destination, kBufferBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    std::printf(
        "mechanism=aiv_text_footprint block_dim=%ld launches=%ld "
        "first_output=0x%04x\n",
        static_cast<long>(block_dim), static_cast<long>(launches),
        output[0]);

    ACL_CHECK(aclrtFree(device_destination));
    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
