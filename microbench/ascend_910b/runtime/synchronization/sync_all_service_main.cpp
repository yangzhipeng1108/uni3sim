// Host driver for the independent Ascend 910B AIV SyncAll benchmark.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

#define CLOCK_CHECK(call) do {                                            \
    if ((call) != 0) {                                                    \
        std::perror(#call);                                               \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_sync_all_service(
    uint32_t block_dim, void* stream, void* gm_cycles);

namespace {

constexpr int32_t kLogicalDeviceId = 0;
constexpr int64_t kMaximumAivBlocks = 48;
constexpr int64_t kAivBlocksPerAicBlock = 2;
constexpr int64_t kCycleCacheLineBytes = 64;
constexpr int64_t kCycleSlotElements =
    kCycleCacheLineBytes / static_cast<int64_t>(sizeof(int64_t));

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

std::string JoinCycles(const std::vector<int64_t>& values) {
    std::string result;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            result += ';';
        }
        result += std::to_string(values[index]);
    }
    return result;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be set explicitly\n");
        std::exit(EXIT_FAILURE);
    }
    if (ParseInteger(device_text, "ASCEND_DEVICE_ID") != kLogicalDeviceId) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must be logical device 0 after visibility "
            "remapping\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::strcmp(argv[1], "--batch") != 0) {
        std::fprintf(stderr, "usage: %s --batch aivBlockDim [...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<int64_t> block_dims;
    block_dims.reserve(static_cast<size_t>(argc - 2));
    for (int argument = 2; argument < argc; ++argument) {
        const int64_t block_dim = ParseInteger(argv[argument], "aivBlockDim");
        if (block_dim <= 0 || block_dim > kMaximumAivBlocks
                || block_dim % 2 != 0) {
            std::fprintf(stderr, "AIV blockDim must be even and in [2, 48]\n");
            return EXIT_FAILURE;
        }
        block_dims.push_back(block_dim);
    }

    const size_t cycle_bytes = static_cast<size_t>(
        kMaximumAivBlocks * kCycleCacheLineBytes);
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_cycles, cycle_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0xff, cycle_bytes));
    ACL_CHECK(aclrtlaunch_sync_all_service(
        static_cast<uint32_t>(
            kMaximumAivBlocks / kAivBlocksPerAicBlock),
        stream, device_cycles));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::printf("block_dim,block_cycles,host_wall_ns,validation\n");
    for (const int64_t block_dim : block_dims) {
        ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0xff, cycle_bytes));
        ACL_CHECK(aclrtlaunch_sync_all_service(
            static_cast<uint32_t>(block_dim / kAivBlocksPerAicBlock),
            stream, device_cycles));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0xff, cycle_bytes));

        struct timespec start;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &start));
        ACL_CHECK(aclrtlaunch_sync_all_service(
            static_cast<uint32_t>(block_dim / kAivBlocksPerAicBlock),
            stream, device_cycles));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        const int64_t wall_ns = static_cast<int64_t>(end.tv_sec - start.tv_sec)
            * 1000000000LL + static_cast<int64_t>(end.tv_nsec - start.tv_nsec);
        if (wall_ns <= 0) {
            std::fprintf(stderr, "host health interval must be positive\n");
            return EXIT_FAILURE;
        }

        std::vector<int64_t> padded_cycles(
            static_cast<size_t>(kMaximumAivBlocks * kCycleSlotElements), -1);
        ACL_CHECK(aclrtMemcpy(
            padded_cycles.data(), cycle_bytes, device_cycles, cycle_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        std::vector<int64_t> cycles;
        cycles.reserve(static_cast<size_t>(block_dim));
        for (int64_t block = 0; block < block_dim; ++block) {
            const int64_t value = padded_cycles[static_cast<size_t>(
                block * kCycleSlotElements)];
            if (value <= 0) {
                std::fprintf(
                    stderr, "invalid cycle %ld at block=%ld blockDim=%ld\n",
                    static_cast<long>(value), static_cast<long>(block),
                    static_cast<long>(block_dim));
                return EXIT_FAILURE;
            }
            cycles.push_back(value);
        }
        std::printf(
            "%ld,%s,%ld,pass\n", static_cast<long>(block_dim),
            JoinCycles(cycles).c_str(), static_cast<long>(wall_ns));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
