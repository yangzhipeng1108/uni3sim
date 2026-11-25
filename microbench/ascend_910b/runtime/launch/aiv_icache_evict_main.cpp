// Host driver for the AIV instruction-cache eviction control.
//
// The footprint sweep (aiv_text_footprint_main.cpp) can only compare a
// process's first launch against later launches of the same binary, so it
// conflates a cold instruction cache with one-time runtime setup and it
// cannot say whether a launch of some *other* kernel puts the text back in
// the cold state.  That second question is the one a serving step asks: a
// decode step runs many distinct kernels, so whether each of them refetches
// its text depends on whether the cores hold more than one kernel text at a
// time.
//
// This driver links two kernels with distinct symbols and identical bodies
// and runs them on one stream in one process, so the one-time setup is paid
// once and drops out of the comparison:
//
//   repeat     A A A A ...   steady-state A follows A
//   alternate  A B A B ...   steady-state A follows B
//
// If alternating restores the per-task cost and the PMU miss rate of the
// cold launches, the other kernel's text evicts this one and every distinct
// kernel in a step fetches its own text.  If alternating stays warm, both
// texts are resident together and only the first use of each is cold.
// Sweeping the pair size locates the footprint at which the two modes
// separate, which is the resident instruction capacity in bytes.
//
// msprof reports Task Duration(us) and aiv_icache_miss_rate per task, so the
// PMU arbitrates which tasks actually missed rather than the driver
// asserting it.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",      \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aiv_text_footprint(
    uint32_t block_dim, void* stream, void* source, void* destination);
extern "C" uint32_t aclrtlaunch_aiv_text_footprint_alt(
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
    if (argc != 4) {
        std::fprintf(
            stderr, "usage: %s <block_dim> <rounds> <repeat|alternate>\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t block_dim = ParsePositiveInteger(argv[1], "block_dim");
    const int64_t rounds = ParsePositiveInteger(argv[2], "rounds");
    bool alternate = false;
    if (std::strcmp(argv[3], "alternate") == 0) {
        alternate = true;
    } else if (std::strcmp(argv[3], "repeat") != 0) {
        std::fprintf(stderr, "mode must be repeat or alternate, got %s\n",
                     argv[3]);
        return EXIT_FAILURE;
    }
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

    // Both modes issue the same number of tasks in the same A-then-other
    // order, so the two captures differ only in which text the intervening
    // task executed.
    for (int64_t round = 0; round < rounds; ++round) {
        ACL_CHECK(aclrtlaunch_aiv_text_footprint(
            static_cast<uint32_t>(block_dim), stream, device_source,
            device_destination));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        if (alternate) {
            ACL_CHECK(aclrtlaunch_aiv_text_footprint_alt(
                static_cast<uint32_t>(block_dim), stream, device_source,
                device_destination));
        } else {
            ACL_CHECK(aclrtlaunch_aiv_text_footprint(
                static_cast<uint32_t>(block_dim), stream, device_source,
                device_destination));
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    std::vector<uint16_t> output(kBufferBytes / sizeof(uint16_t), 0);
    ACL_CHECK(aclrtMemcpy(
        output.data(), kBufferBytes, device_destination, kBufferBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    std::printf(
        "mechanism=aiv_icache_evict block_dim=%ld rounds=%ld mode=%s "
        "first_output=0x%04x\n",
        static_cast<long>(block_dim), static_cast<long>(rounds), argv[3],
        output[0]);

    ACL_CHECK(aclrtFree(device_destination));
    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
