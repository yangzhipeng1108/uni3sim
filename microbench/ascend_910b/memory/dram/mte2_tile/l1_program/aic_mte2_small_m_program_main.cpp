// ACL host for the independent key-98513 A1/B1 MTE2 program probe.
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_small_m_program(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output);

namespace {

constexpr int64_t kActiveAics = 24;
constexpr int64_t kElementBytes = 2;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kASourceElements = 16 * 5120;
constexpr int64_t kBSourceElements = 1536 * 5120;
constexpr int64_t kKSlabs = 10;
constexpr double kSystemCounterTickNs = 20.0;

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

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--trials") != 0) {
        std::fprintf(stderr, "usage: %s --trials COUNT\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[2], "trials");
    if (trials < 3 || trials > 32) {
        std::fprintf(stderr, "trials must be in [3,32]\n");
        return EXIT_FAILURE;
    }

    const int64_t source_programs = trials + 1;
    const size_t a_source_bytes = static_cast<size_t>(
        source_programs * kASourceElements * kElementBytes);
    const size_t b_source_bytes = static_cast<size_t>(
        source_programs * kBSourceElements * kElementBytes);
    const size_t output_bytes = static_cast<size_t>(
        kActiveAics * kCycleSlotBytes);
    void* device_a_source = nullptr;
    void* device_b_source = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_a_source, a_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_b_source, b_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_a_source, a_source_bytes, 0, a_source_bytes));
    ACL_CHECK(aclrtMemset(device_b_source, b_source_bytes, 0, b_source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::printf("trial,max_cycles,duration_ns,validation\n");
    for (int64_t trial = 0; trial <= trials; ++trial) {
        ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
        auto* a_source = static_cast<char*>(device_a_source)
            + trial * kASourceElements * kElementBytes;
        auto* b_source = static_cast<char*>(device_b_source)
            + trial * kBSourceElements * kElementBytes;
        ACL_CHECK(aclrtlaunch_aic_mte2_small_m_program(
            static_cast<uint32_t>(kActiveAics), stream,
            a_source, b_source, device_output));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            output.data(), output_bytes, device_output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t maximum_cycles = 0;
        for (int64_t block = 0; block < kActiveAics; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            if (output[offset] <= 0 || output[offset + 1] != kKSlabs) {
                std::fprintf(stderr, "small-M program validation failed\n");
                return EXIT_FAILURE;
            }
            maximum_cycles = std::max(maximum_cycles, output[offset]);
        }
        if (trial > 0) {
            std::printf(
                "%ld,%ld,%.9f,pass\n", trial, maximum_cycles,
                maximum_cycles * kSystemCounterTickNs);
            std::fflush(stdout);
        }
    }

    ACL_CHECK(aclrtFree(device_a_source));
    ACL_CHECK(aclrtFree(device_b_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
