// ACL host for the 23-AIC, one-A / twenty-B MTE2-only probe.
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

extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_a1_b20_stream(
    uint32_t block_dim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output);

namespace {

constexpr int64_t kActiveAics = 23;
constexpr int64_t kARows = 16;
constexpr int64_t kAColumns = 5120;
constexpr int64_t kBRows = 224;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kExpectedDescriptorsPerWorker = 21;
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

int main(int argc, char** argv) {
    if (argc != 5 || std::strcmp(argv[1], "--trials") != 0
            || std::strcmp(argv[3], "--device-id") != 0) {
        std::fprintf(stderr, "usage: %s --trials COUNT --device-id ID\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[2], "trials");
    const int64_t device_id = ParseInteger(argv[4], "device_id");
    if (trials < 3 || trials > 24 || device_id < 0) {
        std::fprintf(stderr, "trials must be in [3,24] and device ID non-negative\n");
        return EXIT_FAILURE;
    }
    const size_t a_bytes = static_cast<size_t>(kARows * kAColumns * 2);
    const size_t b_bytes = static_cast<size_t>(
        kActiveAics * kBRows * kAColumns * 2);
    const size_t output_bytes = static_cast<size_t>(kActiveAics * kCycleSlotBytes);
    void* a_source = nullptr;
    void* b_source = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(static_cast<int32_t>(device_id)));
    ACL_CHECK(aclrtMalloc(&a_source, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&b_source, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(a_source, a_bytes, 0, a_bytes));
    ACL_CHECK(aclrtMemset(b_source, b_bytes, 0, b_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    auto launch = [&]() {
        ACL_CHECK(aclrtMemset(output, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_aic_mte2_gemm_a1_b20_stream(
            kActiveAics, stream, a_source, b_source, output));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> values(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            values.data(), output_bytes, output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t maximum_cycles = 0;
        for (int64_t block = 0; block < kActiveAics; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            if (values[offset] <= 0
                    || values[offset + 1] != kExpectedDescriptorsPerWorker
                    || values[offset + 3] < values[offset + 2]) {
                std::fprintf(stderr, "invalid block output for block=%ld\n", block);
                std::exit(EXIT_FAILURE);
            }
            maximum_cycles = std::max(maximum_cycles, values[offset]);
        }
        return maximum_cycles;
    };
    for (int64_t warmup = 0; warmup < 4; ++warmup) {
        launch();
    }
    std::printf("trial,max_cycles,duration_ns,validation\n");
    for (int64_t trial = 0; trial < trials; ++trial) {
        const int64_t cycles = launch();
        std::printf("%ld,%ld,%.9f,pass\n", trial, cycles,
                    cycles * kSystemCounterTickNs);
        std::fflush(stdout);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(a_source));
    ACL_CHECK(aclrtFree(b_source));
    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtResetDevice(static_cast<int32_t>(device_id)));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
