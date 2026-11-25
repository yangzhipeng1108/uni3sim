// Host driver for the independent q2 CopyIn first/steady edge primitive.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    const int status = (call);                                             \
    if (status != 0) {                                                     \
    std::fprintf(stderr, "%s failed: %d\n", #call, status);          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

extern "C" uint32_t aclrtlaunch_swiglu_input_pair_loop_edges(
    uint32_t block_dim, void* stream, void* first_source, void* second_source,
    void* output, int64_t repetitions, int64_t pair_order);

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

namespace {

constexpr int64_t kMaximumBlockDim = 48;
constexpr size_t kPayloadBytes = 18432;
constexpr size_t kOutputValuesPerBlock = 16;
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kExpectedSinkPerPair = 0xb4;

int64_t ParsePositive(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "%s must be positive\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s block_dim repetitions pair_order\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t block_dim = ParsePositive(argv[1], "block_dim");
    const int64_t repetitions = ParsePositive(argv[2], "repetitions");
    const int64_t pair_order = ParsePositive(argv[3], "pair_order") - 1;
    if (block_dim > kMaximumBlockDim || (repetitions != 1 && repetitions != 2)
            || pair_order > 1) {
        std::fprintf(stderr, "invalid block_dim, repetitions, or pair_order\n");
        return EXIT_FAILURE;
    }
    const size_t source_bytes = static_cast<size_t>(block_dim) * kPayloadBytes;
    const size_t output_values = static_cast<size_t>(block_dim) * kOutputValuesPerBlock;
    const size_t output_bytes = output_values * sizeof(int64_t);
    std::vector<uint8_t> host_source(source_bytes, 0x5a);
    std::vector<int64_t> host_output(output_values, -1);
    void* first_source = nullptr;
    void* second_source = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclrtMalloc(&first_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&second_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(first_source, source_bytes, host_source.data(), source_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(second_source, source_bytes, host_source.data(), source_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtlaunch_swiglu_input_pair_loop_edges(
        static_cast<uint32_t>(block_dim), stream, first_source, second_source,
        output, repetitions, pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(host_output.data(), output_bytes, output, output_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t worker = 0; worker < block_dim; ++worker) {
        const size_t offset = static_cast<size_t>(worker) * kOutputValuesPerBlock;
        const int64_t control_first = host_output[offset];
        const int64_t control_steady = host_output[offset + 1];
        const int64_t request_first = host_output[offset + 2];
        const int64_t request_steady = host_output[offset + 3];
        const int64_t sink = host_output[offset + 4];
        if (control_first < 0 || request_first <= 0
                || (repetitions == 2
                    && (control_steady < 0 || request_steady <= 0))
                || (repetitions == 1
                    && (control_steady != 0 || request_steady != 0))
                || sink != repetitions * kExpectedSinkPerPair
                || host_output[offset + 5] != repetitions
                || host_output[offset + 6] != pair_order) {
            std::fprintf(
                stderr,
                "device validation failed worker=%lld c1=%lld cs=%lld "
                "r1=%lld rs=%lld sink=%lld reps=%lld order=%lld\n",
                static_cast<long long>(worker),
                static_cast<long long>(control_first),
                static_cast<long long>(control_steady),
                static_cast<long long>(request_first),
                static_cast<long long>(request_steady),
                static_cast<long long>(sink),
                static_cast<long long>(host_output[offset + 5]),
                static_cast<long long>(host_output[offset + 6]));
            return EXIT_FAILURE;
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.9f,%.9f,%.9f,%.9f,%lld,pass\n",
            static_cast<long long>(block_dim), static_cast<long long>(worker),
            static_cast<long long>(repetitions), static_cast<long long>(pair_order),
            static_cast<long long>(control_first),
            static_cast<long long>(control_steady),
            static_cast<long long>(request_first),
            static_cast<long long>(request_steady),
            static_cast<double>(control_first) * kSystemCounterTickNs,
            static_cast<double>(control_steady) * kSystemCounterTickNs,
            static_cast<double>(request_first) * kSystemCounterTickNs,
            static_cast<double>(request_steady) * kSystemCounterTickNs,
            static_cast<long long>(sink));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(first_source));
    ACL_CHECK(aclrtFree(second_source));
    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
