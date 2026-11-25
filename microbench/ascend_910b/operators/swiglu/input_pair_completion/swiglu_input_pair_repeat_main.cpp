// Host driver for the high-repetition paired q2 CopyIn measurement.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",      \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_swiglu_input_pair_repeat(
    uint32_t block_dim, void* stream, void* first_source,
    void* second_source, void* output, int64_t repetitions,
    int64_t phase_order);

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

namespace {

constexpr int64_t kMaximumBlockDim = 48;
constexpr size_t kPayloadBytes = 8192;
constexpr size_t kOutputValuesPerBlock = 8;
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
        std::fprintf(stderr, "usage: %s block_dim repetitions phase_order\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t block_dim = ParsePositive(argv[1], "block_dim");
    const int64_t repetitions = ParsePositive(argv[2], "repetitions");
    const int64_t phase_order = ParsePositive(argv[3], "phase_order") - 1;
    if (block_dim > kMaximumBlockDim || phase_order > 1) {
        std::fprintf(stderr, "block_dim must be <= 48 and phase_order must be 1 or 2\n");
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
    ACL_CHECK(aclrtlaunch_swiglu_input_pair_repeat(
        static_cast<uint32_t>(block_dim), stream, first_source, second_source,
        output, repetitions, phase_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(host_output.data(), output_bytes, output, output_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t worker = 0; worker < block_dim; ++worker) {
        const size_t offset = static_cast<size_t>(worker) * kOutputValuesPerBlock;
        const int64_t small_cycles = host_output[offset];
        const int64_t large_cycles = host_output[offset + 1];
        const int64_t sink = host_output[offset + 2];
        const int64_t recorded_repetitions = host_output[offset + 3];
        const int64_t recorded_order = host_output[offset + 4];
        if (small_cycles <= 0 || large_cycles <= 0
                || sink != 2 * repetitions * kExpectedSinkPerPair
                || recorded_repetitions != repetitions
                || recorded_order != phase_order) {
            std::fprintf(stderr, "validation failed worker=%lld\n",
                         static_cast<long long>(worker));
            return EXIT_FAILURE;
        }
        std::printf("%lld,%lld,%lld,%lld,%.9f,%.9f,pass\n",
                    static_cast<long long>(block_dim),
                    static_cast<long long>(worker),
                    static_cast<long long>(small_cycles),
                    static_cast<long long>(large_cycles),
                    static_cast<double>(small_cycles) * kSystemCounterTickNs / repetitions,
                    static_cast<double>(large_cycles) * kSystemCounterTickNs / repetitions);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(first_source));
    ACL_CHECK(aclrtFree(second_source));
    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
