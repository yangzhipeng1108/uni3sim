// Host driver for the RMSNorm key-0 serial gamma/input completion probe.
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
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_rmsnorm_key0_gamma_input_sequence(
    uint32_t block_dim, void* stream, void* gamma_source,
    void* input_source, void* output, int64_t payload_bytes,
    int64_t stride_bytes, int64_t pair_order);

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

namespace {

constexpr size_t kOutputValuesPerBlock = 8;
constexpr int64_t kBlockDim = 1;
constexpr int64_t kMaximumPayloadBytes = 16384;
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kExpectedSink = 0xb4;

int64_t ParseNonnegative(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

struct Cell {
    int64_t payload_bytes;
    int64_t pair_order;
};

Cell ParseCell(const char* text) {
    const std::string value(text);
    const size_t separator = value.find(':');
    if (separator == std::string::npos
            || value.find(':', separator + 1) != std::string::npos) {
        std::fprintf(
            stderr, "cell must have payload_bytes:pair_order form\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t payload_bytes = ParseNonnegative(
        value.substr(0, separator).c_str(), "payload_bytes");
    const int64_t pair_order = ParseNonnegative(
        value.substr(separator + 1).c_str(), "pair_order");
    if (payload_bytes <= 0 || payload_bytes > kMaximumPayloadBytes
            || payload_bytes % 32 != 0) {
        std::fprintf(
            stderr, "payload_bytes must be a multiple of 32 in [32,16384]\n");
        std::exit(EXIT_FAILURE);
    }
    if (pair_order < 0 || pair_order > 1) {
        std::fprintf(stderr, "pair_order must be zero or one\n");
        std::exit(EXIT_FAILURE);
    }
    return Cell{payload_bytes, pair_order};
}

void RunCell(aclrtStream stream, Cell cell) {
    const size_t stride_bytes = kMaximumPayloadBytes;
    std::vector<uint8_t> host_source(stride_bytes, 0x5a);
    void* gamma_source = nullptr;
    void* input_source = nullptr;
    void* output = nullptr;
    const size_t output_bytes = kOutputValuesPerBlock * sizeof(int64_t);
    ACL_CHECK(aclrtMalloc(
        &gamma_source, stride_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &input_source, stride_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        gamma_source, stride_bytes, host_source.data(), stride_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(
        input_source, stride_bytes, host_source.data(), stride_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtlaunch_rmsnorm_key0_gamma_input_sequence(
        kBlockDim, stream, gamma_source, input_source, output,
        cell.payload_bytes, static_cast<int64_t>(stride_bytes),
        cell.pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> host_output(kOutputValuesPerBlock, -1);
    ACL_CHECK(aclrtMemcpy(
        host_output.data(), output_bytes, output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    const int64_t control_cycles = host_output[0];
    const int64_t request_cycles = host_output[1];
    const int64_t sink = host_output[2];
    if (control_cycles < 0 || request_cycles <= 0
            || sink != kExpectedSink) {
        std::fprintf(
            stderr,
            "validation failed bytes=%lld order=%lld control=%lld "
            "request=%lld sink=%lld\n",
            static_cast<long long>(cell.payload_bytes),
            static_cast<long long>(cell.pair_order),
            static_cast<long long>(control_cycles),
            static_cast<long long>(request_cycles),
            static_cast<long long>(sink));
        std::exit(EXIT_FAILURE);
    }
    std::printf(
        "%lld,%lld,%zu,%lld,%lld,%.9f,%.9f,%lld,pass\n",
        static_cast<long long>(cell.payload_bytes),
        static_cast<long long>(cell.pair_order), stride_bytes,
        static_cast<long long>(control_cycles),
        static_cast<long long>(request_cycles),
        static_cast<double>(control_cycles) * kSystemCounterTickNs,
        static_cast<double>(request_cycles) * kSystemCounterTickNs,
        static_cast<long long>(sink));

    ACL_CHECK(aclrtFree(gamma_source));
    ACL_CHECK(aclrtFree(input_source));
    ACL_CHECK(aclrtFree(output));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(
            stderr, "usage: %s --batch payload_bytes:pair_order...\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    for (int index = 2; index < argc; ++index) {
        RunCell(stream, ParseCell(argv[index]));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
