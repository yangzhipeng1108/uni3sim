#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                               \
    const int status = static_cast<int>(call);                             \
    if (status != 0) {                                                     \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",     \
                     __FILE__, __LINE__, #call, status);                   \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

extern "C" uint32_t aclrtlaunch_add_rmsnorm_key31_residual_chain_depths(
    uint32_t blockDim, void* stream, void* source, void* output,
    uint32_t block_factor, uint32_t last_block_factor, int64_t pair_order);

namespace {

constexpr int64_t kTilesPerRow = 2;
constexpr int64_t kRequestBytes = 16384;
constexpr int64_t kMaximumRequests = 44;
constexpr int64_t kWorkerStrideBytes = kMaximumRequests * kRequestBytes;
constexpr size_t kOutputValuesPerWorker = 8;
constexpr double kSystemCounterTickNs = 20.0;

int64_t ParseInteger(const char* text, const char* name, int64_t maximum) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value <= 0 || value > maximum) {
        std::fprintf(stderr, "%s must be in [1, %lld]\n", name,
                     static_cast<long long>(maximum));
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int64_t ParsePairOrder(const char* text) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < 0 || value > 1) {
        std::fprintf(stderr, "pair_order must be 0 or 1\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

void RunCell(aclrtStream stream, void* source, void* output,
             int64_t block_dim, int64_t block_factor,
             int64_t last_block_factor, int64_t pair_order) {
    ACL_CHECK(aclrtlaunch_add_rmsnorm_key31_residual_chain_depths(
        static_cast<uint32_t>(block_dim), stream, source, output,
        static_cast<uint32_t>(block_factor),
        static_cast<uint32_t>(last_block_factor), pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<int64_t> host_output(
        block_dim * kOutputValuesPerWorker, -1);
    ACL_CHECK(aclrtMemcpy(
        host_output.data(), host_output.size() * sizeof(int64_t), output,
        host_output.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t worker = 0; worker < block_dim; ++worker) {
        const size_t offset = worker * kOutputValuesPerWorker;
        const int64_t control_cycles = host_output[offset];
        const int64_t request_cycles = host_output[offset + 1];
        const int64_t control_sink = host_output[offset + 2];
        const int64_t request_sink = host_output[offset + 3];
        const int64_t request_count = host_output[offset + 4];
        const int64_t request_bytes = host_output[offset + 5];
        const int64_t actual_order = host_output[offset + 6];
        const int64_t expected_count = (
            worker + 1 == block_dim
            ? last_block_factor * kTilesPerRow
            : block_factor * kTilesPerRow);
        if (control_cycles <= 0 || request_cycles <= 0
                || control_sink != 1 || request_sink != 2
                || request_count != expected_count
                || request_bytes != kRequestBytes
                || actual_order != pair_order) {
            std::fprintf(stderr, "validation failed worker=%lld\n",
                         static_cast<long long>(worker));
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.9f,%.9f,%lld,%lld,pass\n",
            static_cast<long long>(block_dim),
            static_cast<long long>(worker),
            static_cast<long long>(pair_order),
            static_cast<long long>(request_count),
            static_cast<long long>(request_bytes),
            static_cast<long long>(control_cycles),
            static_cast<long long>(request_cycles),
            static_cast<double>(control_cycles) * kSystemCounterTickNs,
            static_cast<double>(request_cycles) * kSystemCounterTickNs,
            static_cast<long long>(control_sink),
            static_cast<long long>(request_sink));
    }
}

}  // namespace

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 9 || std::string(argv[1]) != "--block-dim"
            || std::string(argv[3]) != "--block-factor"
            || std::string(argv[5]) != "--last-block-factor"
            || std::string(argv[7]) != "--batch") {
        std::fprintf(
            stderr,
            "usage: %s --block-dim N --block-factor N "
            "--last-block-factor N --batch pair_order...\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t block_dim = ParseInteger(argv[2], "block_dim", 48);
    const int64_t block_factor = ParseInteger(argv[4], "block_factor", 22);
    const int64_t last_block_factor = ParseInteger(
        argv[6], "last_block_factor", block_factor);
    const size_t source_bytes = block_dim * kWorkerStrideBytes;
    std::vector<uint16_t> host_source(
        source_bytes / sizeof(uint16_t), 0x3f80);
    void* source = nullptr;
    void* output = nullptr;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtMalloc(&source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &output, block_dim * kOutputValuesPerWorker * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        source, source_bytes, host_source.data(), source_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtCreateStream(&stream));
    for (int index = 8; index < argc; ++index) {
        RunCell(stream, source, output, block_dim, block_factor,
                last_block_factor, ParsePairOrder(argv[index]));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(source));
    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
