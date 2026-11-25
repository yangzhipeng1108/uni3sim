// ACL host for the RMSNorm key-0 Y/rstd output-sequence probe.
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

extern "C" uint32_t aclrtlaunch_rmsnorm_key0_output_sequence(
    uint32_t block_dim, void* stream, void* y_destination,
    void* rstd_destination, void* output, int64_t payload_bytes,
    int64_t pair_order);

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

namespace {

constexpr int64_t kMaximumBlockDim = 47;
constexpr int64_t kMaximumPayloadBytes = 16384;
constexpr int64_t kWorkerStrideBytes = 16384;
constexpr int64_t kRstdStrideBytes = 32;
constexpr size_t kOutputValuesPerBlock = 8;
constexpr double kSystemCounterTickNs = 20.0;

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
    int64_t block_dim;
    int64_t payload_bytes;
    int64_t pair_order;
};

Cell ParseCell(const char* text) {
    const std::string value(text);
    const size_t first = value.find(':');
    const size_t second = value.find(':', first + 1);
    if (first == std::string::npos || second == std::string::npos
            || value.find(':', second + 1) != std::string::npos) {
        std::fprintf(
            stderr, "cell must have block_dim:payload_bytes:pair_order form\n");
        std::exit(EXIT_FAILURE);
    }
    const Cell cell{
        ParseNonnegative(value.substr(0, first).c_str(), "block_dim"),
        ParseNonnegative(
            value.substr(first + 1, second - first - 1).c_str(),
            "payload_bytes"),
        ParseNonnegative(value.substr(second + 1).c_str(), "pair_order")};
    if (cell.block_dim != 32 && cell.block_dim != 43
            && cell.block_dim != 47) {
        std::fprintf(stderr, "block_dim must be 32, 43, or 47\n");
        std::exit(EXIT_FAILURE);
    }
    if (cell.payload_bytes <= 0
            || cell.payload_bytes > kMaximumPayloadBytes
            || cell.payload_bytes % 32 != 0) {
        std::fprintf(
            stderr, "payload_bytes must be a multiple of 32 in [32,16384]\n");
        std::exit(EXIT_FAILURE);
    }
    if (cell.pair_order < 0 || cell.pair_order > 1) {
        std::fprintf(stderr, "pair_order must be zero or one\n");
        std::exit(EXIT_FAILURE);
    }
    return cell;
}

void RunCell(aclrtStream stream, Cell cell) {
    const size_t y_bytes = (
        static_cast<size_t>(cell.block_dim) * kWorkerStrideBytes);
    const size_t rstd_bytes = (
        static_cast<size_t>(cell.block_dim) * kRstdStrideBytes);
    const size_t output_values = (
        static_cast<size_t>(cell.block_dim) * kOutputValuesPerBlock);
    const size_t output_bytes = output_values * sizeof(int64_t);
    void* y_destination = nullptr;
    void* rstd_destination = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &y_destination, y_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &rstd_destination, rstd_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(y_destination, y_bytes, 0, y_bytes));
    ACL_CHECK(aclrtMemset(rstd_destination, rstd_bytes, 0, rstd_bytes));

    ACL_CHECK(aclrtlaunch_rmsnorm_key0_output_sequence(
        static_cast<uint32_t>(cell.block_dim), stream, y_destination,
        rstd_destination, output, cell.payload_bytes, cell.pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> host_output(output_values, -1);
    std::vector<uint16_t> host_y(y_bytes / sizeof(uint16_t), 0);
    std::vector<float> host_rstd(rstd_bytes / sizeof(float), 0.0f);
    ACL_CHECK(aclrtMemcpy(
        host_output.data(), output_bytes, output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(
        host_y.data(), y_bytes, y_destination, y_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(
        host_rstd.data(), rstd_bytes, rstd_destination, rstd_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t worker = 0; worker < cell.block_dim; ++worker) {
        const size_t output_offset = (
            static_cast<size_t>(worker) * kOutputValuesPerBlock);
        const int64_t control_cycles = host_output[output_offset];
        const int64_t request_cycles = host_output[output_offset + 1];
        const int64_t sink = host_output[output_offset + 2];
        const bool valid = (
            control_cycles >= 0 && request_cycles > 0 && sink == 1
            && host_y[
                worker * kWorkerStrideBytes / sizeof(uint16_t)] == 0x3c00
            && host_rstd[
                worker * kRstdStrideBytes / sizeof(float)] == 1.0f);
        if (!valid) {
            std::fprintf(
                stderr,
                "validation failed worker=%lld block_dim=%lld bytes=%lld "
                "order=%lld control=%lld request=%lld sink=%lld\n",
                static_cast<long long>(worker),
                static_cast<long long>(cell.block_dim),
                static_cast<long long>(cell.payload_bytes),
                static_cast<long long>(cell.pair_order),
                static_cast<long long>(control_cycles),
                static_cast<long long>(request_cycles),
                static_cast<long long>(sink));
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.9f,%.9f,%lld,pass\n",
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(worker),
            static_cast<long long>(cell.payload_bytes),
            static_cast<long long>(cell.pair_order),
            static_cast<long long>(kWorkerStrideBytes),
            static_cast<long long>(control_cycles),
            static_cast<long long>(request_cycles),
            static_cast<double>(control_cycles) * kSystemCounterTickNs,
            static_cast<double>(request_cycles) * kSystemCounterTickNs,
            static_cast<long long>(sink));
    }

    ACL_CHECK(aclrtFree(y_destination));
    ACL_CHECK(aclrtFree(rstd_destination));
    ACL_CHECK(aclrtFree(output));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(
            stderr,
            "usage: %s --batch block_dim:payload_bytes:pair_order...\n",
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
