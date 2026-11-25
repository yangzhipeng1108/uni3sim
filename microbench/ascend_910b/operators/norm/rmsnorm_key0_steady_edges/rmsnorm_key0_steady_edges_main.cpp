// ACL host for the CANN RMSNorm key-0 warmed repeated-row edge probe.
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

extern "C" uint32_t aclrtlaunch_rmsnorm_key0_steady_edges(
    uint32_t block_dim, void* stream, void* source, void* destination,
    void* output, int64_t mode, int64_t payload_bytes, int64_t pair_order);

namespace {

constexpr int64_t kSteadyRequests = 8;
constexpr int64_t kWorkerStrideBytes = 131072;
constexpr size_t kOutputValuesPerBlock = 8;
constexpr double kSystemCounterTickNs = 20.0;

struct Cell {
    int64_t block_dim;
    int64_t mode;
    int64_t payload_bytes;
    int64_t pair_order;
};

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

Cell ParseCell(const char* text) {
    const std::string value(text);
    const size_t first = value.find(':');
    const size_t second = value.find(':', first + 1);
    const size_t third = value.find(':', second + 1);
    if (first == std::string::npos || second == std::string::npos
            || third == std::string::npos
            || value.find(':', third + 1) != std::string::npos) {
        std::fprintf(
            stderr,
            "cell must have block_dim:mode:payload_bytes:pair_order form\n");
        std::exit(EXIT_FAILURE);
    }
    const Cell cell{
        ParseNonnegative(value.substr(0, first).c_str(), "block_dim"),
        ParseNonnegative(
            value.substr(first + 1, second - first - 1).c_str(), "mode"),
        ParseNonnegative(
            value.substr(second + 1, third - second - 1).c_str(),
            "payload_bytes"),
        ParseNonnegative(value.substr(third + 1).c_str(), "pair_order")};
    if ((cell.block_dim != 32 && cell.block_dim != 43
            && cell.block_dim != 47)
            || (cell.mode != 0 && cell.mode != 2)
            || (cell.payload_bytes != 4096
                && cell.payload_bytes != 5120
                && cell.payload_bytes != 10240)
            || cell.pair_order < 0 || cell.pair_order > 1) {
        std::fprintf(stderr, "unsupported probe cell\n");
        std::exit(EXIT_FAILURE);
    }
    return cell;
}

void RunCell(aclrtStream stream, Cell cell) {
    const size_t buffer_bytes = cell.block_dim * kWorkerStrideBytes;
    std::vector<uint16_t> source(buffer_bytes / sizeof(uint16_t), 0x3c00);
    std::vector<uint16_t> destination(buffer_bytes / sizeof(uint16_t), 0);
    void* device_source = nullptr;
    void* device_destination = nullptr;
    void* device_output = nullptr;
    const size_t output_values = cell.block_dim * kOutputValuesPerBlock;
    const size_t output_bytes = output_values * sizeof(int64_t);
    ACL_CHECK(aclrtMalloc(
        &device_source, buffer_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_destination, buffer_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_source, buffer_bytes, source.data(), buffer_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(
        device_destination, buffer_bytes, 0, buffer_bytes));
    ACL_CHECK(aclrtlaunch_rmsnorm_key0_steady_edges(
        static_cast<uint32_t>(cell.block_dim), stream, device_source,
        device_destination, device_output, cell.mode, cell.payload_bytes,
        cell.pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> output(output_values, -1);
    ACL_CHECK(aclrtMemcpy(
        output.data(), output_bytes, device_output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (cell.mode >= 2) {
        ACL_CHECK(aclrtMemcpy(
            destination.data(), buffer_bytes, device_destination,
            buffer_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    const int64_t request_count = (
        cell.mode == 0 || cell.mode == 2 ? kSteadyRequests : 1);
    for (int64_t worker = 0; worker < cell.block_dim; ++worker) {
        const size_t offset = worker * kOutputValuesPerBlock;
        const int64_t control_cycles = output[offset];
        const int64_t target_cycles = output[offset + 1];
        const int64_t sink = output[offset + 2];
        bool valid = control_cycles >= 0 && target_cycles > 0;
        if (cell.mode < 2) {
            valid = valid && sink == request_count + 1;
        } else {
            const size_t destination_offset = (
                worker * kWorkerStrideBytes / sizeof(uint16_t)
                + cell.payload_bytes / sizeof(uint16_t));
            valid = valid && sink == request_count
                && destination[destination_offset] == 0x3c00;
        }
        if (!valid) {
            std::fprintf(
                stderr,
                "validation failed worker=%lld block_dim=%lld mode=%lld "
                "pair_order=%lld control=%lld target=%lld sink=%lld\n",
                static_cast<long long>(worker),
                static_cast<long long>(cell.block_dim),
                static_cast<long long>(cell.mode),
                static_cast<long long>(cell.pair_order),
                static_cast<long long>(control_cycles),
                static_cast<long long>(target_cycles),
                static_cast<long long>(sink));
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.9f,%.9f,%lld,%lld,pass\n",
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(worker),
            static_cast<long long>(cell.mode),
            static_cast<long long>(cell.payload_bytes),
            static_cast<long long>(request_count),
            static_cast<long long>(cell.pair_order),
            static_cast<long long>(kWorkerStrideBytes),
            static_cast<long long>(control_cycles),
            static_cast<long long>(target_cycles),
            static_cast<double>(control_cycles) * kSystemCounterTickNs,
            static_cast<double>(target_cycles) * kSystemCounterTickNs,
            static_cast<long long>(sink),
            static_cast<long long>(
                cell.mode >= 2 ? destination[
                    worker * kWorkerStrideBytes / sizeof(uint16_t)
                    + cell.payload_bytes / sizeof(uint16_t)] : 0));
    }
    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtFree(device_destination));
    ACL_CHECK(aclrtFree(device_output));
}

}  // namespace

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(
            stderr,
            "usage: %s --batch block_dim:mode:payload_bytes:pair_order...\n",
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
