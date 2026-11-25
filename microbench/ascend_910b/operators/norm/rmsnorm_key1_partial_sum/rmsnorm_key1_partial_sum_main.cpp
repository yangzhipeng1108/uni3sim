// Host driver for the CANN 8.5.1 RMSNorm key-1 partial-sum handoff probe.
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

extern "C" uint32_t aclrtlaunch_rmsnorm_key1_partial_sum(
    uint32_t block_dim, void* stream, void* output,
    int64_t partial_count, int64_t pair_order, int64_t iterations);

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

namespace {

constexpr size_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumBlockDim = 48;
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
    int64_t partial_count;
    int64_t pair_order;
    int64_t iterations;
};

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
            "cell must have block_dim:partial_count:pair_order:iterations form\n");
        std::exit(EXIT_FAILURE);
    }
    const Cell cell{
        ParseNonnegative(value.substr(0, first).c_str(), "block_dim"),
        ParseNonnegative(
            value.substr(first + 1, second - first - 1).c_str(),
            "partial_count"),
        ParseNonnegative(
            value.substr(second + 1, third - second - 1).c_str(),
            "pair_order"),
        ParseNonnegative(
            value.substr(third + 1).c_str(), "iterations")};
    if (cell.block_dim <= 0 || cell.block_dim > kMaximumBlockDim) {
        std::fprintf(stderr, "block_dim must be in [1,48]\n");
        std::exit(EXIT_FAILURE);
    }
    if (cell.partial_count != 1 && cell.partial_count != 8
            && cell.partial_count != 32
            && cell.partial_count != 64) {
        std::fprintf(stderr, "partial_count must be 1, 8, 32, or 64\n");
        std::exit(EXIT_FAILURE);
    }
    if (cell.pair_order < 0 || cell.pair_order > 1) {
        std::fprintf(stderr, "pair_order must be zero or one\n");
        std::exit(EXIT_FAILURE);
    }
    if (cell.iterations <= 0 || cell.iterations % 2 != 0) {
        std::fprintf(stderr, "iterations must be positive and even\n");
        std::exit(EXIT_FAILURE);
    }
    return cell;
}

void RunCell(aclrtStream stream, Cell cell) {
    const size_t output_values = (
        static_cast<size_t>(cell.block_dim) * kOutputValuesPerBlock);
    const size_t output_bytes = output_values * sizeof(int64_t);
    void* output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(output, output_bytes, 0, output_bytes));
    ACL_CHECK(aclrtlaunch_rmsnorm_key1_partial_sum(
        static_cast<uint32_t>(cell.block_dim), stream, output,
        cell.partial_count, cell.pair_order, cell.iterations));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> host_output(output_values, -1);
    ACL_CHECK(aclrtMemcpy(
        host_output.data(), output_bytes, output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    const int64_t expected_sink = 4 * cell.partial_count;
    for (int64_t worker = 0; worker < cell.block_dim; ++worker) {
        const size_t offset = (
            static_cast<size_t>(worker) * kOutputValuesPerBlock);
        const int64_t control_cycles = host_output[offset];
        const int64_t target_cycles = host_output[offset + 1];
        const int64_t sink = host_output[offset + 2];
        if (control_cycles <= 0 || target_cycles <= 0
                || sink != expected_sink) {
            std::fprintf(
                stderr,
                "validation failed worker=%lld block_dim=%lld partials=%lld "
                "order=%lld iterations=%lld control=%lld target=%lld sink=%lld\n",
                static_cast<long long>(worker),
                static_cast<long long>(cell.block_dim),
                static_cast<long long>(cell.partial_count),
                static_cast<long long>(cell.pair_order),
                static_cast<long long>(cell.iterations),
                static_cast<long long>(control_cycles),
                static_cast<long long>(target_cycles),
                static_cast<long long>(sink));
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,32,%lld,%lld,%lld,%lld,%.9f,%.9f,%lld,pass\n",
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(worker),
            static_cast<long long>(cell.partial_count),
            static_cast<long long>(cell.pair_order),
            static_cast<long long>(cell.partial_count * 32),
            static_cast<long long>(cell.iterations),
            static_cast<long long>(control_cycles),
            static_cast<long long>(target_cycles),
            static_cast<double>(control_cycles) * kSystemCounterTickNs,
            static_cast<double>(target_cycles) * kSystemCounterTickNs,
            static_cast<long long>(sink));
    }
    ACL_CHECK(aclrtFree(output));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(
            stderr,
            "usage: %s --batch block_dim:partial_count:pair_order:iterations...\n",
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
