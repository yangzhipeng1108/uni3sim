// Host driver for the amplified single-TBuf lifecycle probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

extern "C" uint32_t aclrtlaunch_single_tbuf_lifecycle(
    uint32_t blockDim, void* stream, void* output,
    int64_t payload_bytes, int64_t iterations);

namespace {

constexpr int64_t kMaximumBlockDim = 48;
constexpr int64_t kOutputValuesPerBlock = 8;
constexpr double kSystemCounterTickNs = 20.0;

struct Cell {
    int64_t block_dim;
    int64_t payload_bytes;
    int64_t iterations;
};

int64_t ParsePositive(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

Cell ParseCell(const char* text) {
    const std::string value(text);
    const size_t first = value.find(':');
    const size_t second = value.find(':', first + 1);
    if (first == std::string::npos || second == std::string::npos
            || value.find(':', second + 1) != std::string::npos) {
        std::fprintf(stderr, "cell must have block_dim:payload_bytes:iterations form\n");
        std::exit(EXIT_FAILURE);
    }
    const Cell cell{
        ParsePositive(value.substr(0, first).c_str(), "block_dim"),
        ParsePositive(
            value.substr(first + 1, second - first - 1).c_str(),
            "payload_bytes"),
        ParsePositive(value.substr(second + 1).c_str(), "iterations"),
    };
    const bool valid_block_dim = (
        cell.block_dim == 1 || cell.block_dim == 2 || cell.block_dim == 4
        || cell.block_dim == 8 || cell.block_dim == 16
        || cell.block_dim == 48);
    if (!valid_block_dim || (cell.payload_bytes != 1024
            && cell.payload_bytes != 2048
            && cell.payload_bytes != 110080) || cell.iterations != 64) {
        std::fprintf(stderr, "cell is outside the frozen source domain\n");
        std::exit(EXIT_FAILURE);
    }
    return cell;
}

std::string JoinCycles(const std::vector<int64_t>& cycles) {
    std::string result;
    for (size_t index = 0; index < cycles.size(); ++index) {
        if (index > 0) {
            result += "|";
        }
        result += std::to_string(cycles[index]);
    }
    return result;
}

std::vector<int64_t> RunCell(
        aclrtStream stream, void* device_output, const Cell& cell,
        int64_t iterations) {
    ACL_CHECK(aclrtMemset(
        device_output,
        kMaximumBlockDim * kOutputValuesPerBlock * sizeof(int64_t),
        0xff,
        kMaximumBlockDim * kOutputValuesPerBlock * sizeof(int64_t)));
    ACL_CHECK(aclrtlaunch_single_tbuf_lifecycle(
        static_cast<uint32_t>(cell.block_dim), stream, device_output,
        cell.payload_bytes, iterations));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<int64_t> padded(
        static_cast<size_t>(cell.block_dim) * kOutputValuesPerBlock);
    ACL_CHECK(aclrtMemcpy(
        padded.data(), padded.size() * sizeof(int64_t), device_output,
        padded.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<int64_t> cycles(static_cast<size_t>(cell.block_dim));
    for (int64_t block = 0; block < cell.block_dim; ++block) {
        const int64_t value = padded[
            static_cast<size_t>(block) * kOutputValuesPerBlock];
        if (value <= 0) {
            std::fprintf(stderr, "worker %lld produced invalid cycles\n",
                         static_cast<long long>(block));
            std::exit(EXIT_FAILURE);
        }
        cycles[static_cast<size_t>(block)] = value;
    }
    return cycles;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(stderr, "usage: %s --batch block_dim:payload_bytes:iterations...\n", argv[0]);
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_output,
        kMaximumBlockDim * kOutputValuesPerBlock * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    for (int index = 2; index < argc; ++index) {
        const Cell cell = ParseCell(argv[index]);
        RunCell(stream, device_output, cell, 16);
        const std::vector<int64_t> short_cycles = RunCell(
            stream, device_output, cell, cell.iterations);
        const std::vector<int64_t> long_cycles = RunCell(
            stream, device_output, cell, 4 * cell.iterations);
        for (size_t block = 0; block < short_cycles.size(); ++block) {
            if (long_cycles[block] <= short_cycles[block]) {
                std::fprintf(stderr, "worker %zu has non-increasing cycles\n", block);
                return EXIT_FAILURE;
            }
        }
        const int64_t short_critical = *std::max_element(
            short_cycles.begin(), short_cycles.end());
        const int64_t long_critical = *std::max_element(
            long_cycles.begin(), long_cycles.end());
        const double service_ns = (
            static_cast<double>(long_critical - short_critical)
            * kSystemCounterTickNs / (3.0 * cell.iterations));
        if (!std::isfinite(service_ns) || service_ns <= 0.0) {
            std::fprintf(stderr, "derived lifecycle service is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%lld,%lld,%lld,%s,%s,%.9f,%.9f,pass\n",
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(cell.payload_bytes),
            static_cast<long long>(cell.iterations),
            JoinCycles(short_cycles).c_str(), JoinCycles(long_cycles).c_str(),
            kSystemCounterTickNs, service_ns);
        std::fflush(stdout);
    }
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
