// Host driver for the CANN AIV InitTilingData mechanism benchmark.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
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

extern "C" uint32_t aclrtlaunch_tiling_data_materialization(
    uint32_t block_dim, void* stream, void* tiling, void* cycles,
    void* validation, int64_t payload_bytes, int64_t phase_repetitions,
    int64_t iterations);

namespace {

constexpr int64_t kMaximumBlockDim = 48;
constexpr int64_t kCycleValuesPerBlock = 8;
constexpr int64_t kValidationValuesPerBlock = 16;
constexpr int64_t kMaximumTilingBytes = 1872;
constexpr int64_t kTilingValues =
    (kMaximumTilingBytes + static_cast<int64_t>(sizeof(uint32_t)) - 1)
    / static_cast<int64_t>(sizeof(uint32_t));
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kRepeatedBatchWarmupIterations = 262144;

int64_t ParsePositive(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "%s must be a positive integer, got %s\n",
                     name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int64_t ParseNonnegative(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        std::fprintf(stderr, "%s must be a non-negative integer, got %s\n",
                     name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

std::string Join(const std::vector<int64_t>& values) {
    std::string output;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output += ":";
        }
        output += std::to_string(values[index]);
    }
    return output;
}

std::vector<int64_t> Run(
        aclrtStream stream, void* tiling, void* cycles, void* validation,
        int64_t payload_bytes, uint32_t expected_validation,
        int64_t phase_repetitions, int64_t block_dim, int64_t iterations) {
    const size_t cycle_bytes = static_cast<size_t>(
        kMaximumBlockDim * kCycleValuesPerBlock * sizeof(int64_t));
    const size_t validation_bytes = static_cast<size_t>(
        kMaximumBlockDim * kValidationValuesPerBlock * sizeof(uint32_t));
    ACL_CHECK(aclrtMemset(cycles, cycle_bytes, 0xff, cycle_bytes));
    ACL_CHECK(aclrtMemset(validation, validation_bytes, 0, validation_bytes));
    ACL_CHECK(aclrtlaunch_tiling_data_materialization(
        static_cast<uint32_t>(block_dim), stream, tiling, cycles,
        validation, payload_bytes, phase_repetitions, iterations));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> padded_cycles(
        static_cast<size_t>(block_dim) * kCycleValuesPerBlock);
    std::vector<uint32_t> observed(
        static_cast<size_t>(block_dim) * kValidationValuesPerBlock);
    ACL_CHECK(aclrtMemcpy(
        padded_cycles.data(), padded_cycles.size() * sizeof(int64_t),
        cycles, padded_cycles.size() * sizeof(int64_t),
        ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(
        observed.data(), observed.size() * sizeof(uint32_t),
        validation, observed.size() * sizeof(uint32_t),
        ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<int64_t> result(static_cast<size_t>(block_dim));
    for (int64_t block = 0; block < block_dim; ++block) {
        result[static_cast<size_t>(block)] = padded_cycles[
            static_cast<size_t>(block) * kCycleValuesPerBlock];
        const uint32_t value = observed[
            static_cast<size_t>(block) * kValidationValuesPerBlock];
        if (result[static_cast<size_t>(block)] <= 0
                || value != expected_validation) {
            std::fprintf(
                stderr,
                "block %ld validation failed: cycles=%ld value=%u\n",
                static_cast<long>(block),
                static_cast<long>(result[static_cast<size_t>(block)]), value);
            std::exit(EXIT_FAILURE);
        }
    }
    return result;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclrtSetDevice(static_cast<int>(
        ParseNonnegative(text, "ASCEND_DEVICE_ID"))));
}

int main(int argc, char** argv) {
    const bool repeated_batch = (
        argc >= 2 && std::string(argv[1]) == "--repeated-batch");
    if ((repeated_batch && argc < 6)
            || (!repeated_batch
                && (argc < 5 || std::string(argv[1]) != "--batch"))) {
        std::fprintf(
            stderr,
            "usage: %s --batch <payload_bytes> <iterations> "
            "<block_dim>... | --repeated-batch <payload_bytes> "
            "<phase_repetitions> <iterations> <block_dim>...\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t payload_bytes = ParsePositive(argv[2], "payload_bytes");
    if (payload_bytes != 28 && payload_bytes != 48 && payload_bytes != 56
            && payload_bytes != 80 && payload_bytes != 192
            && payload_bytes != 1872) {
        std::fprintf(
            stderr,
            "payload_bytes must be 28, 48, 56, 80, 192, or 1872\n");
        return EXIT_FAILURE;
    }
    if (repeated_batch && payload_bytes != 80) {
        std::fprintf(
            stderr,
            "repeated-batch is reserved for the 80-byte SwiGLU source "
            "prologue\n");
        return EXIT_FAILURE;
    }
    const int64_t phase_repetitions = (
        repeated_batch
        ? ParsePositive(argv[3], "phase_repetitions")
        : 1);
    const int64_t iterations = ParsePositive(
        argv[repeated_batch ? 4 : 3], "iterations");
    std::vector<int64_t> block_dims;
    for (int index = repeated_batch ? 5 : 4; index < argc; ++index) {
        const int64_t block_dim = ParsePositive(argv[index], "block_dim");
        if (block_dim > kMaximumBlockDim) {
            std::fprintf(stderr, "block_dim must be in 1..48\n");
            return EXIT_FAILURE;
        }
        block_dims.push_back(block_dim);
    }

    uint32_t host_tiling[kTilingValues] = {};
    const uint32_t prefix[] = {
        1, 4, 128, 128, 1024, 48, 694, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
    std::copy(
        std::begin(prefix), std::end(prefix), std::begin(host_tiling));
    uint32_t expected_validation = 0;
    for (int64_t index = 0;
            index < payload_bytes / static_cast<int64_t>(sizeof(uint32_t));
            ++index) {
        expected_validation += host_tiling[index];
    }
    void* device_tiling = nullptr;
    void* device_cycles = nullptr;
    void* device_validation = nullptr;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(
        &device_tiling, kMaximumTilingBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles,
        kMaximumBlockDim * kCycleValuesPerBlock * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_validation,
        kMaximumBlockDim * kValidationValuesPerBlock * sizeof(uint32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_tiling, kMaximumTilingBytes, host_tiling, sizeof(host_tiling),
        ACL_MEMCPY_HOST_TO_DEVICE));

    if (repeated_batch) {
        // The two-phase campaign validates a source composition rule rather
        // than cold device power-state behavior.  Precondition the same
        // source-identical chain once, then discard it before the balanced
        // formal schedule.  No production operator work is executed.
        Run(stream, device_tiling, device_cycles, device_validation,
            payload_bytes, expected_validation, phase_repetitions,
            kMaximumBlockDim, kRepeatedBatchWarmupIterations);
    }

    std::printf(
        "payload_bytes,block_dim,iterations,short_cycles,long_cycles,"
        "tick_ns,service_time_ns,validation\n");
    for (int64_t block_dim : block_dims) {
        Run(stream, device_tiling, device_cycles, device_validation,
            payload_bytes, expected_validation, phase_repetitions,
            block_dim, 16);
        const std::vector<int64_t> short_cycles = Run(
            stream, device_tiling, device_cycles, device_validation,
            payload_bytes, expected_validation, phase_repetitions,
            block_dim, iterations);
        const std::vector<int64_t> long_cycles = Run(
            stream, device_tiling, device_cycles, device_validation,
            payload_bytes, expected_validation, phase_repetitions,
            block_dim, 4 * iterations);
        const int64_t short_critical = *std::max_element(
            short_cycles.begin(), short_cycles.end());
        const int64_t long_critical = *std::max_element(
            long_cycles.begin(), long_cycles.end());
        if (long_critical <= short_critical) {
            std::fprintf(stderr, "long interval must exceed short interval\n");
            return EXIT_FAILURE;
        }
        const double service_ns = static_cast<double>(
            long_critical - short_critical) * kSystemCounterTickNs
            / static_cast<double>(3 * iterations);
        if (!std::isfinite(service_ns) || service_ns <= 0.0) {
            std::fprintf(stderr, "derived service time is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%ld,%ld,%ld,%s,%s,%.1f,%.12f,pass\n",
            static_cast<long>(payload_bytes),
            static_cast<long>(block_dim), static_cast<long>(iterations),
            Join(short_cycles).c_str(), Join(long_cycles).c_str(),
            kSystemCounterTickNs, service_ns);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtFree(device_validation));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_tiling));
    ACL_CHECK(aclrtDestroyStream(stream));
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    ACL_CHECK(aclrtResetDevice(
        static_cast<int>(
            ParseNonnegative(device_text, "ASCEND_DEVICE_ID"))));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
