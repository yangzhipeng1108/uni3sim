// ACL host for the boundary-preserving BF16 elementwise Mul compute probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "elementwise_mul_compute_segments_contract.h"

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                         \
    }                                                                    \
} while (0)

extern "C" uint32_t aclrtlaunch_elementwise_mul_compute_segments(
    uint32_t block_dim, void* stream, void* worker_results, int64_t segment,
    int64_t elements, int64_t iterations);

namespace {

using namespace elementwise_mul_compute_segments_contract;

struct Cell {
    int64_t workers;
    int64_t elements;
    int64_t segment;
};

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr || *text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < 0 || value > 7) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be in [0, 7]\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int64_t ParseInteger(const std::string& text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0') {
        std::fprintf(stderr, "invalid %s: %s\n", name, text.c_str());
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

Cell ParseCell(const char* text) {
    const std::string value(text);
    const size_t first = value.find(':');
    const size_t second = value.find(':', first == std::string::npos
        ? first : first + 1);
    if (first == std::string::npos || second == std::string::npos
            || value.find(':', second + 1) != std::string::npos) {
        std::fprintf(stderr, "cell must have workers:elements:segment form\n");
        std::exit(EXIT_FAILURE);
    }
    const Cell cell{
        ParseInteger(value.substr(0, first), "workers"),
        ParseInteger(value.substr(first + 1, second - first - 1), "elements"),
        ParseInteger(value.substr(second + 1), "segment"),
    };
    if ((cell.workers != 46 && cell.workers != 47 && cell.workers != 48)
            || cell.elements <= 0 || cell.elements > kMaximumElements
            || (cell.segment != kPrefixCastA
                && cell.segment != kSuffixCastBMulCastOut)) {
        std::fprintf(stderr, "unsupported cell: %s\n", text);
        std::exit(EXIT_FAILURE);
    }
    return cell;
}

int64_t RunInterval(
        aclrtStream stream, void* results, Cell cell, int64_t iterations) {
    const size_t cycle_bytes = static_cast<size_t>(
        kMaximumWorkers * kResultValuesPerWorker) * sizeof(int64_t);
    ACL_CHECK(aclrtMemset(results, cycle_bytes, 0xff, cycle_bytes));
    ACL_CHECK(aclrtlaunch_elementwise_mul_compute_segments(
        static_cast<uint32_t>(cell.workers), stream, results, cell.segment,
        cell.elements, iterations));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<int64_t> host_results(
        kMaximumWorkers * kResultValuesPerWorker, -1);
    ACL_CHECK(aclrtMemcpy(
        host_results.data(), cycle_bytes, results, cycle_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t maximum_cycles = -1;
    const int64_t expected_sink = (
        cell.segment == kPrefixCastA ? 1500 : 2250);
    for (int64_t worker = 0; worker < cell.workers; ++worker) {
        const size_t offset = worker * kResultValuesPerWorker;
        const int64_t elapsed = host_results[offset];
        const int64_t sink = host_results[offset + 1];
        if (elapsed < 0 || sink != expected_sink) {
            std::fprintf(
                stderr,
                "worker %ld validation failed: cycles=%ld sink=%ld "
                "expected=%ld\n",
                static_cast<long>(worker), static_cast<long>(elapsed),
                static_cast<long>(sink), static_cast<long>(expected_sink));
            std::exit(EXIT_FAILURE);
        }
        maximum_cycles = std::max(maximum_cycles, elapsed);
    }
    if (maximum_cycles <= 0) {
        std::fprintf(stderr, "all active workers emitted zero cycles\n");
        std::exit(EXIT_FAILURE);
    }
    return maximum_cycles;
}

void RunCell(aclrtStream stream, void* results, Cell cell) {
    for (int warmup = 0; warmup < 3; ++warmup) {
        RunInterval(stream, results, cell, 1);
    }
    const int64_t cycles = RunInterval(stream, results, cell, 1);
    const double service_ns = static_cast<double>(cycles) * kSystemCycleNs;
    if (!std::isfinite(service_ns) || service_ns <= 0.0) {
        std::fprintf(stderr, "invalid service interval\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf(
        "%ld,%ld,%ld,%ld,%.9f,pass\n",
        static_cast<long>(cell.workers), static_cast<long>(cell.elements),
        static_cast<long>(cell.segment), static_cast<long>(cycles),
        service_ns);
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(stderr, "usage: %s --batch workers:elements:segment...\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    void* results = nullptr;
    aclrtStream stream = nullptr;
    const size_t cycle_bytes = static_cast<size_t>(
        kMaximumWorkers * kResultValuesPerWorker) * sizeof(int64_t);
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(&results, cycle_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::printf(
        "workers,elements,segment,cycles,service_ns,validation\n");
    for (int index = 2; index < argc; ++index) {
        RunCell(stream, results, ParseCell(argv[index]));
    }
    ACL_CHECK(aclrtFree(results));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
