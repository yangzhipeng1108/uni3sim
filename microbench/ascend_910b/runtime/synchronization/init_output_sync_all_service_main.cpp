// Host driver for the independent Ascend 910B FIA output-init phase probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_init_output_sync_all_service(
    uint32_t block_dim, void* stream, void* output, void* cycles,
    int64_t elements_per_aiv);

namespace {

constexpr int32_t kLogicalDeviceId = 0;
constexpr int64_t kMaximumAivBlocks = 48;
constexpr int64_t kAivBlocksPerAicBlock = 2;
constexpr int64_t kCycleCacheLineBytes = 64;
constexpr int64_t kCycleSlotElements =
    kCycleCacheLineBytes / static_cast<int64_t>(sizeof(int64_t));
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kTimedLaunches = 32;

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

struct Cell {
    int64_t aiv_block_dim;
    int64_t elements_per_aiv;
};

struct Measurement {
    std::vector<int64_t> cycles;
    double device_elapsed_ns;
};

std::string JoinCycles(const std::vector<int64_t>& values) {
    std::string result;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            result += ';';
        }
        result += std::to_string(values[index]);
    }
    return result;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be set explicitly\n");
        std::exit(EXIT_FAILURE);
    }
    if (ParseNonnegative(device_text, "ASCEND_DEVICE_ID") != 0) {
        std::fprintf(stderr,
                     "ASCEND_DEVICE_ID must be logical device 0 after "
                     "visibility remapping\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 2 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch aiv_block_dim elements_per_aiv "
            "[cell ...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    int64_t maximum_elements = 0;
    for (int argument = 2; argument < argc; argument += 2) {
        const int64_t aiv_block_dim = ParsePositive(
            argv[argument], "aiv_block_dim");
        const int64_t elements_per_aiv = ParsePositive(
            argv[argument + 1], "elements_per_aiv");
        if (aiv_block_dim > kMaximumAivBlocks || aiv_block_dim % 2 != 0
                || elements_per_aiv % 16 != 0) {
            std::fprintf(
                stderr,
                "AIV block dim must be even/in [2,48], and elements must "
                "be a multiple of 16\n");
            return EXIT_FAILURE;
        }
        cells.push_back(Cell{aiv_block_dim, elements_per_aiv});
        maximum_elements = std::max(maximum_elements, elements_per_aiv);
    }

    const size_t output_bytes = static_cast<size_t>(
        kMaximumAivBlocks * maximum_elements * sizeof(uint16_t));
    const size_t cycle_bytes = static_cast<size_t>(
        kMaximumAivBlocks * kCycleCacheLineBytes);
    void* device_output = nullptr;
    void* device_cycles = nullptr;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles, cycle_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtlaunch_init_output_sync_all_service(
        static_cast<uint32_t>(
            kMaximumAivBlocks / kAivBlocksPerAicBlock),
        stream, device_output, device_cycles, maximum_elements));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::printf(
        "aiv_block_dim,elements_per_aiv,block_cycles,tick_ns,"
        "phase_time_ns,device_elapsed_ns,validation\n");
    for (const Cell& cell : cells) {
        auto run = [&]() {
            ACL_CHECK(aclrtMemset(
                device_cycles, cycle_bytes, 0xff, cycle_bytes));
            aclrtEvent start = nullptr;
            aclrtEvent end = nullptr;
            ACL_CHECK(aclrtCreateEventWithFlag(
                &start, ACL_EVENT_TIME_LINE));
            ACL_CHECK(aclrtCreateEventWithFlag(
                &end, ACL_EVENT_TIME_LINE));
            ACL_CHECK(aclrtRecordEvent(start, stream));
            for (int64_t launch = 0; launch < kTimedLaunches; ++launch) {
                ACL_CHECK(aclrtlaunch_init_output_sync_all_service(
                    static_cast<uint32_t>(
                        cell.aiv_block_dim / kAivBlocksPerAicBlock),
                    stream, device_output, device_cycles,
                    cell.elements_per_aiv));
            }
            ACL_CHECK(aclrtRecordEvent(end, stream));
            ACL_CHECK(aclrtSynchronizeEvent(end));
            float elapsed_ms = 0.0f;
            ACL_CHECK(aclrtEventElapsedTime(&elapsed_ms, start, end));
            ACL_CHECK(aclrtDestroyEvent(end));
            ACL_CHECK(aclrtDestroyEvent(start));
            if (!(elapsed_ms > 0.0f)) {
                std::fprintf(stderr, "device event interval was not positive\n");
                std::exit(EXIT_FAILURE);
            }
            std::vector<int64_t> padded_cycles(
                static_cast<size_t>(
                    kMaximumAivBlocks * kCycleSlotElements), -1);
            ACL_CHECK(aclrtMemcpy(
                padded_cycles.data(), cycle_bytes, device_cycles,
                cycle_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
            std::vector<int64_t> cycles;
            cycles.reserve(static_cast<size_t>(cell.aiv_block_dim));
            for (int64_t block = 0; block < cell.aiv_block_dim; ++block) {
                const int64_t value = padded_cycles[static_cast<size_t>(
                    block * kCycleSlotElements)];
                if (value <= 0) {
                    std::fprintf(stderr,
                        "invalid cycle %ld at block=%ld aivBlockDim=%ld\n",
                        static_cast<long>(value), static_cast<long>(block),
                        static_cast<long>(cell.aiv_block_dim));
                    std::exit(EXIT_FAILURE);
                }
                cycles.push_back(value);
            }
            return Measurement{
                std::move(cycles),
                static_cast<double>(elapsed_ms) * 1.0e6
                    / static_cast<double>(kTimedLaunches),
            };
        };
        const Measurement measurement = run();
        const int64_t critical_cycles = *std::max_element(
            measurement.cycles.begin(), measurement.cycles.end());
        const double phase_time_ns = static_cast<double>(critical_cycles)
            * kSystemCounterTickNs;
        if (!std::isfinite(phase_time_ns) || phase_time_ns <= 0.0
                || !std::isfinite(measurement.device_elapsed_ns)
                || measurement.device_elapsed_ns <= 0.0) {
            std::fprintf(stderr, "derived phase time is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "%ld,%ld,%s,%.1f,%.12f,%.12f,pass\n",
            static_cast<long>(cell.aiv_block_dim),
            static_cast<long>(cell.elements_per_aiv),
            JoinCycles(measurement.cycles).c_str(), kSystemCounterTickNs,
            phase_time_ns, measurement.device_elapsed_ns);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
