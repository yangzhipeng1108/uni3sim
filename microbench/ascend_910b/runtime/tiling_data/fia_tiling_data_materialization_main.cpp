// ACL host for the independent mixed-core FIA tiling-data benchmark.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

extern "C" uint32_t aclrtlaunch_fia_tiling_data_materialization(
    uint32_t block_dim, void* stream, void* tiling, void* cycles,
    void* validation, int64_t iterations);

namespace {

constexpr int64_t kMaximumAicBlocks = 24;
constexpr int64_t kAivPerAic = 2;
constexpr int64_t kActorsPerGroup = 3;
constexpr int64_t kMaximumActors = kMaximumAicBlocks * kActorsPerGroup;
constexpr int64_t kCycleSlotElements = 8;
constexpr int64_t kValidationSlotElements = 16;
constexpr int64_t kPayloadWords = 390;
constexpr int64_t kPayloadTransferBytes = 1568;
constexpr double kSystemCounterTickNs = 20.0;
constexpr uint32_t kExpectedValidation = 586;

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
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

struct DomainCycles {
    std::vector<int64_t> aic;
    std::vector<int64_t> aiv;
};

DomainCycles Run(
        aclrtStream stream, void* tiling, void* cycles, void* validation,
        int64_t aic_block_dim, int64_t iterations) {
    const size_t cycle_bytes = static_cast<size_t>(
        kMaximumActors * kCycleSlotElements * sizeof(int64_t));
    const size_t validation_bytes = static_cast<size_t>(
        kMaximumActors * kValidationSlotElements * sizeof(uint32_t));
    ACL_CHECK(aclrtMemset(cycles, cycle_bytes, 0xff, cycle_bytes));
    ACL_CHECK(aclrtMemset(validation, validation_bytes, 0, validation_bytes));
    ACL_CHECK(aclrtlaunch_fia_tiling_data_materialization(
        static_cast<uint32_t>(aic_block_dim), stream, tiling, cycles,
        validation, iterations));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> padded_cycles(
        static_cast<size_t>(kMaximumActors * kCycleSlotElements), -1);
    std::vector<uint32_t> observed(
        static_cast<size_t>(kMaximumActors * kValidationSlotElements), 0);
    ACL_CHECK(aclrtMemcpy(
        padded_cycles.data(), cycle_bytes, cycles, cycle_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(
        observed.data(), validation_bytes, validation, validation_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    DomainCycles result;
    result.aic.reserve(static_cast<size_t>(aic_block_dim));
    result.aiv.reserve(static_cast<size_t>(aic_block_dim * kAivPerAic));
    for (int64_t group = 0; group < aic_block_dim; ++group) {
        for (int64_t actor_in_group = 0;
             actor_in_group < kActorsPerGroup; ++actor_in_group) {
            const int64_t actor = group * kActorsPerGroup + actor_in_group;
            const int64_t cycle = padded_cycles[static_cast<size_t>(
                actor * kCycleSlotElements)];
            const uint32_t value = observed[static_cast<size_t>(
                actor * kValidationSlotElements)];
            if (cycle <= 0 || value != kExpectedValidation) {
                std::fprintf(
                    stderr,
                    "actor %ld validation failed: cycles=%ld value=%u\n",
                    static_cast<long>(actor), static_cast<long>(cycle), value);
                std::exit(EXIT_FAILURE);
            }
            if (actor_in_group == 0) {
                result.aic.push_back(cycle);
            } else {
                result.aiv.push_back(cycle);
            }
        }
    }
    return result;
}

double DifferentialNs(
        const std::vector<int64_t>& short_cycles,
        const std::vector<int64_t>& long_cycles, int64_t iterations) {
    const int64_t short_critical = *std::max_element(
        short_cycles.begin(), short_cycles.end());
    const int64_t long_critical = *std::max_element(
        long_cycles.begin(), long_cycles.end());
    if (long_critical <= short_critical) {
        std::fprintf(stderr, "long interval must exceed short interval\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<double>(long_critical - short_critical)
        * kSystemCounterTickNs / static_cast<double>(3 * iterations);
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(static_cast<int>(
        ParseInteger(device_text, "ASCEND_DEVICE_ID"))));
}

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 2 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch iterations aic_block_dim [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    struct Cell {
        int64_t iterations;
        int64_t aic_block_dim;
    };
    std::vector<Cell> cells;
    for (int argument = 2; argument < argc; argument += 2) {
        const Cell cell{
            ParseInteger(argv[argument], "iterations"),
            ParseInteger(argv[argument + 1], "aic_block_dim"),
        };
        if (cell.iterations <= 0 || cell.aic_block_dim <= 0
                || cell.aic_block_dim > kMaximumAicBlocks) {
            std::fprintf(stderr, "invalid FIA tiling cell\n");
            return EXIT_FAILURE;
        }
        cells.push_back(cell);
    }

    std::vector<uint32_t> host_tiling(kPayloadWords);
    for (int64_t index = 0; index < kPayloadWords; ++index) {
        host_tiling[static_cast<size_t>(index)] =
            static_cast<uint32_t>(index + 1);
    }
    void* device_tiling = nullptr;
    void* device_cycles = nullptr;
    void* device_validation = nullptr;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(
        &device_tiling, kPayloadTransferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles,
        kMaximumActors * kCycleSlotElements * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_validation,
        kMaximumActors * kValidationSlotElements * sizeof(uint32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(
        device_tiling, kPayloadTransferBytes, 0, kPayloadTransferBytes));
    ACL_CHECK(aclrtMemcpy(
        device_tiling, kPayloadTransferBytes,
        host_tiling.data(), host_tiling.size() * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE));

    std::printf(
        "payload_bytes,aic_block_dim,aiv_block_dim,iterations,"
        "short_aic_cycles,long_aic_cycles,short_aiv_cycles,long_aiv_cycles,"
        "tick_ns,aic_service_ns,aiv_service_ns,critical_service_ns,validation\n");
    for (const Cell& cell : cells) {
        Run(stream, device_tiling, device_cycles, device_validation,
            cell.aic_block_dim, 16);
        const DomainCycles short_cycles = Run(
            stream, device_tiling, device_cycles, device_validation,
            cell.aic_block_dim, cell.iterations);
        const DomainCycles long_cycles = Run(
            stream, device_tiling, device_cycles, device_validation,
            cell.aic_block_dim, 4 * cell.iterations);
        const double aic_ns = DifferentialNs(
            short_cycles.aic, long_cycles.aic, cell.iterations);
        const double aiv_ns = DifferentialNs(
            short_cycles.aiv, long_cycles.aiv, cell.iterations);
        const double critical_ns = std::max(aic_ns, aiv_ns);
        if (!std::isfinite(critical_ns) || critical_ns <= 0.0) {
            std::fprintf(stderr, "derived service time is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf(
            "1560,%ld,%ld,%ld,%s,%s,%s,%s,%.1f,%.12f,%.12f,%.12f,pass\n",
            static_cast<long>(cell.aic_block_dim),
            static_cast<long>(cell.aic_block_dim * kAivPerAic),
            static_cast<long>(cell.iterations),
            Join(short_cycles.aic).c_str(), Join(long_cycles.aic).c_str(),
            Join(short_cycles.aiv).c_str(), Join(long_cycles.aiv).c_str(),
            kSystemCounterTickNs, aic_ns, aiv_ns, critical_ns);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtFree(device_validation));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_tiling));
    ACL_CHECK(aclrtDestroyStream(stream));
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    ACL_CHECK(aclrtResetDevice(static_cast<int>(
        ParseInteger(device_text, "ASCEND_DEVICE_ID"))));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
