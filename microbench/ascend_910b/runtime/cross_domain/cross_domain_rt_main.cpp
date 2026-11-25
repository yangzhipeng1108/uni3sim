// Host driver for the independent 1:2 AIC/AIV synchronization benchmark.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "cross_domain_contract.h"

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

#define CLOCK_CHECK(call) do {                                            \
    if ((call) != 0) {                                                    \
        std::perror(#call);                                               \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_cross_domain_rt(
    uint32_t block_dim, void* stream, void* gm_cycles, void* gm_task_counts);

namespace {

constexpr int32_t kLogicalDeviceId = 0;

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

std::vector<int64_t> ParseTaskCounts(
        const char* text, int64_t expected_length) {
    const std::string source(text);
    std::vector<int64_t> values;
    size_t begin = 0;
    while (begin <= source.size()) {
        const size_t end = source.find(';', begin);
        const std::string token = source.substr(
            begin, end == std::string::npos ? end : end - begin);
        if (token.empty()) {
            std::fprintf(stderr, "taskCounts contains an empty value\n");
            std::exit(EXIT_FAILURE);
        }
        const int64_t value = ParseInteger(token.c_str(), "taskCounts");
        if (!cross_domain_contract::IsFrozenTaskCount(value)) {
            std::fprintf(stderr, "taskCounts contains an unfrozen value\n");
            std::exit(EXIT_FAILURE);
        }
        values.push_back(value);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (static_cast<int64_t>(values.size()) != expected_length) {
        std::fprintf(stderr, "taskCounts length disagrees with blockDim\n");
        std::exit(EXIT_FAILURE);
    }
    return values;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be set explicitly\n");
        std::exit(EXIT_FAILURE);
    }
    if (ParseInteger(device_text, "ASCEND_DEVICE_ID") != kLogicalDeviceId) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must be logical device 0 after visibility "
            "remapping\n");
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
            "usage: %s --batch aicBlockDim taskCounts [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int cell_count = (argc - 2) / 2;
    if (cell_count > 128) {
        std::fprintf(stderr, "a batch cannot contain more than 128 cells\n");
        return EXIT_FAILURE;
    }
    struct Cell {
        int64_t block_dim;
        std::vector<int64_t> task_counts;
    };
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    for (int argument = 2; argument < argc; argument += 2) {
        const int64_t block_dim = ParseInteger(
            argv[argument], "aicBlockDim");
        if (block_dim <= 0
                || block_dim > cross_domain_contract::kMaximumAicBlocks) {
            std::fprintf(
                stderr, "invalid frozen cross-domain blockDim at argument %d\n",
                argument);
            return EXIT_FAILURE;
        }
        const Cell cell{
            block_dim,
            ParseTaskCounts(argv[argument + 1], block_dim),
        };
        cells.push_back(cell);
    }

    void* device_cycles = nullptr;
    const size_t cycle_bytes = static_cast<size_t>(
        cross_domain_contract::kMaximumActors
        * cross_domain_contract::kCycleCacheLineBytes);
    ACL_CHECK(aclrtMalloc(
        &device_cycles, cycle_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    void* device_task_counts = nullptr;
    const size_t task_count_bytes = static_cast<size_t>(
        cross_domain_contract::kMaximumAicBlocks * sizeof(int64_t));
    ACL_CHECK(aclrtMalloc(
        &device_task_counts, task_count_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::vector<int64_t> padded_task_counts(
        static_cast<size_t>(cross_domain_contract::kMaximumAicBlocks), 1);

    // One balanced warmup leaves every CrossCore flag consumed before the
    // measured batch starts.
    ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0xff, cycle_bytes));
    ACL_CHECK(aclrtMemcpy(
        device_task_counts, task_count_bytes, padded_task_counts.data(),
        task_count_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtlaunch_cross_domain_rt(
        1, stream, device_cycles, device_task_counts));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    // Production Forward models a steady runtime, not process-first-use
    // initialization.  Prime the shared scheduler once with the maximum
    // frozen geometry before either balanced order begins.
    ACL_CHECK(aclrtMemset(device_cycles, cycle_bytes, 0xff, cycle_bytes));
    std::fill(padded_task_counts.begin(), padded_task_counts.end(), 128);
    ACL_CHECK(aclrtMemcpy(
        device_task_counts, task_count_bytes, padded_task_counts.data(),
        task_count_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtlaunch_cross_domain_rt(
        cross_domain_contract::kMaximumAicBlocks, stream, device_cycles,
        device_task_counts));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::printf(
        "block_dim,task_counts,aic_cycles,aiv0_cycles,aiv1_cycles,"
        "host_wall_ns,validation\n");
    for (const Cell& cell : cells) {
        std::fill(padded_task_counts.begin(), padded_task_counts.end(), 1);
        std::copy(
            cell.task_counts.begin(), cell.task_counts.end(),
            padded_task_counts.begin());
        ACL_CHECK(aclrtMemcpy(
            device_task_counts, task_count_bytes, padded_task_counts.data(),
            task_count_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        // Prime the exact grid/task scheduler state immediately before the
        // measured launch.  The fixed trial preamble above cannot isolate
        // grid-dependent runtime initialization, and no warmup interval is
        // consumed by the mechanism report.
        ACL_CHECK(aclrtMemset(
            device_cycles, cycle_bytes, 0xff, cycle_bytes));
        ACL_CHECK(aclrtlaunch_cross_domain_rt(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_cycles, device_task_counts));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemset(
            device_cycles, cycle_bytes, 0xff, cycle_bytes));
        struct timespec start;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &start));
        ACL_CHECK(aclrtlaunch_cross_domain_rt(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_cycles, device_task_counts));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        const int64_t wall_ns = static_cast<int64_t>(end.tv_sec - start.tv_sec)
            * 1000000000LL + static_cast<int64_t>(end.tv_nsec - start.tv_nsec);
        if (wall_ns <= 0) {
            std::fprintf(stderr, "host health interval must be positive\n");
            return EXIT_FAILURE;
        }
        std::vector<int64_t> padded_cycles(
            static_cast<size_t>(
                cross_domain_contract::kMaximumActors
                * cross_domain_contract::kCycleSlotElements),
            -1);
        ACL_CHECK(aclrtMemcpy(
            padded_cycles.data(), cycle_bytes, device_cycles, cycle_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        std::vector<int64_t> aic;
        std::vector<int64_t> aiv0;
        std::vector<int64_t> aiv1;
        aic.reserve(static_cast<size_t>(cell.block_dim));
        aiv0.reserve(static_cast<size_t>(cell.block_dim));
        aiv1.reserve(static_cast<size_t>(cell.block_dim));
        for (int64_t group = 0; group < cell.block_dim; ++group) {
            const int64_t actor_base =
                group * cross_domain_contract::kActorsPerAicGroup;
            const int64_t actor_indices[] = {
                actor_base, actor_base + 1, actor_base + 2};
            std::vector<int64_t>* destinations[] = {&aic, &aiv0, &aiv1};
            for (int actor = 0; actor < 3; ++actor) {
                const int64_t value = padded_cycles[static_cast<size_t>(
                    actor_indices[actor]
                    * cross_domain_contract::kCycleSlotElements)];
                if (value <= 0) {
                    std::fprintf(
                        stderr,
                        "invalid cycle %ld at group=%ld actor=%d "
                        "blockDim=%ld taskCounts=%s\n",
                        static_cast<long>(value), static_cast<long>(group),
                        actor, static_cast<long>(cell.block_dim),
                        JoinCycles(cell.task_counts).c_str());
                    return EXIT_FAILURE;
                }
                destinations[actor]->push_back(value);
            }
        }
        std::printf(
            "%ld,%s,%s,%s,%s,%ld,ok\n",
            static_cast<long>(cell.block_dim),
            JoinCycles(cell.task_counts).c_str(), JoinCycles(aic).c_str(),
            JoinCycles(aiv0).c_str(), JoinCycles(aiv1).c_str(),
            static_cast<long>(wall_ns));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_task_counts));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
