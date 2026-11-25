// Host driver for the independent FIA AIC/AIV handoff-edge benchmark.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "fia_edge_handoff_contract.h"

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

extern "C" uint32_t aclrtlaunch_fia_edge_handoff_rt(
    uint32_t block_dim, void* stream, void* gm_results, int64_t task_count);

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

const int64_t* Actor(
        const std::vector<int64_t>& results, int64_t group, int64_t actor) {
    const int64_t index =
        group * fia_edge_handoff_contract::kActorsPerAicGroup + actor;
    return &results[static_cast<size_t>(
        index * fia_edge_handoff_contract::kResultSlotElements)];
}

int64_t CriticalEdge(
        const std::vector<int64_t>& results, int64_t block_dim,
        int64_t marker, int edge) {
    int64_t critical = 0;
    for (int64_t group = 0; group < block_dim; ++group) {
        const int64_t* aic = Actor(results, group, 0);
        const int64_t* aiv0 = Actor(results, group, 1);
        const int64_t* aiv1 = Actor(results, group, 2);
        int64_t value = 0;
        if (edge == 0 || edge == 2) {
            value = std::max(
                aiv0[marker] - aic[marker],
                aiv1[marker] - aic[marker]);
        } else if (edge == 1) {
            value = std::min(
                aic[marker] - aiv0[marker],
                aic[marker] - aiv1[marker]);
        } else {
            std::fprintf(stderr, "unknown edge\n");
            std::exit(EXIT_FAILURE);
        }
        if (value <= 0) {
            std::fprintf(stderr, "non-positive cross-actor timestamp delta\n");
            std::exit(EXIT_FAILURE);
        }
        critical = std::max(critical, value);
    }
    return critical;
}

int64_t LaneSkewPpm(
        const std::vector<int64_t>& results, int64_t block_dim,
        int64_t marker) {
    int64_t maximum = 0;
    for (int64_t group = 0; group < block_dim; ++group) {
        const int64_t* aic = Actor(results, group, 0);
        const int64_t* aiv0 = Actor(results, group, 1);
        const int64_t* aiv1 = Actor(results, group, 2);
        const int64_t left = aiv0[marker] - aic[marker];
        const int64_t right = aiv1[marker] - aic[marker];
        const int64_t denominator = std::min(left, right);
        if (denominator <= 0) {
            std::fprintf(stderr, "non-positive lane timestamp delta\n");
            std::exit(EXIT_FAILURE);
        }
        const int64_t ppm = static_cast<int64_t>(std::llround(
            1000000.0 * std::abs(left - right) / denominator));
        maximum = std::max(maximum, ppm);
    }
    return maximum;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be set explicitly\n");
        std::exit(EXIT_FAILURE);
    }
    if (ParseInteger(device_text, "ASCEND_DEVICE_ID") != kLogicalDeviceId) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be logical device 0\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 2 != 0) {
        std::fprintf(
            stderr, "usage: %s --batch aicBlockDim taskCount [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    struct Cell { int64_t block_dim; int64_t task_count; };
    std::vector<Cell> cells;
    for (int argument = 2; argument < argc; argument += 2) {
        const int64_t block_dim = ParseInteger(argv[argument], "aicBlockDim");
        const int64_t task_count = ParseInteger(argv[argument + 1], "taskCount");
        if (block_dim <= 0
                || block_dim > fia_edge_handoff_contract::kMaximumAicGroups
                || !fia_edge_handoff_contract::IsFrozenTaskCount(task_count)) {
            std::fprintf(stderr, "cell is outside the frozen domain\n");
            return EXIT_FAILURE;
        }
        cells.push_back({block_dim, task_count});
    }
    if (cells.size() > 128) {
        std::fprintf(stderr, "a batch cannot contain more than 128 cells\n");
        return EXIT_FAILURE;
    }

    const size_t result_bytes = static_cast<size_t>(
        fia_edge_handoff_contract::kMaximumActors
        * fia_edge_handoff_contract::kResultSlotBytes);
    void* device_results = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_results, result_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    auto launch = [&](int64_t block_dim, int64_t task_count) {
        ACL_CHECK(aclrtMemset(
            device_results, result_bytes, 0xff, result_bytes));
        ACL_CHECK(aclrtlaunch_fia_edge_handoff_rt(
            static_cast<uint32_t>(block_dim), stream,
            device_results, task_count));
        ACL_CHECK(aclrtSynchronizeStream(stream));
    };
    launch(1, 4);
    launch(fia_edge_handoff_contract::kMaximumAicGroups, 128);

    std::printf(
        "block_dim,task_count,c1v1_total_cycles,v1c2_total_cycles,"
        "c2v2_total_cycles,c1v1_prefix1_cycles,v1c2_prefix1_cycles,"
        "c2v2_prefix1_cycles,c1v1_prefix2_cycles,v1c2_prefix2_cycles,"
        "c2v2_prefix2_cycles,c1v1_prefix3_cycles,v1c2_prefix3_cycles,"
        "c2v2_prefix3_cycles,c1v1_lane_skew_ppm,c2v2_lane_skew_ppm,"
        "ring_elapsed_cycles,host_wall_ns,validation\n");
    for (const Cell& cell : cells) {
        launch(cell.block_dim, cell.task_count);
        struct timespec start;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &start));
        launch(cell.block_dim, cell.task_count);
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        const int64_t wall_ns = static_cast<int64_t>(end.tv_sec - start.tv_sec)
            * 1000000000LL + static_cast<int64_t>(end.tv_nsec - start.tv_nsec);
        std::vector<int64_t> results(
            static_cast<size_t>(fia_edge_handoff_contract::kMaximumActors
                                * fia_edge_handoff_contract::kResultSlotElements),
            -1);
        ACL_CHECK(aclrtMemcpy(
            results.data(), result_bytes, device_results, result_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t ring_elapsed = 0;
        for (int64_t group = 0; group < cell.block_dim; ++group) {
            for (int actor = 0; actor < 3; ++actor) {
                const int64_t elapsed = Actor(results, group, actor)[0];
                if (elapsed <= 0) {
                    std::fprintf(stderr, "non-positive actor elapsed cycles\n");
                    return EXIT_FAILURE;
                }
                ring_elapsed = std::max(ring_elapsed, elapsed);
            }
        }
        const int64_t total[] = {
            CriticalEdge(results, cell.block_dim, 1, 0),
            CriticalEdge(results, cell.block_dim, 2, 1),
            CriticalEdge(results, cell.block_dim, 3, 2),
        };
        int64_t prefix[3][3] = {};
        for (int64_t depth = 0; depth < 3; ++depth) {
            const int64_t marker = 4 + 3 * depth;
            for (int edge = 0; edge < 3; ++edge) {
                prefix[depth][edge] = CriticalEdge(
                    results, cell.block_dim, marker + edge, edge);
            }
        }
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,"
            "%ld,%ld,%ld,%ld,ok\n",
            static_cast<long>(cell.block_dim),
            static_cast<long>(cell.task_count),
            static_cast<long>(total[0]), static_cast<long>(total[1]),
            static_cast<long>(total[2]),
            static_cast<long>(prefix[0][0]), static_cast<long>(prefix[0][1]),
            static_cast<long>(prefix[0][2]), static_cast<long>(prefix[1][0]),
            static_cast<long>(prefix[1][1]), static_cast<long>(prefix[1][2]),
            static_cast<long>(prefix[2][0]), static_cast<long>(prefix[2][1]),
            static_cast<long>(prefix[2][2]),
            static_cast<long>(LaneSkewPpm(results, cell.block_dim, 1)),
            static_cast<long>(LaneSkewPpm(results, cell.block_dim, 3)),
            static_cast<long>(ring_elapsed), static_cast<long>(wall_ns));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_results));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
