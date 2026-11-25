// Host driver for the source-bound nonempty FIA controller-span benchmark.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fia_program_span_contract.h"

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error: %s returned %d\n", #call, status); \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_fia_program_span_rt(
    uint32_t block_dim, void* stream, void* gm_results, void* gm_task_counts);

namespace {

constexpr int32_t kLogicalDeviceId = 0;

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

std::vector<int64_t> ParseTaskCounts(
        const char* text, int64_t expected_length) {
    std::vector<int64_t> values;
    const std::string source(text);
    size_t begin = 0;
    while (begin <= source.size()) {
        const size_t end = source.find(';', begin);
        const std::string token = source.substr(
            begin, end == std::string::npos ? end : end - begin);
        const int64_t value = ParseInteger(token.c_str(), "taskCounts");
        if (!fia_program_span_contract::IsFrozenTaskCount(value)) {
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

std::string Join(const std::vector<int64_t>& values) {
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
    if (device_text == nullptr
            || ParseInteger(device_text, "ASCEND_DEVICE_ID")
            != kLogicalDeviceId) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must equal logical device 0\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 2 != 0) {
        std::fprintf(stderr, "usage: %s --batch blockDim taskCounts ...\n", argv[0]);
        return EXIT_FAILURE;
    }
    struct Cell {
        int64_t block_dim;
        std::vector<int64_t> task_counts;
    };
    std::vector<Cell> cells;
    for (int argument = 2; argument < argc; argument += 2) {
        const int64_t block_dim = ParseInteger(argv[argument], "blockDim");
        if (!fia_program_span_contract::IsFrozenAicGroupCount(block_dim)) {
            std::fprintf(stderr, "blockDim is outside the frozen campaign\n");
            return EXIT_FAILURE;
        }
        cells.push_back({
            block_dim,
            ParseTaskCounts(argv[argument + 1], block_dim),
        });
    }

    const size_t result_bytes = static_cast<size_t>(
        fia_program_span_contract::kMaximumActors
        * fia_program_span_contract::kCycleCacheLineBytes);
    const size_t count_bytes = static_cast<size_t>(
        fia_program_span_contract::kMaximumAicGroups * sizeof(int64_t));
    void* device_results = nullptr;
    void* device_task_counts = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_results, result_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_task_counts, count_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    std::vector<int64_t> padded_counts(
        fia_program_span_contract::kMaximumAicGroups, 1);

    std::printf(
        "block_dim,task_counts,aic_cycles,aiv0_cycles,aiv1_cycles,validation\n");
    for (const Cell& cell : cells) {
        // Normalize the host-global mixed-task scheduler history before every
        // cell.  The first campaign showed that two exact-cell warmups alone
        // did not erase whether a 1-group cell followed small or large grids.
        // This fixed maximum-geometry preamble is outside the timed interval.
        std::fill(padded_counts.begin(), padded_counts.end(), 11);
        ACL_CHECK(aclrtMemcpy(
            device_task_counts, count_bytes, padded_counts.data(),
            count_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemset(
            device_results, result_bytes, 0xff, result_bytes));
        ACL_CHECK(aclrtlaunch_fia_program_span_rt(
            fia_program_span_contract::kMaximumAicGroups, stream,
            device_results, device_task_counts));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::fill(padded_counts.begin(), padded_counts.end(), 1);
        std::copy(
            cell.task_counts.begin(), cell.task_counts.end(),
            padded_counts.begin());
        ACL_CHECK(aclrtMemcpy(
            device_task_counts, count_bytes, padded_counts.data(),
            count_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        for (int warmup = 0; warmup < 2; ++warmup) {
            ACL_CHECK(aclrtMemset(
                device_results, result_bytes, 0xff, result_bytes));
            ACL_CHECK(aclrtlaunch_fia_program_span_rt(
                static_cast<uint32_t>(cell.block_dim), stream,
                device_results, device_task_counts));
            ACL_CHECK(aclrtSynchronizeStream(stream));
        }
        ACL_CHECK(aclrtMemset(
            device_results, result_bytes, 0xff, result_bytes));
        ACL_CHECK(aclrtlaunch_fia_program_span_rt(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_results, device_task_counts));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> results(
            fia_program_span_contract::kMaximumActors
            * fia_program_span_contract::kCycleSlotElements, -1);
        ACL_CHECK(aclrtMemcpy(
            results.data(), result_bytes, device_results, result_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        std::vector<int64_t> aic;
        std::vector<int64_t> aiv0;
        std::vector<int64_t> aiv1;
        for (int64_t group = 0; group < cell.block_dim; ++group) {
            const int64_t base = group
                * fia_program_span_contract::kActorsPerAicGroup;
            const int64_t actors[] = {base, base + 1, base + 2};
            std::vector<int64_t>* outputs[] = {&aic, &aiv0, &aiv1};
            for (int actor = 0; actor < 3; ++actor) {
                const int64_t slot = actors[actor]
                    * fia_program_span_contract::kCycleSlotElements;
                const int64_t cycles = results[slot];
                const int64_t checksum = results[slot + 1];
                if (cycles <= 0 || checksum == 0) {
                    std::fprintf(stderr, "invalid device result\n");
                    return EXIT_FAILURE;
                }
                outputs[actor]->push_back(cycles);
            }
        }
        std::printf(
            "%ld,%s,%s,%s,%s,ok\n",
            static_cast<long>(cell.block_dim),
            Join(cell.task_counts).c_str(), Join(aic).c_str(),
            Join(aiv0).c_str(), Join(aiv1).c_str());
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_task_counts));
    ACL_CHECK(aclrtFree(device_results));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
