// ACL host for the independent CANN FIA FlashDecode terminal prelude.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_flash_decode_prelude(
    uint32_t block_dim, void* stream, void* gm_cycles, int64_t mode,
    int64_t active_aic_groups);

namespace {

constexpr int64_t kLogicalDeviceId = 0;
constexpr int64_t kMaximumAicGroups = 24;
constexpr int64_t kActorsPerGroup = 3;
constexpr int64_t kCycleCacheLineBytes = 64;
constexpr int64_t kCycleSlotElements =
    kCycleCacheLineBytes / static_cast<int64_t>(sizeof(int64_t));

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

struct Cell {
    int64_t mode;
    int64_t active_aic_groups;
};

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
            "ASCEND_DEVICE_ID must be logical device 0 after visibility remapping\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 2 != 0) {
        std::fprintf(
            stderr, "usage: %s --batch mode active_aic_groups [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    for (int argument = 2; argument < argc; argument += 2) {
        Cell cell{
            ParseInteger(argv[argument], "mode"),
            ParseInteger(argv[argument + 1], "active_aic_groups"),
        };
        if (cell.mode < 0 || cell.mode > 2
                || cell.active_aic_groups <= 0
                || cell.active_aic_groups > kMaximumAicGroups) {
            std::fprintf(stderr, "invalid prelude cell at argument %d\n", argument);
            return EXIT_FAILURE;
        }
        cells.push_back(cell);
    }

    const size_t output_bytes = static_cast<size_t>(
        kMaximumAicGroups * kActorsPerGroup * kCycleCacheLineBytes);
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_cycles, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMemset(device_cycles, output_bytes, 0xff, output_bytes));
    ACL_CHECK(aclrtlaunch_flash_decode_prelude(
        kMaximumAicGroups, stream, device_cycles, 2, kMaximumAicGroups));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::printf("mode,active_aic_groups,actor_cycles,validation\n");
    for (const Cell& cell : cells) {
        ACL_CHECK(aclrtMemset(device_cycles, output_bytes, 0xff, output_bytes));
        ACL_CHECK(aclrtlaunch_flash_decode_prelude(
            static_cast<uint32_t>(cell.active_aic_groups), stream,
            device_cycles, cell.mode, cell.active_aic_groups));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> padded(
            output_bytes / sizeof(int64_t), -1);
        ACL_CHECK(aclrtMemcpy(
            padded.data(), output_bytes, device_cycles, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        const int64_t actor_count = cell.active_aic_groups * kActorsPerGroup;
        std::vector<int64_t> cycles;
        cycles.reserve(static_cast<size_t>(actor_count));
        for (int64_t actor = 0; actor < actor_count; ++actor) {
            const size_t offset = static_cast<size_t>(
                actor * kCycleSlotElements);
            if (padded[offset] <= 0 || padded[offset + 1] != cell.mode + 1) {
                std::fprintf(
                    stderr,
                    "invalid actor=%ld mode=%ld active_aic_groups=%ld\n",
                    static_cast<long>(actor), static_cast<long>(cell.mode),
                    static_cast<long>(cell.active_aic_groups));
                return EXIT_FAILURE;
            }
            cycles.push_back(padded[offset]);
        }
        std::printf(
            "%ld,%ld,%s,pass\n", static_cast<long>(cell.mode),
            static_cast<long>(cell.active_aic_groups),
            JoinCycles(cycles).c_str());
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
