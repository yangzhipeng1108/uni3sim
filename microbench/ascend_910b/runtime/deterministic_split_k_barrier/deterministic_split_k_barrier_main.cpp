// ACL host driver for the independent deterministic Split-K control probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                 \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_deterministic_split_k_barrier(
    uint32_t block_dim, void* stream, void* gm_result,
    int64_t slab_count, int64_t mode);

namespace {
constexpr int64_t kAicGroups = 24;
constexpr int64_t kActorsPerGroup = 3;
constexpr int64_t kActorCount = kAicGroups * kActorsPerGroup;
constexpr int64_t kSlotElements = 8;
constexpr int64_t kLogicalDeviceId = 0;

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
}

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device = std::getenv("ASCEND_DEVICE_ID");
    if (device == nullptr || ParseInteger(device, "ASCEND_DEVICE_ID") != 0) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be logical device 0\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 2 != 0) {
        std::fprintf(stderr, "usage: %s --batch slabCount mode [cell ...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    struct Cell { int64_t slabs; int64_t mode; };
    std::vector<Cell> cells;
    for (int index = 2; index < argc; index += 2) {
        const int64_t slabs = ParseInteger(argv[index], "slabCount");
        const int64_t mode = ParseInteger(argv[index + 1], "mode");
        if (slabs < 1 || slabs > 2 || (mode != 0 && mode != 1)) {
            std::fprintf(stderr, "cell is outside the frozen source domain\n");
            return EXIT_FAILURE;
        }
        cells.push_back({slabs, mode});
    }
    const size_t result_bytes = static_cast<size_t>(
        kActorCount * kSlotElements * static_cast<int64_t>(sizeof(int64_t)));
    void* device_result = nullptr;
    ACL_CHECK(aclrtMalloc(&device_result, result_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    auto launch = [&](const Cell& cell) {
        ACL_CHECK(aclrtMemset(device_result, result_bytes, 0xff, result_bytes));
        ACL_CHECK(aclrtlaunch_deterministic_split_k_barrier(
            kAicGroups, stream, device_result, cell.slabs, cell.mode));
        ACL_CHECK(aclrtSynchronizeStream(stream));
    };
    launch({1, 0});
    launch({2, 0});
    std::printf("slab_count,mode,max_actor_cycles,max_aic_reverse_wait_cycles,"
                "max_aic_sync_cycles,max_aiv_first_sync_cycles,"
                "max_aiv_second_sync_cycles,host_wall_ns,validation\n");
    for (const Cell& cell : cells) {
        launch(cell);
        struct timespec before {};
        struct timespec after {};
        clock_gettime(CLOCK_MONOTONIC, &before);
        launch(cell);
        clock_gettime(CLOCK_MONOTONIC, &after);
        std::vector<int64_t> values(kActorCount * kSlotElements, -1);
        ACL_CHECK(aclrtMemcpy(values.data(), result_bytes, device_result,
            result_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t elapsed = 0;
        int64_t reverse_wait = 0;
        int64_t aic_sync = 0;
        int64_t aiv_first_sync = 0;
        int64_t aiv_second_sync = 0;
        for (int64_t actor = 0; actor < kActorCount; ++actor) {
            const int64_t* slot = &values[static_cast<size_t>(actor * kSlotElements)];
            if (slot[0] <= 0 || slot[5] != cell.slabs || slot[6] != cell.mode
                    || slot[7] != 1) {
                std::fprintf(stderr, "invalid actor result at actor=%ld\n", static_cast<long>(actor));
                return EXIT_FAILURE;
            }
            elapsed = std::max(elapsed, slot[0]);
            if (actor % kActorsPerGroup == 0) {
                reverse_wait = std::max(reverse_wait, slot[1]);
                aic_sync = std::max(aic_sync, slot[2]);
            } else {
                aiv_first_sync = std::max(aiv_first_sync, slot[3]);
                aiv_second_sync = std::max(aiv_second_sync, slot[4]);
            }
        }
        const int64_t wall_ns = static_cast<int64_t>(after.tv_sec - before.tv_sec)
            * 1000000000LL + static_cast<int64_t>(after.tv_nsec - before.tv_nsec);
        if (wall_ns <= 0 || (cell.mode == 0 && aiv_first_sync <= 0)
                || (cell.slabs == 2 && cell.mode == 0
                    && (reverse_wait <= 0 || aic_sync <= 0 || aiv_second_sync <= 0))) {
            std::fprintf(stderr, "source control sequence did not complete\n");
            return EXIT_FAILURE;
        }
        std::printf("%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,pass\n",
            static_cast<long>(cell.slabs), static_cast<long>(cell.mode),
            static_cast<long>(elapsed), static_cast<long>(reverse_wait),
            static_cast<long>(aic_sync), static_cast<long>(aiv_first_sync),
            static_cast<long>(aiv_second_sync), static_cast<long>(wall_ns));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_result));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
