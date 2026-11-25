// Host driver for the independent loaded CrossCore ring screen.
#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define ACL_CHECK(call) do {                                            \
    int status = static_cast<int>(call);                                \
    if (status != 0) {                                                  \
        std::fprintf(stderr, "%s failed: %d\n", #call, status);       \
        std::exit(EXIT_FAILURE);                                       \
    }                                                                   \
} while (0)

extern "C" uint32_t aclrtlaunch_cross_domain_payload(
    uint32_t block_dim, void* stream, void* gm_kv, void* gm_aiv,
    void* gm_output, void* gm_cycles, int64_t task_count,
    int64_t payload_mode);

namespace {
constexpr int64_t kGroups = 24;
constexpr int64_t kActors = kGroups * 3;
constexpr int64_t kCycleBytes = kActors * 64;
constexpr int64_t kKvBytes = 3 * 12 * 4 * 128 * 8 * 128 * 2;
constexpr int64_t kAivBytes = 12 * 48 * 256 * 2;
constexpr int64_t kPairs = 32;

void* Allocate(size_t bytes) {
    void* value = nullptr;
    ACL_CHECK(aclrtMalloc(&value, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(value, bytes, 0, bytes));
    return value;
}

void Run(aclrtStream stream, void* kv, void* aiv, void* output,
         void* cycles, int64_t task_count, int64_t mode, int64_t pair,
         bool measured) {
    ACL_CHECK(aclrtMemset(cycles, kCycleBytes, 0xff, kCycleBytes));
    ACL_CHECK(aclrtlaunch_cross_domain_payload(
        kGroups, stream, kv, aiv, output, cycles, task_count, mode));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (!measured) return;
    std::vector<int64_t> raw(kCycleBytes / sizeof(int64_t), -1);
    ACL_CHECK(aclrtMemcpy(raw.data(), kCycleBytes, cycles, kCycleBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<int64_t> values;
    values.reserve(kActors);
    for (int64_t actor = 0; actor < kActors; ++actor) {
        const int64_t value = raw[actor * 8];
        if (value <= 0) {
            std::fprintf(stderr, "invalid actor %lld cycle %lld\n",
                         static_cast<long long>(actor),
                         static_cast<long long>(value));
            std::exit(EXIT_FAILURE);
        }
        values.push_back(value);
    }
    std::sort(values.begin(), values.end());
    std::printf("%lld,%lld,%lld,%lld,%lld\n",
                static_cast<long long>(task_count),
                static_cast<long long>(pair),
                static_cast<long long>(mode),
                static_cast<long long>((values[35] + values[36]) / 2),
                static_cast<long long>(values.back()));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s task_count(4|12)\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t task_count = std::strtoll(argv[1], nullptr, 10);
    if (task_count != 4 && task_count != 12) {
        std::fprintf(stderr, "task count outside frozen domain\n");
        return EXIT_FAILURE;
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    void* kv = Allocate(kKvBytes);
    void* aiv = Allocate(kAivBytes);
    void* output = Allocate(kAivBytes);
    void* cycles = Allocate(kCycleBytes);
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    Run(stream, kv, aiv, output, cycles, task_count, 0, -1, false);
    Run(stream, kv, aiv, output, cycles, task_count, 1, -1, false);
    std::puts("task_count,pair,mode,median_actor_cycles,max_actor_cycles");
    for (int64_t pair = 0; pair < kPairs; ++pair) {
        const int64_t first = pair % 2;
        Run(stream, kv, aiv, output, cycles, task_count, first,
            pair, true);
        Run(stream, kv, aiv, output, cycles, task_count, 1 - first,
            pair, true);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(cycles));
    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtFree(aiv));
    ACL_CHECK(aclrtFree(kv));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
