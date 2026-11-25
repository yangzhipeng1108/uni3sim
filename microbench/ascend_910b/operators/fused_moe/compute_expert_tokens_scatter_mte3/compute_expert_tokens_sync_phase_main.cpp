#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define ACL_CHECK(call) do {                                             \
    int status = static_cast<int>(call);                                 \
    if (status != 0) {                                                   \
        std::fprintf(stderr, "%s failed with %d\n", #call, status);   \
        std::exit(EXIT_FAILURE);                                         \
    }                                                                    \
} while (0)

extern "C" uint32_t aclrtlaunch_compute_expert_tokens_sync_phase(
    uint32_t blockDim, void* stream, void* workspace, void* output_cycles,
    int64_t elements_per_worker, int64_t iterations);

static int64_t parse_integer(const char* text) {
    char* end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "expected a positive integer, got %s\n", text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s elements_per_worker iterations\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    constexpr int64_t workers = 48;
    constexpr uint32_t aic_groups = 24;
    const int64_t elements = parse_integer(argv[1]);
    const int64_t iterations = parse_integer(argv[2]);
    if (elements > 512) {
        std::fprintf(stderr, "elements_per_worker must be at most 512\n");
        return EXIT_FAILURE;
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* workspace = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &workspace, workers * elements * sizeof(int32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &output, workers * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtlaunch_compute_expert_tokens_sync_phase(
        aic_groups, stream, workspace, output, elements, iterations));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<int64_t> cycles(workers);
    ACL_CHECK(aclrtMemcpy(
        cycles.data(), cycles.size() * sizeof(int64_t), output,
        cycles.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
    const int64_t maximum = *std::max_element(cycles.begin(), cycles.end());
    if (maximum <= 0) {
        std::fprintf(stderr, "invalid cycle count: %ld\n", maximum);
        return EXIT_FAILURE;
    }
    std::printf(
        "elements_per_worker,iterations,max_cycles,cycles_per_phase\n");
    std::printf("%ld,%ld,%ld,%.9f\n", elements, iterations, maximum,
                static_cast<double>(maximum) / iterations);

    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtFree(workspace));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
