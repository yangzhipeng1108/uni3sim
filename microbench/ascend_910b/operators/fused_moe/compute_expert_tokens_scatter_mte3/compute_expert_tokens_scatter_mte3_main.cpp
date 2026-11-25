#include <acl/acl.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>

#define ACL_CHECK(call) do {                                             \
    int status = static_cast<int>(call);                                 \
    if (status != 0) {                                                   \
        std::fprintf(stderr, "%s failed with %d\n", #call, status);   \
        std::exit(EXIT_FAILURE);                                         \
    }                                                                    \
} while (0)

extern "C" uint32_t aclrtlaunch_compute_expert_tokens_scatter_mte3(
    uint32_t blockDim, void* stream, void* destination, void* output_cycles,
    int64_t block_count, int64_t iterations);

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
        std::fprintf(stderr, "usage: %s block_count iterations\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t block_count = parse_integer(argv[1]);
    const int64_t iterations = parse_integer(argv[2]);
    if (block_count > 256) {
        std::fprintf(stderr, "block_count must be at most 256\n");
        return EXIT_FAILURE;
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* destination = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclrtMalloc(&destination, 8192, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&output, 64, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtlaunch_compute_expert_tokens_scatter_mte3(
        48, stream, destination, output, block_count, iterations));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    int64_t cycles = 0;
    ACL_CHECK(aclrtMemcpy(&cycles, sizeof(cycles), output, sizeof(cycles),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    if (cycles <= 0) {
        std::fprintf(stderr, "invalid cycle count: %ld\n", cycles);
        return EXIT_FAILURE;
    }
    std::printf("block_count,iterations,cycles,cycles_per_request\n");
    std::printf("%ld,%ld,%ld,%.9f\n", block_count, iterations, cycles,
                static_cast<double>(cycles) / iterations);

    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtFree(destination));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
