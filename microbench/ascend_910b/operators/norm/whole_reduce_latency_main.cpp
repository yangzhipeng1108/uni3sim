// Host for the UB-resident WholeReduceSum + barrier microbenchmark.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>


#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)


extern "C" uint32_t aclrtlaunch_whole_reduce_latency(
    uint32_t block_dim, void* stream, void* out,
    int64_t mask, int64_t repeat_times, int64_t n_iters);


static int64_t parse_nonnegative_integer(
        const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        std::fprintf(
            stderr, "%s must be a non-negative integer, got %s\n",
            name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}


static int64_t parse_positive_integer(const char* text, const char* name) {
    const int64_t value = parse_nonnegative_integer(text, name);
    if (value == 0) {
        std::fprintf(stderr, "%s must be positive, got 0\n", name);
        std::exit(EXIT_FAILURE);
    }
    return value;
}


static int device_id() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must explicitly select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(
        parse_nonnegative_integer(text, "ASCEND_DEVICE_ID"));
}


__attribute__((constructor(101))) static void early_init() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id()));
}


int main(int argc, char** argv) {
    if (argc > 4) {
        std::fprintf(
            stderr,
            "usage: %s [mask] [repeat_times] [n_iters]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t mask = argc > 1
        ? parse_positive_integer(argv[1], "mask") : 64;
    const int64_t repeat_times = argc > 2
        ? parse_positive_integer(argv[2], "repeat_times") : 1;
    const int64_t n_iters = argc > 3
        ? parse_positive_integer(argv[3], "n_iters") : 20000;
    if (mask > 64) {
        std::fprintf(stderr, "FP32 WholeReduceSum mask cannot exceed 64\n");
        return EXIT_FAILURE;
    }
    if (repeat_times > 64) {
        std::fprintf(stderr, "repeat_times cannot exceed 64\n");
        return EXIT_FAILURE;
    }

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* output_device = nullptr;
    ACL_CHECK(aclrtMalloc(
        &output_device, 64, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtlaunch_whole_reduce_latency(
        1, stream, output_device, mask, repeat_times, n_iters));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    int64_t cycles = 0;
    ACL_CHECK(aclrtMemcpy(
        &cycles, sizeof(cycles), output_device, sizeof(cycles),
        ACL_MEMCPY_DEVICE_TO_HOST));
    constexpr double kSystemCycleNs = 20.0;
    const double per_iteration_ns = (
        static_cast<double>(cycles) * kSystemCycleNs / n_iters);
    std::printf(
        "whole_reduce_latency: mask=%ld repeats=%ld iterations=%ld "
        "system_cycles=%ld per_iteration_ns=%.3f\n",
        static_cast<long>(mask), static_cast<long>(repeat_times),
        static_cast<long>(n_iters),
        static_cast<long>(cycles), per_iteration_ns);

    ACL_CHECK(aclrtFree(output_device));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
