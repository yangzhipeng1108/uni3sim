// Host for the UB-resident complete ReduceSum + barrier microbenchmark.
#include <acl/acl.h>

#include <cerrno>
#include <cmath>
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


extern "C" uint32_t aclrtlaunch_reduce_sum_latency(
    uint32_t block_dim, void* stream, void* cycles, void* validation,
    int64_t count, int64_t n_iters);


constexpr int64_t kMinimumCount = 32;
constexpr int64_t kMaximumCount = 16384;
constexpr size_t kScalarCacheLineBytes = 64;


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
    if (argc > 3) {
        std::fprintf(
            stderr, "usage: %s [count] [n_iters]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t count = argc > 1
        ? parse_positive_integer(argv[1], "count") : 2048;
    const int64_t n_iters = argc > 2
        ? parse_positive_integer(argv[2], "n_iters") : 20000;
    if (count < kMinimumCount || count > kMaximumCount
            || count % 32 != 0) {
        std::fprintf(
            stderr, "count must be a multiple of 32 in [%ld,%ld], got %ld\n",
            static_cast<long>(kMinimumCount),
            static_cast<long>(kMaximumCount),
            static_cast<long>(count));
        return EXIT_FAILURE;
    }

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* cycles_device = nullptr;
    void* validation_device = nullptr;
    ACL_CHECK(aclrtMalloc(
        &cycles_device, kScalarCacheLineBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &validation_device, kScalarCacheLineBytes,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(
        cycles_device, kScalarCacheLineBytes, 0xff,
        kScalarCacheLineBytes));
    ACL_CHECK(aclrtMemset(
        validation_device, kScalarCacheLineBytes, 0,
        kScalarCacheLineBytes));

    ACL_CHECK(aclrtlaunch_reduce_sum_latency(
        1, stream, cycles_device, validation_device, count, n_iters));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    int64_t cycles = 0;
    float validation = 0.0f;
    ACL_CHECK(aclrtMemcpy(
        &cycles, sizeof(cycles), cycles_device, sizeof(cycles),
        ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(
        &validation, sizeof(validation), validation_device,
        sizeof(validation),
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (cycles <= 0) {
        std::fprintf(
            stderr, "device returned invalid system cycles: %ld\n",
            static_cast<long>(cycles));
        return EXIT_FAILURE;
    }
    const float expected = static_cast<float>(count);
    if (!std::isfinite(validation)
            || std::fabs(validation - expected) > 1.0e-3f) {
        std::fprintf(
            stderr, "reduction validation failed: got %.9f expected %.9f\n",
            validation, expected);
        return EXIT_FAILURE;
    }
    constexpr double kSystemCycleNs = 20.0;
    const double per_iteration_ns = (
        static_cast<double>(cycles) * kSystemCycleNs / n_iters);
    std::printf(
        "reduce_sum_latency: device_id=%d count=%ld iterations=%ld "
        "system_cycles=%ld per_iteration_ns=%.3f "
        "validation=%.6f validation_status=pass\n",
        device_id(),
        static_cast<long>(count), static_cast<long>(n_iters),
        static_cast<long>(cycles), per_iteration_ns, validation);

    ACL_CHECK(aclrtFree(validation_device));
    ACL_CHECK(aclrtFree(cycles_device));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
