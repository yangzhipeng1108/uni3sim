// Host for the UB-resident AIV softmax-vector mechanism microbenchmark.
#include <acl/acl.h>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <vector>

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

extern "C" uint32_t aclrtlaunch_softmax_bench(uint32_t blockDim, void* stream,
    void* out, int64_t n_cols, int64_t n_rows, int64_t n_iters);

static int64_t parse_nonnegative_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        fprintf(stderr, "%s must be a non-negative integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

static int64_t parse_positive_integer(const char* text, const char* name) {
    const int64_t value = parse_nonnegative_integer(text, name);
    if (value == 0) {
        fprintf(stderr, "%s must be positive, got 0\n", name);
        std::exit(EXIT_FAILURE);
    }
    return value;
}

static int device_id() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        fprintf(stderr, "ASCEND_DEVICE_ID must explicitly select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(parse_nonnegative_integer(text, "ASCEND_DEVICE_ID"));
}

__attribute__((constructor(101))) static void early_init() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id()));
}

int main(int argc, char** argv) {
    if (argc > 5) {
        fprintf(stderr, "usage: %s [n_cols] [n_rows] [block_dim] [n_iters]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int64_t n_cols = (argc > 1)
        ? parse_positive_integer(argv[1], "n_cols") : 512;
    int64_t n_rows = (argc > 2)
        ? parse_positive_integer(argv[2], "n_rows") : 1;
    int64_t blkdim = (argc > 3)
        ? parse_positive_integer(argv[3], "block_dim") : 1;
    int64_t n_iters = (argc > 4)
        ? parse_positive_integer(argv[4], "n_iters") : 20000;
    if (blkdim > 64) {
        fprintf(stderr, "block_dim must not exceed the 64-entry output buffer\n");
        return EXIT_FAILURE;
    }
    if (n_cols > 32768) {
        fprintf(stderr, "n_cols must not exceed the validated UB limit 32768\n");
        return EXIT_FAILURE;
    }

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void *d_out;
    ACL_CHECK(aclrtMalloc(
        &d_out, 64 * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));

    // warmup
    ACL_CHECK(aclrtlaunch_softmax_bench(
        (uint32_t)blkdim, stream, d_out, n_cols, n_rows, 100));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    struct timespec a, b;
    CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &a));
    ACL_CHECK(aclrtlaunch_softmax_bench(
        (uint32_t)blkdim, stream, d_out, n_cols, n_rows, n_iters));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &b));
    double wall_ns = (b.tv_sec-a.tv_sec)*1e9 + (b.tv_nsec-a.tv_nsec);

    double per_iter_ns = wall_ns / (double)n_iters;     // one softmax pass of n_rows rows
    double per_row_ns  = per_iter_ns / (double)n_rows;
    double per_elem_ns = per_row_ns / (double)n_cols;
    double elem_per_us = 1000.0 / per_elem_ns;

    std::vector<int64_t> block_cycles(static_cast<size_t>(blkdim));
    ACL_CHECK(aclrtMemcpy(
        block_cycles.data(), block_cycles.size() * sizeof(int64_t),
        d_out, block_cycles.size() * sizeof(int64_t),
        ACL_MEMCPY_DEVICE_TO_HOST));
    const int64_t critical_cycles = *std::max_element(
        block_cycles.begin(), block_cycles.end());
    constexpr double kSystemCycleNs = 20.0;
    const double device_per_row_ns = (
        static_cast<double>(critical_cycles) * kSystemCycleNs
        / static_cast<double>(n_iters * n_rows));

    printf("cols=%ld rows=%ld blk=%ld per_iter=%.1fns per_row=%.1fns "
           "per_elem=%.4fns elem/us=%.0f device_per_row=%.3fns\n",
           (long)n_cols, (long)n_rows, (long)blkdim,
           per_iter_ns, per_row_ns, per_elem_ns, elem_per_us,
           device_per_row_ns);

    ACL_CHECK(aclrtFree(d_out));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id()));
    ACL_CHECK(aclFinalize());
    return 0;
}
