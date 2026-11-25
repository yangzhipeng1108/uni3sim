// Host for rmsnorm_full microbench: sweep hidden, measure per-iter ns
// via GetSystemCycle (kernel-internal, no host overhead) + wall-clock sanity.
#include <acl/acl.h>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>

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

extern "C" uint32_t aclrtlaunch_rmsnorm_full(uint32_t blockDim, void* stream,
    void* gm_src, void* gm_dst, void* out, int64_t n_elems, int64_t n_iters);

static int64_t parse_positive_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0) {
        fprintf(stderr, "%s must be a positive integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

__attribute__((constructor(101))) static void early_init() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc > 3) {
        fprintf(stderr, "usage: %s [n_elems] [n_iters]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int64_t n_elems = (argc > 1)
        ? parse_positive_integer(argv[1], "n_elems") : 2048;
    int64_t n_iters = (argc > 2)
        ? parse_positive_integer(argv[2], "n_iters") : 20000;

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    size_t buf_bytes = (size_t)n_elems * sizeof(float);
    void *d_src, *d_dst, *d_out;
    ACL_CHECK(aclrtMalloc(&d_src, buf_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_dst, buf_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_out, 64, ACL_MEM_MALLOC_HUGE_FIRST));

    // warmup
    ACL_CHECK(aclrtlaunch_rmsnorm_full(
        1, stream, d_src, d_dst, d_out, n_elems, 100));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // timed (kernel-internal via GetSystemCycle)
    ACL_CHECK(aclrtlaunch_rmsnorm_full(
        1, stream, d_src, d_dst, d_out, n_elems, n_iters));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Read cycle count from device (GetSystemCycle @50MHz = 20ns/tick)
    int64_t cycles = 0;
    ACL_CHECK(aclrtMemcpy(
        &cycles, sizeof(cycles), d_out, sizeof(cycles),
        ACL_MEMCPY_DEVICE_TO_HOST));
    double per_iter_ns_kern = (double)cycles / n_iters * 20.0;  // 20ns/tick

    // Wall-clock sanity
    struct timespec a, b;
    CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &a));
    ACL_CHECK(aclrtlaunch_rmsnorm_full(
        1, stream, d_src, d_dst, d_out, n_elems, n_iters));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &b));
    double per_iter_us_wall = ((b.tv_sec-a.tv_sec)*1e9 + (b.tv_nsec-a.tv_nsec)) / n_iters / 1000.0;

    printf("rmsnorm_full: hidden=%ld kern=%.1f ns  wall=%.3f us  (DataCopy+ReduceSum+Muls+DataCopy over %ld fp32)\n",
           (long)n_elems, per_iter_ns_kern, per_iter_us_wall, (long)n_elems);

    ACL_CHECK(aclrtFree(d_src));
    ACL_CHECK(aclrtFree(d_dst));
    ACL_CHECK(aclrtFree(d_out));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
