// Host for the combined DataCopy + complete ReduceSum diagnostic.
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

extern "C" uint32_t aclrtlaunch_norm_reduce(uint32_t blockDim, void* stream,
    void* gm_src, void* out, int64_t n_elems, int64_t n_iters);

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
    void *d_src, *d_out;
    ACL_CHECK(aclrtMalloc(&d_src, buf_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_out, 64, ACL_MEM_MALLOC_HUGE_FIRST));

    auto run = [&](int64_t iters, double& per_iter_ps) {
        struct timespec a, b;
        // warmup
        ACL_CHECK(aclrtlaunch_norm_reduce(
            1, stream, d_src, d_out, n_elems, 100));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // timed
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &a));
        ACL_CHECK(aclrtlaunch_norm_reduce(
            1, stream, d_src, d_out, n_elems, iters));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &b));
        double wall_ns = (b.tv_sec-a.tv_sec)*1e9 + (b.tv_nsec-a.tv_nsec);
        per_iter_ps = wall_ns / (double)iters * 1000.0;  // ps (picoseconds)
    };

    double ps;
    run(n_iters, ps);
    double per_iter_ns = ps / 1000.0;
    printf("norm_reduce: n_elems=%ld per_iter=%.1f ns  (DataCopy + ReduceSum over %ld fp32)\n",
           (long)n_elems, per_iter_ns, (long)n_elems);

    ACL_CHECK(aclrtFree(d_src));
    ACL_CHECK(aclrtFree(d_out));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
