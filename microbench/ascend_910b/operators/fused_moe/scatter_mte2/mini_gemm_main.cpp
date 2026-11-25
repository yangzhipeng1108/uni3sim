// Host for the replicated-AIC mini-GEMM scan.
// Args: n_experts N Mc K b_stride_kb logical_device_id
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <limits>

#ifndef BM
#error "BM must be supplied by CMake through BM_VAL"
#endif
#ifndef BN
#error "BN must be supplied by CMake through BN_VAL"
#endif
#ifndef BK
#error "BK must be supplied by CMake through BK_VAL"
#endif

namespace {
constexpr int64_t kBuildM = BM;
constexpr int64_t kBuildN = BN;
constexpr int64_t kBuildK = BK;
constexpr uint32_t kBlockDim = 24;
constexpr size_t kElementBytes = sizeof(uint16_t);

static_assert(kBuildM > 0 && kBuildN > 0 && kBuildK > 0);

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

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

static int32_t parse_logical_device_id(const char* text) {
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0
            || value > std::numeric_limits<int32_t>::max()) {
        std::fprintf(stderr,
            "logical_device_id must be a non-negative int32, got %s\n", text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int32_t>(value);
}

static size_t checked_multiply(size_t lhs, size_t rhs, const char* name) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        std::fprintf(stderr, "%s exceeds size_t\n", name);
        std::exit(EXIT_FAILURE);
    }
    return lhs * rhs;
}

static size_t checked_add(size_t lhs, size_t rhs, const char* name) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        std::fprintf(stderr, "%s exceeds size_t\n", name);
        std::exit(EXIT_FAILURE);
    }
    return lhs + rhs;
}

static void require_build_shape(int64_t m, int64_t n, int64_t k) {
    if (m != kBuildM || n != kBuildN || k != kBuildK) {
        std::fprintf(stderr,
            "runtime shape Mc=%ld,N=%ld,K=%ld does not match compiled "
            "BM=%ld,BN=%ld,BK=%ld; rebuild with matching *_VAL values\n",
            static_cast<long>(m), static_cast<long>(n), static_cast<long>(k),
            static_cast<long>(kBuildM), static_cast<long>(kBuildN),
            static_cast<long>(kBuildK));
        std::exit(EXIT_FAILURE);
    }
}
}  // namespace

extern "C" uint32_t aclrtlaunch_mini_gemm(uint32_t blockDim, void* stream,
    void* aGM, void* bGM, void* cGM, void* out,
    int64_t n_experts, int64_t b_stride_elems);

int main(int argc, char** argv) {
    if (argc > 7) {
        std::fprintf(stderr,
            "usage: %s [experts] [N] [Mc] [K] [stride_kb] "
            "[logical_device_id]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int64_t n_experts = (argc > 1)
        ? parse_positive_integer(argv[1], "experts") : 128;
    int64_t N = (argc > 2) ? parse_positive_integer(argv[2], "N") : kBuildN;
    int64_t Mc = (argc > 3) ? parse_positive_integer(argv[3], "Mc") : kBuildM;
    int64_t K = (argc > 4) ? parse_positive_integer(argv[4], "K") : kBuildK;
    int64_t stride_kb = (argc > 5)
        ? parse_positive_integer(argv[5], "stride_kb") : 16600;
    int32_t logical_device_id = (argc > 6)
        ? parse_logical_device_id(argv[6]) : 0;
    require_build_shape(Mc, N, K);

    const size_t a_elements = checked_multiply(
        static_cast<size_t>(Mc), static_cast<size_t>(K), "A elements");
    const size_t b_elements = checked_multiply(
        static_cast<size_t>(N), static_cast<size_t>(K), "B elements");
    const size_t c_elements_per_aic = checked_multiply(
        static_cast<size_t>(Mc), static_cast<size_t>(N), "C elements per AIC");
    const size_t b_stride_bytes = checked_multiply(
        static_cast<size_t>(stride_kb), 1024, "B stride bytes");
    const size_t b_bytes = checked_multiply(
        b_elements, kElementBytes, "B bytes");
    if (b_stride_bytes < b_bytes || b_stride_bytes % kElementBytes != 0) {
        std::fprintf(stderr,
            "stride_kb must cover one B matrix exactly in bytes: stride=%zu, "
            "B=%zu\n", b_stride_bytes, b_bytes);
        return EXIT_FAILURE;
    }
    const size_t b_stride_elements = b_stride_bytes / kElementBytes;
    if (b_stride_elements > static_cast<size_t>(
            std::numeric_limits<int64_t>::max())) {
        std::fprintf(stderr, "B stride elements exceeds int64\n");
        return EXIT_FAILURE;
    }

    const size_t a_bytes = checked_multiply(
        a_elements, kElementBytes, "A bytes");
    const size_t b_total = checked_multiply(
        checked_multiply(static_cast<size_t>(n_experts), b_stride_elements,
            "all expert B elements"),
        kElementBytes, "all expert B bytes");
    const size_t c_bytes = checked_multiply(
        checked_multiply(c_elements_per_aic, kBlockDim, "all AIC C elements"),
        kElementBytes, "all AIC C bytes");
    const size_t logical_bytes_per_aic = checked_multiply(
        static_cast<size_t>(n_experts),
        checked_multiply(checked_add(a_elements, b_elements,
            "logical elements per expert"), kElementBytes,
            "logical bytes per expert"),
        "logical bytes per AIC scan");
    const size_t replicated_logical_bytes = checked_multiply(
        logical_bytes_per_aic, kBlockDim, "replicated logical bytes");

    ACL_CHECK(aclInit(nullptr));
    uint32_t device_count = 0;
    ACL_CHECK(aclrtGetDeviceCount(&device_count));
    if (static_cast<uint32_t>(logical_device_id) >= device_count) {
        std::fprintf(stderr,
            "logical_device_id %d is outside visible device count %u\n",
            logical_device_id, device_count);
        ACL_CHECK(aclFinalize());
        return EXIT_FAILURE;
    }
    ACL_CHECK(aclrtSetDevice(logical_device_id));

    void *d_a, *d_b, *d_c, *d_out;
    ACL_CHECK(aclrtMalloc(&d_a, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_b, b_total, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_c, c_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &d_out, kBlockDim * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(d_a, a_bytes, 0, a_bytes));
    ACL_CHECK(aclrtMemset(d_b, b_total, 0, b_total));

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    // warmup launch
    ACL_CHECK(aclrtlaunch_mini_gemm(
        kBlockDim, stream, d_a, d_b, d_c, d_out, n_experts,
        static_cast<int64_t>(b_stride_elements)));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    // timed launch
    ACL_CHECK(aclrtlaunch_mini_gemm(
        kBlockDim, stream, d_a, d_b, d_c, d_out, n_experts,
        static_cast<int64_t>(b_stride_elements)));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    int64_t cyc[kBlockDim] = {0};
    ACL_CHECK(aclrtMemcpy(
        cyc, sizeof(cyc), d_out, sizeof(cyc), ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t cycles = 0;
    for (uint32_t i = 0; i < kBlockDim; i++) {
        if (cyc[i] <= 0) {
            std::fprintf(stderr, "AIC %u returned non-positive cycles %ld\n",
                i, static_cast<long>(cyc[i]));
            return EXIT_FAILURE;
        }
        cycles = std::max(cycles, cyc[i]);
    }
    const double seconds = static_cast<double>(cycles) / 50.0e6;
    const double replicated_logical_bandwidth_gb_s =
        static_cast<double>(replicated_logical_bytes) / seconds / 1e9;
    std::printf(
        "logical_device_id,n_experts,N,Mc,K,stride_kb,block_dim,max_cycles,"
        "logical_bytes_per_aic_scan,replicated_logical_bytes_all_aic,"
        "replicated_logical_bandwidth_gb_s\n");
    std::printf("%d,%ld,%ld,%ld,%ld,%ld,%u,%ld,%zu,%zu,%.9g\n",
        logical_device_id, static_cast<long>(n_experts), static_cast<long>(N),
        static_cast<long>(Mc), static_cast<long>(K),
        static_cast<long>(stride_kb), kBlockDim, static_cast<long>(cycles),
        logical_bytes_per_aic, replicated_logical_bytes,
        replicated_logical_bandwidth_gb_s);

    ACL_CHECK(aclrtFree(d_a));
    ACL_CHECK(aclrtFree(d_b));
    ACL_CHECK(aclrtFree(d_c));
    ACL_CHECK(aclrtFree(d_out));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(logical_device_id));
    ACL_CHECK(aclFinalize());
    return 0;
}
