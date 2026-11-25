// ACL driver for the independent five-N full-A MatMulV2 Scalar control.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

namespace {
constexpr int64_t kLogicalDevice = 0;
constexpr int64_t kAicWorkers = 23;
constexpr int64_t kBaseM = 16;
constexpr int64_t kBaseN = 256;
constexpr int64_t kWorkerNTiles = 5;
constexpr int64_t kWorkerN = kBaseN * kWorkerNTiles;
constexpr int64_t kMaximumK = 4096;
constexpr int64_t kResultStride = 16;
constexpr double kCycleNs = 20.0;

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}
}  // namespace

extern "C" uint32_t aclrtlaunch_matmul_v2_full_a_five_n_scalar_control(
    uint32_t block_dim, void* stream, void* a_gm, void* b_gm, void* c_gm,
    void* result, int64_t logical_m, int64_t projection_k,
    int64_t static_no_tail);

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device = std::getenv("ASCEND_DEVICE_ID");
    if (device == nullptr || ParseInteger(device, "ASCEND_DEVICE_ID") != 0) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be logical device 0\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDevice));
}

int main(int argc, char** argv) {
    if (argc != 9 || std::strcmp(argv[1], "--m") != 0
            || std::strcmp(argv[3], "--k") != 0
            || std::strcmp(argv[5], "--static-no-tail") != 0
            || std::strcmp(argv[7], "--trials") != 0) {
        std::fprintf(stderr,
            "usage: %s --m {1|16} --k {2048|4096} --static-no-tail {0|1} --trials COUNT\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t logical_m = ParseInteger(argv[2], "m");
    const int64_t projection_k = ParseInteger(argv[4], "k");
    const int64_t static_no_tail = ParseInteger(argv[6], "static_no_tail");
    const int64_t trials = ParseInteger(argv[8], "trials");
    if ((logical_m != 1 && logical_m != kBaseM)
            || (projection_k != 2048 && projection_k != 4096)
            || (static_no_tail != 0 && static_no_tail != 1)
            || (static_no_tail == 1 && logical_m != kBaseM)
            || (static_no_tail == 0 && logical_m != 1)
            || trials < 1 || trials > 32) {
        std::fprintf(stderr, "probe arguments are outside the frozen domain\n");
        return EXIT_FAILURE;
    }
    const size_t a_bytes = static_cast<size_t>(
        kAicWorkers * kBaseM * kMaximumK * sizeof(uint16_t));
    const size_t b_bytes = static_cast<size_t>(
        kAicWorkers * kWorkerN * kMaximumK * sizeof(uint16_t));
    const size_t c_bytes = static_cast<size_t>(
        kAicWorkers * kBaseM * kWorkerN * sizeof(uint16_t));
    const size_t result_bytes = static_cast<size_t>(
        kAicWorkers * kResultStride * sizeof(int64_t));
    void* a_gm = nullptr;
    void* b_gm = nullptr;
    void* c_gm = nullptr;
    void* result_gm = nullptr;
    ACL_CHECK(aclrtMalloc(&a_gm, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&b_gm, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&c_gm, c_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&result_gm, result_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(a_gm, a_bytes, 0, a_bytes));
    ACL_CHECK(aclrtMemset(b_gm, b_bytes, 0, b_bytes));
    ACL_CHECK(aclrtMemset(c_gm, c_bytes, 0, c_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    auto launch = [&]() {
        ACL_CHECK(aclrtMemset(result_gm, result_bytes, 0, result_bytes));
        ACL_CHECK(aclrtlaunch_matmul_v2_full_a_five_n_scalar_control(
            kAicWorkers, stream, a_gm, b_gm, c_gm, result_gm,
            logical_m, projection_k, static_no_tail));
        ACL_CHECK(aclrtSynchronizeStream(stream));
    };
    for (int warmup = 0; warmup < 3; ++warmup) {
        launch();
    }
    std::printf("trial,worker,m,k,static_no_tail,depth_a1,depth_b1,"
                "step_ka,step_kb,base_n,worker_n_tiles,program_cycles,"
                "program_ns,validation\n");
    for (int64_t trial = 0; trial < trials; ++trial) {
        launch();
        std::vector<int64_t> result(kAicWorkers * kResultStride, 0);
        ACL_CHECK(aclrtMemcpy(result.data(), result_bytes, result_gm,
            result_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        for (int64_t worker = 0; worker < kAicWorkers; ++worker) {
            const int64_t* row = result.data() + worker * kResultStride;
            if (row[0] <= 0 || row[1] != logical_m || row[2] != projection_k
                    || row[3] != static_no_tail || row[4] != kAicWorkers
                    || row[5] != 1 || row[6] != 2
                    || row[7] != projection_k / 64 || row[8] != 4
                    || row[9] != kBaseN || row[10] != kWorkerNTiles) {
                std::fprintf(stderr, "worker result violates source contract\n");
                return EXIT_FAILURE;
            }
            std::printf("%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,pass\n",
                static_cast<long>(trial), static_cast<long>(worker),
                static_cast<long>(logical_m), static_cast<long>(projection_k),
                static_cast<long>(static_no_tail), static_cast<long>(row[5]),
                static_cast<long>(row[6]), static_cast<long>(row[7]),
                static_cast<long>(row[8]), static_cast<long>(row[9]),
                static_cast<long>(row[10]), static_cast<long>(row[0]),
                row[0] * kCycleNs);
        }
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(result_gm));
    ACL_CHECK(aclrtFree(c_gm));
    ACL_CHECK(aclrtFree(b_gm));
    ACL_CHECK(aclrtFree(a_gm));
    ACL_CHECK(aclrtResetDevice(kLogicalDevice));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
