// ACL host for the MatMulV2 partial-M asynchronous-admission probe.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int32_t kLogicalDevice = 0;
constexpr int64_t kLaunchedAicCount = 22;
constexpr int64_t kPhysicalAicCount = 24;
constexpr int64_t kBaseM = 48;
constexpr int64_t kBaseN = 96;
constexpr int64_t kMaximumK = 4096;
constexpr int64_t kOutputTiles = 2;
constexpr int64_t kResultStride = 16;
constexpr double kCycleNs = 20.0;

#define ACL_CHECK(call) do {                                             \
    const int status = static_cast<int>(call);                           \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                 \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

size_t CheckedBf16Bytes(int64_t elements, const char* name) {
    if (elements <= 0 || static_cast<uint64_t>(elements)
            > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
        std::fprintf(stderr, "%s has an invalid allocation size\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(elements) * sizeof(uint16_t);
}

}  // namespace

extern "C" uint32_t aclrtlaunch_matmul_v2_m_tail_admission(
    uint32_t block_dim, void* stream, void* a_gm, void* b_gm, void* c_gm,
    void* result, int64_t logical_m, int64_t projection_k, int64_t step_k,
    int64_t static_no_tail);

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0'
            || ParseInteger(device_text, "ASCEND_DEVICE_ID") != kLogicalDevice) {
        std::fprintf(stderr,
            "ASCEND_DEVICE_ID must be logical device 0 after remapping\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDevice));
}

int main(int argc, char** argv) {
    if (argc != 9 || std::strcmp(argv[1], "--m") != 0
            || std::strcmp(argv[3], "--k") != 0
            || std::strcmp(argv[5], "--step-k") != 0
            || std::strcmp(argv[7], "--static-no-tail") != 0) {
        std::fprintf(stderr,
            "usage: %s --m {40|48} --k {512|1024|2048|4096} --step-k {4|8} --static-no-tail {0|1}\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t logical_m = ParseInteger(argv[2], "m");
    const int64_t projection_k = ParseInteger(argv[4], "k");
    const int64_t step_k = ParseInteger(argv[6], "step-k");
    const int64_t static_no_tail = ParseInteger(
        argv[8], "static-no-tail");
    if ((logical_m != 40 && logical_m != 48)
            || (projection_k != 512 && projection_k != 1024
                && projection_k != 2048 && projection_k != 4096)
            || (step_k != 4 && step_k != 8)
            || (static_no_tail != 0 && static_no_tail != 1)
            || (static_no_tail == 1 && logical_m != 48)) {
        std::fprintf(stderr, "unsupported probe cell\n");
        return EXIT_FAILURE;
    }

    const size_t a_bytes = CheckedBf16Bytes(
        kPhysicalAicCount * kBaseM * kMaximumK, "A");
    const size_t b_bytes = CheckedBf16Bytes(
        kOutputTiles * kBaseN * kMaximumK, "B");
    const size_t c_bytes = CheckedBf16Bytes(
        kPhysicalAicCount * kOutputTiles * kBaseM * kBaseN, "C");
    const size_t result_bytes = static_cast<size_t>(
        kPhysicalAicCount * kResultStride) * sizeof(int64_t);
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

    for (int warmup = 0; warmup < 2; ++warmup) {
        ACL_CHECK(aclrtMemset(result_gm, result_bytes, 0, result_bytes));
        ACL_CHECK(aclrtlaunch_matmul_v2_m_tail_admission(
            kLaunchedAicCount, stream, a_gm, b_gm, c_gm, result_gm,
            logical_m, projection_k, step_k, static_no_tail));
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }
    std::vector<int64_t> result(kPhysicalAicCount * kResultStride, 0);
    ACL_CHECK(aclrtMemcpy(result.data(), result_bytes, result_gm, result_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::printf(
        "worker,m,k,step_k,static_no_tail,output_tile,post_cycles,post_ns,"
        "program_cycles,program_ns,validation\n");
    bool valid = true;
    int64_t active_workers = 0;
    for (int64_t worker = 0; worker < kPhysicalAicCount; ++worker) {
        const int64_t* row = result.data() + worker * kResultStride;
        const bool metadata_valid = row[2] == logical_m
            && row[3] == projection_k && row[4] == step_k
            && row[5] == 2 * step_k && row[6] == 2 * step_k
            && row[7] == kBaseM && row[8] == kBaseN && row[9] == 64
            && row[10] == static_no_tail;
        if (!metadata_valid) {
            continue;
        }
        ++active_workers;
        for (int64_t output_tile = 0; output_tile < kOutputTiles;
                ++output_tile) {
            // IterateAll<false> is asynchronous and can return within one
            // GetSystemCycle tick.  The metadata written after PIPE_ALL is
            // the completion check; a zero-cycle admission is valid data.
            const bool row_valid = row[output_tile] >= 0;
            std::printf(
                "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%ld,%.9f,%s\n",
                worker, logical_m, projection_k, step_k, static_no_tail,
                output_tile,
                row[output_tile], row[output_tile] * kCycleNs,
                row[11], row[11] * kCycleNs,
                row_valid ? "pass" : "fail");
            valid = valid && row_valid;
        }
    }
    valid = valid && active_workers == kLaunchedAicCount;

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(result_gm));
    ACL_CHECK(aclrtFree(c_gm));
    ACL_CHECK(aclrtFree(b_gm));
    ACL_CHECK(aclrtFree(a_gm));
    ACL_CHECK(aclrtResetDevice(kLogicalDevice));
    ACL_CHECK(aclFinalize());
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
