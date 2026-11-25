// ACL host for the MatMulV2 base-K lifecycle probe.
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
constexpr int64_t kAicCount = 24;
constexpr int64_t kBaseM = 96;
constexpr int64_t kBaseN = 128;
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

extern "C" uint32_t aclrtlaunch_matmul_v2_narrow_stage_ladder(
    uint32_t block_dim, void* stream, void* a_gm, void* b_gm, void* c_gm,
    void* result, int64_t projection_k, int64_t base_m);

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
    if (argc != 5 || std::strcmp(argv[1], "--k") != 0
            || std::strcmp(argv[3], "--base-m") != 0) {
        std::fprintf(stderr,
            "usage: %s --k {1536|4096} --base-m {16|96} --aic AIC --base-n BN\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t projection_k = ParseInteger(argv[2], "k");
    const int64_t base_m = ParseInteger(argv[4], "base_m");
    if ((projection_k != 1536 && projection_k != 4096)
            || (base_m != 16 && base_m != 96)) {
        std::fprintf(stderr, "unsupported probe cell\n");
        return EXIT_FAILURE;
    }

    const int64_t cell_aic =
        (projection_k == 1536) ? 23 : 24;
    const int64_t cell_base_n =
        (base_m == 96) ? 128 : ((projection_k == 1536) ? 224 : 176);
    const size_t a_bytes = CheckedBf16Bytes(
        24 * 96 * kMaximumK, "A");
    const size_t b_bytes = CheckedBf16Bytes(
        kOutputTiles * 256 * kMaximumK, "B");
    const size_t c_bytes = CheckedBf16Bytes(
        24 * kOutputTiles * 96 * 256, "C");
    const size_t result_bytes = static_cast<size_t>(
        kAicCount * kResultStride) * sizeof(int64_t);
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

    for (int warmup = 0; warmup < 3; ++warmup) {
        ACL_CHECK(aclrtMemset(result_gm, result_bytes, 0, result_bytes));
        ACL_CHECK(aclrtlaunch_matmul_v2_narrow_stage_ladder(
            (uint32_t)cell_aic, stream, a_gm, b_gm, c_gm, result_gm,
            projection_k, base_m));
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }
    std::vector<int64_t> result(kAicCount * kResultStride, 0);
    ACL_CHECK(aclrtMemcpy(result.data(), result_bytes, result_gm, result_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::printf(
        "worker,k,base_m,step_ka,step_kb,depth_a1,depth_b1,program_cycles,program_ns,validation\n");
    bool valid = true;
    for (int64_t worker = 0; worker < cell_aic; ++worker) {
        const int64_t* row = result.data() + worker * kResultStride;
        const int64_t expected_step_ka =
            (base_m == 96) ? (512 / 64) : (projection_k / 64);
        const int64_t expected_depth_a1 =
            (base_m == 96) ? (2 * expected_step_ka) : expected_step_ka;
        const bool row_valid = row[0] > 0 && row[1] == projection_k
            && row[2] == 64 && row[3] == expected_step_ka
            && row[4] * 64 == 256 && row[5] == expected_depth_a1
            && row[6] == 2 * row[4]
            && row[7] == base_m && row[8] == cell_base_n;
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%s\n",
            worker, projection_k, base_m, row[3], row[4], row[5], row[6],
            row[0], row[0] * kCycleNs, row_valid ? "pass" : "fail");
        valid = valid && row_valid;
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(result_gm));
    ACL_CHECK(aclrtFree(c_gm));
    ACL_CHECK(aclrtFree(b_gm));
    ACL_CHECK(aclrtFree(a_gm));
    ACL_CHECK(aclrtResetDevice(kLogicalDevice));
    ACL_CHECK(aclFinalize());
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
