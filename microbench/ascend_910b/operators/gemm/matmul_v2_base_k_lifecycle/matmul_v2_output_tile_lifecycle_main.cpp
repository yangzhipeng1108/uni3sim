// ACL host for the MatMulV2 output-tile lifecycle probe.
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
constexpr int64_t kMaximumAicCount = 24;
constexpr int64_t kBaseM = 128;
constexpr int64_t kMaximumBaseN = 128;
constexpr int64_t kMaximumK = 4096;
constexpr int64_t kMaximumOutputTiles = 9;
constexpr int64_t kResultStride = 16;
constexpr double kCycleNs = 20.0;

#define ACL_CHECK(call) do {                                             \
    const int status = static_cast<int>(call);                           \
    if (status != 0) {                                                   \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
            __FILE__, __LINE__, #call, status);                         \
        std::exit(EXIT_FAILURE);                                        \
    }                                                                    \
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

bool IsSupportedK(int64_t value) {
    return value == 1024 || value == 2048 || value == 4096;
}

bool IsSupportedOutputTiles(int64_t value) {
    return value == 1 || value == 2 || value == 4 || value == 6
        || value == 9;
}

}  // namespace

extern "C" uint32_t aclrtlaunch_matmul_v2_output_tile_lifecycle(
    uint32_t block_dim, void* stream, void* a_gm, void* b_gm, void* c_gm,
    void* result, int64_t projection_k, int64_t base_n, int64_t output_tiles,
    int64_t active_aics, int64_t force_tile_drain);

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0'
            || ParseInteger(device_text, "ASCEND_DEVICE_ID")
                != kLogicalDevice) {
        std::fprintf(stderr,
            "ASCEND_DEVICE_ID must be logical device 0 after remapping\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDevice));
}

int main(int argc, char** argv) {
    if (argc != 11 || std::strcmp(argv[1], "--base-n") != 0
            || std::strcmp(argv[3], "--workers") != 0
            || std::strcmp(argv[5], "--k") != 0
            || std::strcmp(argv[7], "--output-tiles") != 0
            || std::strcmp(argv[9], "--force-tile-drain") != 0) {
        std::fprintf(stderr,
            "usage: %s --base-n {112|128} --workers {22|24} --k {1024|2048|4096} --output-tiles {1|2|4|6|9} --force-tile-drain {0|1}\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t base_n = ParseInteger(argv[2], "base-n");
    const int64_t active_aics = ParseInteger(argv[4], "workers");
    const int64_t projection_k = ParseInteger(argv[6], "k");
    const int64_t output_tiles = ParseInteger(argv[8], "output-tiles");
    const int64_t force_tile_drain = ParseInteger(
        argv[10], "force-tile-drain");
    if ((base_n != 112 && base_n != 128)
            || (active_aics != 22 && active_aics != 24)
            || !IsSupportedK(projection_k)
            || !IsSupportedOutputTiles(output_tiles)
            || (force_tile_drain != 0 && force_tile_drain != 1)) {
        std::fprintf(stderr, "unsupported probe cell\n");
        return EXIT_FAILURE;
    }

    const size_t a_bytes = CheckedBf16Bytes(
        kMaximumAicCount * kBaseM * kMaximumK, "A");
    const size_t b_bytes = CheckedBf16Bytes(
        kMaximumOutputTiles * kMaximumBaseN * kMaximumK, "B");
    const size_t c_bytes = CheckedBf16Bytes(
        kMaximumAicCount * kMaximumOutputTiles * kBaseM * kMaximumBaseN,
        "C");
    const size_t result_bytes = static_cast<size_t>(
        kMaximumAicCount * kResultStride) * sizeof(int64_t);
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
        ACL_CHECK(aclrtlaunch_matmul_v2_output_tile_lifecycle(
            static_cast<uint32_t>(active_aics), stream,
            a_gm, b_gm, c_gm, result_gm,
            projection_k, base_n, output_tiles, active_aics,
            force_tile_drain));
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }
    std::vector<int64_t> result(kMaximumAicCount * kResultStride, 0);
    ACL_CHECK(aclrtMemcpy(result.data(), result_bytes, result_gm, result_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::printf(
        "worker,base_n,workers,k,output_tiles,force_tile_drain,step_ka,step_kb,depth_a1,depth_b1,program_cycles,program_ns,validation\n");
    bool valid = true;
    int64_t populated_workers = 0;
    for (int64_t worker = 0; worker < kMaximumAicCount; ++worker) {
        const int64_t* row = result.data() + worker * kResultStride;
        const bool populated = row[1] == projection_k && row[2] == base_n
            && row[3] == output_tiles && row[4] == active_aics
            && row[5] == 2 && row[6] == 2 && row[7] == 4 && row[8] == 4
            && row[9] == force_tile_drain && row[10] == kBaseM
            && row[11] == 128;
        if (!populated) {
            continue;
        }
        ++populated_workers;
        const bool row_valid = row[0] > 0;
        std::printf(
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%s\n",
            worker, base_n, active_aics, projection_k, output_tiles,
            force_tile_drain, row[5], row[6], row[7], row[8], row[0],
            row[0] * kCycleNs, row_valid ? "pass" : "fail");
        valid = valid && row_valid;
    }
    valid = valid && populated_workers == active_aics;

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(result_gm));
    ACL_CHECK(aclrtFree(c_gm));
    ACL_CHECK(aclrtFree(b_gm));
    ACL_CHECK(aclrtFree(a_gm));
    ACL_CHECK(aclrtResetDevice(kLogicalDevice));
    ACL_CHECK(aclFinalize());
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
