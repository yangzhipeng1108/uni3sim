// Host driver for the M16/M64 GroupedMatmul asynchronous-admission probe.
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
constexpr int64_t kMaximumIssues = 32;
constexpr int64_t kResultStride = 48;
constexpr int64_t kM = 64;
constexpr int64_t kMinimumM = 16;
constexpr int64_t kN = 256;
constexpr int64_t kMaximumK = 2048;
constexpr int64_t kProjectionW1 = 1;
constexpr int64_t kProjectionW2 = 2;
constexpr int64_t kProgramM16 = 1;
constexpr int64_t kProgramM64 = 2;
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

int64_t ParseProjection(const char* text) {
    if (std::strcmp(text, "w1") == 0) {
        return kProjectionW1;
    }
    if (std::strcmp(text, "w2") == 0) {
        return kProjectionW2;
    }
    std::fprintf(stderr, "--projection must be w1 or w2\n");
    std::exit(EXIT_FAILURE);
}

int64_t ParseProgram(const char* text) {
    if (std::strcmp(text, "m16") == 0) {
        return kProgramM16;
    }
    if (std::strcmp(text, "m64") == 0) {
        return kProgramM64;
    }
    std::fprintf(stderr, "--program must be m16 or m64\n");
    std::exit(EXIT_FAILURE);
}

int64_t ProjectionK(int64_t projection) {
    return projection == kProjectionW1 ? 2048 : 512;
}

const char* ProjectionName(int64_t projection) {
    return projection == kProjectionW1 ? "w1" : "w2";
}

const char* ProgramName(int64_t program) {
    return program == kProgramM16 ? "m16" : "m64";
}

int64_t ProgramBaseM(int64_t program) {
    return program == kProgramM16 ? kMinimumM : kM;
}

int64_t ProgramDepthA1(int64_t program) {
    return program == kProgramM16 ? 120 : 24;
}

int64_t ProgramStepKa(int64_t program) {
    return program == kProgramM16 ? 60 : 12;
}

bool AllowedIssueCount(int64_t issues) {
    return issues == 1 || issues == 2 || issues == 4 || issues == 8
        || issues == 16 || issues == 32;
}

}  // namespace

extern "C" uint32_t aclrtlaunch_grouped_matmul_dispatch_admission(
    uint32_t block_dim, void* stream, void* a_gm, void* b_gm, void* c_gm,
    void* result, int64_t issue_count, int64_t projection, int64_t program);

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
    if (argc != 7 || std::strcmp(argv[1], "--program") != 0
            || std::strcmp(argv[3], "--projection") != 0
            || std::strcmp(argv[5], "--issues") != 0) {
        std::fprintf(stderr,
            "usage: %s --program {m16|m64} --projection {w1|w2} --issues {1|2|4|8|16|32}\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t program = ParseProgram(argv[2]);
    const int64_t projection = ParseProjection(argv[4]);
    const int64_t issues = ParseInteger(argv[6], "issues");
    if (!AllowedIssueCount(issues)) {
        std::fprintf(stderr, "issues must be 1, 2, 4, 8, 16, or 32\n");
        return EXIT_FAILURE;
    }
    const int64_t projection_k = ProjectionK(projection);
    const size_t a_bytes = CheckedBf16Bytes(kAicCount * kM * kMaximumK, "A");
    const size_t b_bytes = CheckedBf16Bytes(kN * kMaximumK, "B");
    const size_t c_bytes = CheckedBf16Bytes(
        kAicCount * kMaximumIssues * kM * kN, "C");
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
    ACL_CHECK(aclrtMemset(result_gm, result_bytes, 0, result_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    // Warm one finite sequence. The measured launch remains one source pass.
    ACL_CHECK(aclrtlaunch_grouped_matmul_dispatch_admission(
        kAicCount, stream, a_gm, b_gm, c_gm, result_gm, issues, projection,
        program));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemset(result_gm, result_bytes, 0, result_bytes));
    ACL_CHECK(aclrtlaunch_grouped_matmul_dispatch_admission(
        kAicCount, stream, a_gm, b_gm, c_gm, result_gm, issues, projection,
        program));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<int64_t> result(kAicCount * kResultStride, 0);
    ACL_CHECK(aclrtMemcpy(result.data(), result_bytes, result_gm, result_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::printf("worker,program,projection,issue_index,post_cycles,post_ns,base_m,base_n,base_k,depth_a1,depth_b1,step_ka,step_kb,db_l0a,db_l0b,db_l0c,validation\n");
    bool valid = true;
    for (int64_t worker = 0; worker < kAicCount; ++worker) {
        const int64_t* row = result.data() + worker * kResultStride;
        const bool metadata_valid = row[32] == ProgramBaseM(program)
            && row[33] == 256 && row[34] == 64
            && row[35] == ProgramDepthA1(program) && row[36] == 8
            && row[37] == ProgramStepKa(program) && row[38] == 4 && row[39] == 2
            && row[40] == 2 && row[41] == 1 && row[42] == projection
            && row[43] == issues && row[44] == program;
        for (int64_t issue = 0; issue < issues; ++issue) {
            const bool row_valid = metadata_valid && row[issue] > 0;
            std::printf("%ld,%s,%s,%ld,%ld,%.9f,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%s\n",
                static_cast<long>(worker), ProgramName(program), ProjectionName(projection),
                static_cast<long>(issue), static_cast<long>(row[issue]),
                row[issue] * kCycleNs, row[32], row[33], row[34], row[35],
                row[36], row[37], row[38], row[39], row[40], row[41],
                row_valid ? "pass" : "fail");
            valid = valid && row_valid;
        }
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
