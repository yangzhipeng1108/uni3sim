// ACL host for the fixed M64/N5120/K1536 MatMulV2 ingress probe.
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    int status = static_cast<int>(call);                                  \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_m64_source_order(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output);
extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_m64_ingress_completion(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output);

namespace {

constexpr int64_t kElementBytes = 2;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kActiveAics = 23;
constexpr int64_t kARows = 64;
constexpr int64_t kAColumns = 1536;
constexpr int64_t kBRows = 224;
constexpr int64_t kNWorkers = 23;
constexpr double kSystemCounterTickNs = 20.0;

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

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--trials") != 0) {
        std::fprintf(stderr, "usage: %s --trials COUNT\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[2], "trials");
    if (trials < 3 || trials > 24) {
        std::fprintf(stderr, "trials must be in [3,24]\n");
        return EXIT_FAILURE;
    }

    const int64_t source_programs = 2 * (trials + 1);
    const size_t a_program_bytes = static_cast<size_t>(
        kARows * kAColumns * kElementBytes);
    const size_t b_program_bytes = static_cast<size_t>(
        kNWorkers * kBRows * kAColumns * kElementBytes);
    const size_t a_source_bytes = static_cast<size_t>(
        source_programs) * a_program_bytes;
    const size_t b_source_bytes = static_cast<size_t>(
        source_programs) * b_program_bytes;
    const size_t output_bytes = static_cast<size_t>(
        kActiveAics * kCycleSlotBytes);
    void* device_a_source = nullptr;
    void* device_b_source = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_a_source, a_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_b_source, b_source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_a_source, a_source_bytes, 0, a_source_bytes));
    ACL_CHECK(aclrtMemset(device_b_source, b_source_bytes, 0, b_source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto launch = [&](int64_t program, int64_t mode) -> int64_t {
        ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
        auto* a_source = static_cast<char*>(device_a_source)
            + program * a_program_bytes;
        auto* b_source = static_cast<char*>(device_b_source)
            + program * b_program_bytes;
        if (mode == 0) {
            ACL_CHECK(aclrtlaunch_aic_mte2_gemm_m64_source_order(
                kActiveAics, stream, a_source, b_source, device_output));
        } else if (mode == 1) {
            ACL_CHECK(aclrtlaunch_aic_mte2_gemm_m64_ingress_completion(
                kActiveAics, stream, a_source, b_source, device_output));
        } else {
            std::fprintf(stderr, "unsupported ingress mode %ld\n", mode);
            std::exit(EXIT_FAILURE);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            output.data(), output_bytes, device_output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t maximum_cycles = 0;
        bool validation_failed = false;
        const int64_t expected_count = mode == 0 ? 7 : 24;
        for (int64_t block = 0; block < kActiveAics; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            const bool invalid_cycles = mode == 0
                ? output[offset] < 0
                : output[offset] <= 0;
            if (invalid_cycles
                    || output[offset + 1] != expected_count) {
                std::fprintf(
                    stderr,
                    "ingress block validation failed: block=%ld mode=%ld "
                    "cycles=%ld count=%ld expected_count=%ld "
                    "start=%ld end=%ld\n",
                    block, mode, output[offset], output[offset + 1],
                    expected_count, output[offset + 2], output[offset + 3]);
                validation_failed = true;
            }
            maximum_cycles = std::max(maximum_cycles, output[offset]);
        }
        if (validation_failed) {
            std::exit(EXIT_FAILURE);
        }
        if (maximum_cycles <= 0) {
            std::fprintf(
                stderr,
                "ingress launch validation failed: mode=%ld "
                "maximum_cycles=%ld\n",
                mode, maximum_cycles);
            std::exit(EXIT_FAILURE);
        }
        return maximum_cycles;
    };

    launch(2 * trials, 0);
    launch(2 * trials + 1, 1);
    std::printf("trial,mode,max_cycles,duration_ns,validation\n");
    for (int64_t trial = 0; trial < trials; ++trial) {
        for (int64_t position = 0; position < 2; ++position) {
            const int64_t mode = (trial + position) & 1;
            const int64_t program = 2 * trial + mode;
            const int64_t maximum_cycles = launch(program, mode);
            std::printf(
                "%ld,%s,%ld,%.9f,pass\n", trial,
                mode == 0 ? "mte2_only" : "mte2_mte1_completion",
                maximum_cycles, maximum_cycles * kSystemCounterTickNs);
            std::fflush(stdout);
        }
    }

    ACL_CHECK(aclrtFree(device_a_source));
    ACL_CHECK(aclrtFree(device_b_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
