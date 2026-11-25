// ACL host for the independent source-shaped MatMulV2 L1 descriptor program.
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
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

extern "C" uint32_t aclrtlaunch_aic_mte2_l1_program(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t programs);

namespace {

constexpr int64_t kActiveAics = 24;
constexpr int64_t kElementBytes = 2;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kAHeight = 128;
constexpr int64_t kAWidth = 512;
constexpr int64_t kBHeight = 256;
constexpr int64_t kBWidth = 240;
constexpr int64_t kADescriptorsPerProgram = 112;
constexpr int64_t kBDescriptorsPerProgram = 168;
constexpr int64_t kSourceTiles = kActiveAics;
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kABytes = kAHeight * kAWidth * kElementBytes;
constexpr int64_t kBBytes = kBHeight * kBWidth * kElementBytes;
constexpr int64_t kL1Bytes = 2 * (kABytes + kBBytes);
constexpr int64_t kProgramBytes = (
    kADescriptorsPerProgram * kABytes + kBDescriptorsPerProgram * kBBytes);
static_assert(kL1Bytes <= 512 * 1024, "program must fit the 910B AIC L1");

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

struct Timing {
    int64_t cycles;
    int64_t wall_ns;
};

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc != 5 || std::strcmp(argv[1], "--short-programs") != 0
            || std::strcmp(argv[3], "--long-programs") != 0) {
        std::fprintf(
            stderr, "usage: %s --short-programs COUNT --long-programs COUNT\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t short_programs = ParseInteger(argv[2], "short-programs");
    const int64_t long_programs = ParseInteger(argv[4], "long-programs");
    if (short_programs <= 0 || long_programs <= short_programs) {
        std::fprintf(stderr, "program counts must be positive and increasing\n");
        return EXIT_FAILURE;
    }

    void* device_a_source = nullptr;
    void* device_b_source = nullptr;
    void* device_output = nullptr;
    const size_t a_source_bytes = static_cast<size_t>(kSourceTiles * kABytes);
    const size_t b_source_bytes = static_cast<size_t>(kSourceTiles * kBBytes);
    const size_t output_bytes = static_cast<size_t>(
        kActiveAics * kCycleSlotBytes);
    ACL_CHECK(aclrtMalloc(&device_a_source, a_source_bytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_b_source, b_source_bytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&device_output, output_bytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(device_a_source, a_source_bytes, 0, a_source_bytes));
    ACL_CHECK(aclrtMemset(device_b_source, b_source_bytes, 0, b_source_bytes));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto run = [&](int64_t programs) -> Timing {
        ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_aic_mte2_l1_program(
            static_cast<uint32_t>(kActiveAics), stream, device_a_source,
            device_b_source, device_output, 1));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        struct timespec begin;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &begin));
        ACL_CHECK(aclrtlaunch_aic_mte2_l1_program(
            static_cast<uint32_t>(kActiveAics), stream, device_a_source,
            device_b_source, device_output, programs));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(output.data(), output_bytes, device_output,
                              output_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t cycles = 0;
        for (int64_t block = 0; block < kActiveAics; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            if (output[offset] <= 0 || output[offset + 1] != 1) {
                std::fprintf(stderr, "L1 program block %ld failed validation\n",
                             block);
                std::exit(EXIT_FAILURE);
            }
            cycles = std::max(cycles, output[offset]);
        }
        return Timing{
            cycles,
            static_cast<int64_t>(end.tv_sec - begin.tv_sec) * 1000000000LL
                + static_cast<int64_t>(end.tv_nsec - begin.tv_nsec),
        };
    };

    const Timing short_timing = run(short_programs);
    const Timing long_timing = run(long_programs);
    if (long_timing.cycles <= short_timing.cycles
            || long_timing.wall_ns <= short_timing.wall_ns) {
        std::fprintf(stderr, "L1 program differential interval did not increase\n");
        return EXIT_FAILURE;
    }
    const double cycles_per_program = static_cast<double>(
        long_timing.cycles - short_timing.cycles) /
        static_cast<double>(long_programs - short_programs);
    const double ns_per_program = cycles_per_program * kSystemCounterTickNs;
    const double gb_per_s_per_aic = kProgramBytes / ns_per_program;
    if (!std::isfinite(ns_per_program) || ns_per_program <= 0.0
            || !std::isfinite(gb_per_s_per_aic) || gb_per_s_per_aic <= 0.0) {
        std::fprintf(stderr, "L1 program derived result is invalid\n");
        return EXIT_FAILURE;
    }
    std::printf(
        "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,%.9f,pass\n",
        kActiveAics, short_programs, long_programs,
        kADescriptorsPerProgram, kBDescriptorsPerProgram, kABytes, kBBytes,
        kL1Bytes, kProgramBytes, short_timing.cycles, short_timing.wall_ns,
        long_timing.cycles, long_timing.wall_ns, kSystemCounterTickNs,
        ns_per_program, gb_per_s_per_aic);
    std::fflush(stdout);

    ACL_CHECK(aclrtFree(device_a_source));
    ACL_CHECK(aclrtFree(device_b_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
