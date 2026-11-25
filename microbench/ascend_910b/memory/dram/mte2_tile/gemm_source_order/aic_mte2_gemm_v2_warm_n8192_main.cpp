// ACL host for the warm 22-AIC MatMulV2 MTE2-only source program.
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
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",     \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_v2_warm_n8192(
    uint32_t block_dim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output);

namespace {

constexpr int64_t kActiveAics = 22;
constexpr int64_t kFullOutputWorkers = 20;
constexpr int64_t kFullOutputTiles = 3;
constexpr int64_t kTailOutputTiles = 2;
constexpr int64_t kLogicalNTiles = 64;
constexpr int64_t kMRows = 16;
constexpr int64_t kNRows = 128;
constexpr int64_t kSourceRowStride = 8192;
constexpr int64_t kKSlabs = 16;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kSameProgramWarmups = 5;
constexpr int64_t kHostChunkBytes = 16 * 1024 * 1024;
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

int64_t OutputTilesForWorker(int64_t block) {
    return block < kFullOutputWorkers ? kFullOutputTiles : kTailOutputTiles;
}

void InitializeFromHost(void* destination, size_t bytes) {
    void* host_chunk = nullptr;
    ACL_CHECK(aclrtMallocHost(&host_chunk, kHostChunkBytes));
    std::memset(host_chunk, 0, kHostChunkBytes);
    for (size_t offset = 0; offset < bytes; offset += kHostChunkBytes) {
        const size_t chunk_bytes = std::min(
            static_cast<size_t>(kHostChunkBytes), bytes - offset);
        ACL_CHECK(aclrtMemcpy(
            static_cast<char*>(destination) + offset, chunk_bytes, host_chunk,
            chunk_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ACL_CHECK(aclrtFreeHost(host_chunk));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::strcmp(argv[1], "--trials") != 0
            || std::strcmp(argv[3], "--device-id") != 0) {
        std::fprintf(stderr, "usage: %s --trials COUNT --device-id ID\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[2], "trials");
    const int64_t device_id = ParseInteger(argv[4], "device_id");
    if (trials < 3 || trials > 24 || device_id < 0) {
        std::fprintf(stderr, "trials must be in [3,24] and device ID non-negative\n");
        return EXIT_FAILURE;
    }
    const size_t a_bytes = static_cast<size_t>(
        kMRows * kSourceRowStride * 2);
    const size_t b_bytes = static_cast<size_t>(
        kLogicalNTiles * kNRows * kSourceRowStride * 2);
    const size_t output_bytes = static_cast<size_t>(kActiveAics * kCycleSlotBytes);
    void* a_source = nullptr;
    void* b_source = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(static_cast<int32_t>(device_id)));
    ACL_CHECK(aclrtMalloc(&a_source, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&b_source, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    InitializeFromHost(a_source, a_bytes);
    InitializeFromHost(b_source, b_bytes);
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    auto validate_output = [&]() {
        std::vector<int64_t> values(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            values.data(), output_bytes, output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t maximum_cycles = 0;
        for (int64_t block = 0; block < kActiveAics; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            const int64_t output_tiles = OutputTilesForWorker(block);
            if (values[offset] <= 0
                    || values[offset + 1] != output_tiles * kKSlabs * 2
                    || values[offset + 2] != output_tiles) {
                std::fprintf(stderr, "invalid block output for block=%ld\n", block);
                std::exit(EXIT_FAILURE);
            }
            maximum_cycles = std::max(maximum_cycles, values[offset]);
        }
        return maximum_cycles;
    };
    auto reset_and_launch = [&]() {
        ACL_CHECK(aclrtMemset(output, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_aic_mte2_gemm_v2_warm_n8192(
            kActiveAics, stream, a_source, b_source, output));
    };
    auto launch_untimed = [&]() {
        reset_and_launch();
        ACL_CHECK(aclrtSynchronizeStream(stream));
        return validate_output();
    };
    for (int64_t warmup = 0; warmup < kSameProgramWarmups; ++warmup) {
        launch_untimed();
    }
    std::printf(
        "trial,max_cycles,pipe_all_duration_ns,event_duration_ns,validation\n");
    for (int64_t trial = 0; trial < trials; ++trial) {
        ACL_CHECK(aclrtMemset(output, output_bytes, 0, output_bytes));
        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        ACL_CHECK(aclrtCreateEventWithFlag(&start, ACL_EVENT_TIME_LINE));
        ACL_CHECK(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE));
        ACL_CHECK(aclrtRecordEvent(start, stream));
        ACL_CHECK(aclrtlaunch_aic_mte2_gemm_v2_warm_n8192(
            kActiveAics, stream, a_source, b_source, output));
        ACL_CHECK(aclrtRecordEvent(end, stream));
        ACL_CHECK(aclrtSynchronizeEvent(end));
        float elapsed_ms = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&elapsed_ms, start, end));
        ACL_CHECK(aclrtDestroyEvent(end));
        ACL_CHECK(aclrtDestroyEvent(start));
        if (!(elapsed_ms > 0.0f)) {
            std::fprintf(stderr, "ACL Event duration was not positive\n");
            std::exit(EXIT_FAILURE);
        }
        const int64_t cycles = validate_output();
        std::printf("%ld,%ld,%.9f,%.9f,pass\n", trial, cycles,
                    cycles * kSystemCounterTickNs,
                    static_cast<double>(elapsed_ms) * 1.0e6);
        std::fflush(stdout);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(a_source));
    ACL_CHECK(aclrtFree(b_source));
    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtResetDevice(static_cast<int32_t>(device_id)));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
