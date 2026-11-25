// ACL host for the full-A / five-N-tile MTE2-only probe.
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
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\\n",   \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_matmul_v2_full_a_k4096_n96(
    uint32_t block_dim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_clean_eviction, void* gm_output);

namespace {

constexpr int64_t kActiveAics = 23;
constexpr int64_t kLogicalNTiles = 115;
constexpr int64_t kARows = 16;
constexpr int64_t kAColumns = 4096;
constexpr int64_t kBRows = 96;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kExpectedDescriptorsPerWorker = 41;
constexpr int64_t kBDescriptorBytes = 96 * 512 * 2;
constexpr int64_t kCleanEvictionDescriptorsPerWorker = 128;
constexpr int64_t kCleanEvictionBytesPerWorker = (
    kCleanEvictionDescriptorsPerWorker * kBDescriptorBytes);
constexpr int64_t kCleanEvictionBytes = (
    kActiveAics * kCleanEvictionBytesPerWorker);
constexpr int64_t kHostChunkBytes = 16 * 1024 * 1024;
constexpr double kSystemCounterTickNs = 20.0;

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
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
        std::fprintf(stderr, "usage: %s --trials COUNT --device-id ID\\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[2], "trials");
    const int64_t device_id = ParseInteger(argv[4], "device_id");
    if (trials < 3 || trials > 24 || device_id < 0) {
        std::fprintf(stderr, "trials must be in [3,24] and device ID non-negative\\n");
        return EXIT_FAILURE;
    }
    const size_t a_bytes = static_cast<size_t>(kARows * kAColumns * 2);
    const size_t b_bytes = static_cast<size_t>(
        kLogicalNTiles * kBRows * kAColumns * 2);
    const size_t output_bytes = static_cast<size_t>(kActiveAics * kCycleSlotBytes);
    void* a_source = nullptr;
    void* b_source = nullptr;
    void* clean_eviction = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(static_cast<int32_t>(device_id)));
    ACL_CHECK(aclrtMalloc(&a_source, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&b_source, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &clean_eviction, kCleanEvictionBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    InitializeFromHost(a_source, a_bytes);
    InitializeFromHost(b_source, b_bytes);
    InitializeFromHost(clean_eviction, kCleanEvictionBytes);
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    auto launch = [&]() {
        ACL_CHECK(aclrtMemset(output, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_aic_mte2_matmul_v2_full_a_k4096_n96(
            kActiveAics, stream, a_source, b_source, clean_eviction, output));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> values(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            values.data(), output_bytes, output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t maximum_cycles = 0;
        for (int64_t block = 0; block < kActiveAics; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            if (values[offset] <= 0
                    || values[offset + 1] != kExpectedDescriptorsPerWorker
                    || values[offset + 3] < values[offset + 2]) {
                std::fprintf(stderr, "invalid block output for block=%ld\\n", block);
                std::exit(EXIT_FAILURE);
            }
            maximum_cycles = std::max(maximum_cycles, values[offset]);
        }
        return maximum_cycles;
    };
    // The first launch verifies descriptor count before the three measured
    // independent cold-source runs. Each launch begins with a fresh clean
    // eviction prelude inside the device program.
    launch();
    std::printf("trial,max_cycles,duration_ns,validation\\n");
    for (int64_t trial = 0; trial < trials; ++trial) {
        const int64_t cycles = launch();
        std::printf("%ld,%ld,%.9f,pass\\n", trial, cycles,
                    cycles * kSystemCounterTickNs);
        std::fflush(stdout);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(a_source));
    ACL_CHECK(aclrtFree(b_source));
    ACL_CHECK(aclrtFree(clean_eviction));
    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtResetDevice(static_cast<int32_t>(device_id)));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
