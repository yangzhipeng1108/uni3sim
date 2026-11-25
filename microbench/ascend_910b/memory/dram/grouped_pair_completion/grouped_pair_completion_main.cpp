// ACL host for the finite GroupedMatmul GM-to-L1 completion-span probe.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
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

extern "C" uint32_t aclrtlaunch_grouped_pair_completion(
    uint32_t blockDim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_output, int64_t pairs, int64_t b_source_tiles_per_worker);

namespace {

#ifndef GROUPED_PAIR_BASE_M
#define GROUPED_PAIR_BASE_M 16
#endif
#ifndef GROUPED_PAIR_ACTIVE_M
#define GROUPED_PAIR_ACTIVE_M GROUPED_PAIR_BASE_M
#endif
#ifndef GROUPED_PAIR_B_ROW_STRIDE
#define GROUPED_PAIR_B_ROW_STRIDE 256
#endif

constexpr int64_t kActiveAics = 24;
constexpr int64_t kBaseM = GROUPED_PAIR_BASE_M;
constexpr int64_t kActiveM = GROUPED_PAIR_ACTIVE_M;
constexpr int64_t kBSourceRowStride = GROUPED_PAIR_B_ROW_STRIDE;
constexpr int64_t kATileBytes = kBaseM * 64 * 2;
constexpr int64_t kBTileBytes = 64 * 256 * 2;
constexpr int64_t kBSourceTileBytes = ((64 - 1) * kBSourceRowStride + 256) * 2;
constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kEvictionFootprintBytes = 512LL * 1024 * 1024;
constexpr int64_t kHostChunkBytes = 16LL * 1024 * 1024;
constexpr double kSystemCounterTickNs = 20.0;
static_assert(kBaseM == 16 || kBaseM == 64 || kBaseM == 128,
              "grouped pair probe supports only M16, M64, and M128 source tiles");
static_assert(kActiveM == 16 || kActiveM == 32 || kActiveM == 48
              || kActiveM == kBaseM,
              "grouped pair probe supports M16/M32/M48 tails or the full tile");
static_assert(kActiveM <= kBaseM,
              "grouped pair active rows cannot exceed the base-M allocation");
static_assert(kBSourceRowStride >= 256 && kBSourceRowStride <= 65535,
              "grouped pair B source-row stride is outside ND2NZ bounds");

int64_t SourceTilesPerWorker(int64_t pairs) {
    // A 512-tile ring touches 384 MiB across 24 AICs and therefore remains
    // larger than the shared 192 MiB L2 while bounding sparse-stride storage.
    return kBSourceRowStride == 256 ? pairs : std::min<int64_t>(pairs, 512);
}

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

void InitializeFromHost(void* destination, int64_t bytes) {
    void* host_chunk = nullptr;
    ACL_CHECK(aclrtMallocHost(
        &host_chunk, static_cast<size_t>(kHostChunkBytes)));
    std::memset(host_chunk, 0, static_cast<size_t>(kHostChunkBytes));
    for (int64_t offset = 0; offset < bytes; offset += kHostChunkBytes) {
        const int64_t copy_bytes = std::min(kHostChunkBytes, bytes - offset);
        ACL_CHECK(aclrtMemcpy(
            static_cast<char*>(destination) + offset,
            static_cast<size_t>(copy_bytes), host_chunk,
            static_cast<size_t>(copy_bytes), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ACL_CHECK(aclrtFreeHost(host_chunk));
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--pairs") != 0) {
        std::fprintf(
            stderr,
            "usage: %s --pairs {16|32|44|48|64|128|256|512|1024|2048|3072|4096|6144|8192|12288|16384|24576}\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t pairs = ParseInteger(argv[2], "pairs");
    if (pairs != 16 && pairs != 32 && pairs != 44 && pairs != 48
            && pairs != 64 && pairs != 128 && pairs != 256
            && pairs != 512 && pairs != 1024 && pairs != 2048
            && pairs != 3072 && pairs != 4096 && pairs != 6144
            && pairs != 8192 && pairs != 12288 && pairs != 16384
            && pairs != 24576) {
        std::fprintf(
            stderr,
            "pairs must be one of 16, 32, 44, 48, 64, 128, 256, 512, "
            "1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576\n");
        return EXIT_FAILURE;
    }
    const int64_t eviction_pairs = (
        (kEvictionFootprintBytes / (kActiveAics * kBTileBytes) + 3) / 4 * 4);
    const int64_t eviction_source_tiles = SourceTilesPerWorker(eviction_pairs);
    const int64_t timed_source_tiles = SourceTilesPerWorker(pairs);
    const int64_t eviction_bytes = (
        eviction_source_tiles * kActiveAics * kBSourceTileBytes);
    const int64_t timed_bytes = (
        timed_source_tiles * kActiveAics * kBSourceTileBytes);
    if (eviction_bytes > std::numeric_limits<int64_t>::max() - timed_bytes) {
        std::fprintf(stderr, "B allocation size overflow\n");
        return EXIT_FAILURE;
    }

    void* device_a_source = nullptr;
    void* device_b_source = nullptr;
    void* device_output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_a_source, static_cast<size_t>(kActiveAics * kATileBytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_b_source, static_cast<size_t>(eviction_bytes + timed_bytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output,
        static_cast<size_t>(kActiveAics * kCycleSlotBytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    InitializeFromHost(device_a_source, kActiveAics * kATileBytes);
    InitializeFromHost(device_b_source, eviction_bytes + timed_bytes);
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    const size_t output_bytes = static_cast<size_t>(
        kActiveAics * kCycleSlotBytes);
    ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));
    ACL_CHECK(aclrtlaunch_grouped_pair_completion(
        static_cast<uint32_t>(kActiveAics), stream, device_a_source,
        device_b_source, device_output, eviction_pairs,
        eviction_source_tiles));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemset(device_output, output_bytes, 0, output_bytes));

    void* timed_b_source = static_cast<void*>(
        static_cast<char*>(device_b_source) + eviction_bytes);
    struct timespec begin;
    struct timespec end;
    CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &begin));
    ACL_CHECK(aclrtlaunch_grouped_pair_completion(
        static_cast<uint32_t>(kActiveAics), stream, device_a_source,
        timed_b_source, device_output, pairs, timed_source_tiles));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
    const int64_t wall_ns = (
        static_cast<int64_t>(end.tv_sec - begin.tv_sec) * 1000000000LL
        + static_cast<int64_t>(end.tv_nsec - begin.tv_nsec));

    std::vector<int64_t> output(output_bytes / sizeof(int64_t), 0);
    ACL_CHECK(aclrtMemcpy(
        output.data(), output_bytes, device_output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t minimum_cycles = std::numeric_limits<int64_t>::max();
    int64_t maximum_cycles = 0;
    for (int64_t block = 0; block < kActiveAics; ++block) {
        const size_t offset = static_cast<size_t>(
            block * kCycleSlotBytes / sizeof(int64_t));
        if (output[offset] <= 0 || output[offset + 1] != 1) {
            std::fprintf(stderr, "completion block %ld failed validation\n", block);
            return EXIT_FAILURE;
        }
        minimum_cycles = std::min(minimum_cycles, output[offset]);
        maximum_cycles = std::max(maximum_cycles, output[offset]);
    }
    const double completion_span_ns = maximum_cycles * kSystemCounterTickNs;
    if (wall_ns <= 0 || !std::isfinite(completion_span_ns)
            || completion_span_ns <= 0.0) {
        std::fprintf(stderr, "completion measurement is invalid\n");
        return EXIT_FAILURE;
    }
    std::puts(
        "pairs,active_aics,a_rows,b_columns,base_k,b_source_row_stride,"
        "b_source_tiles_per_worker,a_residency,b_residency,"
        "drain_interval,eviction_bytes,min_cycles,max_cycles,tick_ns,"
        "completion_span_ns,wall_ns,validation");
    std::printf(
        "%ld,%ld,%ld,256,64,%ld,%ld,resident,cold,4,%ld,%ld,%ld,%.1f,%.1f,%ld,pass\n",
        pairs, kActiveAics, kActiveM, kBSourceRowStride,
        timed_source_tiles, eviction_bytes, minimum_cycles, maximum_cycles,
        kSystemCounterTickNs, completion_span_ns, wall_ns);

    ACL_CHECK(aclrtFree(device_a_source));
    ACL_CHECK(aclrtFree(device_b_source));
    ACL_CHECK(aclrtFree(device_output));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
