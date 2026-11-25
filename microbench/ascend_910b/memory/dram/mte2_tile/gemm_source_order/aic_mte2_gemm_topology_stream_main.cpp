// ACL host for the parameterised MatMulV2 A1/B1 MTE2-only source probe.
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
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",      \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_aic_mte2_gemm_topology_stream(
    uint32_t block_dim, void* stream, void* gm_a_source, void* gm_b_source,
    void* gm_topology, void* gm_output);

namespace {

constexpr int64_t kCycleSlotBytes = 64;
constexpr int64_t kElementBytes = 2;
constexpr int64_t kL1Bytes = 512 * 1024;
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kTopologyFields = 11;

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

int64_t Align16(int64_t value) { return (value + 15) / 16 * 16; }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 14) {
        std::fprintf(
            stderr,
            "usage: %s TRIALS DEVICE_ID ACTIVE_AICS A_ROWS A_COLUMNS B_ROWS "
            "B_COLUMNS SOURCE_ROW_STRIDE A1_DEPTH B1_DEPTH A_DESCRIPTORS "
            "B_DESCRIPTORS WARMUPS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int64_t trials = ParseInteger(argv[1], "trials");
    const int64_t device_id = ParseInteger(argv[2], "device_id");
    const int64_t active_aics = ParseInteger(argv[3], "active_aics");
    const int64_t a_rows = ParseInteger(argv[4], "a_rows");
    const int64_t a_columns = ParseInteger(argv[5], "a_columns");
    const int64_t b_rows = ParseInteger(argv[6], "b_rows");
    const int64_t b_columns = ParseInteger(argv[7], "b_columns");
    const int64_t stride = ParseInteger(argv[8], "source_row_stride");
    const int64_t a1_depth = ParseInteger(argv[9], "a1_depth");
    const int64_t b1_depth = ParseInteger(argv[10], "b1_depth");
    const int64_t a_descriptors = ParseInteger(argv[11], "a_descriptors");
    const int64_t b_descriptors = ParseInteger(argv[12], "b_descriptors");
    const int64_t warmups = ParseInteger(argv[13], "warmups");
    const int64_t a_tile_rows = Align16(a_rows);
    const int64_t b_tile_rows = Align16(b_rows);
    const int64_t l1_bytes =
        a1_depth * a_tile_rows * a_columns * kElementBytes
        + b1_depth * b_tile_rows * b_columns * kElementBytes;
    if (trials < 3 || trials > 24 || device_id < 0
            || active_aics <= 0 || active_aics > 24
            || a_rows <= 0 || a_rows > 512 || b_rows <= 0 || b_rows > 512
            || a_columns < 16 || a_columns > 8192 || a_columns % 16 != 0
            || b_columns < 16 || b_columns > 8192 || b_columns % 16 != 0
            || stride < a_columns || stride < b_columns || stride > 65535
            || a1_depth < 1 || a1_depth > 2 || b1_depth < 1 || b1_depth > 2
            || a_descriptors <= 0 || a_descriptors > 4096
            || b_descriptors <= 0 || b_descriptors > 4096
            || warmups < 1 || warmups > 16
            || l1_bytes > kL1Bytes) {
        std::fprintf(stderr, "invalid topology\n");
        return EXIT_FAILURE;
    }
    const size_t a_bytes = static_cast<size_t>(
        (active_aics * a_rows + 16) * stride * kElementBytes);
    const size_t b_bytes = static_cast<size_t>(
        (active_aics * b_rows + 16) * stride * kElementBytes);
    const size_t topology_bytes =
        static_cast<size_t>(kTopologyFields * sizeof(int64_t));
    const size_t output_bytes =
        static_cast<size_t>(active_aics * kCycleSlotBytes);
    std::vector<int64_t> topology = {
        a_rows, a_columns, b_rows, b_columns, stride, a1_depth, b1_depth,
        a_descriptors, b_descriptors, a_tile_rows, b_tile_rows};
    void* a_source = nullptr;
    void* b_source = nullptr;
    void* topology_device = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(static_cast<int32_t>(device_id)));
    ACL_CHECK(aclrtMalloc(&a_source, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&b_source, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &topology_device, topology_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(a_source, a_bytes, 0, a_bytes));
    ACL_CHECK(aclrtMemset(b_source, b_bytes, 0, b_bytes));
    ACL_CHECK(aclrtMemcpy(
        topology_device, topology_bytes, topology.data(), topology_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    auto launch = [&]() {
        ACL_CHECK(aclrtMemset(output, output_bytes, 0, output_bytes));
        ACL_CHECK(aclrtlaunch_aic_mte2_gemm_topology_stream(
            static_cast<uint32_t>(active_aics), stream, a_source, b_source,
            topology_device, output));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> values(output_bytes / sizeof(int64_t), 0);
        ACL_CHECK(aclrtMemcpy(
            values.data(), output_bytes, output, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        int64_t maximum_cycles = 0;
        for (int64_t block = 0; block < active_aics; ++block) {
            const size_t offset = static_cast<size_t>(
                block * kCycleSlotBytes / sizeof(int64_t));
            if (values[offset] <= 0
                    || values[offset + 1] != a_descriptors + b_descriptors
                    || values[offset + 3] < values[offset + 2]) {
                std::fprintf(
                    stderr, "invalid block output for block=%ld\n", block);
                std::exit(EXIT_FAILURE);
            }
            maximum_cycles = std::max(maximum_cycles, values[offset]);
        }
        return maximum_cycles;
    };
    for (int64_t warmup = 0; warmup < warmups; ++warmup) {
        launch();
    }
    std::printf("trial,max_cycles,duration_ns,validation\n");
    for (int64_t trial = 0; trial < trials; ++trial) {
        const int64_t cycles = launch();
        std::printf("%ld,%ld,%.9f,pass\n", trial, cycles,
                    cycles * kSystemCounterTickNs);
        std::fflush(stdout);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(a_source));
    ACL_CHECK(aclrtFree(b_source));
    ACL_CHECK(aclrtFree(topology_device));
    ACL_CHECK(aclrtFree(output));
    ACL_CHECK(aclrtResetDevice(static_cast<int32_t>(device_id)));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
