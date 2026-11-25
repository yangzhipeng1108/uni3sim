// Host driver for the source-independent q2 row-stride geometry sweep.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                              \
    const int status = static_cast<int>(call);                            \
    if (status != 0) {                                                    \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",    \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

extern "C" uint32_t aclrtlaunch_swiglu_input_pair_geometry(
    uint32_t block_dim, void* stream, void* first_source,
    void* second_source, void* output, int64_t row_count,
    int64_t row_bytes, int64_t source_stride, int64_t base_phase,
    int64_t pair_order);

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

namespace {

constexpr size_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumBlockDim = 48;
constexpr int64_t kMaximumRows = 4;
constexpr int64_t kMaximumRowBytes = 8192;
constexpr int64_t kMaximumStrideRatio = 32;
constexpr int64_t kMaximumBasePhase = 512;
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kExpectedSink = 0xb4;

int64_t ParseNonnegative(const std::string& text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || value < 0) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

struct Cell {
    int64_t block_dim;
    int64_t row_count;
    int64_t row_bytes;
    int64_t stride_ratio;
    int64_t base_phase;
    int64_t pair_order;
};

Cell ParseCell(const char* text) {
    const std::string value(text);
    std::vector<std::string> fields;
    size_t start = 0;
    while (true) {
        const size_t separator = value.find(':', start);
        fields.push_back(value.substr(start, separator - start));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    if (fields.size() != 6) {
        std::fprintf(
            stderr,
            "cell must have block_dim:row_count:row_bytes:stride_ratio:"
            "base_phase:pair_order form\n");
        std::exit(EXIT_FAILURE);
    }
    const Cell cell = {
        ParseNonnegative(fields[0], "block_dim"),
        ParseNonnegative(fields[1], "row_count"),
        ParseNonnegative(fields[2], "row_bytes"),
        ParseNonnegative(fields[3], "stride_ratio"),
        ParseNonnegative(fields[4], "base_phase"),
        ParseNonnegative(fields[5], "pair_order"),
    };
    if (cell.block_dim <= 0 || cell.block_dim > kMaximumBlockDim
            || cell.row_count <= 0 || cell.row_count > kMaximumRows
            || cell.row_bytes <= 0 || cell.row_bytes > kMaximumRowBytes
            || cell.row_bytes % 32 != 0
            || cell.stride_ratio <= 0
            || cell.stride_ratio > kMaximumStrideRatio
            || cell.base_phase < 0 || cell.base_phase > kMaximumBasePhase
            || cell.base_phase % 32 != 0
            || cell.pair_order < 0 || cell.pair_order > 1) {
        std::fprintf(stderr, "cell has an out-of-contract geometry\n");
        std::exit(EXIT_FAILURE);
    }
    if (cell.row_bytes > std::numeric_limits<int64_t>::max()
            / cell.stride_ratio) {
        std::fprintf(stderr, "cell source stride overflows\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t source_stride = cell.row_bytes * cell.stride_ratio;
    const int64_t payload_bytes = cell.row_count * cell.row_bytes;
    if (payload_bytes > 32768 || source_stride < cell.row_bytes) {
        std::fprintf(stderr, "cell has an invalid source geometry\n");
        std::exit(EXIT_FAILURE);
    }
    return cell;
}

void RunCell(aclrtStream stream, const Cell cell) {
    const int64_t source_stride = cell.row_bytes * cell.stride_ratio;
    const int64_t worker_span = (
        (cell.row_count - 1) * source_stride + cell.row_bytes);
    const int64_t source_bytes_signed = (
        cell.base_phase + cell.block_dim * worker_span);
    if (source_bytes_signed <= 0) {
        std::fprintf(stderr, "source allocation overflowed\n");
        std::exit(EXIT_FAILURE);
    }
    const size_t source_bytes = static_cast<size_t>(source_bytes_signed);
    const size_t output_values = (
        static_cast<size_t>(cell.block_dim) * kOutputValuesPerBlock);
    const size_t output_bytes = output_values * sizeof(int64_t);
    std::vector<uint8_t> host_source(source_bytes, 0x5a);
    void* first_source = nullptr;
    void* second_source = nullptr;
    void* output = nullptr;
    ACL_CHECK(aclrtMalloc(
        &first_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &second_source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        first_source, source_bytes, host_source.data(), source_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(
        second_source, source_bytes, host_source.data(), source_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtlaunch_swiglu_input_pair_geometry(
        static_cast<uint32_t>(cell.block_dim), stream, first_source,
        second_source, output, cell.row_count, cell.row_bytes,
        source_stride, cell.base_phase, cell.pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> host_output(output_values, -1);
    ACL_CHECK(aclrtMemcpy(
        host_output.data(), output_bytes, output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t block = 0; block < cell.block_dim; ++block) {
        const size_t offset = (
            static_cast<size_t>(block) * kOutputValuesPerBlock);
        const int64_t control_cycles = host_output[offset];
        const int64_t request_cycles = host_output[offset + 1];
        const int64_t sink = host_output[offset + 2];
        if (control_cycles < 0 || request_cycles <= 0
                || sink != kExpectedSink) {
            std::fprintf(stderr, "device validation failed\n");
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.9f,%.9f,"
            "%lld,pass\n",
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(block),
            static_cast<long long>(cell.row_count),
            static_cast<long long>(cell.row_bytes),
            static_cast<long long>(source_stride),
            static_cast<long long>(cell.base_phase),
            static_cast<long long>(cell.pair_order),
            static_cast<long long>(control_cycles),
            static_cast<long long>(request_cycles),
            static_cast<double>(control_cycles) * kSystemCounterTickNs,
            static_cast<double>(request_cycles) * kSystemCounterTickNs,
            static_cast<long long>(sink));
    }
    ACL_CHECK(aclrtFree(first_source));
    ACL_CHECK(aclrtFree(second_source));
    ACL_CHECK(aclrtFree(output));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(
            stderr,
            "usage: %s --batch block_dim:row_count:row_bytes:"
            "stride_ratio:base_phase:pair_order...\n", argv[0]);
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    for (int index = 2; index < argc; ++index) {
        RunCell(stream, ParseCell(argv[index]));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
