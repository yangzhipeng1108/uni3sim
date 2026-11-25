// Host driver for the AIV first-GetValue dependency-curve benchmark.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

extern "C" uint32_t aclrtlaunch_first_get_value(
    uint32_t blockDim, void* stream, void* tiling, void* buffer,
    void* output, int64_t request_count);

constexpr size_t kBufferBytes = 4096;
constexpr size_t kStrideBytes = sizeof(int32_t);
constexpr size_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumBlockDim = 48;
constexpr double kSystemCounterTickNs = 20.0;
constexpr int64_t kExpectedTilingSum = 2027;

__attribute__((constructor(101))) static void early_acl_init() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

static int64_t parse_nonnegative(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

struct Cell {
    int64_t block_dim;
    int64_t request_count;
};

static Cell parse_cell(const char* text) {
    const std::string value(text);
    const size_t delimiter = value.find(':');
    if (delimiter == std::string::npos
            || delimiter == 0 || delimiter + 1 == value.size()) {
        std::fprintf(stderr, "cell must have block_dim:request_count form\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t block_dim = parse_nonnegative(
        value.substr(0, delimiter).c_str(), "block_dim");
    const int64_t request_count = parse_nonnegative(
        value.substr(delimiter + 1).c_str(), "request_count");
    if (block_dim <= 0 || block_dim > kMaximumBlockDim) {
        std::fprintf(stderr, "block_dim must be in 1..48\n");
        std::exit(EXIT_FAILURE);
    }
    if (request_count > 1) {
        std::fprintf(stderr, "request_count must be zero or one\n");
        std::exit(EXIT_FAILURE);
    }
    return Cell{block_dim, request_count};
}

static void run_cell(aclrtStream stream, void* device_tiling, Cell cell) {
    const size_t elements = kBufferBytes / sizeof(int32_t);
    std::vector<int32_t> host_buffer(elements, 0);
    for (size_t index = 0; index < elements; ++index) {
        host_buffer[index] = static_cast<int32_t>(index + 1);
    }

    void* device_buffer = nullptr;
    void* device_output = nullptr;
    const size_t output_values = static_cast<size_t>(cell.block_dim) * kOutputValuesPerBlock;
    const size_t output_bytes = output_values * sizeof(int64_t);
    ACL_CHECK(aclrtMalloc(
        &device_buffer, kBufferBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_buffer, kBufferBytes, host_buffer.data(), kBufferBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtlaunch_first_get_value(
        static_cast<uint32_t>(cell.block_dim), stream, device_tiling,
        device_buffer, device_output, cell.request_count));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> output(output_values, -1);
    ACL_CHECK(aclrtMemcpy(
        output.data(), output_bytes, device_output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t block = 0; block < cell.block_dim; ++block) {
        const size_t output_offset = static_cast<size_t>(block) * kOutputValuesPerBlock;
        const int64_t cycles = output[output_offset];
        const int64_t sink = output[output_offset + 1];
        const int64_t expected_value = (
            cell.request_count == 0 ? 0 : host_buffer[block]);
        const int64_t expected = expected_value + kExpectedTilingSum;
        const bool valid = (
            cycles >= 0 && (cell.request_count == 0 || cycles > 0)
            && sink == expected);
        if (!valid) {
            std::fprintf(
                stderr,
                "measurement validation failed for block=%lld N=%lld "
                "cycles=%lld sink=%lld expected=%lld\n",
                static_cast<long long>(block),
                static_cast<long long>(cell.request_count),
                static_cast<long long>(cycles),
                static_cast<long long>(sink),
                static_cast<long long>(expected));
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%zu,%zu,%zu,%lld,%.9f,%lld,pass\n",
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(block),
            static_cast<long long>(cell.request_count), kBufferBytes,
            kStrideBytes, elements, static_cast<long long>(cycles),
            static_cast<double>(cycles) * kSystemCounterTickNs,
            static_cast<long long>(sink));
    }

    ACL_CHECK(aclrtFree(device_buffer));
    ACL_CHECK(aclrtFree(device_output));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(
            stderr, "usage: %s --batch block_dim:request_count...\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const uint32_t host_tiling[7] = {1, 4, 128, 128, 1024, 48, 694};
    void* device_tiling = nullptr;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtMalloc(&device_tiling, 32, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_tiling, 32, host_tiling, sizeof(host_tiling),
        ACL_MEMCPY_HOST_TO_DEVICE));
    for (int index = 2; index < argc; ++index) {
        run_cell(stream, device_tiling, parse_cell(argv[index]));
    }
    ACL_CHECK(aclrtFree(device_tiling));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
