// Host driver for the ReshapeAndCache key-200 hard-event chain probe.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define ACL_CHECK(call) do {                                               \
    const int status = static_cast<int>(call);                             \
    if (status != 0) {                                                     \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",     \
                     __FILE__, __LINE__, #call, status);                   \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

extern "C" uint32_t aclrtlaunch_reshape_cache_equal_chain(
    uint32_t blockDim, void* stream, void* key_source, void* value_source,
    void* key_destination, void* value_destination, void* output,
    int64_t payload_bytes, int64_t stride_bytes, int64_t pair_order);

namespace {

constexpr size_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumBlockDim = 48;
constexpr int64_t kStrideBytes = 4096;
constexpr double kSystemCounterTickNs = 20.0;

struct Cell {
    int64_t block_dim;
    int64_t payload_bytes;
    int64_t pair_order;
    int64_t warmup_launches;
};

int64_t ParseNonnegative(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

Cell ParseCell(const char* text) {
    const std::string value(text);
    const size_t first = value.find(':');
    const size_t second = value.find(':', first + 1);
    const size_t third = value.find(':', second + 1);
    if (first == std::string::npos || second == std::string::npos
            || (third != std::string::npos
                && value.find(':', third + 1) != std::string::npos)) {
        std::fprintf(
            stderr,
            "cell must have block_dim:payload_bytes:pair_order[:warmup_launches] form\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t block_dim = ParseNonnegative(
        value.substr(0, first).c_str(), "block_dim");
    const int64_t payload_bytes = ParseNonnegative(
        value.substr(first + 1, second - first - 1).c_str(),
        "payload_bytes");
    const size_t pair_end = (
        third == std::string::npos ? value.size() : third);
    const int64_t pair_order = ParseNonnegative(
        value.substr(second + 1, pair_end - second - 1).c_str(),
        "pair_order");
    const int64_t warmup_launches = (
        third == std::string::npos ? 0 : ParseNonnegative(
            value.substr(third + 1).c_str(), "warmup_launches"));
    if (block_dim != 1 && block_dim != 2 && block_dim != 4
            && block_dim != 8 && block_dim != 16 && block_dim != 48) {
        std::fprintf(stderr, "block_dim must be 1, 2, 4, 8, 16, or 48\n");
        std::exit(EXIT_FAILURE);
    }
    if (payload_bytes != 1024 && payload_bytes != 2048) {
        std::fprintf(stderr, "payload_bytes must be 1024 or 2048\n");
        std::exit(EXIT_FAILURE);
    }
    if (pair_order != 0 && pair_order != 1) {
        std::fprintf(stderr, "pair_order must be zero or one\n");
        std::exit(EXIT_FAILURE);
    }
    if (warmup_launches != 0 && warmup_launches != 4
            && warmup_launches != 8 && warmup_launches != 16) {
        std::fprintf(
            stderr, "warmup_launches must be zero, 4, 8, or 16\n");
        std::exit(EXIT_FAILURE);
    }
    return Cell{block_dim, payload_bytes, pair_order, warmup_launches};
}

void RunCell(aclrtStream stream, Cell cell) {
    const size_t buffer_bytes = (
        static_cast<size_t>(cell.block_dim) * kStrideBytes);
    std::vector<uint16_t> host_key(buffer_bytes / sizeof(uint16_t), 0x3c00);
    std::vector<uint16_t> host_value(buffer_bytes / sizeof(uint16_t), 0x4000);
    void* device_key_source = nullptr;
    void* device_value_source = nullptr;
    void* device_key_destination = nullptr;
    void* device_value_destination = nullptr;
    void* device_output = nullptr;
    const size_t output_values = (
        static_cast<size_t>(cell.block_dim) * kOutputValuesPerBlock);
    const size_t output_bytes = output_values * sizeof(int64_t);
    for (void** pointer : {
            &device_key_source, &device_value_source,
            &device_key_destination, &device_value_destination}) {
        ACL_CHECK(aclrtMalloc(pointer, buffer_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ACL_CHECK(aclrtMalloc(
        &device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_key_source, buffer_bytes, host_key.data(), buffer_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(
        device_value_source, buffer_bytes, host_value.data(), buffer_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(device_key_destination, buffer_bytes, 0, buffer_bytes));
    ACL_CHECK(aclrtMemset(
        device_value_destination, buffer_bytes, 0, buffer_bytes));

    for (int64_t launch = 0; launch <= cell.warmup_launches; ++launch) {
        ACL_CHECK(aclrtlaunch_reshape_cache_equal_chain(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_key_source, device_value_source,
            device_key_destination, device_value_destination, device_output,
            cell.payload_bytes, kStrideBytes, cell.pair_order));
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    std::vector<int64_t> output(output_values, -1);
    ACL_CHECK(aclrtMemcpy(
        output.data(), output_bytes, device_output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t block = 0; block < cell.block_dim; ++block) {
        uint16_t key_head = 0;
        uint16_t value_head = 0;
        ACL_CHECK(aclrtMemcpy(
            &key_head, sizeof(key_head),
            static_cast<uint8_t*>(device_key_destination)
                + block * kStrideBytes,
            sizeof(key_head), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(
            &value_head, sizeof(value_head),
            static_cast<uint8_t*>(device_value_destination)
                + block * kStrideBytes,
            sizeof(value_head), ACL_MEMCPY_DEVICE_TO_HOST));
        const size_t offset = block * kOutputValuesPerBlock;
        const int64_t control_cycles = output[offset];
        const int64_t request_cycles = output[offset + 1];
        const int64_t sink = output[offset + 2];
        const bool valid = (
            control_cycles >= 0 && request_cycles > 0 && sink == 2
            && key_head == 0x3c00 && value_head == 0x4000);
        if (!valid) {
            std::fprintf(stderr, "validation failed at worker %lld\n",
                         static_cast<long long>(block));
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%zu,%lld,%lld,%.9f,%.9f,%lld,%u,%u,pass\n",
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(block),
            static_cast<long long>(cell.payload_bytes),
            static_cast<long long>(cell.pair_order),
            static_cast<size_t>(kStrideBytes),
            static_cast<long long>(control_cycles),
            static_cast<long long>(request_cycles),
            static_cast<double>(control_cycles) * kSystemCounterTickNs,
            static_cast<double>(request_cycles) * kSystemCounterTickNs,
            static_cast<long long>(sink), static_cast<unsigned>(key_head),
            static_cast<unsigned>(value_head));
    }

    for (void* pointer : {
            device_key_source, device_value_source,
            device_key_destination, device_value_destination,
            device_output}) {
        ACL_CHECK(aclrtFree(pointer));
    }
}

}  // namespace

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(
            stderr,
            "usage: %s --batch block_dim:payload_bytes:pair_order[:warmup_launches]...\n",
            argv[0]);
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
