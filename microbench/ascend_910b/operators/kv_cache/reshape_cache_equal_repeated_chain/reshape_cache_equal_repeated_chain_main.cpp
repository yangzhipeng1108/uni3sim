// Host driver for the repeated ReshapeAndCache key-200 chain probe.
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

extern "C" uint32_t aclrtlaunch_reshape_cache_equal_repeated_chain(
    uint32_t blockDim, void* stream, void* key_source, void* value_source,
    void* key_destination, void* value_destination, void* output,
    int64_t payload_bytes, int64_t stride_bytes, int64_t repeat_count,
    int64_t pair_order);

namespace {

constexpr size_t kOutputValuesPerBlock = 8;
constexpr int64_t kBlockDim = 48;
constexpr int64_t kDefaultPayloadBytes = 2048;
constexpr int64_t kStrideBytes = 4096;
constexpr double kSystemCounterTickNs = 20.0;

struct Cell {
    int64_t repeat_count;
    int64_t pair_order;
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
    const size_t separator = value.find(':');
    if (separator == std::string::npos
            || value.find(':', separator + 1) != std::string::npos) {
        std::fprintf(stderr, "cell must have repeat_count:pair_order form\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t repeat_count = ParseNonnegative(
        value.substr(0, separator).c_str(), "repeat_count");
    const int64_t pair_order = ParseNonnegative(
        value.substr(separator + 1).c_str(), "pair_order");
    if (repeat_count != 1 && repeat_count != 2
            && repeat_count != 6 && repeat_count != 22) {
        std::fprintf(stderr, "repeat_count must be 1, 2, 6, or 22\n");
        std::exit(EXIT_FAILURE);
    }
    if (pair_order != 0 && pair_order != 1) {
        std::fprintf(stderr, "pair_order must be zero or one\n");
        std::exit(EXIT_FAILURE);
    }
    return Cell{repeat_count, pair_order};
}

void RunCell(aclrtStream stream, Cell cell, int64_t payload_bytes) {
    const size_t buffer_bytes = static_cast<size_t>(
        kBlockDim * cell.repeat_count * kStrideBytes);
    std::vector<uint16_t> host_key(buffer_bytes / sizeof(uint16_t), 0x3c00);
    std::vector<uint16_t> host_value(buffer_bytes / sizeof(uint16_t), 0x4000);
    void* device_key_source = nullptr;
    void* device_value_source = nullptr;
    void* device_key_destination = nullptr;
    void* device_value_destination = nullptr;
    void* device_output = nullptr;
    const size_t output_values = kBlockDim * kOutputValuesPerBlock;
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

    ACL_CHECK(aclrtlaunch_reshape_cache_equal_repeated_chain(
        static_cast<uint32_t>(kBlockDim), stream,
        device_key_source, device_value_source,
        device_key_destination, device_value_destination, device_output,
        payload_bytes, kStrideBytes, cell.repeat_count, cell.pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> output(output_values, -1);
    ACL_CHECK(aclrtMemcpy(
        output.data(), output_bytes, device_output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t block = 0; block < kBlockDim; ++block) {
        const size_t final_offset = static_cast<size_t>(
            (block * cell.repeat_count + cell.repeat_count - 1)
            * kStrideBytes);
        uint16_t key_head = 0;
        uint16_t value_head = 0;
        ACL_CHECK(aclrtMemcpy(
            &key_head, sizeof(key_head),
            static_cast<uint8_t*>(device_key_destination) + final_offset,
            sizeof(key_head), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(
            &value_head, sizeof(value_head),
            static_cast<uint8_t*>(device_value_destination) + final_offset,
            sizeof(value_head), ACL_MEMCPY_DEVICE_TO_HOST));
        const size_t offset = block * kOutputValuesPerBlock;
        const int64_t control_cycles = output[offset];
        const int64_t request_cycles = output[offset + 1];
        const int64_t sink = output[offset + 2];
        const bool valid = (
            control_cycles >= 0 && request_cycles > 0 && sink == 3
            && key_head == 0x3c00 && value_head == 0x4000);
        if (!valid) {
            std::fprintf(stderr, "validation failed at worker %lld\n",
                         static_cast<long long>(block));
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.9f,%.9f,%lld,%u,%u,pass\n",
            static_cast<long long>(kBlockDim),
            static_cast<long long>(block),
            static_cast<long long>(payload_bytes),
            static_cast<long long>(cell.repeat_count),
            static_cast<long long>(cell.pair_order),
            static_cast<long long>(kStrideBytes),
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
    int argument_index = 1;
    int64_t payload_bytes = kDefaultPayloadBytes;
    if (argc >= 3 && std::string(argv[argument_index]) == "--payload-bytes") {
        payload_bytes = ParseNonnegative(argv[argument_index + 1],
                                         "payload_bytes");
        argument_index += 2;
    }
    if ((payload_bytes != 1024 && payload_bytes != 2048)
            || argc <= argument_index + 1
            || std::string(argv[argument_index]) != "--batch") {
        std::fprintf(
            stderr,
            "usage: %s [--payload-bytes 1024|2048] --batch "
            "repeat_count:pair_order...\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    for (int index = argument_index + 1; index < argc; ++index) {
        RunCell(stream, ParseCell(argv[index]), payload_bytes);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
