// Host driver for the repeated ReshapeAndCache key-202 queue-pair probe.
#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

extern "C" uint32_t aclrtlaunch_reshape_cache_key202_repeated_pair(
    uint32_t blockDim, void* stream, void* key_source, void* value_source,
    void* key_destination, void* value_destination, void* output,
    int64_t payload_bytes, int64_t stride_bytes, int64_t repeat_count,
    int64_t pair_order);

namespace {

constexpr int64_t kBlockDim = 48;
constexpr int64_t kPayloadBytes = 2048;
constexpr int64_t kStrideBytes = 2048;
constexpr int64_t kRepeatCount = 22;
constexpr size_t kOutputValuesPerBlock = 8;
constexpr double kSystemCounterTickNs = 20.0;

int64_t ParsePairOrder(const char* text) {
    const std::string value(text);
    if (value == "0") {
        return 0;
    }
    if (value == "1") {
        return 1;
    }
    std::fprintf(stderr, "pair_order must be zero or one\n");
    std::exit(EXIT_FAILURE);
}

void RunCell(aclrtStream stream, int64_t pair_order) {
    const size_t bytes_per_launch = static_cast<size_t>(
        kBlockDim * kRepeatCount * kStrideBytes);
    const size_t buffer_bytes = 2 * bytes_per_launch;
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

    // Warm the runtime and instruction path on a disjoint address range.
    // The measured half remains untouched by the queue-pair kernel.
    ACL_CHECK(aclrtlaunch_reshape_cache_key202_repeated_pair(
        static_cast<uint32_t>(kBlockDim), stream,
        device_key_source, device_value_source,
        device_key_destination, device_value_destination, device_output,
        kPayloadBytes, kStrideBytes, kRepeatCount, pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtlaunch_reshape_cache_key202_repeated_pair(
        static_cast<uint32_t>(kBlockDim), stream,
        static_cast<uint8_t*>(device_key_source) + bytes_per_launch,
        static_cast<uint8_t*>(device_value_source) + bytes_per_launch,
        static_cast<uint8_t*>(device_key_destination) + bytes_per_launch,
        static_cast<uint8_t*>(device_value_destination) + bytes_per_launch,
        device_output, kPayloadBytes, kStrideBytes, kRepeatCount, pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> output(output_values, -1);
    ACL_CHECK(aclrtMemcpy(
        output.data(), output_bytes, device_output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t block = 0; block < kBlockDim; ++block) {
        const size_t final_offset = static_cast<size_t>(
            bytes_per_launch
            + (block * kRepeatCount + kRepeatCount - 1) * kStrideBytes);
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
        const size_t output_offset = block * kOutputValuesPerBlock;
        const int64_t control_cycles = output[output_offset];
        const int64_t request_cycles = output[output_offset + 1];
        const int64_t sink = output[output_offset + 2];
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
            static_cast<long long>(kPayloadBytes),
            static_cast<long long>(kRepeatCount),
            static_cast<long long>(pair_order),
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
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(stderr, "usage: %s --batch pair_order...\n", argv[0]);
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    for (int index = 2; index < argc; ++index) {
        RunCell(stream, ParsePairOrder(argv[index]));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
