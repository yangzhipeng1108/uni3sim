// Host driver for the Ascend 910B RoPE key-24 steady-edge probe.
#include <acl/acl.h>

#include <cerrno>
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

extern "C" uint32_t aclrtlaunch_rope_key24_steady_edges(
    uint32_t blockDim, void* stream, void* source, void* destination,
    void* output, int64_t mode, int64_t payload_bytes,
    int64_t request_count, int64_t pair_order);

namespace {

constexpr int64_t kBlockDim = 47;
constexpr int64_t kWorkerStrideBytes = 131072;
constexpr size_t kOutputValuesPerBlock = 8;
constexpr double kSystemCounterTickNs = 20.0;

struct Cell {
    int64_t mode;
    int64_t payload_bytes;
    int64_t request_count;
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
    std::vector<int64_t> fields;
    size_t begin = 0;
    while (true) {
        const size_t separator = value.find(':', begin);
        fields.push_back(ParseNonnegative(
            value.substr(begin, separator - begin).c_str(), "cell field"));
        if (separator == std::string::npos) {
            break;
        }
        begin = separator + 1;
    }
    if (fields.size() != 4) {
        std::fprintf(
            stderr, "cell must have mode:payload:count:pair_order form\n");
        std::exit(EXIT_FAILURE);
    }
    const Cell cell{fields[0], fields[1], fields[2], fields[3]};
    const bool valid_payload = (
        cell.payload_bytes == 128 || cell.payload_bytes == 2048
        || cell.payload_bytes == 4096);
    const bool valid_count = (
        cell.request_count == 1 || cell.request_count == 2
        || cell.request_count == 4 || cell.request_count == 8
        || cell.request_count == 20 || cell.request_count == 21);
    if (cell.pair_order < 0 || cell.pair_order > 1
            || (cell.mode == 0
                && (!valid_payload || cell.request_count != 1))
            || (cell.mode == 1
                && (cell.payload_bytes != 4096 || !valid_count))
            || (cell.mode == 2
                && (!valid_payload || cell.request_count != 1))
            || (cell.mode == 3
                && (cell.payload_bytes != 0 || cell.request_count != 16))
            || cell.mode < 0 || cell.mode > 3) {
        std::fprintf(stderr, "unsupported probe cell\n");
        std::exit(EXIT_FAILURE);
    }
    return cell;
}

void RunCell(aclrtStream stream, Cell cell) {
    const size_t buffer_bytes = kBlockDim * kWorkerStrideBytes;
    std::vector<uint16_t> host_source(
        buffer_bytes / sizeof(uint16_t), 0x3c00);
    std::vector<uint16_t> host_destination(
        buffer_bytes / sizeof(uint16_t), 0);
    void* device_source = nullptr;
    void* device_destination = nullptr;
    void* device_output = nullptr;
    const size_t output_values = kBlockDim * kOutputValuesPerBlock;
    const size_t output_bytes = output_values * sizeof(int64_t);
    ACL_CHECK(aclrtMalloc(
        &device_source, buffer_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_destination, buffer_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_source, buffer_bytes, host_source.data(), buffer_bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(device_destination, buffer_bytes, 0, buffer_bytes));

    ACL_CHECK(aclrtlaunch_rope_key24_steady_edges(
        static_cast<uint32_t>(kBlockDim), stream, device_source,
        device_destination, device_output, cell.mode, cell.payload_bytes,
        cell.request_count, cell.pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> output(output_values, -1);
    ACL_CHECK(aclrtMemcpy(
        output.data(), output_bytes, device_output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (cell.mode == 2) {
        ACL_CHECK(aclrtMemcpy(
            host_destination.data(), buffer_bytes, device_destination,
            buffer_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    for (int64_t worker = 0; worker < kBlockDim; ++worker) {
        const size_t output_offset = worker * kOutputValuesPerBlock;
        const int64_t control_cycles = output[output_offset];
        const int64_t request_cycles = output[output_offset + 1];
        const int64_t sink = output[output_offset + 2];
        bool valid = control_cycles >= 0 && request_cycles > 0 && sink > 0;
        if (cell.mode == 2) {
            const size_t destination_offset = (
                worker * kWorkerStrideBytes / sizeof(uint16_t));
            valid = valid && host_destination[destination_offset] == 0x3c00;
        }
        if (!valid) {
            std::fprintf(
                stderr,
                "validation failed at worker %lld mode=%lld payload=%lld "
                "count=%lld pair_order=%lld control=%lld request=%lld "
                "sink=%lld\n",
                static_cast<long long>(worker),
                static_cast<long long>(cell.mode),
                static_cast<long long>(cell.payload_bytes),
                static_cast<long long>(cell.request_count),
                static_cast<long long>(cell.pair_order),
                static_cast<long long>(control_cycles),
                static_cast<long long>(request_cycles),
                static_cast<long long>(sink));
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.9f,%.9f,%lld,pass\n",
            static_cast<long long>(kBlockDim),
            static_cast<long long>(worker),
            static_cast<long long>(cell.mode),
            static_cast<long long>(cell.payload_bytes),
            static_cast<long long>(cell.request_count),
            static_cast<long long>(cell.pair_order),
            static_cast<long long>(kWorkerStrideBytes),
            static_cast<long long>(control_cycles),
            static_cast<long long>(request_cycles),
            static_cast<double>(control_cycles) * kSystemCounterTickNs,
            static_cast<double>(request_cycles) * kSystemCounterTickNs,
            static_cast<long long>(sink));
    }

    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtFree(device_destination));
    ACL_CHECK(aclrtFree(device_output));
}

}  // namespace

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(
            stderr, "usage: %s --batch mode:payload:count:pair_order...\n",
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
