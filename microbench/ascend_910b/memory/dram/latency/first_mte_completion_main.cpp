// Host driver for the first AIV MTE request-to-completion benchmark.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

extern "C" uint32_t aclrtlaunch_first_mte_completion(
    uint32_t blockDim, void* stream, void* tiling, void* source,
    void* destination, void* output, int64_t direction,
    int64_t tile_bytes, int64_t stride_bytes, int64_t pair_order);

constexpr size_t kOutputValuesPerBlock = 8;
constexpr int64_t kMaximumBlockDim = 48;
constexpr int64_t kMaximumTileBytes = 32768;
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
    int64_t direction;
    int64_t tile_bytes;
    int64_t pair_order;
};

static Cell parse_cell(const char* text) {
    const std::string value(text);
    size_t positions[3];
    positions[0] = value.find(':');
    positions[1] = value.find(':', positions[0] + 1);
    positions[2] = value.find(':', positions[1] + 1);
    if (positions[0] == std::string::npos
            || positions[1] == std::string::npos
            || positions[2] == std::string::npos
            || value.find(':', positions[2] + 1) != std::string::npos) {
        std::fprintf(
            stderr,
            "cell must have block_dim:direction:tile_bytes:pair_order form\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t block_dim = parse_nonnegative(
        value.substr(0, positions[0]).c_str(), "block_dim");
    const int64_t direction = parse_nonnegative(
        value.substr(positions[0] + 1,
                     positions[1] - positions[0] - 1).c_str(), "direction");
    const int64_t tile_bytes = parse_nonnegative(
        value.substr(positions[1] + 1,
                     positions[2] - positions[1] - 1).c_str(), "tile_bytes");
    const int64_t pair_order = parse_nonnegative(
        value.substr(positions[2] + 1).c_str(), "pair_order");
    if (block_dim <= 0 || block_dim > kMaximumBlockDim) {
        std::fprintf(stderr, "block_dim must be in 1..48\n");
        std::exit(EXIT_FAILURE);
    }
    if (direction < 0 || direction > 8) {
        std::fprintf(
            stderr,
            "direction must be 0 (MTE2->MTE3), 1 (MTE3->MTE2), "
            "2 (MTE2->V), 3 (MTE3->V), 4 (TQue MTE2->V), or "
            "5 (TQue V->MTE3), 6 (TQue DataCopyPad MTE2->V), or "
            "7 (TQue DataCopyPad V->MTE3), or "
            "8 (Gather index DataCopyPad MTE2->Scalar->GetValue)\n");
        std::exit(EXIT_FAILURE);
    }
    const bool valid_tile_bytes = (
        direction == 8
        ? (tile_bytes >= 4 && tile_bytes <= 32 && tile_bytes % 4 == 0)
        : (tile_bytes > 0 && tile_bytes <= kMaximumTileBytes
           && tile_bytes % 32 == 0));
    if (!valid_tile_bytes) {
        std::fprintf(
            stderr,
            "tile_bytes must be 4-byte aligned in [4,32] for direction 8 "
            "or a multiple of 32 in [32,32768] otherwise\n");
        std::exit(EXIT_FAILURE);
    }
    if (pair_order < 0 || pair_order > 1) {
        std::fprintf(stderr, "pair_order must be zero or one\n");
        std::exit(EXIT_FAILURE);
    }
    return Cell{block_dim, direction, tile_bytes, pair_order};
}

static void run_cell(aclrtStream stream, void* device_tiling, Cell cell) {
    const size_t stride_bytes = static_cast<size_t>(kMaximumTileBytes);
    const size_t buffer_bytes = (
        static_cast<size_t>(cell.block_dim) * stride_bytes);
    std::vector<uint8_t> host_source(buffer_bytes, 0x5a);
    void* device_source = nullptr;
    void* device_destination = nullptr;
    void* device_output = nullptr;
    const size_t output_values = (
        static_cast<size_t>(cell.block_dim) * kOutputValuesPerBlock);
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
    ACL_CHECK(aclrtMemset(
        device_destination, buffer_bytes, 0, buffer_bytes));

    ACL_CHECK(aclrtlaunch_first_mte_completion(
        static_cast<uint32_t>(cell.block_dim), stream, device_tiling,
        device_source, device_destination, device_output, cell.direction,
        cell.tile_bytes, static_cast<int64_t>(stride_bytes),
        cell.pair_order));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> output(output_values, -1);
    ACL_CHECK(aclrtMemcpy(
        output.data(), output_bytes, device_output, output_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<uint16_t> destination_heads(
        static_cast<size_t>(cell.block_dim), 0);
    for (int64_t block = 0; block < cell.block_dim; ++block) {
        ACL_CHECK(aclrtMemcpy(
            &destination_heads[block], sizeof(uint16_t),
            static_cast<uint8_t*>(device_destination)
                + static_cast<size_t>(block) * stride_bytes,
            sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    }
    for (int64_t block = 0; block < cell.block_dim; ++block) {
        const size_t output_offset = (
            static_cast<size_t>(block) * kOutputValuesPerBlock);
        const int64_t control_cycles = output[output_offset];
        const int64_t request_cycles = output[output_offset + 1];
        const int64_t sink = output[output_offset + 2];
        const int64_t expected_sink = (
            kExpectedTilingSum
            + ((cell.direction == 0 || cell.direction == 2
                || cell.direction == 4 || cell.direction == 6) ? 0x5a : 0)
            + (cell.direction == 8
               ? (cell.tile_bytes / 4) * 0x5a5a5a5aLL : 0));
        const uint16_t expected_destination = (
            (cell.direction == 1 || cell.direction == 3
                || cell.direction == 5 || cell.direction == 7) ? 0x3c00 : 0);
        const bool valid = (
            control_cycles >= 0 && request_cycles > 0
            && sink == expected_sink
            && destination_heads[block] == expected_destination);
        if (!valid) {
            std::fprintf(
                stderr,
                "validation failed block=%lld direction=%lld bytes=%lld "
                "order=%lld control_cycles=%lld request_cycles=%lld "
                "sink=%lld destination=%u\n",
                static_cast<long long>(block),
                static_cast<long long>(cell.direction),
                static_cast<long long>(cell.tile_bytes),
                static_cast<long long>(cell.pair_order),
                static_cast<long long>(control_cycles),
                static_cast<long long>(request_cycles),
                static_cast<long long>(sink),
                static_cast<unsigned>(destination_heads[block]));
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%lld,%zu,%lld,%lld,%.9f,%.9f,%lld,%u,pass\n",
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(block),
            static_cast<long long>(cell.direction),
            static_cast<long long>(cell.tile_bytes),
            static_cast<long long>(cell.pair_order), stride_bytes,
            static_cast<long long>(control_cycles),
            static_cast<long long>(request_cycles),
            static_cast<double>(control_cycles) * kSystemCounterTickNs,
            static_cast<double>(request_cycles) * kSystemCounterTickNs,
            static_cast<long long>(sink),
            static_cast<unsigned>(destination_heads[block]));
    }

    ACL_CHECK(aclrtFree(device_source));
    ACL_CHECK(aclrtFree(device_destination));
    ACL_CHECK(aclrtFree(device_output));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(
            stderr,
            "usage: %s --batch block_dim:direction:tile_bytes:pair_order...\n",
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
