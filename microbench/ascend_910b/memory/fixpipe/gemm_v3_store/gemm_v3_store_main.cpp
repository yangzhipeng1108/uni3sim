#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "gemm_v3_store_contract.h"

#define ACL_CHECK(call) do {                                             \
    int status = static_cast<int>(call);                                 \
    if (status != 0) {                                                   \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",   \
                     __FILE__, __LINE__, #call, status);                 \
        std::exit(EXIT_FAILURE);                                         \
    }                                                                    \
} while (0)

extern "C" uint32_t aclrtlaunch_gemm_v3_store(
    uint32_t block_dim, void* stream, void* gm_a, void* gm_b,
    void* gm_destination, void* gm_cycles, int64_t dtype, int64_t m,
    int64_t n, int64_t invocations);

namespace {

struct Cell {
    int64_t dtype;
    int64_t m;
    int64_t n;
    int64_t block_dim;
};

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

uint16_t FloatToBfloat16(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be binary32");
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

std::string JoinCycles(const std::vector<int64_t>& values) {
    std::string result;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            result += ';';
        }
        result += std::to_string(values[index]);
    }
    return result;
}

void ValidateValue(
        void* device_destination, const Cell& cell, int64_t invocations,
        int64_t block_index, int64_t invocation, int64_t row,
        int64_t column) {
    const int64_t tile_elements = cell.m * cell.n;
    const int64_t block_elements = tile_elements * (invocations + 1);
    const int64_t element_index = block_index * block_elements
        + invocation * tile_elements + row * cell.n + column;
    const int64_t element_bytes =
        gemm_v3_store_contract::output_element_bytes(cell.dtype);
    auto* source = static_cast<uint8_t*>(device_destination)
        + element_index * element_bytes;
    const float expected = static_cast<float>(16 * (row + 1));
    if (cell.dtype == gemm_v3_store_contract::kBfloat16) {
        uint16_t observed = 0;
        ACL_CHECK(aclrtMemcpy(
            &observed, sizeof(observed), source, sizeof(observed),
            ACL_MEMCPY_DEVICE_TO_HOST));
        if (observed != FloatToBfloat16(expected)) {
            std::fprintf(
                stderr,
                "BF16 validation failed at block=%lld invocation=%lld "
                "row=%lld column=%lld\n",
                static_cast<long long>(block_index),
                static_cast<long long>(invocation),
                static_cast<long long>(row),
                static_cast<long long>(column));
            std::exit(EXIT_FAILURE);
        }
    } else {
        float observed = 0.0f;
        ACL_CHECK(aclrtMemcpy(
            &observed, sizeof(observed), source, sizeof(observed),
            ACL_MEMCPY_DEVICE_TO_HOST));
        if (observed != expected) {
            std::fprintf(
                stderr,
                "FP32 validation failed at block=%lld invocation=%lld "
                "row=%lld column=%lld\n",
                static_cast<long long>(block_index),
                static_cast<long long>(invocation),
                static_cast<long long>(row),
                static_cast<long long>(column));
            std::exit(EXIT_FAILURE);
        }
    }
}

std::vector<int64_t> RunInterval(
        aclrtStream stream, void* device_a, void* device_b,
        void* device_destination, size_t destination_bytes,
        void* device_cycles, const Cell& cell, int64_t invocations) {
    ACL_CHECK(aclrtMemset(
        device_destination, destination_bytes, 0xff, destination_bytes));
    ACL_CHECK(aclrtMemset(
        device_cycles,
        gemm_v3_store_contract::kMaximumBlocks
            * gemm_v3_store_contract::kCycleCacheLineBytes,
        0xff,
        gemm_v3_store_contract::kMaximumBlocks
            * gemm_v3_store_contract::kCycleCacheLineBytes));
    ACL_CHECK(aclrtlaunch_gemm_v3_store(
        static_cast<uint32_t>(cell.block_dim), stream, device_a, device_b,
        device_destination, device_cycles, cell.dtype, cell.m, cell.n,
        invocations));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<int64_t> padded(
        static_cast<size_t>(gemm_v3_store_contract::kMaximumBlocks
            * gemm_v3_store_contract::kCycleSlotElements), -1);
    ACL_CHECK(aclrtMemcpy(
        padded.data(), padded.size() * sizeof(int64_t), device_cycles,
        padded.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<int64_t> cycles;
    cycles.reserve(static_cast<size_t>(cell.block_dim));
    for (int64_t block = 0; block < cell.block_dim; ++block) {
        const int64_t value = padded[static_cast<size_t>(
            block * gemm_v3_store_contract::kCycleSlotElements)];
        if (value <= 0) {
            std::fprintf(stderr, "invalid cycle count %lld\n",
                static_cast<long long>(value));
            std::exit(EXIT_FAILURE);
        }
        cycles.push_back(value);
        for (int64_t invocation : {int64_t{0}, invocations - 1}) {
            ValidateValue(
                device_destination, cell, invocations, block, invocation,
                0, 0);
            ValidateValue(
                device_destination, cell, invocations, block, invocation,
                cell.m - 1, cell.n - 1);
        }
    }
    return cycles;
}

}  // namespace

int main(int argc, char** argv) {
    const bool steady_campaign = argc >= 2
        && std::strcmp(argv[1], "--batch") == 0;
    const bool finite_campaign = argc >= 2
        && std::strcmp(argv[1], "--finite-batch") == 0;
    const bool long_stream_campaign = argc >= 2
        && std::strcmp(argv[1], "--long-stream-batch") == 0;
    if (argc < 6
            || (!steady_campaign && !finite_campaign && !long_stream_campaign)
            || (argc - 2) % 4 != 0) {
        std::fprintf(
            stderr, "usage: %s (--batch|--finite-batch|--long-stream-batch) "
            "dtype m n blockDim [cell ...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || ParseInteger(device_text, "ASCEND_DEVICE_ID") != 0) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must be logical device 0 after visibility "
            "remapping\n");
        return EXIT_FAILURE;
    }

    std::vector<Cell> cells;
    for (int argument = 2; argument < argc; argument += 4) {
        Cell cell{
            ParseInteger(argv[argument], "dtype"),
            ParseInteger(argv[argument + 1], "m"),
            ParseInteger(argv[argument + 2], "n"),
            ParseInteger(argv[argument + 3], "blockDim"),
        };
        const bool valid = steady_campaign
            ? gemm_v3_store_contract::is_frozen_cell(
                cell.dtype, cell.m, cell.n, cell.block_dim)
            : finite_campaign
                ? gemm_v3_store_contract::is_frozen_finite_cell(
                    cell.dtype, cell.m, cell.n, cell.block_dim)
                : gemm_v3_store_contract::is_frozen_long_stream_cell(
                    cell.dtype, cell.m, cell.n, cell.block_dim);
        if (!valid) {
            std::fprintf(stderr, "cell is outside the frozen campaign\n");
            return EXIT_FAILURE;
        }
        cells.push_back(cell);
    }

    constexpr size_t kMaximumAElements =
        gemm_v3_store_contract::kMaximumM * gemm_v3_store_contract::kK;
    constexpr size_t kMaximumBElements =
        gemm_v3_store_contract::kMaximumN * gemm_v3_store_contract::kK;
    std::vector<uint16_t> host_a(kMaximumAElements, 0);
    std::vector<uint16_t> host_b(kMaximumBElements, FloatToBfloat16(1.0f));
    for (int64_t row = 0; row < gemm_v3_store_contract::kMaximumM; ++row) {
        const uint16_t value = FloatToBfloat16(static_cast<float>(row + 1));
        for (int64_t k = 0; k < gemm_v3_store_contract::kK; ++k) {
            host_a[static_cast<size_t>(row * gemm_v3_store_contract::kK + k)] = value;
        }
    }

    size_t maximum_destination_bytes = 0;
    const int64_t short_invocations = steady_campaign
        ? gemm_v3_store_contract::kShortInvocations
        : finite_campaign
            ? gemm_v3_store_contract::kFiniteShortInvocations
            : gemm_v3_store_contract::kLongStreamShortInvocations;
    const int64_t long_invocations = steady_campaign
        ? gemm_v3_store_contract::kLongInvocations
        : finite_campaign
            ? gemm_v3_store_contract::kFiniteLongInvocations
            : gemm_v3_store_contract::kLongStreamLongInvocations;
    for (const Cell& cell : cells) {
        const int64_t bytes = gemm_v3_store_contract::block_elements(
            cell.m, cell.n, long_invocations)
            * cell.block_dim
            * gemm_v3_store_contract::output_element_bytes(cell.dtype);
        if (bytes <= 0
                || static_cast<uint64_t>(bytes)
                    > std::numeric_limits<size_t>::max()) {
            std::fprintf(stderr, "destination extent overflow\n");
            return EXIT_FAILURE;
        }
        maximum_destination_bytes = std::max(
            maximum_destination_bytes, static_cast<size_t>(bytes));
    }

    void* device_a = nullptr;
    void* device_b = nullptr;
    void* device_destination = nullptr;
    void* device_cycles = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    ACL_CHECK(aclrtMalloc(
        &device_a, host_a.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_b, host_b.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_destination, maximum_destination_bytes,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_cycles,
        gemm_v3_store_contract::kMaximumBlocks
            * gemm_v3_store_contract::kCycleCacheLineBytes,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_a, host_a.size() * sizeof(uint16_t), host_a.data(),
        host_a.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(
        device_b, host_b.size() * sizeof(uint16_t), host_b.data(),
        host_b.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::puts(
        "dtype,m,n,block_dim,output_bytes,l0c_bytes,short_invocations,"
        "long_invocations,short_cycles,long_cycles,validation");
    for (const Cell& cell : cells) {
        const std::vector<int64_t> short_cycles = RunInterval(
            stream, device_a, device_b, device_destination,
            maximum_destination_bytes, device_cycles, cell, short_invocations);
        const std::vector<int64_t> long_cycles = RunInterval(
            stream, device_a, device_b, device_destination,
            maximum_destination_bytes, device_cycles, cell, long_invocations);
        std::printf(
            "%s,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%s,%s,exact\n",
            cell.dtype == gemm_v3_store_contract::kBfloat16 ? "bf16" : "fp32",
            static_cast<long long>(cell.m),
            static_cast<long long>(cell.n),
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(cell.m * cell.n
                * gemm_v3_store_contract::output_element_bytes(cell.dtype)),
            static_cast<long long>(cell.m * cell.n * sizeof(float)),
            static_cast<long long>(short_invocations),
            static_cast<long long>(long_invocations),
            JoinCycles(short_cycles).c_str(), JoinCycles(long_cycles).c_str());
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtFree(device_destination));
    ACL_CHECK(aclrtFree(device_b));
    ACL_CHECK(aclrtFree(device_a));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
