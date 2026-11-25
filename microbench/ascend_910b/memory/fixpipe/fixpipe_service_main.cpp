// Host driver for the independent FP32 L0C-to-GM Fixpipe service benchmark.
#include <acl/acl.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <vector>

#include "fixpipe_service_contract.h"

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

extern "C" uint32_t aclrtlaunch_fixpipe_service(
    uint32_t blockDim, void* stream, void* gm_a, void* gm_b,
    void* gm_destination, void* gm_cycles, int64_t m, int64_t n,
    int64_t src_stride, int64_t dst_stride, int64_t layout, int64_t mode,
    int64_t stream_depth, int64_t destination_slots, int64_t invocations);

static int64_t parse_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

constexpr int32_t kLogicalDeviceId = 0;

__attribute__((constructor(101))) static void early_init() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0') {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be set explicitly\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t parsed_device = parse_integer(
        device_text, "ASCEND_DEVICE_ID");
    if (parsed_device != kLogicalDeviceId) {
        std::fprintf(
            stderr,
            "ASCEND_DEVICE_ID must be logical device 0 after visibility "
            "remapping\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

struct Cell {
    int64_t layout;
    int64_t mode;
    int64_t stream_depth;
    int64_t m;
    int64_t n;
    int64_t src_stride;
    int64_t dst_stride;
    int64_t block_dim;
    int64_t destination_slots;
    int64_t base_invocations;
};

struct Interval {
    std::vector<int64_t> block_cycles;
    int64_t host_wall_ns;
};

static std::string join_cycles(const std::vector<int64_t>& values) {
    std::string result;
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            result += ';';
        }
        result += std::to_string(values[index]);
    }
    return result;
}

static uint16_t float_to_bfloat16(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be binary32");
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static size_t checked_multiply(size_t left, size_t right, const char* label) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        std::fprintf(stderr, "%s size multiplication overflow\n", label);
        std::exit(EXIT_FAILURE);
    }
    return left * right;
}

int main(int argc, char** argv) {
    if (argc < 12 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 10 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch layout mode depth m n srcStride dstStride "
            "blockDim destinationSlots baseInvocations [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int cell_count = (argc - 2) / 10;
    if (cell_count > 512) {
        std::fprintf(stderr, "a batch cannot contain more than 512 cells\n");
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(cell_count));
    size_t maximum_destination_bytes = 0;
    for (int argument = 2; argument < argc; argument += 10) {
        Cell cell{
            parse_integer(argv[argument], "layout"),
            parse_integer(argv[argument + 1], "mode"),
            parse_integer(argv[argument + 2], "stream_depth"),
            parse_integer(argv[argument + 3], "m"),
            parse_integer(argv[argument + 4], "n"),
            parse_integer(argv[argument + 5], "srcStride"),
            parse_integer(argv[argument + 6], "dstStride"),
            parse_integer(argv[argument + 7], "blockDim"),
            parse_integer(argv[argument + 8], "destinationSlots"),
            parse_integer(argv[argument + 9], "baseInvocations"),
        };
        const bool valid_shape =
            cell.m >= 1 && cell.m <= 128 && cell.n >= 1 && cell.n <= 128;
        const bool valid_depth = cell.stream_depth == 1;
        if (!valid_shape || cell.layout != 0 || cell.mode != 0 || !valid_depth
                || (cell.block_dim != 1 && cell.block_dim != 24)) {
            std::fprintf(stderr, "invalid frozen Fixpipe cell at argument %d\n", argument);
            return EXIT_FAILURE;
        }
        const int64_t expected_src_stride = (cell.m + 15) / 16 * 16;
        fixpipe_service_contract::InvocationCounts invocation_counts{0, 0};
        if (!fixpipe_service_contract::frozen_invocation_counts(
                cell.m, cell.n, cell.stream_depth, &invocation_counts)) {
            std::fprintf(stderr, "frozen invocation arithmetic overflow\n");
            return EXIT_FAILURE;
        }
        if (cell.src_stride != expected_src_stride
                || cell.dst_stride < cell.n || cell.dst_stride % 32 != 0
                || cell.dst_stride > 8192
                || cell.destination_slots <= 0
                || cell.destination_slots > invocation_counts.short_count
                || cell.base_invocations != invocation_counts.short_count) {
            std::fprintf(
                stderr, "cell changed frozen stride or invocation count\n");
            return EXIT_FAILURE;
        }
        int64_t tile_span_elements = 0;
        fixpipe_service_contract::DestinationExtent destination_extent{
            0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (!fixpipe_service_contract::checked_row_major_span(
                    cell.m, cell.n, cell.dst_stride, &tile_span_elements)
                || !fixpipe_service_contract::checked_destination_extent(
                    cell.block_dim - 1, invocation_counts.long_count,
                    cell.destination_slots,
                    tile_span_elements, &destination_extent)) {
            std::fprintf(stderr, "destination extent arithmetic overflow\n");
            return EXIT_FAILURE;
        }
        const size_t destination_bytes = checked_multiply(
            static_cast<size_t>(destination_extent.destination_end_offset),
            sizeof(float), "destination bytes");
        if (destination_bytes > maximum_destination_bytes) {
            maximum_destination_bytes = destination_bytes;
        }
        cells.push_back(cell);
    }

    constexpr int64_t kMaximumInputRows = 128;
    constexpr int64_t kOracleK = 16;
    constexpr size_t kMaximumInputElements =
        static_cast<size_t>(kMaximumInputRows * kOracleK);
    std::vector<uint16_t> host_a(kMaximumInputElements, 0);
    const uint16_t oracle_one = float_to_bfloat16(1.0f);
    std::vector<uint16_t> host_b(kMaximumInputElements, oracle_one);
    for (int64_t row = 0; row < kMaximumInputRows; ++row) {
        const uint16_t row_value = float_to_bfloat16(
            static_cast<float>(row + 1));
        for (int64_t k_index = 0; k_index < kOracleK; ++k_index) {
            host_a[static_cast<size_t>(row * kOracleK + k_index)] = row_value;
        }
    }
    void* device_a = nullptr;
    void* device_b = nullptr;
    void* device_destination = nullptr;
    void* device_cycles = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_a, host_a.size() * sizeof(uint16_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_b, host_b.size() * sizeof(uint16_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &device_destination, maximum_destination_bytes,
        ACL_MEM_MALLOC_HUGE_FIRST));
    constexpr uintptr_t kDestinationAllocationAlignmentBytes =
        static_cast<uintptr_t>(
            fixpipe_service_contract::kBlockAlignmentElements)
        * sizeof(float);
    if (reinterpret_cast<uintptr_t>(device_destination)
            % kDestinationAllocationAlignmentBytes != 0) {
        std::fprintf(
            stderr, "destination allocation is not 4096-byte aligned\n");
        return EXIT_FAILURE;
    }
    ACL_CHECK(aclrtMalloc(
        &device_cycles,
        fixpipe_service_contract::kMaximumBlocks
            * fixpipe_service_contract::kCycleCacheLineBytes,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        device_a, host_a.size() * sizeof(uint16_t), host_a.data(),
        host_a.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(
        device_b, host_b.size() * sizeof(uint16_t), host_b.data(),
        host_b.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    auto run_interval = [&](const Cell& cell, int64_t invocations) {
        int64_t tile_span_elements = 0;
        fixpipe_service_contract::DestinationExtent last_extent{
            0, 0, 0, 0, 0, 0, 0, 0, 0};
        if (!fixpipe_service_contract::checked_row_major_span(
                    cell.m, cell.n, cell.dst_stride, &tile_span_elements)
                || !fixpipe_service_contract::checked_destination_extent(
                    cell.block_dim - 1, invocations,
                    cell.destination_slots, tile_span_elements,
                    &last_extent)) {
            std::fprintf(stderr, "output extent arithmetic overflow\n");
            std::exit(EXIT_FAILURE);
        }
        const size_t output_elements = static_cast<size_t>(
            last_extent.destination_end_offset);
        const size_t output_bytes = checked_multiply(
            output_elements, sizeof(float), "output bytes");
        if (output_bytes > maximum_destination_bytes) {
            std::fprintf(stderr, "output exceeds destination allocation\n");
            std::exit(EXIT_FAILURE);
        }
        ACL_CHECK(aclrtMemset(
            device_destination, output_bytes, 0xff, output_bytes));
        ACL_CHECK(aclrtMemset(
            device_cycles,
            fixpipe_service_contract::kMaximumBlocks
                * fixpipe_service_contract::kCycleCacheLineBytes,
            0xff,
            fixpipe_service_contract::kMaximumBlocks
                * fixpipe_service_contract::kCycleCacheLineBytes));
        struct timespec start;
        struct timespec end;
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &start));
        ACL_CHECK(aclrtlaunch_fixpipe_service(
            static_cast<uint32_t>(cell.block_dim), stream,
            device_a, device_b, device_destination, device_cycles,
            cell.m, cell.n, cell.src_stride, cell.dst_stride,
            cell.layout, cell.mode, cell.stream_depth,
            cell.destination_slots, invocations));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        CLOCK_CHECK(clock_gettime(CLOCK_MONOTONIC, &end));
        const int64_t wall_ns = static_cast<int64_t>(end.tv_sec - start.tv_sec)
            * 1000000000LL + static_cast<int64_t>(end.tv_nsec - start.tv_nsec);
        if (wall_ns <= 0) {
            std::fprintf(stderr, "host health interval must be positive\n");
            std::exit(EXIT_FAILURE);
        }

        std::vector<int64_t> padded_cycles(
            static_cast<size_t>(
                fixpipe_service_contract::kMaximumBlocks
                * fixpipe_service_contract::kCycleSlotElements),
            -1);
        ACL_CHECK(aclrtMemcpy(
            padded_cycles.data(), padded_cycles.size() * sizeof(int64_t),
            device_cycles, padded_cycles.size() * sizeof(int64_t),
            ACL_MEMCPY_DEVICE_TO_HOST));
        std::vector<int64_t> block_cycles(
            static_cast<size_t>(cell.block_dim), -1);
        for (size_t block_index = 0;
                block_index < block_cycles.size(); ++block_index) {
            block_cycles[block_index] = padded_cycles[
                block_index * static_cast<size_t>(
                    fixpipe_service_contract::kCycleSlotElements)];
        }
        for (size_t block_index = 0;
                block_index < block_cycles.size(); ++block_index) {
            const int64_t value = block_cycles[block_index];
            if (value <= 0) {
                std::fprintf(
                    stderr,
                    "Fixpipe returned invalid cycle count %ld at block %zu "
                    "for layout=%ld mode=%ld depth=%ld m=%ld n=%ld "
                    "blockDim=%ld invocations=%ld\n",
                    value, block_index, cell.layout, cell.mode,
                    cell.stream_depth, cell.m, cell.n, cell.block_dim,
                    invocations);
                std::exit(EXIT_FAILURE);
            }
        }
        std::vector<uint32_t> output(output_elements);
        ACL_CHECK(aclrtMemcpy(
            output.data(), output_bytes, device_destination, output_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        const uint32_t guard_value = 0xffffffffU;
        for (int64_t block = 0; block < cell.block_dim; ++block) {
            fixpipe_service_contract::DestinationExtent extent{
                0, 0, 0, 0, 0, 0, 0, 0, 0};
            if (!fixpipe_service_contract::checked_destination_extent(
                        block, invocations, cell.destination_slots,
                        tile_span_elements, &extent)) {
                std::fprintf(stderr, "validation extent arithmetic overflow\n");
                std::exit(EXIT_FAILURE);
            }
            for (int64_t slot = 0; slot <= extent.slot_count; ++slot) {
                const size_t tile_base = static_cast<size_t>(
                    slot == extent.slot_count
                        ? extent.warmup_destination_offset
                        : extent.block_offset
                            + slot * extent.slot_pitch_elements
                            + fixpipe_service_contract::
                                kDestinationBaseOffsetElements);
                const size_t leading_guard_base = tile_base
                    - static_cast<size_t>(
                        fixpipe_service_contract::kDestinationGuardElements);
                for (int64_t guard = 0;
                        guard < fixpipe_service_contract::kDestinationGuardElements;
                        ++guard) {
                    const size_t leading = leading_guard_base
                        + static_cast<size_t>(guard);
                    const size_t trailing = tile_base
                        + static_cast<size_t>(tile_span_elements + guard);
                    if (output[leading] != guard_value
                            || output[trailing] != guard_value) {
                        std::fprintf(
                            stderr,
                            "Fixpipe destination guard changed at block=%ld "
                            "slot=%ld guard=%ld\n",
                            block, slot, guard);
                        std::exit(EXIT_FAILURE);
                    }
                }
                for (int64_t row = 0; row < cell.m; ++row) {
                    for (int64_t column = 0; column < cell.n; ++column) {
                        const size_t physical = static_cast<size_t>(
                            row * cell.dst_stride + column);
                        if (physical >= static_cast<size_t>(tile_span_elements)) {
                            std::fprintf(stderr, "validation layout overflow\n");
                            std::exit(EXIT_FAILURE);
                        }
                        const float expected = static_cast<float>(
                            kOracleK * (row + 1));
                        uint32_t expected_bits = 0;
                        std::memcpy(
                            &expected_bits, &expected, sizeof(expected_bits));
                        const uint32_t observed_bits = output[
                            tile_base + physical];
                        if (observed_bits != expected_bits) {
                            std::fprintf(
                                stderr,
                                "Fixpipe tile validation failed at block=%ld "
                                "slot=%ld row=%ld column=%ld physical=%zu: "
                                "got 0x%08x expected 0x%08x\n",
                                block, slot, row, column, physical,
                                observed_bits, expected_bits);
                            std::exit(EXIT_FAILURE);
                        }
                    }
                    if (row + 1 < cell.m) {
                        for (int64_t gap = cell.n;
                                gap < cell.dst_stride; ++gap) {
                            const size_t physical = static_cast<size_t>(
                                row * cell.dst_stride + gap);
                            if (output[tile_base + physical] != guard_value) {
                                std::fprintf(
                                    stderr,
                                    "Fixpipe row gap changed at block=%ld "
                                    "slot=%ld row=%ld gap=%ld\n",
                                    block, slot, row, gap);
                                std::exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
            }
        }
        return Interval{block_cycles, wall_ns};
    };

    for (const Cell& cell : cells) {
        fixpipe_service_contract::InvocationCounts invocation_counts{0, 0};
        if (!fixpipe_service_contract::frozen_invocation_counts(
                cell.m, cell.n, cell.stream_depth, &invocation_counts)
                || cell.base_invocations != invocation_counts.short_count) {
            std::fprintf(stderr, "cell invocation contract changed\n");
            return EXIT_FAILURE;
        }
        const int64_t middle_invocations =
            invocation_counts.long_count
            / fixpipe_service_contract::kMiddleIntervalMultiplier;
        if (middle_invocations
                != fixpipe_service_contract::kMiddleIntervalMultiplier
                    * cell.base_invocations) {
            std::fprintf(stderr, "middle invocation contract changed\n");
            return EXIT_FAILURE;
        }
        const int64_t long_invocations = invocation_counts.long_count;
        size_t base_logical_bytes = checked_multiply(
            static_cast<size_t>(cell.base_invocations),
            static_cast<size_t>(cell.m), "base logical M");
        base_logical_bytes = checked_multiply(
            base_logical_bytes, static_cast<size_t>(cell.n),
            "base logical N");
        base_logical_bytes = checked_multiply(
            base_logical_bytes, sizeof(float), "base logical bytes");
        const size_t middle_logical_bytes = checked_multiply(
            base_logical_bytes, 2, "middle logical bytes");
        const size_t long_logical_bytes = checked_multiply(
            base_logical_bytes, 4, "long logical bytes");
        const Interval base_interval = run_interval(
            cell, cell.base_invocations);
        const Interval middle_interval = run_interval(
            cell, middle_invocations);
        const Interval long_interval = run_interval(cell, long_invocations);
        const char* layout_name = "row_major";
        const char* mode_name = "serial";
        std::printf(
            "%s,%ld,%ld,%ld,%ld,%s,%ld,%ld,%ld,%ld,%ld,%ld,"
            "%zu,%zu,%zu,%s,%s,%s,%ld,%ld,%ld,"
            "GetSystemCycle,pipe_fix_completion,pass\n",
            layout_name, cell.m, cell.n, cell.src_stride, cell.dst_stride,
            mode_name, cell.stream_depth, cell.block_dim,
            cell.destination_slots, cell.base_invocations,
            middle_invocations, long_invocations,
            base_logical_bytes, middle_logical_bytes, long_logical_bytes,
            join_cycles(base_interval.block_cycles).c_str(),
            join_cycles(middle_interval.block_cycles).c_str(),
            join_cycles(long_interval.block_cycles).c_str(),
            base_interval.host_wall_ns, middle_interval.host_wall_ns,
            long_interval.host_wall_ns);
        std::fflush(stdout);
    }

    ACL_CHECK(aclrtFree(device_a));
    ACL_CHECK(aclrtFree(device_b));
    ACL_CHECK(aclrtFree(device_destination));
    ACL_CHECK(aclrtFree(device_cycles));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
