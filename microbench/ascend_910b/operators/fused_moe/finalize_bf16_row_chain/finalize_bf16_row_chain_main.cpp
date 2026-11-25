// ACL host for the non-production BF16 indexed-row chain primitive.
#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define ACL_CHECK(call) do {                                               \
    const int status = static_cast<int>(call);                             \
    if (status != 0) {                                                     \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",      \
                     __FILE__, __LINE__, #call, status);                  \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

extern "C" uint32_t aclrtlaunch_finalize_bf16_row_chain(
    uint32_t block_dim, void* stream, void* gm_source, void* gm_indices,
    void* gm_output, void* gm_results, int64_t row_elements,
    int64_t source_groups, int64_t iterations, int64_t source_mode);

namespace {

constexpr int64_t kResultStrideElements = 8;
constexpr int64_t kRowsPerOutput = 2;
constexpr int64_t kSourceGroups = 4096;
constexpr double kSystemCounterTickNs = 20.0;

int64_t ParseInteger(const char* text, const char* label) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer, got %s\n", label, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

bool IsSupportedRows(int64_t value) {
    return value == 6080 || value == 6208;
}

struct Cell {
    int64_t row_elements;
    int64_t source_mode;
    int64_t block_dim;
    int64_t iterations;
};

size_t CheckedBytes(int64_t elements, size_t element_bytes, const char* label) {
    if (elements <= 0 || static_cast<uint64_t>(elements)
            > std::numeric_limits<size_t>::max() / element_bytes) {
        std::fprintf(stderr, "%s allocation overflows\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(elements) * element_bytes;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr || *device_text == '\0'
            || ParseInteger(device_text, "ASCEND_DEVICE_ID") != 0) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be remapped logical device 0\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
}

int main(int argc, char** argv) {
    if (argc < 6 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 4 != 0) {
        std::fprintf(stderr,
            "usage: %s --batch ROW_ELEMENTS MODE BLOCK_DIM ITERATIONS [cell ...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    for (int argument = 2; argument < argc; argument += 4) {
        const Cell cell = {
            ParseInteger(argv[argument], "row_elements"),
            ParseInteger(argv[argument + 1], "source_mode"),
            ParseInteger(argv[argument + 2], "block_dim"),
            ParseInteger(argv[argument + 3], "iterations"),
        };
        if (!IsSupportedRows(cell.row_elements)
                || cell.source_mode < 0 || cell.source_mode > 1
                || cell.block_dim != 2 || cell.iterations <= 0
                || cell.iterations * 4 > kSourceGroups) {
            std::fprintf(stderr, "cell is outside the frozen source domain\n");
            return EXIT_FAILURE;
        }
        cells.push_back(cell);
    }

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    std::puts("row_elements,source_mode,block_dim,iterations,source_groups,"
              "short_max_cycles,long_max_cycles,cycles_per_row_chain,"
              "ns_per_row_chain,validation");
    for (const Cell& cell : cells) {
        const int64_t source_elements = cell.block_dim * kSourceGroups
            * kRowsPerOutput * cell.row_elements;
        const int64_t index_elements = cell.block_dim * kSourceGroups
            * kRowsPerOutput;
        const int64_t output_elements = cell.block_dim * cell.iterations * 4
            * cell.row_elements;
        void* source = nullptr;
        void* indices = nullptr;
        void* output = nullptr;
        void* results = nullptr;
        const size_t source_bytes = CheckedBytes(source_elements, sizeof(uint16_t), "source");
        const size_t index_bytes = CheckedBytes(index_elements, sizeof(int32_t), "indices");
        const size_t output_bytes = CheckedBytes(output_elements, sizeof(uint16_t), "output");
        ACL_CHECK(aclrtMalloc(&source, source_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(&indices, index_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(&results,
            CheckedBytes(cell.block_dim * kResultStrideElements, sizeof(int64_t), "results"),
            ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(source, source_bytes, 0x3c, source_bytes));
        ACL_CHECK(aclrtMemset(output, output_bytes, 0, output_bytes));
        std::vector<int32_t> host_indices(static_cast<size_t>(index_elements));
        for (int64_t item = 0; item < index_elements; item += kRowsPerOutput) {
            host_indices[static_cast<size_t>(item)] = 0;
            host_indices[static_cast<size_t>(item + 1)] = 1;
        }
        ACL_CHECK(aclrtMemcpy(indices, index_bytes, host_indices.data(), index_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));

        const auto run = [&](int64_t iterations) {
            ACL_CHECK(aclrtMemset(results,
                CheckedBytes(cell.block_dim * kResultStrideElements, sizeof(int64_t), "results"),
                0, CheckedBytes(cell.block_dim * kResultStrideElements, sizeof(int64_t), "results")));
            ACL_CHECK(aclrtlaunch_finalize_bf16_row_chain(
                static_cast<uint32_t>(cell.block_dim), stream, source, indices,
                output, results, cell.row_elements, kSourceGroups, iterations,
                cell.source_mode));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<int64_t> host_results(
                static_cast<size_t>(cell.block_dim * kResultStrideElements));
            ACL_CHECK(aclrtMemcpy(host_results.data(),
                host_results.size() * sizeof(int64_t), results,
                host_results.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST));
            int64_t maximum_cycles = 0;
            for (int64_t worker = 0; worker < cell.block_dim; ++worker) {
                const size_t offset = static_cast<size_t>(worker * kResultStrideElements);
                if (host_results[offset] <= 0 || host_results[offset + 1] != iterations
                        || host_results[offset + 2] != cell.source_mode) {
                    std::fprintf(stderr, "row-chain validation failed\n");
                    std::exit(EXIT_FAILURE);
                }
                maximum_cycles = std::max(maximum_cycles, host_results[offset]);
            }
            uint16_t final_value = 0;
            const size_t final_offset = static_cast<size_t>(
                (cell.block_dim * iterations - 1) * cell.row_elements * sizeof(uint16_t));
            ACL_CHECK(aclrtMemcpy(&final_value, sizeof(final_value),
                static_cast<uint8_t*>(output) + final_offset, sizeof(final_value),
                ACL_MEMCPY_DEVICE_TO_HOST));
            if (final_value == 0) {
                std::fprintf(stderr, "row-chain output validation failed\n");
                std::exit(EXIT_FAILURE);
            }
            return maximum_cycles;
        };

        run(cell.iterations);
        const int64_t short_cycles = run(cell.iterations);
        const int64_t long_cycles = run(cell.iterations * 4);
        if (long_cycles <= short_cycles) {
            std::fprintf(stderr, "row-chain differential interval did not increase\n");
            return EXIT_FAILURE;
        }
        const double cycles_per_chain = static_cast<double>(long_cycles - short_cycles)
            / (3.0 * cell.iterations);
        const double ns_per_chain = cycles_per_chain * kSystemCounterTickNs;
        if (!std::isfinite(ns_per_chain) || ns_per_chain <= 0.0) {
            std::fprintf(stderr, "row-chain duration is invalid\n");
            return EXIT_FAILURE;
        }
        std::printf("%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.9f,%.9f,pass\n",
            static_cast<long>(cell.row_elements), static_cast<long>(cell.source_mode),
            static_cast<long>(cell.block_dim), static_cast<long>(cell.iterations),
            static_cast<long>(kSourceGroups), static_cast<long>(short_cycles),
            static_cast<long>(long_cycles), cycles_per_chain, ns_per_chain);
        ACL_CHECK(aclrtFree(results));
        ACL_CHECK(aclrtFree(output));
        ACL_CHECK(aclrtFree(indices));
        ACL_CHECK(aclrtFree(source));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
