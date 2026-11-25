// Host launcher for the CANN 8.5.1 FIA local-row-sum mechanism mirror.
#include <acl/acl.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "block_reduce_sum_contract.h"

#define ACL_CHECK(call) do {                                               \
    int status = static_cast<int>(call);                                   \
    if (status != 0) {                                                     \
        std::fprintf(stderr, "ACL error at %s:%d: %s returned %d\n",     \
                     __FILE__, __LINE__, #call, status);                   \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

extern "C" uint32_t aclrtlaunch_block_reduce_sum_service(
    uint32_t block_dim, void* stream, void* cycles, void* validation,
    uint32_t row_chunk_round, uint32_t column_count,
    uint32_t column_alignment, uint32_t iterations);

namespace {

using namespace block_reduce_sum_contract;

uint32_t ParsePositive(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    unsigned long value = std::strtoul(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value == 0 || value > UINT32_MAX) {
        std::fprintf(stderr, "%s must be a positive uint32, got %s\n",
                     name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<uint32_t>(value);
}

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must be explicit\n");
        std::exit(EXIT_FAILURE);
    }
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < 0 || value > INT32_MAX) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID is invalid: %s\n", text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

struct Cell {
    uint32_t row_chunk_round;
    uint32_t column_count;
    uint32_t column_alignment;
    uint32_t iterations;
};

Cell ParseCell(char** argv, int offset) {
    Cell cell{
        ParsePositive(argv[offset], "row_chunk_round"),
        ParsePositive(argv[offset + 1], "column_count"),
        ParsePositive(argv[offset + 2], "column_alignment"),
        ParsePositive(argv[offset + 3], "iterations"),
    };
    if (!IsSupportedCell(
            cell.row_chunk_round, cell.column_count,
            cell.column_alignment)) {
        std::fprintf(
            stderr,
            "unsupported cell rows=%u columns=%u alignment=%u\n",
            cell.row_chunk_round, cell.column_count,
            cell.column_alignment);
        std::exit(EXIT_FAILURE);
    }
    return cell;
}

void RunCell(
        aclrtStream stream, void* cycles_device, void* validation_device,
        const Cell& cell) {
    ACL_CHECK(aclrtMemset(cycles_device, 64, 0xff, 64));
    ACL_CHECK(aclrtMemset(
        validation_device, kRowOutputElements * sizeof(float), 0,
        kRowOutputElements * sizeof(float)));
    ACL_CHECK(aclrtlaunch_block_reduce_sum_service(
        1, stream, cycles_device, validation_device,
        cell.row_chunk_round, cell.column_count,
        cell.column_alignment, cell.iterations));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    int64_t system_cycles = 0;
    std::vector<float> validation(cell.row_chunk_round);
    ACL_CHECK(aclrtMemcpy(
        &system_cycles, sizeof(system_cycles), cycles_device,
        sizeof(system_cycles), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(
        validation.data(), validation.size() * sizeof(float),
        validation_device, validation.size() * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (system_cycles <= 0) {
        std::fprintf(stderr, "device returned invalid system cycles\n");
        std::exit(EXIT_FAILURE);
    }
    const float expected = static_cast<float>(cell.column_count);
    for (uint32_t row = 0; row < cell.row_chunk_round; ++row) {
        if (!std::isfinite(validation[row])
                || std::fabs(validation[row] - expected) > 1.0e-3f) {
            std::fprintf(
                stderr,
                "row-sum validation failed at row %u: got %.9f expected %.9f\n",
                row, validation[row], expected);
            std::exit(EXIT_FAILURE);
        }
    }
    const double per_iteration_ns =
        static_cast<double>(system_cycles) * kSystemCounterTickNs
        / cell.iterations;
    std::printf(
        "%u,%u,%u,%u,%u,%ld,%.9f,%.6f,%.6f,pass\n",
        cell.row_chunk_round, cell.column_count,
        RoundUp(cell.column_count, cell.column_alignment),
        cell.column_alignment, cell.iterations,
        static_cast<long>(system_cycles), per_iteration_ns,
        validation.front(), validation.back());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6 || std::strcmp(argv[1], "--batch") != 0
            || (argc - 2) % 4 != 0) {
        std::fprintf(
            stderr,
            "usage: %s --batch ROWS COLUMNS ALIGNMENT ITERATIONS [...]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    for (int offset = 2; offset < argc; offset += 4) {
        cells.push_back(ParseCell(argv, offset));
    }

    const int device_id = DeviceId();
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* cycles_device = nullptr;
    void* validation_device = nullptr;
    ACL_CHECK(aclrtMalloc(
        &cycles_device, 64, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &validation_device, kRowOutputElements * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST));

    std::puts(
        "row_chunk_round,column_count,column_count_round,column_alignment,"
        "iterations,system_cycles,per_iteration_ns,first_row_sum,"
        "last_row_sum,validation");
    for (const Cell& cell : cells) {
        RunCell(stream, cycles_device, validation_device, cell);
    }

    ACL_CHECK(aclrtFree(validation_device));
    ACL_CHECK(aclrtFree(cycles_device));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
