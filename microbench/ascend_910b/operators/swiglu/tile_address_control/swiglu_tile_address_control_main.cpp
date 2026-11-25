// Host driver for the independent CANN SwiGLU tile-address control probe.
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

extern "C" uint32_t aclrtlaunch_swiglu_tile_address_control(
    uint32_t block_dim, void* stream, void* output, int64_t branch,
    int64_t mode, int64_t repetitions, int64_t column_tile_count,
    int64_t base_row_length, int64_t combined_column_length,
    int64_t base_column_length, int64_t column_length,
    int64_t base_row_tile_count, int64_t base_column_tile_count);

namespace {

constexpr int64_t kMaximumWorkers = 48;
constexpr int64_t kOutputValuesPerWorker = 8;
constexpr double kSystemCounterTickNs = 20.0;
// These values select legal CANN source control paths.  They deliberately
// cross the kernel boundary as dynamic parameters: CaclOneTileParam receives
// its fields from runtime tiling data, so baking them into this probe would
// let the compiler turn its dynamic quotient/remainder and address DAG into
// a different program.
constexpr int64_t kColumnTileCount = 128;
constexpr int64_t kBaseRowLength = 37;
constexpr int64_t kCombinedColumnLength = 12288;
constexpr int64_t kBaseColumnLength = 6144;
constexpr int64_t kColumnLength = 6144;
constexpr int64_t kBaseColumnTileCount = 64;

int64_t ParsePositive(const std::string& text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int64_t ParseNonnegative(const std::string& text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || value < 0) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

int DeviceId() {
    const char* text = std::getenv("ASCEND_DEVICE_ID");
    if (text == nullptr) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must select an idle NPU\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(
        ParseNonnegative(std::string(text), "ASCEND_DEVICE_ID"));
}

struct Cell {
    int64_t block_dim;
    int64_t branch;
    int64_t mode;
    int64_t repetitions;
};

Cell ParseCell(const char* text) {
    const std::string value(text);
    const size_t first = value.find(':');
    const size_t second = value.find(':', first + 1);
    const size_t third = value.find(':', second + 1);
    if (first == std::string::npos || second == std::string::npos
            || third == std::string::npos
            || value.find(':', third + 1) != std::string::npos) {
        std::fprintf(stderr,
                     "cell must be workers:branch:mode:repetitions\n");
        std::exit(EXIT_FAILURE);
    }
    const Cell cell{
        ParsePositive(value.substr(0, first), "workers"),
        ParsePositive(value.substr(first + 1, second - first - 1), "branch") - 1,
        ParsePositive(value.substr(second + 1, third - second - 1), "mode") - 1,
        ParsePositive(value.substr(third + 1), "repetitions"),
    };
    if (cell.block_dim > kMaximumWorkers || cell.branch > 3 || cell.mode > 1) {
        std::fprintf(stderr, "cell is outside the source-control domain\n");
        std::exit(EXIT_FAILURE);
    }
    return cell;
}

void RunCell(aclrtStream stream, const Cell cell) {
    const size_t output_values = static_cast<size_t>(cell.block_dim)
        * kOutputValuesPerWorker;
    const size_t output_bytes = output_values * sizeof(uint64_t);
    void* output = nullptr;
    ACL_CHECK(aclrtMalloc(&output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(output, output_bytes, 0, output_bytes));
    ACL_CHECK(aclrtlaunch_swiglu_tile_address_control(
        static_cast<uint32_t>(cell.block_dim), stream, output, cell.branch,
        cell.mode, cell.repetitions, kColumnTileCount, kBaseRowLength,
        kCombinedColumnLength, kBaseColumnLength, kColumnLength,
        cell.branch < 2 ? cell.repetitions + 1 : 1,
        kBaseColumnTileCount));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<uint64_t> host_output(output_values, 0);
    ACL_CHECK(aclrtMemcpy(host_output.data(), output_bytes, output,
                          output_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t worker = 0; worker < cell.block_dim; ++worker) {
        const size_t index = static_cast<size_t>(worker)
            * kOutputValuesPerWorker;
        const uint64_t cycles = host_output[index];
        const uint64_t checksum = host_output[index + 1];
        const bool valid = cycles > 0 && checksum > 0
            && host_output[index + 2] == static_cast<uint64_t>(cell.branch)
            && host_output[index + 3] == static_cast<uint64_t>(cell.mode)
            && host_output[index + 4] == static_cast<uint64_t>(cell.repetitions)
            && host_output[index + 5] == static_cast<uint64_t>(cell.block_dim);
        if (!valid) {
            std::fprintf(stderr, "device control validation failed\n");
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "%lld,%lld,%lld,%lld,%lld,%llu,%llu,%.9f,pass\n",
            static_cast<long long>(cell.block_dim),
            static_cast<long long>(worker),
            static_cast<long long>(cell.branch),
            static_cast<long long>(cell.mode),
            static_cast<long long>(cell.repetitions),
            static_cast<unsigned long long>(cycles),
            static_cast<unsigned long long>(checksum),
            static_cast<double>(cycles) * kSystemCounterTickNs);
    }
    ACL_CHECK(aclrtFree(output));
}

}  // namespace

__attribute__((constructor(101))) static void EarlyAclInit() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(DeviceId()));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--batch") {
        std::fprintf(stderr, "usage: %s --batch workers:branch:mode:repetitions...\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    for (int index = 2; index < argc; ++index) {
        RunCell(stream, ParseCell(argv[index]));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(DeviceId()));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
