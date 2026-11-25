// Host driver for the 48-AIV elementwise source-control lifecycle probe.
#include <acl/acl.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "elementwise_control_contract.h"

#define ACL_CHECK(call) do {                                               \
    const int status = static_cast<int>(call);                             \
    if (status != 0) {                                                     \
        std::fprintf(stderr, "ACL error: %s returned %d\n", #call, status); \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

extern "C" uint32_t aclrtlaunch_elementwise_control(
    uint32_t block_dim, void* stream, void* gm_results,
    int64_t tile_count, int64_t variant);

namespace {

constexpr int32_t kLogicalDeviceId = 0;

struct Cell {
    int64_t variant;
    int64_t tile_count;
    std::string variant_name;
};

int64_t ParseInteger(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        std::fprintf(stderr, "%s must be an integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int64_t>(value);
}

Cell ParseCell(const char* text) {
    const std::string value(text);
    const size_t separator = value.find(':');
    if (separator == std::string::npos
            || value.find(':', separator + 1) != std::string::npos) {
        std::fprintf(stderr, "cell must be variant:tile_count\n");
        std::exit(EXIT_FAILURE);
    }
    const std::string variant_name = value.substr(0, separator);
    int64_t variant = -1;
    if (variant_name == "control_baseline") {
        variant = 0;
    } else if (variant_name == "dag_tbuf") {
        variant = 1;
    } else if (variant_name == "full_lifecycle") {
        variant = 2;
    } else {
        std::fprintf(stderr, "unknown variant\n");
        std::exit(EXIT_FAILURE);
    }
    const int64_t tile_count = ParseInteger(
        value.substr(separator + 1).c_str(), "tile_count");
    if (!elementwise_control_contract::IsFrozenTileCount(tile_count)) {
        std::fprintf(stderr, "tile_count is outside the frozen campaign\n");
        std::exit(EXIT_FAILURE);
    }
    return Cell{variant, tile_count, variant_name};
}

std::string Join(
        const std::vector<int64_t>& values, int64_t field_index) {
    std::string result;
    for (int64_t worker = 0;
            worker < elementwise_control_contract::kAivWorkers; ++worker) {
        if (worker != 0) {
            result += ';';
        }
        result += std::to_string(values[
            worker * elementwise_control_contract::kResultSlotElements
            + field_index]);
    }
    return result;
}

}  // namespace

__attribute__((constructor(101))) static void EarlyInit() {
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (device_text == nullptr
            || ParseInteger(device_text, "ASCEND_DEVICE_ID")
            != kLogicalDeviceId) {
        std::fprintf(stderr, "ASCEND_DEVICE_ID must equal logical device 0\n");
        std::exit(EXIT_FAILURE);
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(kLogicalDeviceId));
}

int main(int argc, char** argv) {
    if (argc < 3 || std::strcmp(argv[1], "--batch") != 0) {
        std::fprintf(stderr, "usage: %s --batch variant:tile_count ...\n", argv[0]);
        return EXIT_FAILURE;
    }
    std::vector<Cell> cells;
    for (int argument = 2; argument < argc; ++argument) {
        cells.push_back(ParseCell(argv[argument]));
    }
    const size_t result_bytes = static_cast<size_t>(
        elementwise_control_contract::kAivWorkers
        * elementwise_control_contract::kResultSlotBytes);
    void* device_results = nullptr;
    ACL_CHECK(aclrtMalloc(
        &device_results, result_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::printf("variant,tile_count,worker_cycles,worker_checksums,validation\n");
    for (const Cell& cell : cells) {
        ACL_CHECK(aclrtMemset(device_results, result_bytes, 0xff, result_bytes));
        ACL_CHECK(aclrtlaunch_elementwise_control(
            elementwise_control_contract::kAivWorkers, stream,
            device_results, cell.tile_count, cell.variant));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<int64_t> results(
            elementwise_control_contract::kAivWorkers
            * elementwise_control_contract::kResultSlotElements, -1);
        ACL_CHECK(aclrtMemcpy(
            results.data(), result_bytes, device_results, result_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        for (int64_t worker = 0;
                worker < elementwise_control_contract::kAivWorkers; ++worker) {
            const int64_t offset = worker
                * elementwise_control_contract::kResultSlotElements;
            if (results[offset] <= 0 || results[offset + 1] == 0) {
                std::fprintf(stderr, "invalid result at worker %lld\n",
                             static_cast<long long>(worker));
                return EXIT_FAILURE;
            }
        }
        std::printf("%s,%lld,%s,%s,ok\n", cell.variant_name.c_str(),
                    static_cast<long long>(cell.tile_count),
                    Join(results, 0).c_str(), Join(results, 1).c_str());
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtFree(device_results));
    ACL_CHECK(aclrtResetDevice(kLogicalDeviceId));
    ACL_CHECK(aclFinalize());
    return EXIT_SUCCESS;
}
