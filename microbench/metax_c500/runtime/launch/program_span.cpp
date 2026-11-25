/*
 * MetaX C500 low-wave program-span mechanism benchmark.
 *
 * This benchmark is intentionally operator-independent.  Three register-only
 * bodies separate dynamic dependent work from static code footprint:
 *
 *   looped:   COUNT dependent add_f32 operations in a retained loop;
 *   unrolled: COUNT dependent add_f32 operations emitted statically;
 *   cold:     one executed add_f32 plus a runtime-disabled COUNT-add body.
 *
 * The cold body changes the retained device-code footprint without executing
 * the added operations.  No production operator, tensor traffic, shared
 * memory, barrier, or target-kernel duration is used.
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kWarmupLaunchesPerConfiguration = 8;

enum class Body : int {
    kLooped = 0,
    kUnrolled = 1,
    kCold = 2,
};

struct PlanRow {
    std::string sample_id;
    int trial;
    int order_position;
    Body body;
    int count;
    int grid_x;
    int grid_y;
    int grid_z;
    int block_x;
    int block_y;
    int block_z;
};

struct Configuration {
    Body body;
    int count;
    int grid_x;
    int grid_y;
    int grid_z;
    int block_x;
    int block_y;
    int block_z;

    bool operator<(const Configuration& other) const {
        return std::tie(
            body, count, grid_x, grid_y, grid_z,
            block_x, block_y, block_z) < std::tie(
            other.body, other.count, other.grid_x, other.grid_y,
            other.grid_z, other.block_x, other.block_y, other.block_z);
    }
};

__device__ __forceinline__ float dependent_add(float value) {
    float output;
    const float addend = 0.0000152587890625f;
    asm volatile("add_f32 %0, %1, %2"
                 : "=r"(output) : "r"(value), "r"(addend));
    return output;
}

template <int Count>
struct UnrolledChain {
    __device__ __forceinline__ static float run(float value) {
        return UnrolledChain<Count - 1>::run(dependent_add(value));
    }
};

template <>
struct UnrolledChain<0> {
    __device__ __forceinline__ static float run(float value) {
        return value;
    }
};

__global__ void looped_chain_kernel(int count) {
    float value = 1.0009765625f
        + static_cast<float>(threadIdx.x & 1U) * 0.0009765625f;
    #pragma unroll 1
    for (int iteration = 0; iteration < count; ++iteration) {
        value = dependent_add(value);
    }
    asm volatile("" : : "r"(value));
}

template <int Count>
__global__ void unrolled_chain_kernel() {
    float value = 1.0009765625f
        + static_cast<float>(threadIdx.x & 1U) * 0.0009765625f;
    value = UnrolledChain<Count>::run(value);
    asm volatile("" : : "r"(value));
}

template <int Count>
__global__ void cold_footprint_kernel(int execute_cold_body) {
    float value = 1.0009765625f
        + static_cast<float>(threadIdx.x & 1U) * 0.0009765625f;
    value = dependent_add(value);
    if (execute_cold_body != 0) {
        value = UnrolledChain<Count>::run(value);
    }
    asm volatile("" : : "r"(value));
}

template <int Count>
void launch_unrolled(dim3 grid, dim3 block) {
    unrolled_chain_kernel<Count><<<grid, block>>>();
}

template <int Count>
void launch_cold(dim3 grid, dim3 block) {
    cold_footprint_kernel<Count><<<grid, block>>>(0);
}

void require_supported_count(int count) {
    if (count != 1 && count != 8 && count != 32
            && count != 128 && count != 512) {
        std::fprintf(stderr, "unsupported program-span count: %d\n", count);
        std::exit(EXIT_FAILURE);
    }
}

void launch_body(Body body, int count, dim3 grid, dim3 block) {
    require_supported_count(count);
    if (body == Body::kLooped) {
        looped_chain_kernel<<<grid, block>>>(count);
        return;
    }
    if (body == Body::kUnrolled) {
        switch (count) {
            case 1: launch_unrolled<1>(grid, block); return;
            case 8: launch_unrolled<8>(grid, block); return;
            case 32: launch_unrolled<32>(grid, block); return;
            case 128: launch_unrolled<128>(grid, block); return;
            case 512: launch_unrolled<512>(grid, block); return;
            default: std::abort();
        }
    }
    if (body == Body::kCold) {
        switch (count) {
            case 1: launch_cold<1>(grid, block); return;
            case 8: launch_cold<8>(grid, block); return;
            case 32: launch_cold<32>(grid, block); return;
            case 128: launch_cold<128>(grid, block); return;
            case 512: launch_cold<512>(grid, block); return;
            default: std::abort();
        }
    }
    std::fprintf(stderr, "unknown program-span body\n");
    std::exit(EXIT_FAILURE);
}

Body parse_body(const std::string& text) {
    if (text == "looped") {
        return Body::kLooped;
    }
    if (text == "unrolled") {
        return Body::kUnrolled;
    }
    if (text == "cold") {
        return Body::kCold;
    }
    std::fprintf(stderr, "unknown program-span body: %s\n", text.c_str());
    std::exit(EXIT_FAILURE);
}

const char* body_name(Body body) {
    if (body == Body::kLooped) {
        return "looped";
    }
    if (body == Body::kUnrolled) {
        return "unrolled";
    }
    if (body == Body::kCold) {
        return "cold";
    }
    std::fprintf(stderr, "unknown program-span body\n");
    std::exit(EXIT_FAILURE);
}

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

void validate_plan_row(const PlanRow& row) {
    if (row.sample_id.empty() || row.sample_id.size() > 96) {
        std::fprintf(stderr, "sample_id length is invalid\n");
        std::exit(EXIT_FAILURE);
    }
    if (row.trial <= 0 || row.order_position < 0) {
        std::fprintf(stderr, "trial/order_position is invalid\n");
        std::exit(EXIT_FAILURE);
    }
    require_supported_count(row.count);
    const int axes[] = {
        row.grid_x, row.grid_y, row.grid_z,
        row.block_x, row.block_y, row.block_z,
    };
    for (int axis : axes) {
        if (axis <= 0) {
            std::fprintf(stderr, "launch axes must be positive\n");
            std::exit(EXIT_FAILURE);
        }
    }
    const long long block_size = (
        static_cast<long long>(row.block_x) * row.block_y * row.block_z);
    if (block_size > 1024) {
        std::fprintf(stderr, "block size exceeds 1024 threads\n");
        std::exit(EXIT_FAILURE);
    }
}

std::vector<PlanRow> read_plan() {
    std::vector<PlanRow> rows;
    std::set<std::string> sample_ids;
    std::string sample_id;
    int trial = 0;
    int order_position = 0;
    std::string body_text;
    int count = 0;
    int grid_x = 0;
    int grid_y = 0;
    int grid_z = 0;
    int block_x = 0;
    int block_y = 0;
    int block_z = 0;
    while (std::cin >> sample_id >> trial >> order_position >> body_text
            >> count >> grid_x >> grid_y >> grid_z
            >> block_x >> block_y >> block_z) {
        PlanRow row {
            sample_id, trial, order_position, parse_body(body_text), count,
            grid_x, grid_y, grid_z, block_x, block_y, block_z,
        };
        validate_plan_row(row);
        if (!sample_ids.insert(sample_id).second) {
            std::fprintf(stderr, "duplicate sample_id: %s\n", sample_id.c_str());
            std::exit(EXIT_FAILURE);
        }
        rows.push_back(row);
    }
    if (!std::cin.eof() || rows.empty()) {
        std::fprintf(stderr, "program-span plan is empty or malformed\n");
        std::exit(EXIT_FAILURE);
    }
    return rows;
}

Configuration configuration(const PlanRow& row) {
    return Configuration {
        row.body, row.count,
        row.grid_x, row.grid_y, row.grid_z,
        row.block_x, row.block_y, row.block_z,
    };
}

void launch_configuration(const Configuration& value) {
    launch_body(
        value.body, value.count,
        dim3(value.grid_x, value.grid_y, value.grid_z),
        dim3(value.block_x, value.block_y, value.block_z));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--device-id") != 0) {
        std::fprintf(
            stderr, "usage: %s --device-id DEVICE_ID < plan\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[2], "device-id");
    CHECK(mcSetDevice(device_id));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device_id));
    if (properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize) {
        std::fprintf(
            stderr, "expected C500 104 AP/wave64, got %d AP/wave%d\n",
            properties.multiProcessorCount, properties.warpSize);
        return EXIT_FAILURE;
    }
    char pci_bus_id[32] = {};
    CHECK(mcDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_id));
    for (char* character = pci_bus_id; *character != '\0'; ++character) {
        *character = static_cast<char>(std::tolower(
            static_cast<unsigned char>(*character)));
    }

    const std::vector<PlanRow> plan = read_plan();
    std::set<Configuration> configurations;
    for (const PlanRow& row : plan) {
        configurations.insert(configuration(row));
    }
    for (const Configuration& value : configurations) {
        for (int warmup = 0;
                warmup < kWarmupLaunchesPerConfiguration; ++warmup) {
            launch_configuration(value);
        }
    }
    CHECK(mcDeviceSynchronize());

    std::printf(
        "metadata,1,%s,%d,%s,%d,%d,%d\n",
        properties.name, device_id, pci_bus_id,
        properties.multiProcessorCount, properties.warpSize,
        properties.clockRate);
    for (const PlanRow& row : plan) {
        launch_body(
            row.body, row.count,
            dim3(row.grid_x, row.grid_y, row.grid_z),
            dim3(row.block_x, row.block_y, row.block_z));
        CHECK(mcDeviceSynchronize());
        std::printf(
            "sample,%s,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d\n",
            row.sample_id.c_str(), row.trial, row.order_position,
            body_name(row.body), row.count,
            row.grid_x, row.grid_y, row.grid_z,
            row.block_x, row.block_y, row.block_z);
    }
    std::fflush(stdout);
    return EXIT_SUCCESS;
}
