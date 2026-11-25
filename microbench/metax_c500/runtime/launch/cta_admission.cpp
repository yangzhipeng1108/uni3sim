/*
 * MetaX C500 resource-matched CTA admission benchmark.
 *
 * This benchmark contains no production operator.  Every launch uses 256
 * threads and 32 KiB of dynamic shared memory.  A deliberately large set of
 * simultaneously-live scalar values asks mxcc for register pressure close to
 * the observed 238 registers/thread target; mcFuncGetAttributes is the
 * authoritative report of the binary that was actually launched.
 *
 * Admission is measured without a grid-wide barrier.  Each CTA leader enters
 * an atomic active set, updates the peak, remains live for one fixed clock64
 * window, and leaves.  Later waves never wait for unscheduled CTAs, so an
 * over-subscribed grid cannot deadlock.  entered/finished counts prove that
 * every planned CTA completed.  No kernel or wall duration is reported.
 */

#include <mc_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kExpectedL2Bytes = 8 * 1024 * 1024;
constexpr int kBlockThreads = 256;
constexpr int kDynamicSharedBytes = 32 * 1024;
constexpr int kTargetRegistersPerThread = 238;
constexpr int kRegisterPressureScalars = 224;
constexpr unsigned long long kAdmissionHoldCycles = 2250000ULL;
constexpr int kSchemaVersion = 2;

struct PlanRow {
    std::string sample_id;
    int trial;
    int order_position;
    std::string geometry_id;
    int grid_x;
    int grid_y;
    int grid_z;
};

struct Counters {
    unsigned int active;
    unsigned int peak;
    unsigned int entered;
    unsigned int finished;
    unsigned int observed_grid_x;
    unsigned int observed_grid_y;
    unsigned int observed_grid_z;
    unsigned int observed_block_x;
    unsigned int geometry_mismatch;
};

__device__ __forceinline__ float register_pressure(float seed) {
    float values[kRegisterPressureScalars];
    #pragma unroll
    for (int index = 0; index < kRegisterPressureScalars; ++index) {
        const float addend = static_cast<float>(index + 1) * 0.000001f;
        asm volatile("add_f32 %0, %1, %2"
                     : "=r"(values[index]) : "r"(seed), "r"(addend));
    }
    float checksum = 0.0f;
    #pragma unroll
    for (int index = 0; index < kRegisterPressureScalars; ++index) {
        asm volatile("add_f32 %0, %1, %2"
                     : "=r"(checksum) : "r"(checksum), "r"(values[index]));
    }
    return checksum;
}

__global__ __launch_bounds__(kBlockThreads, 1) void c500_cta_admission(
        Counters* counters, float* sinks,
        unsigned long long hold_cycles,
        unsigned int expected_grid_x, unsigned int expected_grid_y,
        unsigned int expected_grid_z) {
    extern __shared__ unsigned char dynamic_shared[];
    const float seed = 1.0f
        + static_cast<float>(threadIdx.x) * 0.000003814697265625f;
    const float checksum = register_pressure(seed);
    dynamic_shared[threadIdx.x] = static_cast<unsigned char>(threadIdx.x);
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }

    const unsigned int ticket = atomicAdd(&counters->entered, 1U);
    if (ticket == 0U) {
        counters->observed_grid_x = gridDim.x;
        counters->observed_grid_y = gridDim.y;
        counters->observed_grid_z = gridDim.z;
        counters->observed_block_x = blockDim.x;
    }
    if (gridDim.x != expected_grid_x || gridDim.y != expected_grid_y
            || gridDim.z != expected_grid_z || blockDim.x != kBlockThreads
            || blockDim.y != 1U || blockDim.z != 1U) {
        atomicExch(&counters->geometry_mismatch, 1U);
    }
    const unsigned int active = atomicAdd(&counters->active, 1U) + 1U;
    atomicMax(&counters->peak, active);
    const unsigned long long start = clock64();
    while (clock64() - start < hold_cycles) {
        asm volatile("" ::: "memory");
    }
    atomicSub(&counters->active, 1U);
    atomicAdd(&counters->finished, 1U);

    const unsigned int flattened_block = static_cast<unsigned int>(
        blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
    sinks[flattened_block] = checksum + static_cast<float>(ticket & 1U);
}

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

void require_token(const std::string& value, const char* label) {
    if (value.empty() || value.size() > 96) {
        std::fprintf(stderr, "%s length is invalid\n", label);
        std::exit(EXIT_FAILURE);
    }
    for (char character : value) {
        const bool valid = (
            (character >= 'a' && character <= 'z')
            || (character >= 'A' && character <= 'Z')
            || (character >= '0' && character <= '9')
            || character == '_' || character == '-');
        if (!valid) {
            std::fprintf(stderr, "%s contains an invalid character\n", label);
            std::exit(EXIT_FAILURE);
        }
    }
}

std::vector<PlanRow> read_plan(const char* path) {
    std::ifstream stream(path);
    if (!stream) {
        std::fprintf(stderr, "cannot open CTA admission plan\n");
        std::exit(EXIT_FAILURE);
    }
    std::vector<PlanRow> rows;
    std::set<std::string> sample_ids;
    PlanRow row {};
    while (stream >> row.sample_id >> row.trial >> row.order_position
                  >> row.geometry_id >> row.grid_x >> row.grid_y
                  >> row.grid_z) {
        require_token(row.sample_id, "sample_id");
        require_token(row.geometry_id, "geometry_id");
        if (row.trial <= 0 || row.order_position < 0
                || row.grid_x <= 0 || row.grid_y <= 0 || row.grid_z <= 0) {
            std::fprintf(stderr, "CTA admission plan row is invalid\n");
            std::exit(EXIT_FAILURE);
        }
        const long long total = static_cast<long long>(row.grid_x)
            * row.grid_y * row.grid_z;
        if (total <= 0 || total > 4096) {
            std::fprintf(stderr, "CTA admission grid size is invalid\n");
            std::exit(EXIT_FAILURE);
        }
        if (!sample_ids.insert(row.sample_id).second) {
            std::fprintf(stderr, "CTA admission sample_id is duplicated\n");
            std::exit(EXIT_FAILURE);
        }
        if (row.order_position != static_cast<int>(rows.size())) {
            std::fprintf(stderr, "CTA admission plan order is not contiguous\n");
            std::exit(EXIT_FAILURE);
        }
        rows.push_back(row);
    }
    if (!stream.eof() || rows.empty()) {
        std::fprintf(stderr, "CTA admission plan is incomplete\n");
        std::exit(EXIT_FAILURE);
    }
    return rows;
}

void require_device(int device_id, mcDeviceProp_t* properties) {
    CHECK(mcSetDevice(device_id));
    CHECK(mcGetDeviceProperties(properties, device_id));
    if (std::strcmp(properties->name, "MetaX C500") != 0
            || properties->multiProcessorCount != kExpectedApCount
            || properties->warpSize != kExpectedWaveSize
            || properties->l2CacheSize != kExpectedL2Bytes) {
        std::fprintf(stderr, "expected MetaX C500 AP=104 wave=64 L2=8MiB\n");
        std::exit(EXIT_FAILURE);
    }
}

void read_resources(
        mcFuncAttributes* attributes, int* resident_blocks_per_ap) {
    CHECK(mcFuncGetAttributes(
        attributes, reinterpret_cast<const void*>(c500_cta_admission)));
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        resident_blocks_per_ap, c500_cta_admission,
        kBlockThreads, kDynamicSharedBytes));
    if (attributes->numRegs <= 0 || attributes->sharedSizeBytes != 0
            || *resident_blocks_per_ap <= 0) {
        std::fprintf(stderr, "CTA admission binary resource report is invalid\n");
        std::exit(EXIT_FAILURE);
    }
}

void print_metadata(
        int device_id, const mcDeviceProp_t& properties,
        const mcFuncAttributes& attributes, int resident_blocks_per_ap) {
    int runtime_version = 0;
    int driver_version = 0;
    CHECK(mcRuntimeGetVersion(&runtime_version));
    CHECK(mcDriverGetVersion(&driver_version));
    if (runtime_version <= 0 || driver_version <= 0) {
        std::fprintf(stderr, "MACA runtime/driver version is invalid\n");
        std::exit(EXIT_FAILURE);
    }
    std::printf(
        "metadata,%d,metax_c500,cta_admission,%d,%s,%d,%d,%d,"
        "%d,%d,%d,%d,%zu,%d,%llu,%d,%d,%d\n",
        kSchemaVersion, device_id, properties.name,
        properties.multiProcessorCount, properties.warpSize,
        properties.l2CacheSize, kBlockThreads, kDynamicSharedBytes,
        kTargetRegistersPerThread, attributes.numRegs,
        static_cast<std::size_t>(attributes.sharedSizeBytes),
        resident_blocks_per_ap,
        static_cast<unsigned long long>(kAdmissionHoldCycles),
        kRegisterPressureScalars, runtime_version, driver_version);
}

void run_plan(int device_id, const char* plan_path) {
    mcDeviceProp_t properties {};
    require_device(device_id, &properties);
    mcFuncAttributes attributes {};
    int resident_blocks_per_ap = 0;
    read_resources(&attributes, &resident_blocks_per_ap);
    const std::vector<PlanRow> plan = read_plan(plan_path);

    int maximum_ctas = 0;
    for (const PlanRow& row : plan) {
        const int total = row.grid_x * row.grid_y * row.grid_z;
        if (total > maximum_ctas) {
            maximum_ctas = total;
        }
    }
    Counters* counters = nullptr;
    float* sinks = nullptr;
    CHECK(mcMalloc(reinterpret_cast<void**>(&counters), sizeof(Counters)));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&sinks), maximum_ctas * sizeof(float)));

    print_metadata(
        device_id, properties, attributes, resident_blocks_per_ap);
    for (const PlanRow& row : plan) {
        const int total_ctas = row.grid_x * row.grid_y * row.grid_z;
        CHECK(mcMemset(counters, 0, sizeof(Counters)));
        CHECK(mcMemset(sinks, 0, total_ctas * sizeof(float)));
        c500_cta_admission<<<
            dim3(row.grid_x, row.grid_y, row.grid_z), kBlockThreads,
            kDynamicSharedBytes>>>(
                counters, sinks, kAdmissionHoldCycles,
                static_cast<unsigned int>(row.grid_x),
                static_cast<unsigned int>(row.grid_y),
                static_cast<unsigned int>(row.grid_z));
        CHECK(mcGetLastError());
        CHECK(mcDeviceSynchronize());

        Counters observed {};
        std::vector<float> host_sinks(total_ctas);
        CHECK(mcMemcpy(
            &observed, counters, sizeof(Counters), mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(
            host_sinks.data(), sinks, total_ctas * sizeof(float),
            mcMemcpyDeviceToHost));
        double checksum = 0.0;
        for (float value : host_sinks) {
            if (!std::isfinite(value) || value <= 0.0f) {
                std::fprintf(
                    stderr, "CTA admission sink coverage is invalid\n");
                std::exit(EXIT_FAILURE);
            }
            checksum += value;
        }
        if (observed.active != 0U
                || observed.entered != static_cast<unsigned int>(total_ctas)
                || observed.finished != static_cast<unsigned int>(total_ctas)
                || observed.peak == 0U
                || observed.peak > observed.entered
                || observed.observed_grid_x
                    != static_cast<unsigned int>(row.grid_x)
                || observed.observed_grid_y
                    != static_cast<unsigned int>(row.grid_y)
                || observed.observed_grid_z
                    != static_cast<unsigned int>(row.grid_z)
                || observed.observed_block_x
                    != static_cast<unsigned int>(kBlockThreads)
                || observed.geometry_mismatch != 0U
                || !std::isfinite(checksum) || checksum <= 0.0) {
            std::fprintf(stderr, "CTA admission completion invariant failed\n");
            std::exit(EXIT_FAILURE);
        }
        std::printf(
            "sample,%s,%d,%d,%s,%d,%d,%d,%d,%u,%u,%u,%.17g\n",
            row.sample_id.c_str(), row.trial, row.order_position,
            row.geometry_id.c_str(), row.grid_x, row.grid_y, row.grid_z,
            total_ctas, observed.peak, observed.entered,
            observed.finished, checksum);
    }
    std::fflush(stdout);
    CHECK(mcFree(sinks));
    CHECK(mcFree(counters));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::strcmp(argv[1], "--device-id") != 0
            || std::strcmp(argv[3], "--plan") != 0) {
        std::fprintf(
            stderr, "usage: %s --device-id DEVICE_ID --plan PLAN\n", argv[0]);
        return EXIT_FAILURE;
    }
    run_plan(parse_nonnegative_int(argv[2], "device-id"), argv[4]);
    return EXIT_SUCCESS;
}
