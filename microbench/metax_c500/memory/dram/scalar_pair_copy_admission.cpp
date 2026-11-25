/*
 * MetaX C500 scalar-pair copy/admission mechanism benchmark.
 *
 * This is a generic two-input/two-output BF16-width copy primitive, not a
 * production kernel: it has no slot mapping, cache layout, head indexing,
 * tensor framework, or operator arithmetic.  A block has 512 threads; each
 * active thread copies one scalar from each independently-addressed input to
 * its matching output, and then advances by the block width.  Thus one,
 * two, three, and four source-loop rounds expose the hardware's ordinary
 * active-wave admission and load/store overlap boundary without encoding an
 * operator case or an empirical duration lookup.
 *
 * The campaign holds the scalar access form and data placement fixed while
 * varying source-loop rounds and CTA grid.  Timing is solely an independent
 * hardware-mechanism observation; it must not be compared to a production
 * duration or used as a per-payload service table.
 *
 * Build inside the prescribed C500 profile container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native -O3 \
 *       -o scalar_pair_copy_admission scalar_pair_copy_admission.cpp
 */

#include <mc_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CHECK(call) do {                                                   \
    const mcError_t status = (call);                                       \
    if (status != mcSuccess) {                                             \
        std::fprintf(stderr, "GPU error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, mcGetErrorString(status));        \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kBlockThreads = 512;
constexpr int kLaunchesPerBatch = 32;
constexpr int kWarmupsPerCell = 4;
constexpr int kSamplesPerCell = 17;
constexpr int kMaximumElements = 2048;
constexpr int kMaximumGrid = 1024;

struct Cell {
    int elements;
    int grid;
    int sample;
};

struct ResourceMetadata {
    int registers_per_thread;
    std::size_t static_shared_bytes;
    int resident_ctas_per_ap;
};

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

__global__ void scalar_pair_copy_admission(
        const volatile std::uint16_t* key,
        const volatile std::uint16_t* value,
        volatile std::uint16_t* key_output,
        volatile std::uint16_t* value_output,
        std::uint32_t* sink,
        int elements_per_cta) {
    const int base = blockIdx.x * kMaximumElements;
    std::uint32_t checksum = static_cast<std::uint32_t>(threadIdx.x + 1);
    #pragma unroll 1
    for (int element = threadIdx.x; element < elements_per_cta;
         element += blockDim.x) {
        const std::uint16_t key_value = key[base + element];
        const std::uint16_t value_value = value[base + element];
        key_output[base + element] = key_value;
        value_output[base + element] = value_value;
        checksum ^= (static_cast<std::uint32_t>(key_value) << 16)
            | static_cast<std::uint32_t>(value_value);
    }
    if (threadIdx.x == 0) {
        sink[blockIdx.x] = checksum;
    }
}

void launch(
        int grid, int elements,
        const std::uint16_t* key, const std::uint16_t* value,
        std::uint16_t* key_output, std::uint16_t* value_output,
        std::uint32_t* sink) {
    scalar_pair_copy_admission<<<grid, kBlockThreads>>>(
        key, value, key_output, value_output, sink, elements);
}

float record_batch(
        mcEvent_t start, mcEvent_t end, int grid, int elements,
        const std::uint16_t* key, const std::uint16_t* value,
        std::uint16_t* key_output, std::uint16_t* value_output,
        std::uint32_t* sink) {
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(grid, elements, key, value, key_output, value_output, sink);
    }
    CHECK(mcEventRecord(start));
    for (int index = 0; index < kLaunchesPerBatch; ++index) {
        launch(grid, elements, key, value, key_output, value_output, sink);
    }
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0F;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
        std::fprintf(stderr, "invalid mcEvent duration: %.9g ms\n", elapsed_ms);
        std::exit(EXIT_FAILURE);
    }
    return elapsed_ms;
}

ResourceMetadata resource_metadata() {
    mcFuncAttributes attributes {};
    CHECK(mcFuncGetAttributes(
        &attributes,
        reinterpret_cast<const void*>(scalar_pair_copy_admission)));
    int resident_ctas_per_ap = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas_per_ap, scalar_pair_copy_admission,
        kBlockThreads, 0));
    if (attributes.numRegs <= 0 || attributes.sharedSizeBytes != 0
            || resident_ctas_per_ap <= 0) {
        std::fprintf(stderr, "scalar-pair binary resource report is invalid\n");
        std::exit(EXIT_FAILURE);
    }
    return {
        attributes.numRegs,
        attributes.sharedSizeBytes,
        resident_ctas_per_ap,
    };
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--device-id"
            || std::string(argv[3]) != "--trial") {
        std::fprintf(
            stderr, "usage: %s --device-id DEVICE_ID --trial TRIAL\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[2], "device-id");
    const int trial = parse_nonnegative_int(argv[4], "trial");
    if (trial < 1 || trial > 3) {
        std::fprintf(stderr, "trial must be one of 1, 2, or 3\n");
        return EXIT_FAILURE;
    }
    CHECK(mcSetDevice(device_id));
    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device_id));
    if (properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize
            || properties.clockRate <= 0) {
        std::fprintf(
            stderr, "expected C500 104 AP/wave64 and a positive clock\n");
        return EXIT_FAILURE;
    }
    const ResourceMetadata resources = resource_metadata();

    const std::size_t element_count = (
        static_cast<std::size_t>(kMaximumGrid) * kMaximumElements);
    const std::size_t scalar_bytes = element_count * sizeof(std::uint16_t);
    std::vector<std::uint16_t> host(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        host[index] = static_cast<std::uint16_t>(
            1U + ((index * 37U) ^ (index >> 3U)) % 65534U);
    }
    std::uint16_t* key = nullptr;
    std::uint16_t* value = nullptr;
    std::uint16_t* key_output = nullptr;
    std::uint16_t* value_output = nullptr;
    std::uint32_t* sink = nullptr;
    CHECK(mcMalloc(reinterpret_cast<void**>(&key), scalar_bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&value), scalar_bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&key_output), scalar_bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&value_output), scalar_bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&sink),
                   static_cast<std::size_t>(kMaximumGrid) * sizeof(*sink)));
    CHECK(mcMemcpy(key, host.data(), scalar_bytes, mcMemcpyHostToDevice));
    std::reverse(host.begin(), host.end());
    CHECK(mcMemcpy(value, host.data(), scalar_bytes, mcMemcpyHostToDevice));

    const int element_counts[] = {512, 1024, 1536, 2048};
    const int grids[] = {64, 104, 208, 256, 416, 1024};
    for (int elements : element_counts) {
        for (int grid : grids) {
            for (int warmup = 0; warmup < kWarmupsPerCell; ++warmup) {
                launch(grid, elements, key, value, key_output, value_output,
                       sink);
            }
        }
    }
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());

    std::vector<Cell> plan;
    for (int elements : element_counts) {
        for (int grid : grids) {
            for (int sample = 0; sample < kSamplesPerCell; ++sample) {
                plan.push_back({elements, grid, sample});
            }
        }
    }
    std::mt19937 generator(964381U + 7919U * static_cast<unsigned>(trial));
    std::shuffle(plan.begin(), plan.end(), generator);

    std::printf(
        "metadata,1,MetaX C500,%d,%d,%d,%d,%d,%d,%d,%d,%zu\n",
        device_id, properties.multiProcessorCount, properties.warpSize,
        properties.clockRate, trial, kLaunchesPerBatch,
        resources.registers_per_thread, resources.resident_ctas_per_ap,
        resources.static_shared_bytes);
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    for (std::size_t order = 0; order < plan.size(); ++order) {
        const Cell& cell = plan[order];
        const float elapsed_ms = record_batch(
            start, end, cell.grid, cell.elements, key, value,
            key_output, value_output, sink);
        std::printf("sample,%zu,%d,%d,%d,%a\n", order, cell.elements,
                    cell.grid, cell.sample, static_cast<double>(elapsed_ms));
    }
    CHECK(mcEventDestroy(end));
    CHECK(mcEventDestroy(start));

    CHECK(mcMemcpy(host.data(), key_output, scalar_bytes, mcMemcpyDeviceToHost));
    if (host[0] == 0) {
        std::fprintf(stderr, "scalar-pair copy did not write output\n");
        return EXIT_FAILURE;
    }
    CHECK(mcFree(sink));
    CHECK(mcFree(value_output));
    CHECK(mcFree(key_output));
    CHECK(mcFree(value));
    CHECK(mcFree(key));
    return EXIT_SUCCESS;
}
