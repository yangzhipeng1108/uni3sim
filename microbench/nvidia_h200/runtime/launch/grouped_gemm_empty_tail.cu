/*
 * H200 owner copy of microbench/nvidia_a800/runtime/launch/grouped_gemm_empty_tail.cu
 *
 * Kernel body, grids (1280, 8192, 32768), active prefix (320 rows,
 * num_pid_n 64), hold cycles and trial order are unchanged.  Hardware
 * coupling only:
 *   - kSmCount 108 -> 132; device check compute capability 9.0 / 132 SMs.
 *   - Resident CTAs per SM stay 10, the W2 residency of the H200 BM16
 *     grouped-GEMM launch (Triton 3.6 compile facts: 45 registers,
 *     12288-B shared memory -> 10 CTAs/SM, same as A800).  With 228-KiB SM
 *     shared memory and a 1-KiB per-CTA reservation, 15-KiB dynamic shared
 *     memory would admit 14 CTAs, so the occupancy-setting dynamic
 *     allocation becomes 21 KiB (floor(233472 / 22528) = 10).
 *   - warm-up grid 1080 (10 x 108) -> 1320 (10 x 132).
 */
/*
 * Operator-independent SM80 source-ordered empty-tail probe.
 *
 * A contiguous prefix of CTAs holds its resident slot for a fixed number of
 * clock cycles.  The remaining CTAs take an immediate return.  This isolates
 * whether an empty launch tail drains after a busy prefix without executing
 * any production operator, tensor-core instruction, or production-shaped
 * memory stream.
 */

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kThreads = 128;
constexpr int kSmCount = 132;
constexpr int kDynamicSharedBytes = 21 * 1024;

void check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(call) check((call), #call)

int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < std::numeric_limits<int>::min()
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be an integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

__global__ void source_ordered_empty_tail(
        const unsigned int* num_tokens_post_padded,
        unsigned int num_pid_n, unsigned long long hold_cycles,
        unsigned long long* sink) {
    const unsigned int pid = blockIdx.x;
    const unsigned int pid_m = pid / num_pid_n;
    if (pid_m * 16U >= *num_tokens_post_padded) {
        return;
    }
    const unsigned long long start = clock64();
    unsigned int value = pid * 1664525U + threadIdx.x + 1013904223U;
    while (clock64() - start < hold_cycles) {
        value = value * 1664525U + 1013904223U;
        value ^= value >> 13;
    }
    if (threadIdx.x == 0) {
        sink[pid] = value;
    }
}

float measure(int grid, unsigned int active_rows, unsigned int num_pid_n,
              unsigned long long hold_cycles,
              const unsigned int* device_active_rows,
              unsigned long long* sink) {
    CHECK(cudaMemcpy(
        const_cast<unsigned int*>(device_active_rows), &active_rows,
        sizeof(active_rows), cudaMemcpyHostToDevice));
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    source_ordered_empty_tail<<<grid, kThreads, kDynamicSharedBytes>>>(
        device_active_rows, num_pid_n, hold_cycles, sink);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaEventDestroy(start));
    return elapsed_ms;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS OWNER BASELINE\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = parse_int(argv[1], "device");
    const int trials = parse_int(argv[2], "trials");
    if (device < 0 || trials < 3 || trials > 11 || argv[3][0] == '\0'
            || argv[4][0] == '\0') {
        std::fprintf(stderr, "invalid device, trials, owner, or baseline\n");
        return EXIT_FAILURE;
    }
    CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "probe requires the 132-SM sm_90 target\n");
        return EXIT_FAILURE;
    }
    CHECK(cudaFuncSetAttribute(
        source_ordered_empty_tail,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedBytes));
    int occupancy = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, source_ordered_empty_tail, kThreads,
        kDynamicSharedBytes));
    if (occupancy != 10) {
        std::fprintf(stderr, "expected occupancy 10, got %d\n", occupancy);
        return EXIT_FAILURE;
    }
    constexpr int maximum_grid = 32768;
    unsigned int* device_active_rows = nullptr;
    unsigned long long* sink = nullptr;
    CHECK(cudaMalloc(&device_active_rows, sizeof(*device_active_rows)));
    CHECK(cudaMalloc(&sink, maximum_grid * sizeof(*sink)));
    const unsigned int warmup_active_rows = 320;
    CHECK(cudaMemcpy(
        device_active_rows, &warmup_active_rows,
        sizeof(warmup_active_rows), cudaMemcpyHostToDevice));
    for (int warmup = 0; warmup < 3; ++warmup) {
        source_ordered_empty_tail<<<1320, kThreads, kDynamicSharedBytes>>>(
            device_active_rows, 64, 14000, sink);
    }
    CHECK(cudaDeviceSynchronize());

    std::printf("meta,device,%d\n", device);
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,sm_count,%d\n", properties.multiProcessorCount);
    std::printf("meta,resident_ctas_per_sm,%d\n", occupancy);
    std::printf("meta,threads_per_cta,%d\n", kThreads);
    std::printf("meta,dynamic_shared_bytes,%d\n", kDynamicSharedBytes);
    std::printf("meta,owner,%s\n", argv[3]);
    std::printf("meta,baseline,%s\n", argv[4]);
    std::printf("data,trial,grid,active_rows,num_pid_n,hold_cycles,event_duration_ns\n");
    constexpr int grids[] = {1280, 8192, 32768};
    constexpr unsigned int active_rows[] = {0, 320};
    constexpr unsigned int num_pid_n = 64;
    constexpr unsigned long long hold_cycles[] = {0, 7000, 14000, 28000};
    for (int trial = 0; trial < trials; ++trial) {
        for (unsigned long long cycles : hold_cycles) {
            for (int grid : grids) {
                for (unsigned int rows : active_rows) {
                    if (rows / 16U * num_pid_n
                            > static_cast<unsigned int>(grid)) {
                        continue;
                    }
                    const float elapsed_ms = measure(
                        grid, rows, num_pid_n, cycles,
                        device_active_rows, sink);
                    std::printf("data,%d,%d,%u,%u,%llu,%.3f\n", trial,
                                grid, rows, num_pid_n, cycles,
                                elapsed_ms * 1.0e6F);
                }
            }
        }
    }
    CHECK(cudaFree(device_active_rows));
    CHECK(cudaFree(sink));
    return EXIT_SUCCESS;
}
