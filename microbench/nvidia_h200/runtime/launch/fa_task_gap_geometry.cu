/*
 * H200 owner copy of microbench/nvidia_a800/runtime/launch/fa_task_gap_geometry.cu.
 * Kernel bodies (main geometry: 128 threads, 81920 B dynamic + 1 KiB static
 * shared memory; combine geometry: 128 threads, 1072 B static), the clock64 /
 * globaltimer CTA envelope, 10 warm-up launches and the CSV are the A800 ones.
 * Hardware coupling changed (and nothing else):
 *   - build: nvcc -O3 -std=c++17 -arch=sm_90 -cudart static;
 *   - device check: sm_90 with 132 SMs instead of sm_80 with 108 SMs.
 * The A800 source ships no runner or report; the H200 procedure is scripted in
 * fa_task_gap_geometry_runner.py / fa_task_gap_geometry_report.py.
 */
/*
 * Independent SM80 task-shell probe for source-closed FA geometry.
 *
 * NSYS observes the launch-wide GPU-kernel interval. Each CTA records its
 * clock64 envelope around a matched shared-memory body. The difference is a
 * candidate launch-wide fill/drain span only when it is invariant under a
 * separately selected in-CTA spin length. No FlashAttention tensor,
 * production binary, or production/evaluation duration is read.
 */

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr int kThreads = 128;
constexpr int kMainDynamicSharedBytes = 81920;
constexpr int kMainStaticWords = 256;
constexpr int kCombineStaticWords = 268;

struct CtaTiming {
    std::uint64_t internal_envelope_cycles;
    std::uint64_t global_begin_ns;
    std::uint64_t global_end_ns;
};

__device__ __forceinline__ std::uint64_t globaltimer_ns() {
    std::uint64_t timestamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
    return timestamp;
}

void check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(call) check((call), #call)

int parse_positive(const char* text, const char* name) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

int parse_nonnegative(const char* text, const char* name) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0
            || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "%s must be a nonnegative integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

template <bool kMain>
__global__ __launch_bounds__(kThreads, 1)
void task_gap_geometry(
        CtaTiming* timings, int body_cycles, std::uint32_t salt) {
    constexpr int static_words = kMain ? kMainStaticWords : kCombineStaticWords;
    __shared__ std::uint32_t static_shared[static_words];
    extern __shared__ std::uint32_t dynamic_shared[];
    const int lane = threadIdx.x;
    std::uint32_t value = static_cast<std::uint32_t>(
        (blockIdx.x + 1) * 0x9e3779b9U ^ (lane + 1) ^ salt);
    const std::uint64_t global_begin = globaltimer_ns();
    const std::uint64_t begin = clock64();
    static_shared[lane % static_words] = value;
    if constexpr (kMain) {
        dynamic_shared[lane] = value ^ 0xa5a5a5a5U;
    }
    __syncthreads();
    const std::uint64_t spin_begin = clock64();
    while (clock64() - spin_begin < static_cast<std::uint64_t>(body_cycles)) {
        value = value * 1664525U + 1013904223U;
    }
    static_shared[(lane * 17) % static_words] ^= value;
    if constexpr (kMain) {
        dynamic_shared[(lane * 13) % kThreads] ^= value;
    }
    __syncthreads();
    const std::uint64_t end = clock64();
    const std::uint64_t global_end = globaltimer_ns();
    if (lane == 0) {
        timings[blockIdx.x] = CtaTiming{
            end - begin, global_begin, global_end};
    }
}

void launch(bool main_geometry, CtaTiming* timings, int grid,
            int body_cycles, std::uint32_t salt) {
    if (main_geometry) {
        task_gap_geometry<true><<<grid, kThreads, kMainDynamicSharedBytes>>>(
            timings, body_cycles, salt);
    } else {
        task_gap_geometry<false><<<grid, kThreads>>>(
            timings, body_cycles, salt);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
            "usage: %s DEVICE MAIN_GEOMETRY GRID BODY_CYCLES SAMPLES OUTPUT_CSV\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    const int device = parse_nonnegative(argv[1], "DEVICE");
    const int main_flag = parse_nonnegative(argv[2], "MAIN_GEOMETRY");
    if (main_flag != 0 && main_flag != 1) {
        std::fprintf(stderr, "MAIN_GEOMETRY must be zero or one\n");
        return EXIT_FAILURE;
    }
    const bool main_geometry = main_flag == 1;
    const int grid = parse_positive(argv[3], "GRID");
    const int body_cycles = parse_nonnegative(argv[4], "BODY_CYCLES");
    const int samples = parse_positive(argv[5], "SAMPLES");
    const char* output_path = argv[6];
    CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.minor != 0
            || properties.multiProcessorCount != 132) {
        std::fprintf(stderr, "probe requires the 132-SM sm_90 target\n");
        return EXIT_FAILURE;
    }
    CHECK(cudaFuncSetAttribute(task_gap_geometry<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kMainDynamicSharedBytes));
    CtaTiming* device_timings = nullptr;
    CHECK(cudaMalloc(&device_timings,
        static_cast<std::size_t>(grid) * sizeof(*device_timings)));
    auto* host_timings = new CtaTiming[grid];
    for (int warmup = 0; warmup < 10; ++warmup) {
        launch(main_geometry, device_timings, grid, body_cycles,
               static_cast<std::uint32_t>(warmup));
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    FILE* output = std::fopen(output_path, "w");
    if (output == nullptr) {
        std::perror("fopen output");
        return EXIT_FAILURE;
    }
    std::fprintf(output,
        "sample,cta,internal_envelope_cycles,global_begin_ns,global_end_ns\n");
    for (int sample = 0; sample < samples; ++sample) {
        launch(main_geometry, device_timings, grid, body_cycles,
               static_cast<std::uint32_t>(sample + 100));
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(host_timings, device_timings,
            static_cast<std::size_t>(grid) * sizeof(*host_timings),
            cudaMemcpyDeviceToHost));
        for (int cta = 0; cta < grid; ++cta) {
            const CtaTiming timing = host_timings[cta];
            std::fprintf(output, "%d,%d,%llu,%llu,%llu\n", sample, cta,
                static_cast<unsigned long long>(
                    timing.internal_envelope_cycles),
                static_cast<unsigned long long>(timing.global_begin_ns),
                static_cast<unsigned long long>(timing.global_end_ns));
        }
    }
    if (std::fclose(output) != 0) {
        std::perror("fclose output");
        return EXIT_FAILURE;
    }
    delete[] host_timings;
    CHECK(cudaFree(device_timings));
    return EXIT_SUCCESS;
}
