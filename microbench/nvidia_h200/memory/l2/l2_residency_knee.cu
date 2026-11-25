/*
 * NVIDIA H200 L2 operand-residency knee mechanism benchmark.
 *
 * Owner copy of microbench/nvidia_a800/memory/l2/l2_residency_knee.cu.  The
 * 2R1W kernel, warmup, traffic budget, timing and output are unchanged.
 * Hardware coupling changed for H200 only: build for sm_90 (A800: sm_80);
 * kResidentCtas 108 * 8 -> 132 * 8 (eight 128-thread CTAs per SM on the
 * 132-SM chip); device name "NVIDIA H200" (A800: "NVIDIA A800-SXM4-80GB").
 * The runner redraws the footprint sweep around the 60 MiB H200 L2
 * (device reports 62914560 B).  Comments below naming the GA100 40 MiB L2
 * are kept from the source.
 *
 * A single-pass two-read / one-write stream re-executes over the same three
 * buffers.  Warmup passes establish the steady state that a repeated
 * same-operand launch reaches; the timed passes then report the achieved
 * logical byte rate for that footprint.  Sweeping the total footprint across
 * the GA100 40 MiB L2 capacity locates the footprint above which the steady
 * re-read stops being served by L2, which is the only quantity this probe
 * claims.
 *
 * Reads use ld.global.cg so they bypass L1 and allocate in L2, matching the
 * coalesced sector stream the accepted L2 read envelope was measured with.
 * Writes use the default write-back policy so the dirty-line pressure of a
 * real single-pass vector stream is present.
 *
 * It contains no production operator, no normalization or activation
 * arithmetic, and no production or evaluation duration: the consumer is a
 * fold of the loaded words, and the reported quantity is bytes per second at
 * a footprint.  It identifies neither partitions, request credits, nor a
 * per-operator correction.
 *
 * Build:
 *   nvcc -O3 -arch=sm_90 -cudart static -o l2_residency_knee l2_residency_knee.cu
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do {                                              \
    const cudaError_t error = (call);                                      \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(error));       \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

constexpr int kThreads = 128;
constexpr int kWarmupPasses = 5;
constexpr long long kResidentCtas = 132 * 8;
constexpr long long kTrafficBudgetBytes = 2LL * 1024 * 1024 * 1024;
constexpr const char* kExpectedDevice = "NVIDIA H200";

__device__ __forceinline__ uint4 load_l2(const uint4* address) {
    uint4 value;
    asm volatile(
        "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(address));
    return value;
}

/* A 2R1W stream over vec4 words, repeated ``passes`` times over the same
 * three buffers inside one launch.  Each word is touched exactly once per
 * pass, so the launch's logical traffic is passes * 3 * per-buffer footprint.
 * Keeping the repetition inside the kernel makes the timed region
 * bandwidth-dominated at every footprint instead of launch-dominated at the
 * small ones; successive passes carry no data dependency because the probe
 * measures traffic service, not a reduction. */
__global__ void stream_2r1w(const uint4* __restrict__ left,
                            const uint4* __restrict__ right,
                            uint4* __restrict__ out,
                            long long words, int passes) {
    const long long stride =
        static_cast<long long>(gridDim.x) * blockDim.x;
    for (int pass = 0; pass < passes; ++pass) {
        for (long long index = blockIdx.x * blockDim.x + threadIdx.x;
             index < words; index += stride) {
            const uint4 a = load_l2(left + index);
            const uint4 b = load_l2(right + index);
            uint4 value;
            value.x = a.x + b.x + static_cast<unsigned int>(pass);
            value.y = a.y + b.y;
            value.z = a.z + b.z;
            value.w = a.w + b.w;
            out[index] = value;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "usage: %s <timed_passes> <footprint_mib_csv>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int timed_passes = std::atoi(argv[1]);
    if (timed_passes < 1 || timed_passes > 64) {
        std::fprintf(stderr, "timed_passes must be within 1..64\n");
        return EXIT_FAILURE;
    }

    std::vector<int> footprints;
    for (const char* cursor = argv[2]; *cursor != '\0';) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value <= 0 || value % 3 != 0) {
            std::fprintf(stderr,
                         "footprints must be positive multiples of 3 MiB\n");
            return EXIT_FAILURE;
        }
        footprints.push_back(static_cast<int>(value));
        cursor = (*end == ',') ? end + 1 : end;
    }

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    if (std::strcmp(properties.name, kExpectedDevice) != 0) {
        std::fprintf(stderr, "unexpected device: %s\n", properties.name);
        return EXIT_FAILURE;
    }

    std::printf("total_footprint_mib,buffer_bytes,timed_logical_bytes,"
                "in_kernel_passes,trial_index,duration_ms\n");
    for (const int total_mib : footprints) {
        const size_t buffer_bytes =
            static_cast<size_t>(total_mib) * 1024u * 1024u / 3u;
        const long long words = static_cast<long long>(buffer_bytes / 16u);
        uint4* left = nullptr;
        uint4* right = nullptr;
        uint4* out = nullptr;
        CHECK_CUDA(cudaMalloc(&left, buffer_bytes));
        CHECK_CUDA(cudaMalloc(&right, buffer_bytes));
        CHECK_CUDA(cudaMalloc(&out, buffer_bytes));
        CHECK_CUDA(cudaMemset(left, 0x11, buffer_bytes));
        CHECK_CUDA(cudaMemset(right, 0x22, buffer_bytes));
        CHECK_CUDA(cudaMemset(out, 0x00, buffer_bytes));

        const int grid = static_cast<int>(
            std::min<long long>((words + kThreads - 1) / kThreads,
                                kResidentCtas));
        // Hold the timed logical traffic near a constant budget so launch
        // cost is negligible and every footprint is compared in the same
        // bandwidth-dominated regime.
        const long long pass_bytes = static_cast<long long>(buffer_bytes) * 3;
        const int passes = static_cast<int>(
            std::max<long long>(1, kTrafficBudgetBytes / pass_bytes));
        for (int warmup = 0; warmup < kWarmupPasses; ++warmup) {
            stream_2r1w<<<grid, kThreads>>>(left, right, out, words, 1);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start;
        cudaEvent_t stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        for (int pass = 0; pass < timed_passes; ++pass) {
            CHECK_CUDA(cudaEventRecord(start));
            stream_2r1w<<<grid, kThreads>>>(left, right, out, words, passes);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float milliseconds = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
            std::printf("%d,%zu,%lld,%d,%d,%.6f\n", total_mib, buffer_bytes,
                        pass_bytes * passes, passes, pass,
                        static_cast<double>(milliseconds));
        }
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(left));
        CHECK_CUDA(cudaFree(right));
        CHECK_CUDA(cudaFree(out));
    }
    return EXIT_SUCCESS;
}
