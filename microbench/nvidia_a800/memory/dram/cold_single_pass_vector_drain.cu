/*
 * NVIDIA A800 cold single-pass vector-stream drain mechanism benchmark.
 *
 * The generic GPU vector path prices a single-pass operand stream as a
 * first-request access latency repeated once per source loop round and taken
 * serially.  That representation has no measured basis: a 128-thread CTA
 * issues every round's loads before consuming any of them, so a round's
 * requests leave as one burst and successive rounds overlap rather than
 * retiring one access latency at a time.  The published segmented SiLU
 * service already encodes that burst behaviour for its own 1024-thread
 * source, which is why only the generic path still serializes.
 *
 * This probe measures the missing quantity for the generic class: the
 * achieved logical byte rate of a COLD single-pass two-read / one-write vec4
 * stream, indexed by its CTA grid and its total footprint.  Cold is the
 * point -- every timed launch is preceded by an explicit L2-eviction sweep,
 * so the operand stream is a compulsory miss exactly as it is in the
 * canonical capture and in any launch whose producer did not just run.  A
 * single in-kernel pass is likewise the point: repeating passes inside the
 * kernel (as the L2 residency knee probe deliberately does) would measure a
 * warm re-read instead.
 *
 * Reads use ld.global.cg so they bypass L1 and allocate in L2, matching the
 * coalesced sector stream the accepted L2 read envelope was measured with.
 * Writes use the default write-back policy so the dirty-line pressure of a
 * real single-pass vector stream is present.
 *
 * It contains no production operator, no normalization or activation
 * arithmetic, and no production or evaluation duration: the consumer is a
 * sum of the loaded words, and the reported quantity is a duration at a
 * (grid, footprint) launch geometry.  It identifies neither partitions,
 * request credits, nor a per-operator correction.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o cold_single_pass_vector_drain \
 *        cold_single_pass_vector_drain.cu
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
/* Eight times the 40 MiB GA100 L2, matching the eviction extent the
 * canonical A800 timing runner requires of its own pre-NVTX sweep. */
constexpr size_t kFlushBytes = 320u * 1024u * 1024u;
constexpr const char* kExpectedDevice = "NVIDIA A800-SXM4-80GB";

__device__ __forceinline__ uint4 load_l2(const uint4* address) {
    uint4 value;
    asm volatile(
        "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(address));
    return value;
}

/* Overwrite a buffer far larger than L2 so no line of the measured stream
 * survives into the timed launch. */
__global__ void evict_l2(uint4* __restrict__ scratch, long long words) {
    const long long stride =
        static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long index = blockIdx.x * blockDim.x + threadIdx.x;
         index < words; index += stride) {
        uint4 value;
        value.x = static_cast<unsigned int>(index);
        value.y = value.x;
        value.z = value.x;
        value.w = value.x;
        scratch[index] = value;
    }
}

/* One single-pass 2R1W grid-stride stream over vec4 words.  Each word is
 * touched exactly once, so the launch's logical traffic is exactly three
 * times the per-buffer footprint. */
__global__ void cold_stream_2r1w(const uint4* __restrict__ left,
                                 const uint4* __restrict__ right,
                                 uint4* __restrict__ out,
                                 long long words) {
    const long long stride =
        static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long index = blockIdx.x * blockDim.x + threadIdx.x;
         index < words; index += stride) {
        const uint4 a = load_l2(left + index);
        const uint4 b = load_l2(right + index);
        uint4 value;
        value.x = a.x + b.x;
        value.y = a.y + b.y;
        value.z = a.z + b.z;
        value.w = a.w + b.w;
        out[index] = value;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s <trials> <grid_csv> <footprint_mib_csv>\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    const int trials = std::atoi(argv[1]);
    if (trials < 1 || trials > 64) {
        std::fprintf(stderr, "trials must be within 1..64\n");
        return EXIT_FAILURE;
    }

    std::vector<int> grids;
    for (const char* cursor = argv[2]; *cursor != '\0';) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value <= 0 || value > 65535) {
            std::fprintf(stderr, "grids must be within 1..65535\n");
            return EXIT_FAILURE;
        }
        grids.push_back(static_cast<int>(value));
        cursor = (*end == ',') ? end + 1 : end;
    }

    std::vector<int> footprints;
    for (const char* cursor = argv[3]; *cursor != '\0';) {
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

    uint4* flush = nullptr;
    CHECK_CUDA(cudaMalloc(&flush, kFlushBytes));
    const long long flush_words =
        static_cast<long long>(kFlushBytes / sizeof(uint4));

    std::printf("grid_ctas,threads_per_cta,total_footprint_mib,buffer_bytes,"
                "logical_bytes,trial_index,duration_ms\n");
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

        for (const int grid : grids) {
            cudaEvent_t start;
            cudaEvent_t stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            for (int trial = 0; trial < trials; ++trial) {
                /* Cold every time: the sweep is outside the timed region and
                 * is synchronized before the event is recorded. */
                evict_l2<<<1024, 256>>>(flush, flush_words);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaEventRecord(start));
                cold_stream_2r1w<<<grid, kThreads>>>(left, right, out, words);
                CHECK_CUDA(cudaEventRecord(stop));
                CHECK_CUDA(cudaEventSynchronize(stop));
                float milliseconds = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
                std::printf("%d,%d,%d,%zu,%lld,%d,%.6f\n", grid, kThreads,
                            total_mib, buffer_bytes,
                            static_cast<long long>(buffer_bytes) * 3, trial,
                            static_cast<double>(milliseconds));
            }
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
        }
        CHECK_CUDA(cudaFree(left));
        CHECK_CUDA(cudaFree(right));
        CHECK_CUDA(cudaFree(out));
    }
    CHECK_CUDA(cudaFree(flush));
    return EXIT_SUCCESS;
}
