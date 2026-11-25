/* Per-SM DRAM bandwidth saturation benchmark for NVIDIA H200.
 *
 * Owner copy of microbench/nvidia_a800/memory/dram/bw_per_sm.cu.  The kernels,
 * the incompressible 2 GiB initialization, the timing and the output are
 * unchanged.  Hardware coupling changed for H200 only:
 *   - build with -arch=sm_90 (A800: sm_80);
 *   - the active-SM sweep ends at the 132-SM full chip instead of the 108-SM
 *     A800 full chip (the point list replaces 108 with 132);
 *   - the H200 runner also passes threads_per_block=1024 through the existing
 *     command-line argument, because one 256-thread CTA per SM does not hold
 *     enough bytes in flight to saturate HBM3e.  No code path changes.
 *
 * Build: nvcc -O3 -arch=sm_90 -cudart static -o bw_per_sm bw_per_sm.cu
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

static int parse_positive_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value <= 0 || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a positive integer, got %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

#define CUDA_CHECK(call) do {                                               \
    cudaError_t error = (call);                                             \
    if (error != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

__device__ __forceinline__ unsigned mix_bits(unsigned value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__global__ void initialize_incompressible(float4* data, long long count) {
    const long long thread =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride =
        static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long index = thread; index < count; index += stride) {
        unsigned seed = mix_bits(static_cast<unsigned>(index));
        data[index] = make_float4(
            static_cast<float>((seed & 0xffffU) + 1U),
            static_cast<float>(((seed >> 16) & 0xffffU) + 1U),
            static_cast<float>((mix_bits(seed) & 0xffffU) + 1U),
            static_cast<float>(((mix_bits(seed) >> 16) & 0xffffU) + 1U));
    }
}

__global__ void bypass_l1_kernel(
    const float4* __restrict__ data,
    float* __restrict__ sink,
    long long element_count,
    int iterations
) {
    const long long thread =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride =
        static_cast<long long>(blockDim.x) * gridDim.x;
    float sum = 0.0F;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (long long index = thread; index < element_count; index += stride) {
            float4 value;
            asm volatile(
                "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=f"(value.x), "=f"(value.y),
                  "=f"(value.z), "=f"(value.w)
                : "l"(data + index));
            sum += value.x + value.y + value.z + value.w;
        }
    }
    if (thread == 0) {
        sink[0] = sum;
    }
}

__global__ void cached_kernel(
    const float4* __restrict__ data,
    float* __restrict__ sink,
    long long element_count,
    int iterations
) {
    const long long thread =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride =
        static_cast<long long>(blockDim.x) * gridDim.x;
    float sum = 0.0F;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (long long index = thread; index < element_count; index += stride) {
            float4 value;
            asm volatile(
                "ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=f"(value.x), "=f"(value.y),
                  "=f"(value.z), "=f"(value.w)
                : "l"(data + index));
            sum += value.x + value.y + value.z + value.w;
        }
    }
    if (thread == 0) {
        sink[0] = sum;
    }
}

static double measure(
    const float4* data,
    float* sink,
    long long element_count,
    int grid_size,
    int block_size,
    int iterations,
    bool bypass_l1
) {
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    if (bypass_l1) {
        bypass_l1_kernel<<<grid_size, block_size>>>(
            data, sink, element_count, iterations);
    } else {
        cached_kernel<<<grid_size, block_size>>>(
            data, sink, element_count, iterations);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    if (elapsed_ms <= 0.0F) {
        std::fprintf(stderr, "non-positive benchmark duration\n");
        std::exit(EXIT_FAILURE);
    }
    const double bytes = static_cast<double>(element_count)
        * sizeof(float4) * iterations;
    return bytes / (elapsed_ms / 1000.0) / 1.0e9;
}

static void run_sweep(
    const char* label,
    const float4* data,
    float* sink,
    long long element_count,
    int block_size,
    int sm_count,
    bool bypass_l1
) {
    const int points[] = {
        1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 132,
    };
    std::printf("\n%s\n%8s %14s %14s\n", label, "SMs", "total GB/s", "per-SM GB/s");
    for (const int active_sms : points) {
        if (active_sms > sm_count) {
            continue;
        }
        const int iterations = active_sms <= 4 ? 3 : (active_sms <= 16 ? 5 : 10);
        const double total = measure(
            data, sink, element_count, active_sms, block_size,
            iterations, bypass_l1);
        std::printf("%8d %14.2f %14.2f\n",
                    active_sms, total, total / active_sms);
    }
}

int main(int argc, char** argv) {
    if (argc > 3) {
        std::fprintf(stderr, "usage: %s [buffer_MB] [threads_per_block]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int buffer_mb = argc > 1
        ? parse_positive_integer(argv[1], "buffer_MB") : 2048;
    const int block_size = argc > 2
        ? parse_positive_integer(argv[2], "threads_per_block") : 256;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    const size_t buffer_bytes =
        static_cast<size_t>(buffer_mb) * 1024 * 1024;
    const long long element_count = buffer_bytes / sizeof(float4);
    float4* data = nullptr;
    float* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&data, buffer_bytes));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    initialize_incompressible<<<properties.multiProcessorCount * 4, 256>>>(
        data, element_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    run_sweep("L1-bypass", data, sink, element_count,
              block_size, properties.multiProcessorCount, true);
    run_sweep("L1-cached", data, sink, element_count,
              block_size, properties.multiProcessorCount, false);
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(sink));
    return EXIT_SUCCESS;
}
