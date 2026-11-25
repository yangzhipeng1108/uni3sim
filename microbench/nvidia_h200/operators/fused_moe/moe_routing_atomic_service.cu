/*
 * H200 owner copy of
 * microbench/nvidia_a800/operators/fused_moe/moe_routing_atomic_service.cu.
 * The source is unchanged apart from this header: the cases carry no SM
 * coupling (the 128-CTA E32 grid is below both 108 and 132 SMs) and the
 * build target (-arch=sm_90) is set by the H200 runner.
 *
 * Independent MoE-routing atomic-service probe.
 *
 * This is deliberately not a Fused-MoE implementation: it has no expert
 * weights, activation tensors, tensor-core instructions, sort output, or
 * production input.  It isolates the two atomic mechanisms selected by the
 * vLLM routing source: one 1024-thread CTA that updates shared counters, and
 * a 256-thread CTA grid that updates global counters.  Counter destinations
 * use a deterministic uniform synthetic histogram so the mechanism is
 * reproducible without consuming a production routing histogram.
 */

#include <cuda_runtime.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(error));           \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

struct Case {
    const char* name;
    int mode;
    int items;
    int experts;
    int grid;
    int block;
};

constexpr int SHARED_MODE = 0;
constexpr int GLOBAL_MODE = 1;
constexpr Case CASES[] = {
    {"shared_cta_e128_r262144", SHARED_MODE, 262144, 128, 1, 1024},
    {"global_grid1024_e128_r262144", GLOBAL_MODE, 262144, 128, 1024, 256},
    {"global_grid128_e32_r32768", GLOBAL_MODE, 32768, 32, 128, 256},
    {"global_grid256_e8_r65536", GLOBAL_MODE, 65536, 8, 256, 256},
};
constexpr int CASE_COUNT = sizeof(CASES) / sizeof(CASES[0]);

static bool parse_int(const char* text, int* output) {
    errno = 0;
    char* end = nullptr;
    const long value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value < INT_MIN || value > INT_MAX) {
        return false;
    }
    *output = (int)value;
    return true;
}

__device__ __forceinline__ int counter_index(int item, int experts) {
    // Coprime odd stride produces a fixed uniform synthetic histogram for
    // every power-of-two expert count in this probe.
    return (item * 17) & (experts - 1);
}

__global__ void shared_atomic_service(int items, int experts, int* checksum) {
    extern __shared__ int counters[];
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        counters[expert] = 0;
    }
    __syncthreads();
    for (int item = threadIdx.x; item < items; item += blockDim.x) {
        atomicAdd(&counters[counter_index(item, experts)], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int expert = 0; expert < experts; ++expert) total += counters[expert];
        *checksum = total;
    }
}

__global__ void global_atomic_service(
        int items, int experts, int* counters) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < items) {
        atomicAdd(&counters[counter_index(item, experts)], 1);
    }
}

static void run_case(const Case& spec, int samples) {
    int* counters = nullptr;
    CUDA_CHECK(cudaMalloc(&counters, (size_t)spec.experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(counters, 0, (size_t)spec.experts * sizeof(int)));
    if (spec.mode == SHARED_MODE) {
        shared_atomic_service<<<spec.grid, spec.block,
            (size_t)spec.experts * sizeof(int)>>>(
                spec.items, spec.experts, counters);
    } else {
        global_atomic_service<<<spec.grid, spec.block>>>(
            spec.items, spec.experts, counters);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(counters, 0, (size_t)spec.experts * sizeof(int)));

    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaMemsetAsync(
            counters, 0, (size_t)spec.experts * sizeof(int)));
        CUDA_CHECK(cudaEventRecord(start));
        if (spec.mode == SHARED_MODE) {
            shared_atomic_service<<<spec.grid, spec.block,
                (size_t)spec.experts * sizeof(int)>>>(
                    spec.items, spec.experts, counters);
        } else {
            global_atomic_service<<<spec.grid, spec.block>>>(
                spec.items, spec.experts, counters);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
        if (!(milliseconds > 0.0f)) {
            fprintf(stderr, "non-positive event time\n");
            exit(EXIT_FAILURE);
        }
        const double rate = spec.items / (milliseconds * 1.0e-3);
        printf("%s,%d,%d,%d,%d,%d,%d,%.9g,%.9g\n", spec.name, sample,
               spec.mode, spec.items, spec.experts, spec.grid, spec.block,
               milliseconds, rate);
    }
    int* host_counters = static_cast<int*>(
        malloc((size_t)spec.experts * sizeof(int)));
    if (host_counters == nullptr) {
        fprintf(stderr, "host counter allocation failed\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(
        host_counters, counters, (size_t)spec.experts * sizeof(int),
        cudaMemcpyDeviceToHost));
    int observed_total = 0;
    for (int expert = 0; expert < spec.experts; ++expert) {
        observed_total += host_counters[expert];
    }
    free(host_counters);
    if (observed_total != spec.items) {
        fprintf(stderr, "atomic checksum does not conserve items\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "validated case=%s total=%d\n", spec.name,
            observed_total);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(counters));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s CASE_ID SAMPLES\n", argv[0]);
        return EXIT_FAILURE;
    }
    int case_id = 0;
    int samples = 0;
    if (!parse_int(argv[1], &case_id) || !parse_int(argv[2], &samples)
            || case_id < 0 || case_id >= CASE_COUNT || samples <= 0) {
        fprintf(stderr, "CASE_ID and positive SAMPLES must be valid\n");
        return EXIT_FAILURE;
    }
    run_case(CASES[case_id], samples);
    return EXIT_SUCCESS;
}
