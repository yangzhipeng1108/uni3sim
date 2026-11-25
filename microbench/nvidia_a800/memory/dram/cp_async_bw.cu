/**
 * cp.async microbenchmark v2: maximize outstanding to match FA.
 * 
 * FA issues many cp.async per thread before commit/wait.
 * This version pipelines: issue 4 SMEM chunks before waiting.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(error));           \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

static int parse_positive_integer(const char* text, const char* name) {
    char* end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value <= 0 || value > INT_MAX) {
        fprintf(stderr, "%s must be a positive integer, got %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cp_async_wait_group(int n) {
    // Wait until at most n groups pending
    if (n == 0) asm volatile("cp.async.wait_group 0;");
    else if (n == 1) asm volatile("cp.async.wait_group 1;");
    else if (n == 2) asm volatile("cp.async.wait_group 2;");
    else if (n == 3) asm volatile("cp.async.wait_group 3;");
    else asm volatile("trap;");
}

#define SMEM_SIZE (32 * 1024)
#define COPY_SIZE 16
#define PIPELINE_STAGES 4  // match FA's double-buffer depth

__global__ void cp_async_pipelined_kernel(const char* __restrict__ gmem_in,
                                           int n_bytes_per_cta) {
    __shared__ char smem[SMEM_SIZE];  // single buffer, reuse each iteration

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int n_threads = blockDim.x;
    const int copies_per_thread = SMEM_SIZE / COPY_SIZE / n_threads;

    const char* my_gmem = gmem_in + (long long)block_id * n_bytes_per_cta;

    int offset = 0;

    // Pipeline: issue several groups before waiting
    while (offset < n_bytes_per_cta) {
        int chunk = (SMEM_SIZE < n_bytes_per_cta - offset) ? SMEM_SIZE : (n_bytes_per_cta - offset);
        char* dst = smem;

        for (int c = 0; c < copies_per_thread; c++) {
            int byte_offset = (tid * copies_per_thread + c) * COPY_SIZE;
            if (byte_offset < chunk) {
                cp_async_16(dst + byte_offset, my_gmem + offset + byte_offset);
            }
        }
        cp_async_commit();

        // Wait for oldest group if pipeline full
        if (offset >= PIPELINE_STAGES * SMEM_SIZE) {
            cp_async_wait_group(PIPELINE_STAGES - 1);
            __syncthreads();
        }

        offset += SMEM_SIZE;
    }

    // Drain pipeline
    cp_async_wait_group(0);
    __syncthreads();
}

int main(int argc, char** argv) {
    if (argc > 3) {
        fprintf(stderr, "usage: %s [size_mb] [max_grid]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int size_mb = (argc > 1)
        ? parse_positive_integer(argv[1], "size_mb") : 256;
    int max_grid = (argc > 2)
        ? parse_positive_integer(argv[2], "max_grid") : 64;

    long long total_bytes = (long long)size_mb * 1024 * 1024;
    char* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, total_bytes));
    CUDA_CHECK(cudaMemset(d_buf, 0x42, total_bytes));

    const int block_size = 128;  // match FA

    // Set max SMEM
    CUDA_CHECK(cudaFuncSetAttribute(
        cp_async_pipelined_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 0));

    // Warmup
    int n_bytes = total_bytes;
    cp_async_pipelined_kernel<<<1, block_size>>>(d_buf, n_bytes);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("cp.async pipelined DRAM→SMEM BW (%d MB, block=%d, pipeline=%d)\n",
           size_mb, block_size, PIPELINE_STAGES);
    printf("%-8s %10s %10s %10s\n", "grid", "time_ms", "BW_GBs", "per_SM_GBs");
    printf("----------------------------------------------\n");

    // A800 exposes 108 SMs.  The last three points separate under-filled,
    // one-CTA-per-SM, and multi-CTA residency without using FA work.
    int grid_sizes[] = {1, 2, 4, 8, 16, 32, 64, 108, 216, 432};
    for (int i = 0; i < 10; i++) {
        int grid = grid_sizes[i];
        if (grid > max_grid) break;

        // Keep every CTA base and every iteration on a full 32 KiB chunk.
        // cp.async requires 16-byte aligned source and destination addresses;
        // an arbitrary total_bytes / grid quotient violates that contract.
        int per_cta = (int)((total_bytes / grid) / SMEM_SIZE) * SMEM_SIZE;
        if (per_cta < SMEM_SIZE * 2) per_cta = SMEM_SIZE * 2;

        for (int w = 0; w < 3; w++)
            cp_async_pipelined_kernel<<<grid, block_size>>>(d_buf, per_cta);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        int reps = 10;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < reps; r++)
            cp_async_pipelined_kernel<<<grid, block_size>>>(d_buf, per_cta);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        elapsed_ms /= reps;

        long long actual = (long long)per_cta * grid;
        double bw = (double)actual / (elapsed_ms * 1e-3) / 1e9;
        printf("%-8d %10.3f %10.1f %10.1f\n", grid, elapsed_ms, bw, bw / grid);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    printf("\nPE_MAX = per_SM_BW / (peak_DRAM / SM_sat) = per_SM_BW / 26.2\n");
    CUDA_CHECK(cudaFree(d_buf));
    return 0;
}
