/*
 * Microbenchmark v2: test cp.async (ldgsts) vs normal ld.global xbar counting.
 *
 * Test: N blocks read same 1MB via cp.async into shared memory.
 * Compare xbar_read with the normal load version.
 *
 * Build: nvcc -O3 -arch=sm_80 -o l2_hit_counter_v2 l2_hit_counter_v2.cu
 * Profile: ncu --kernel-name regex:"read_.*" --launch-skip 1 \
 *     --metrics l1tex__m_xbar2l1tex_read_bytes.sum,dram__bytes_read.sum,lts__t_sector_hit_rate.pct,sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum \
 *     ./l2_hit_counter_v2
 */
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                           \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, cudaGetErrorString(error));       \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

// Normal load: each block reads same 1MB into registers
__global__ void read_normal_shared(const float* __restrict__ buf, float* out, int n_elements) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n_elements; i += blockDim.x) {
        sum += buf[i];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// cp.async: each block copies same 1MB into shared memory via cp.async
__global__ void read_cpasync_shared(const float* __restrict__ buf, float* out, int n_elements) {
    extern __shared__ float smem[];
    
    // Use cp.async to load global → shared memory
    // cp.async bypasses L1 cache
    auto pipe = cuda::make_pipeline();
    
    const int chunk = 128;  // 128 floats = 512 bytes per cp.async
    float sum = 0.0f;
    
    for (int base = 0; base < n_elements; base += blockDim.x * chunk) {
        int idx = base + threadIdx.x * chunk;
        int smem_idx = threadIdx.x * chunk;
        
        if (idx + chunk <= n_elements) {
            pipe.producer_acquire();
            // cp.async: global → shared (L1 bypass)
            cuda::memcpy_async(&smem[smem_idx], &buf[idx], 
                              cuda::aligned_size_t<16>(chunk * sizeof(float)), pipe);
            pipe.producer_commit();
            pipe.consumer_wait();
            
            // Read from shared memory
            for (int j = 0; j < chunk; j++) {
                sum += smem[smem_idx + j];
            }
            pipe.consumer_release();
        }
    }
    
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

int main() {
    const int DATA_SIZE = 1 * 1024 * 1024;  // 1MB
    const int N_ELEMENTS = DATA_SIZE / sizeof(float);
    const int N_BLOCKS = 108;
    const int BLOCK_SIZE = 128;  // smaller for shared memory
    const int SMEM_SIZE = BLOCK_SIZE * 128 * sizeof(float);  // 128 floats per thread

    float *d_buf, *d_out;
    CUDA_CHECK(cudaMalloc(&d_buf, DATA_SIZE));
    CUDA_CHECK(cudaMalloc(&d_out, N_BLOCKS * sizeof(float)));

    // Warm up L2
    read_normal_shared<<<1, BLOCK_SIZE>>>(d_buf, d_out, N_ELEMENTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("=== Test 1: Normal ld.global, 108 blocks read SAME 1MB ===\n");
    read_normal_shared<<<N_BLOCKS, BLOCK_SIZE>>>(d_buf, d_out, N_ELEMENTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warm up L2 again
    read_normal_shared<<<1, BLOCK_SIZE>>>(d_buf, d_out, N_ELEMENTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("=== Test 2: cp.async (ldgsts), 108 blocks read SAME 1MB ===\n");
    read_cpasync_shared<<<N_BLOCKS, BLOCK_SIZE, SMEM_SIZE>>>(d_buf, d_out, N_ELEMENTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
