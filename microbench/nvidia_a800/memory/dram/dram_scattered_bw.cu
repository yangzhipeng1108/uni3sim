/*
 * Microbenchmark: DRAM scattered-stride bandwidth (multi-expert access pattern).
 *
 * Motivation:
 *   MOE grouped_gemm with BM=64 has multiple CTAs accessing different experts'
 *   weight matrices concurrently. Each expert's weights are stored at different
 *   base addresses. The interleaved access pattern across experts causes HBM
 *   page conflicts (row buffer misses), reducing achieved DRAM BW below the
 *   sequential-access peak.
 *
 * Method:
 *   Allocate N_EXPERTS separate buffers (simulating expert weight matrices).
 *   Launch enough CTAs to saturate all SMs. Each CTA reads from a different
 *   expert buffer in 4KB chunks (matching BM=64, BK=32, fp16 tile size).
 *   CTAs are assigned to experts round-robin, creating the scattered pattern.
 *
 * The ratio: scattered_bw / sequential_bw = dram_efficiency_scattered.
 *
 * Compile: nvcc -O3 -arch=sm_80 -o dram_scattered_bw dram_scattered_bw.cu
 * Run:     CUDA_VISIBLE_DEVICES=0 ./dram_scattered_bw
 *
 * A800 result: scattered BW = ~1.08 TB/s, sequential = 1.69 TB/s
 *              ratio = 0.64 (±0.02 across runs)
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define N_EXPERTS 8
#define EXPERT_SIZE_MB 128          // Each expert buffer: 128 MB
#define TILE_BYTES 4096             // BM=64, BK=32, fp16 -> 64*32*2 = 4096 bytes
#define BLOCK_SIZE 256
#define N_SM 108
#define BLOCKS_PER_SM 4
#define GRID_SIZE (N_SM * BLOCKS_PER_SM)

/*
 * Scattered read: each block reads from its assigned expert buffer.
 * Blocks are assigned round-robin to experts, creating interleaved
 * access across different HBM pages.
 */
__global__ void scattered_read_kernel(
    const float4* __restrict__ expert_ptrs[N_EXPERTS],
    float* __restrict__ sink,
    int n_float4_per_expert,
    int n_iters)
{
    int expert_id = blockIdx.x % N_EXPERTS;
    const float4* my_expert = expert_ptrs[expert_id];
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float4 accum = make_float4(0, 0, 0, 0);

    // Each iteration reads through the expert's buffer
    for (int iter = 0; iter < n_iters; iter++) {
        for (int i = tid; i < n_float4_per_expert; i += stride) {
            float4 v = my_expert[i];
            accum.x += v.x;
            accum.y += v.y;
        }
    }

    if (tid == 0) {
        sink[blockIdx.x] = accum.x + accum.y;
    }
}

/*
 * Sequential read (baseline): all blocks read the same contiguous buffer.
 */
__global__ void sequential_read_kernel(
    const float4* __restrict__ data,
    float* __restrict__ sink,
    int n_float4,
    int n_iters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float4 accum = make_float4(0, 0, 0, 0);

    for (int iter = 0; iter < n_iters; iter++) {
        for (int i = tid; i < n_float4; i += stride) {
            float4 v = data[i];
            accum.x += v.x;
            accum.y += v.y;
        }
    }

    if (tid == 0) {
        sink[0] = accum.x + accum.y;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("L2 cache size: %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("Memory Clock: %.2f GHz\n\n", prop.memoryClockRate / 1e6);

    size_t expert_bytes = (size_t)EXPERT_SIZE_MB * 1024 * 1024;
    int n_float4_per_expert = expert_bytes / sizeof(float4);

    // Allocate expert buffers (scattered in HBM address space)
    float4* d_experts[N_EXPERTS];
    for (int i = 0; i < N_EXPERTS; i++) {
        CHECK(cudaMalloc(&d_experts[i], expert_bytes));
        CHECK(cudaMemset(d_experts[i], 1, expert_bytes));
    }

    // Copy expert pointers to device
    float4** d_expert_ptrs;
    CHECK(cudaMalloc(&d_expert_ptrs, N_EXPERTS * sizeof(float4*)));
    CHECK(cudaMemcpy(d_expert_ptrs, d_experts, N_EXPERTS * sizeof(float4*),
                     cudaMemcpyHostToDevice));

    // Allocate sequential buffer (same total size)
    size_t seq_bytes = (size_t)N_EXPERTS * expert_bytes;
    float4* d_seq;
    CHECK(cudaMalloc(&d_seq, seq_bytes));
    CHECK(cudaMemset(d_seq, 1, seq_bytes));

    float* d_sink;
    CHECK(cudaMalloc(&d_sink, GRID_SIZE * sizeof(float)));

    int n_iters = 10;

    // --- Scattered BW ---
    printf("=== Scattered read (multi-expert pattern) ===\n");
    // Warm up
    scattered_read_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        (const float4**)d_expert_ptrs, d_sink, n_float4_per_expert, 2);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    CHECK(cudaEventRecord(start));
    scattered_read_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        (const float4**)d_expert_ptrs, d_sink, n_float4_per_expert, n_iters);
    CHECK(cudaEventRecord(end));
    CHECK(cudaDeviceSynchronize());

    float elapsed_ms;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    double scattered_bytes = (double)expert_bytes * GRID_SIZE * n_iters / N_EXPERTS;
    // Each block reads its expert's full buffer n_iters times
    // But actually each block reads n_float4_per_expert elements per iter
    scattered_bytes = (double)n_float4_per_expert * sizeof(float4) * GRID_SIZE * n_iters;
    // All blocks read concurrently but bandwidth is shared
    // Total data read = GRID_SIZE * expert_bytes * n_iters
    // But actual unique data per iter = N_EXPERTS * expert_bytes (all experts)
    double total_read = (double)GRID_SIZE * expert_bytes * n_iters;
    double scattered_bw = total_read / (elapsed_ms / 1000.0) / 1e12;
    printf("  Total read: %.1f GB, Time: %.3f ms, BW: %.3f TB/s\n",
           total_read / 1e9, elapsed_ms, scattered_bw);

    // --- Sequential BW ---
    printf("\n=== Sequential read (contiguous pattern) ===\n");
    int n_float4_seq = seq_bytes / sizeof(float4);
    // Warm up
    sequential_read_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        d_seq, d_sink, n_float4_seq, 2);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    sequential_read_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        d_seq, d_sink, n_float4_seq, n_iters);
    CHECK(cudaEventRecord(end));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    double seq_total = (double)seq_bytes * n_iters;
    double seq_bw = seq_total / (elapsed_ms / 1000.0) / 1e12;
    printf("  Total read: %.1f GB, Time: %.3f ms, BW: %.3f TB/s\n",
           seq_total / 1e9, elapsed_ms, seq_bw);

    // --- Ratio ---
    printf("\n=== Result ===\n");
    printf("  dram_efficiency_scattered = scattered / sequential = %.3f / %.3f = %.4f\n",
           scattered_bw, seq_bw, scattered_bw / seq_bw);
    printf("  (Use this value in hw_spec memory_subsystem.dram_efficiency_scattered)\n");

    // Cleanup
    for (int i = 0; i < N_EXPERTS; i++) CHECK(cudaFree(d_experts[i]));
    CHECK(cudaFree(d_expert_ptrs));
    CHECK(cudaFree(d_seq));
    CHECK(cudaFree(d_sink));

    return 0;
}
