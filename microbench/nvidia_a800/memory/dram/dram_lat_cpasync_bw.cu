/*
 * Microbenchmark for A800 hardware constants.
 * Compile: nvcc -O3 -arch=sm_80 -o microbench scripts/microbench.cu
 * Run: CUDA_VISIBLE_DEVICES=0 ./microbench
 *
 * Measures:
 * 1. C_launch: empty kernel dispatch overhead
 * 2. L2 BW: read bandwidth with L1 bypass (ld.global.cs)
 * 3. DRAM BW: read bandwidth with large working set
 * 4. L1 BW: read bandwidth with default loads (L1 cached) for reference
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm>

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* ============================================================
 * Kernel 1: Empty kernel for C_launch measurement
 * ============================================================ */
__global__ void empty_kernel() {}

/* ============================================================
 * Kernel 2: Read BW kernel — L1 BYPASS via ld.global.cg (L2 only)
 * .cg = "Cache at Global level": bypasses L1, caches in L2 only.
 * .cs = "Cache Streaming": still hits L1, just lower eviction priority.
 * ============================================================ */
__device__ __forceinline__ float4 load_l2_only(const float4* addr) {
    float4 ret;
    asm volatile(
        "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
        : "l"(addr)
    );
    return ret;
}

__global__ void bw_read_l2_kernel(const float4* __restrict__ data,
                                   float* __restrict__ sink,
                                   int n_float4,
                                   int n_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float4 accum = make_float4(0, 0, 0, 0);

    for (int iter = 0; iter < n_iters; iter++) {
        for (int i = tid; i < n_float4; i += stride) {
            float4 v = load_l2_only(&data[i]);
            accum.x += v.x;
            accum.y += v.y;
            accum.z += v.z;
            accum.w += v.w;
        }
    }

    if (tid == 0) {
        sink[0] = accum.x + accum.y + accum.z + accum.w;
    }
}

/* ============================================================
 * Kernel 3: Read BW kernel — DEFAULT loads (L1 cached) for reference
 * ============================================================ */
__global__ void bw_read_l1_kernel(const float4* __restrict__ data,
                                   float* __restrict__ sink,
                                   int n_float4,
                                   int n_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float4 accum = make_float4(0, 0, 0, 0);

    for (int iter = 0; iter < n_iters; iter++) {
        for (int i = tid; i < n_float4; i += stride) {
            float4 v = data[i];  // default load — goes through L1
            accum.x += v.x;
            accum.y += v.y;
            accum.z += v.z;
            accum.w += v.w;
        }
    }

    if (tid == 0) {
        sink[0] = accum.x + accum.y + accum.z + accum.w;
    }
}

/* ============================================================
 * Benchmark functions
 * ============================================================ */

void bench_launch_overhead() {
    printf("======================================================================\n");
    printf("  Benchmark 1: C_launch (empty kernel dispatch overhead)\n");
    printf("======================================================================\n");

    // Warm up
    for (int i = 0; i < 100; i++) {
        empty_kernel<<<1, 1>>>();
    }
    CHECK(cudaDeviceSynchronize());

    const int N = 1000;
    cudaEvent_t starts[N], ends[N];
    for (int i = 0; i < N; i++) {
        CHECK(cudaEventCreate(&starts[i]));
        CHECK(cudaEventCreate(&ends[i]));
    }

    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < N; i++) {
        CHECK(cudaEventRecord(starts[i]));
        empty_kernel<<<1, 1>>>();
        CHECK(cudaEventRecord(ends[i]));
    }
    CHECK(cudaDeviceSynchronize());

    float times[N];
    for (int i = 0; i < N; i++) {
        CHECK(cudaEventElapsedTime(&times[i], starts[i], ends[i]));
        times[i] *= 1000.0f; // ms -> us
    }

    std::sort(times + 100, times + N);
    int start_idx = 100;
    int count = N - start_idx;
    float sum = 0;
    for (int i = start_idx; i < N; i++) sum += times[i];
    float mean = sum / count;
    float median = times[start_idx + count / 2];
    float p10 = times[start_idx + count / 10];
    float p90 = times[start_idx + count * 9 / 10];

    printf("  n=%d  mean=%.2f us  median=%.2f us  P10=%.2f us  P90=%.2f us\n",
           count, mean, median, p10, p90);
    printf("  -> C_launch = %.2f us\n\n", median);

    for (int i = 0; i < N; i++) {
        CHECK(cudaEventDestroy(starts[i]));
        CHECK(cudaEventDestroy(ends[i]));
    }
}

// mode: 0 = L2 bypass (ld.cs), 1 = L1 cached (default)
void bench_bw(const char* label, size_t size_bytes, int n_iters, int mode) {
    if (mode != 0 && mode != 1) {
        fprintf(stderr, "unsupported bandwidth mode: %d\n", mode);
        exit(EXIT_FAILURE);
    }
    int n_float4 = size_bytes / sizeof(float4);
    size_t alloc_bytes = (size_t)n_float4 * sizeof(float4);

    float4* d_data;
    float* d_sink;
    CHECK(cudaMalloc(&d_data, alloc_bytes));
    CHECK(cudaMalloc(&d_sink, sizeof(float)));
    CHECK(cudaMemset(d_data, 1, alloc_bytes));

    int block_size = 256;
    int grid_size = 108 * 4;  // 4 blocks per SM

    // Warm up
    for (int i = 0; i < 20; i++) {
        if (mode == 0)
            bw_read_l2_kernel<<<grid_size, block_size>>>(d_data, d_sink, n_float4, 1);
        else if (mode == 1)
            bw_read_l1_kernel<<<grid_size, block_size>>>(d_data, d_sink, n_float4, 1);
        else
            abort();
    }
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    if (mode == 0)
        bw_read_l2_kernel<<<grid_size, block_size>>>(d_data, d_sink, n_float4, n_iters);
    else if (mode == 1)
        bw_read_l1_kernel<<<grid_size, block_size>>>(d_data, d_sink, n_float4, n_iters);
    else
        abort();
    CHECK(cudaEventRecord(end));
    CHECK(cudaDeviceSynchronize());

    float elapsed_ms;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    double elapsed_s = elapsed_ms / 1000.0;
    double total_bytes_read = (double)alloc_bytes * n_iters;
    double bw_tb = total_bytes_read / elapsed_s / 1e12;

    printf("  %-12s  WS=%5.0f MB  iters=%4d  time=%8.3f ms  BW=%.3f TB/s\n",
           label, alloc_bytes / 1e6, n_iters, elapsed_ms, bw_tb);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_sink));
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("L2 cache size: %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Clock: %.2f GHz\n\n", prop.clockRate / 1e6);

    // 1. Launch overhead
    bench_launch_overhead();

    // 2. L2 BW — ld.global.cs bypass L1
    printf("======================================================================\n");
    printf("  Benchmark 2: L2 BW (ld.global.cg — L1 bypass, L2 only)\n");
    printf("======================================================================\n");
    bench_bw("L2_1MB",    1 * 1024 * 1024, 2000, 0);
    bench_bw("L2_4MB",    4 * 1024 * 1024, 500, 0);
    bench_bw("L2_8MB",    8 * 1024 * 1024, 300, 0);
    bench_bw("L2_16MB",  16 * 1024 * 1024, 200, 0);
    bench_bw("L2_20MB",  20 * 1024 * 1024, 200, 0);
    bench_bw("L2_32MB",  32 * 1024 * 1024, 100, 0);
    bench_bw("L2_38MB",  38 * 1024 * 1024, 100, 0);
    printf("\n");

    // 3. DRAM BW — ld.global.cs (L1 bypass, working set >> L2)
    printf("======================================================================\n");
    printf("  Benchmark 3: DRAM BW (working set >> L2)\n");
    printf("======================================================================\n");
    bench_bw("DRAM_64MB",   64 * 1024 * 1024, 50, 0);
    bench_bw("DRAM_128MB", 128 * 1024 * 1024, 30, 0);
    bench_bw("DRAM_256MB", 256 * 1024 * 1024, 20, 0);
    bench_bw("DRAM_512MB", 512 * 1024 * 1024, 10, 0);
    bench_bw("DRAM_1GB",  1024 * 1024 * 1024, 5, 0);
    printf("\n");

    // 4. L1 BW reference — default loads (cached in L1)
    printf("======================================================================\n");
    printf("  Benchmark 4: L1 BW reference (default loads, L1 cached)\n");
    printf("======================================================================\n");
    bench_bw("L1_1MB",    1 * 1024 * 1024, 2000, 1);
    bench_bw("L1_4MB",    4 * 1024 * 1024, 500, 1);
    bench_bw("L1_8MB",    8 * 1024 * 1024, 300, 1);
    bench_bw("L1_16MB",  16 * 1024 * 1024, 200, 1);
    printf("\n");

    // 5. Full sweep — L2 bypass only
    printf("======================================================================\n");
    printf("  Benchmark 5: BW vs working set sweep (ld.global.cs)\n");
    printf("======================================================================\n");
    int sizes_mb[] = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64, 96, 128, 256, 512};
    int n_sizes = sizeof(sizes_mb) / sizeof(sizes_mb[0]);
    for (int i = 0; i < n_sizes; i++) {
        int mb = sizes_mb[i];
        int iters = mb <= 40 ? 200 : (mb <= 128 ? 50 : 10);
        char label[32];
        snprintf(label, sizeof(label), "%dMB", mb);
        bench_bw(label, (size_t)mb * 1024 * 1024, iters, 0);
    }

    return 0;
}
