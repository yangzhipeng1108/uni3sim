/*
 * Microbenchmark: DRAM row-buffer conflict efficiency curve.
 *
 * Replicates the access pattern from dram_scattered_bw.cu (each CTA reads
 * its entire assigned buffer independently) while sweeping N_streams to
 * produce a continuous dram_eff(N_streams) curve.
 *
 * Why this pattern: the original microbench showed 0.64 efficiency at 8
 * streams vs 1 stream. With cooperative striding (each addr read once),
 * there's no conflict. The 0.64 comes from L2 thrashing when multiple
 * CTAs independently sweep through different large (>L2) buffers —
 * their interleaved accesses evict each other from L2, forcing DRAM
 * re-reads. This IS the physical effect in MOE grouped-GEMM: each
 * expert's weight is too large for L2, and concurrent reading across
 * experts causes L2 conflict misses beyond the cold misses.
 *
 * Sweep axes:
 *   N_streams ∈ {1, 2, 4, 8, 16, 32}
 *
 * For each N_streams:
 *   Sequential baseline: all CTAs cooperatively stride through N_streams
 *   contiguous buffers (total = N × 128 MB). Each address read once.
 *   Scattered: CTAs round-robin to N_streams separate buffers, each CTA
 *   independently reads its entire buffer. Multiple CTAs per buffer cause
 *   L2 set conflicts between different buffers' addresses.
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o dram_conflict_curve dram_conflict_curve.cu
 * Run:
 *   CUDA_VISIBLE_DEVICES=0 ./dram_conflict_curve
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define BUFFER_SIZE_MB 128
#define BLOCK_SIZE 256
#define N_SM 108
#define BLOCKS_PER_SM 4
#define GRID_SIZE (N_SM * BLOCKS_PER_SM)
#define MAX_STREAMS 32
#define N_WARMUP 2
#define N_MEASURE 10

/*
 * Scattered: each CTA independently reads its ENTIRE assigned buffer.
 * Multiple CTAs per stream all read the same buffer independently,
 * creating inter-stream L2 set conflicts.
 */
__global__ void scattered_read_kernel(
    const float4* __restrict__* __restrict__ stream_ptrs,
    float* __restrict__ sink,
    int n_streams,
    int n_float4_per_buf,
    int n_iters)
{
    int stream_id = blockIdx.x % n_streams;
    const float4* my_buf = stream_ptrs[stream_id];
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int iter = 0; iter < n_iters; iter++) {
        for (int i = tid; i < n_float4_per_buf; i += stride) {
            float4 v = my_buf[i];
            acc.x += v.x;
            acc.y += v.y;
        }
    }
    if (tid == 0) sink[blockIdx.x] = acc.x + acc.y;
}

/*
 * Sequential baseline: all CTAs cooperatively stride through one buffer.
 * This gives the theoretical DRAM peak for this working set size.
 */
__global__ void sequential_read_kernel(
    const float4* __restrict__ data,
    float* __restrict__ sink,
    int n_float4,
    int n_iters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int iter = 0; iter < n_iters; iter++) {
        for (int i = tid; i < n_float4; i += stride) {
            float4 v = data[i];
            acc.x += v.x;
            acc.y += v.y;
        }
    }
    if (tid == 0) sink[0] = acc.x + acc.y;
}

typedef struct {
    double bw_tb_s;
    float elapsed_ms;
} BenchResult;

BenchResult measure_sequential(float4* d_buf, float* d_sink,
                                int n_float4, int n_iters)
{
    sequential_read_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_buf, d_sink, n_float4, N_WARMUP);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    CHECK(cudaEventRecord(start));
    sequential_read_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_buf, d_sink, n_float4, n_iters);
    CHECK(cudaEventRecord(end));
    CHECK(cudaDeviceSynchronize());

    float elapsed_ms;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));

    double total_bytes = (double)n_float4 * sizeof(float4) * n_iters;
    BenchResult r;
    r.bw_tb_s = total_bytes / (elapsed_ms / 1000.0) / 1e12;
    r.elapsed_ms = elapsed_ms;
    return r;
}

BenchResult measure_scattered(float4** d_ptrs_dev, float* d_sink,
                               int n_streams, int n_float4_per_buf, int n_iters)
{
    scattered_read_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        (const float4**)d_ptrs_dev, d_sink, n_streams, n_float4_per_buf, N_WARMUP);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    CHECK(cudaEventRecord(start));
    scattered_read_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(
        (const float4**)d_ptrs_dev, d_sink, n_streams, n_float4_per_buf, n_iters);
    CHECK(cudaEventRecord(end));
    CHECK(cudaDeviceSynchronize());

    float elapsed_ms;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));

    /* Total bytes: each CTA reads entire buffer n_iters times.
     * But CTAs share buffers, so actual unique reads depend on L2 behavior.
     * Report as if each CTA independently reads the full buffer. */
    double total_bytes = (double)GRID_SIZE * n_float4_per_buf * sizeof(float4) * n_iters;
    BenchResult r;
    r.bw_tb_s = total_bytes / (elapsed_ms / 1000.0) / 1e12;
    r.elapsed_ms = elapsed_ms;
    return r;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("L2 cache size: %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("Memory Clock: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Grid: %d blocks (%d SMs x %d blocks/SM)\n\n", GRID_SIZE, N_SM, BLOCKS_PER_SM);

    size_t buf_bytes = (size_t)BUFFER_SIZE_MB * 1024 * 1024;
    int n_float4_per_buf = buf_bytes / sizeof(float4);

    /* Allocate MAX_STREAMS separate buffers */
    float4* d_streams[MAX_STREAMS];
    float4* h_ptrs[MAX_STREAMS];
    for (int i = 0; i < MAX_STREAMS; i++) {
        CHECK(cudaMalloc(&d_streams[i], buf_bytes));
        CHECK(cudaMemset(d_streams[i], 1, buf_bytes));
        h_ptrs[i] = d_streams[i];
    }

    float4** d_ptrs;
    CHECK(cudaMalloc(&d_ptrs, MAX_STREAMS * sizeof(float4*)));
    CHECK(cudaMemcpy(d_ptrs, h_ptrs, MAX_STREAMS * sizeof(float4*), cudaMemcpyHostToDevice));

    /* Sequential baseline: one big contiguous buffer */
    size_t seq_bytes = 8 * buf_bytes;  /* 1 GB — matches 8-stream total */
    float4* d_seq;
    CHECK(cudaMalloc(&d_seq, seq_bytes));
    CHECK(cudaMemset(d_seq, 1, seq_bytes));
    int seq_n_f4 = seq_bytes / sizeof(float4);

    float* d_sink;
    CHECK(cudaMalloc(&d_sink, GRID_SIZE * sizeof(float)));

    /* ─── Baselines ─── */
    printf("=== Baselines (cooperative stride, each addr read once) ===\n");
    BenchResult seq = measure_sequential(d_seq, d_sink, seq_n_f4, N_MEASURE);
    printf("Sequential (1 GB): %.3f TB/s, %.2f ms\n", seq.bw_tb_s, seq.elapsed_ms);
    BenchResult seq_128 = measure_sequential(d_streams[0], d_sink, n_float4_per_buf, N_MEASURE);
    printf("Sequential (128 MB): %.3f TB/s, %.2f ms\n\n", seq_128.bw_tb_s, seq_128.elapsed_ms);

    /* 1-stream scattered = each CTA independently reads the same buffer
     * (= massive redundant reads, L2 thrashing within one address space) */
    printf("=== 1-stream reference (432 CTAs, each reads entire 128 MB) ===\n");
    float4** d_ptrs_1;
    CHECK(cudaMalloc(&d_ptrs_1, sizeof(float4*)));
    CHECK(cudaMemcpy(d_ptrs_1, &d_streams[0], sizeof(float4*), cudaMemcpyHostToDevice));
    BenchResult ref1 = measure_scattered(d_ptrs_1, d_sink, 1, n_float4_per_buf, N_MEASURE);
    printf("  BW (per-CTA perspective): %.3f TB/s, Time: %.2f ms\n", ref1.bw_tb_s, ref1.elapsed_ms);
    /* Actual unique DRAM reads = 128 MB per iter (L2 absorbs duplicates within-stream) */
    double actual_unique_1 = (double)buf_bytes * N_MEASURE;
    double actual_dram_bw_1 = actual_unique_1 / (ref1.elapsed_ms / 1000.0) / 1e12;
    printf("  Unique DRAM BW: %.3f TB/s (128 MB × %d / %.2f ms)\n\n",
           actual_dram_bw_1, N_MEASURE, ref1.elapsed_ms);
    CHECK(cudaFree(d_ptrs_1));

    /* ─── N_streams sweep ─── */
    const int n_streams_sweep[] = {1, 2, 4, 8, 16, 32};
    const int N_NS = sizeof(n_streams_sweep) / sizeof(n_streams_sweep[0]);

    printf("=== N_streams sweep ===\n");
    printf("Each CTA independently reads its entire assigned buffer (128 MB).\n");
    printf("CTAs assigned round-robin to N_streams buffers.\n\n");
    printf("%-12s %-12s %-14s %-14s %-12s\n",
           "n_streams", "CTAs/stream", "apparent_TB_s", "unique_TB_s", "eff_vs_seq");
    printf("%-12s %-12s %-14s %-14s %-12s\n",
           "----------", "----------", "------------", "------------", "----------");

    for (int i = 0; i < N_NS; i++) {
        int ns = n_streams_sweep[i];
        CHECK(cudaMemcpy(d_ptrs, h_ptrs, ns * sizeof(float4*), cudaMemcpyHostToDevice));

        BenchResult r = measure_scattered(d_ptrs, d_sink, ns, n_float4_per_buf, N_MEASURE);

        int ctas_per_stream = GRID_SIZE / ns;
        /* Unique bytes: N_streams × 128 MB per iter (each buffer read at least once) */
        double unique_bytes = (double)ns * buf_bytes * N_MEASURE;
        double unique_bw = unique_bytes / (r.elapsed_ms / 1000.0) / 1e12;
        double eff = unique_bw / seq.bw_tb_s;

        printf("%-12d %-12d %-14.3f %-14.3f %-12.4f\n",
               ns, ctas_per_stream, r.bw_tb_s, unique_bw, eff);
        fprintf(stderr, "RESULT n_streams=%d ctas_per_stream=%d unique_bw_TB_s=%.3f eff=%.4f\n",
                ns, ctas_per_stream, unique_bw, eff);
    }

    printf("\n=== Summary ===\n");
    printf("Sequential baseline: %.3f TB/s\n", seq.bw_tb_s);
    printf("eff = unique_dram_bw / seq_bw.\n");
    printf("eff < 1.0 means inter-stream L2 conflict misses degrade DRAM efficiency.\n");
    printf("Use RESULT lines (stderr) for model integration.\n");

    /* Cleanup */
    for (int i = 0; i < MAX_STREAMS; i++) CHECK(cudaFree(d_streams[i]));
    CHECK(cudaFree(d_ptrs));
    CHECK(cudaFree(d_seq));
    CHECK(cudaFree(d_sink));

    return 0;
}
