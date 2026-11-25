/*
 * Microbenchmark: L2 streaming prefetch bandwidth curve.
 *
 * Measures how L2 achieved BW varies with (active SMs, access pattern),
 * reporting raw mechanism throughput as a function of access pattern and
 * producer concurrency.  It does not select a Forward threshold or correction.
 *
 * Physical hypothesis: Ampere L2 has a hardware prefetcher that activates
 * under sequential (streaming) access.  The measured throughput depends on
 * access predictability and system-wide L2 request concurrency.
 *
 * Method:
 *   For each (N_blocks, pattern):
 *     1. Allocate L2-resident buffer (~32 MB < 40 MB L2)
 *     2. Launch N_blocks CTAs (1 per SM to control concurrency)
 *     3. Each CTA reads the buffer using the specified pattern
 *     4. Measure achieved L2 read BW (cudaEvent timing)
 *     5. Report raw bandwidth and its fraction of the documented L2 datapath
 *        peak (40 slices × 2 sub-partitions × 64 B/clk = 5120 B/clk)
 *
 * Sweep axes:
 *   N_blocks  ∈ {1, 4, 16, 32, 54, 72, 108, 216}
 *   pattern   ∈ {streaming, strided_256B, strided_4KB, random}
 *
 * Output format (CSV-parseable on stderr):
 *   RESULT n_blocks=108 pattern=streaming bw_TB_s=2.80 fraction_of_spec=0.48
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -o l2_prefetch_curve l2_prefetch_curve.cu
 * Run:
 *   CUDA_VISIBLE_DEVICES=0 ./l2_prefetch_curve
 *
 * Alternative: profile with NCU for sector-level metrics:
 *   ncu --kernel-name regex:"l2_read_" \
 *       --metrics lts__t_bytes.sum.per_second,lts__t_sector_hit_rate.pct,\
 *  sm__cycles_elapsed.avg.per_second --csv ./l2_prefetch_curve
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

#define BUF_ROWS 2048
#define BUF_COLS 4096
/* 2048 × 4096 × 4 bytes = 32 MB, fits in 40 MB L2 */
#define BUF_ELEMS ((long)BUF_ROWS * BUF_COLS)
#define BLOCK_SIZE 256
#define N_WARMUP 2
#define N_MEASURE 5

enum AccessPattern {
    STREAMING = 0,
    STRIDED_256B = 1,
    STRIDED_4KB = 2,
    RANDOM_PATTERN = 3,
};

static const char* pattern_names[] = {
    "streaming", "strided_256B", "strided_4KB", "random"
};

/*
 * Streaming: each CTA reads its assigned column range sequentially (row-major
 * over a contiguous column window). Maximally prefetchable.
 * float4 vectorized loads for memory-level parallelism.
 */
__global__ void l2_read_streaming(
    const float4* __restrict__ buf,
    float* __restrict__ out,
    int n_float4,
    int n_iters)
{
    int n_blocks = gridDim.x;
    int chunk = n_float4 / n_blocks;
    int my_start = blockIdx.x * chunk;
    int my_end = (blockIdx.x == n_blocks - 1) ? n_float4 : my_start + chunk;

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int it = 0; it < n_iters; it++) {
        for (int i = my_start + threadIdx.x; i < my_end; i += blockDim.x) {
            float4 v = buf[i];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc.x + acc.y + acc.z + acc.w;
}

/*
 * Strided: each CTA reads with a fixed stride (in float4 units).
 * stride_f4 = stride_bytes / sizeof(float4).
 * This scatters L2 requests across partitions, reducing prefetch effectiveness.
 */
__global__ void l2_read_strided(
    const float4* __restrict__ buf,
    float* __restrict__ out,
    int n_float4,
    int stride_f4,
    int n_iters)
{
    int n_blocks = gridDim.x;
    int elems_per_block = n_float4 / n_blocks;
    int my_count = (blockIdx.x == n_blocks - 1)
                   ? (n_float4 - (n_blocks - 1) * elems_per_block)
                   : elems_per_block;

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int it = 0; it < n_iters; it++) {
        /* Start at a block-unique offset, stride through the buffer */
        int base = (blockIdx.x * 17) % n_float4;  /* pseudo-random start per CTA */
        for (int count = threadIdx.x; count < my_count; count += blockDim.x) {
            int idx = (base + count * stride_f4) % n_float4;
            float4 v = buf[idx];
            acc.x += v.x; acc.y += v.y;
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc.x + acc.y + acc.z + acc.w;
}

/*
 * Random: each CTA reads L2-resident data at pseudo-random offsets.
 * Uses a simple LCG to generate indices. Worst case for prefetcher.
 */
__global__ void l2_read_random(
    const float4* __restrict__ buf,
    float* __restrict__ out,
    int n_float4,
    int n_reads_per_block,
    int n_iters)
{
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    /* LCG seed: unique per (block, thread) */
    unsigned int rng = blockIdx.x * 1099087573u + threadIdx.x * 2654435761u + 1;

    for (int it = 0; it < n_iters; it++) {
        for (int count = threadIdx.x; count < n_reads_per_block; count += blockDim.x) {
            rng = rng * 1103515245u + 12345u;
            int idx = (int)(rng % (unsigned int)n_float4);
            float4 v = buf[idx];
            acc.x += v.x; acc.y += v.y;
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc.x + acc.y + acc.z + acc.w;
}

typedef struct {
    double bw_tb_s;
    double elapsed_ms;
} BenchResult;

BenchResult run_pattern(
    const float4* d_buf, float* d_out,
    int n_float4, int n_blocks,
    AccessPattern pattern, int n_iters)
{
    int reads_per_block = n_float4 / n_blocks;

    /* Warmup */
    for (int w = 0; w < N_WARMUP; w++) {
        switch (pattern) {
        case STREAMING:
            l2_read_streaming<<<n_blocks, BLOCK_SIZE>>>(
                d_buf, d_out, n_float4, 1);
            break;
        case STRIDED_256B:
            l2_read_strided<<<n_blocks, BLOCK_SIZE>>>(
                d_buf, d_out, n_float4, 256 / (int)sizeof(float4), 1);
            break;
        case STRIDED_4KB:
            l2_read_strided<<<n_blocks, BLOCK_SIZE>>>(
                d_buf, d_out, n_float4, 4096 / (int)sizeof(float4), 1);
            break;
        case RANDOM_PATTERN:
            l2_read_random<<<n_blocks, BLOCK_SIZE>>>(
                d_buf, d_out, n_float4, reads_per_block, 1);
            break;
        default:
            fprintf(stderr, "unknown access pattern: %d\n", (int)pattern);
            exit(EXIT_FAILURE);
        }
        CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    CHECK(cudaEventRecord(start));
    switch (pattern) {
    case STREAMING:
        l2_read_streaming<<<n_blocks, BLOCK_SIZE>>>(
            d_buf, d_out, n_float4, n_iters);
        break;
    case STRIDED_256B:
        l2_read_strided<<<n_blocks, BLOCK_SIZE>>>(
            d_buf, d_out, n_float4, 256 / (int)sizeof(float4), n_iters);
        break;
    case STRIDED_4KB:
        l2_read_strided<<<n_blocks, BLOCK_SIZE>>>(
            d_buf, d_out, n_float4, 4096 / (int)sizeof(float4), n_iters);
        break;
    case RANDOM_PATTERN:
        l2_read_random<<<n_blocks, BLOCK_SIZE>>>(
            d_buf, d_out, n_float4, reads_per_block, n_iters);
        break;
    default:
        fprintf(stderr, "unknown access pattern: %d\n", (int)pattern);
        exit(EXIT_FAILURE);
    }
    CHECK(cudaEventRecord(end));
    CHECK(cudaDeviceSynchronize());

    float elapsed_ms;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));

    double total_bytes;
    if (pattern == RANDOM_PATTERN) {
        total_bytes = (double)n_blocks * reads_per_block * sizeof(float4) * n_iters;
    } else {
        total_bytes = (double)n_float4 * sizeof(float4) * n_iters;
    }

    BenchResult r;
    r.elapsed_ms = elapsed_ms;
    r.bw_tb_s = total_bytes / (elapsed_ms / 1000.0) / 1e12;
    return r;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("L2 cache size: %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("SM clock: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Buffer: %ld elements × %zu bytes = %.1f MB\n\n",
           BUF_ELEMS, sizeof(float), BUF_ELEMS * sizeof(float) / 1e6);

    /* Documented GA100 L2 datapath peak: 80 partitions × 64 B/clock. */
    double sm_freq_hz = prop.clockRate * 1000.0;  /* clockRate is in kHz */
    double spec_l2_bw_tb_s = 5120.0 * sm_freq_hz / 1e12;
    printf("L2 datapath peak (5120 B/clk × %.3f GHz): %.3f TB/s\n\n",
           sm_freq_hz / 1e9, spec_l2_bw_tb_s);

    /* Allocate L2-resident buffer as float4 for vectorized loads */
    int n_float4 = BUF_ELEMS / 4;  /* BUF_ELEMS floats = BUF_ELEMS/4 float4s */
    float4* d_buf;
    CHECK(cudaMalloc(&d_buf, (long)n_float4 * sizeof(float4)));
    CHECK(cudaMemset(d_buf, 0, (long)n_float4 * sizeof(float4)));

    /* Warmup: load into L2 */
    float* d_out;
    int max_blocks = 216;
    CHECK(cudaMalloc(&d_out, max_blocks * sizeof(float)));

    const int n_blocks_sweep[] = {1, 4, 16, 32, 54, 72, 108, 216};
    const int N_NB = sizeof(n_blocks_sweep) / sizeof(n_blocks_sweep[0]);
    const AccessPattern patterns[] = {STREAMING, STRIDED_256B, STRIDED_4KB, RANDOM_PATTERN};
    const int N_PAT = sizeof(patterns) / sizeof(patterns[0]);

    printf("%-12s %-14s %-12s %-12s %-12s\n",
           "n_blocks", "pattern", "bw_TB_s", "frac_spec", "time_ms");
    printf("%-12s %-14s %-12s %-12s %-12s\n",
           "----------", "------------", "----------", "----------", "----------");

    for (int nb_i = 0; nb_i < N_NB; nb_i++) {
        int n_blocks = n_blocks_sweep[nb_i];

        for (int p_i = 0; p_i < N_PAT; p_i++) {
            AccessPattern pat = patterns[p_i];

            /* Scale iterations: fewer blocks → more iters to keep timing stable */
            int n_iters = 20 * 108 / n_blocks;
            if (n_iters < 4) n_iters = 4;
            if (n_iters > 500) n_iters = 500;

            BenchResult r = run_pattern(d_buf, d_out, n_float4,
                                         n_blocks, pat, n_iters);
            double fraction_of_spec = r.bw_tb_s / spec_l2_bw_tb_s;

            printf("%-12d %-14s %-12.3f %-12.4f %-12.2f\n",
                   n_blocks, pattern_names[p_i], r.bw_tb_s,
                   fraction_of_spec, r.elapsed_ms);

            fprintf(stderr,
                    "RESULT n_blocks=%d pattern=%s bw_TB_s=%.3f "
                    "fraction_of_spec=%.4f\n",
                    n_blocks, pattern_names[p_i], r.bw_tb_s,
                    fraction_of_spec);
        }
    }

    printf("\n=== Summary ===\n");
    printf("L2 datapath peak: %.3f TB/s (5120 B/clk × %.3f GHz)\n",
           spec_l2_bw_tb_s, sm_freq_hz / 1e9);
    printf("RESULT lines are mechanism observations, not Forward parameters.\n");

    CHECK(cudaFree(d_buf));
    CHECK(cudaFree(d_out));

    return 0;
}
