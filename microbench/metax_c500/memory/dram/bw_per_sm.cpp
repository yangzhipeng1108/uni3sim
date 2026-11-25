/*
 * Microbenchmark: per-SM DRAM bandwidth cap.
 *
 * Measures achieved DRAM BW as a function of active SM count.
 * Two modes:
 *   (a) L1-bypass: models FA ldgsts path
 *   (b) L1-cached (default load): models GEMM/MOE path
 *
 * Output: per-SM BW saturation curve → derive bw_per_sm_cap for each path.
 *
 * Physical reasoning:
 *   Each SM has a finite number of outstanding DRAM requests (MSHRs).
 *   With few SMs active, per-SM BW = total_BW / active_SMs is bounded
 *   by hardware outstanding limit. The BW per SM plateaus at the cap:
 *     bw_per_sm_cap = outstanding × cacheline / dram_latency
 *
 * Compile (MetaX C500 / MACA):
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -o bw_per_sm bw_per_sm.cpp
 *
 * Run:
 *   ./bw_per_sm [buffer_MB] [threads_per_block]
 *   Default: buffer=2048 MB, threads_per_block=256
 */
#include <mc_runtime.h>
#define gpuMalloc              mcMalloc
#define gpuFree                mcFree
#define gpuMemset              mcMemset
#define gpuDeviceSynchronize   mcDeviceSynchronize
#define gpuEventCreate         mcEventCreate
#define gpuEventRecord         mcEventRecord
#define gpuEventElapsedTime    mcEventElapsedTime
#define gpuEventDestroy        mcEventDestroy
#define gpuGetDeviceProperties mcGetDeviceProperties
#define gpuDeviceProp          mcDeviceProp_t
#define gpuEvent_t             mcEvent_t
#define gpuError_t             mcError_t
#define gpuSuccess             mcSuccess
#define gpuGetErrorString      mcGetErrorString

#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>

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

#define CHECK(call) do { \
    gpuError_t err = call; \
    if (err != gpuSuccess) { \
        fprintf(stderr, "GPU error at %s:%d: %s\n", __FILE__, __LINE__, \
                gpuGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// L1-bypass streaming read kernel
// On NVIDIA: ld.global.cg bypasses L1, hits L2 only → DRAM on miss
// On C500: inline asm may not be available; use volatile pointer cast
__global__ void bw_kernel_bypass(const float4* __restrict__ data,
                                  float* __restrict__ sink,
                                  long long n_float4,
                                  int n_iters) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    float4 accum = make_float4(0, 0, 0, 0);

    // Volatile disables L1 caching on most architectures
    const volatile float* vdata = (const volatile float*)data;

    for (int iter = 0; iter < n_iters; iter++) {
        for (long long i = tid; i < n_float4; i += stride) {
            long long base = i * 4;
            accum.x += vdata[base];
            accum.y += vdata[base + 1];
            accum.z += vdata[base + 2];
            accum.w += vdata[base + 3];
        }
    }

    if (tid == 0)
        sink[0] = accum.x + accum.y + accum.z + accum.w;
}

// L1-cached streaming read kernel (default load)
__global__ void bw_kernel_cached(const float4* __restrict__ data,
                                  float* __restrict__ sink,
                                  long long n_float4,
                                  int n_iters) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    float4 accum = make_float4(0, 0, 0, 0);

    for (int iter = 0; iter < n_iters; iter++) {
        for (long long i = tid; i < n_float4; i += stride) {
            float4 v = data[i];
            accum.x += v.x;
            accum.y += v.y;
            accum.z += v.z;
            accum.w += v.w;
        }
    }

    if (tid == 0)
        sink[0] = accum.x + accum.y + accum.z + accum.w;
}

struct BenchResult {
    int active_sms;
    double total_bw_gbs;
    double per_sm_bw_gbs;
};

BenchResult measure_bw(const float4* d_data, float* d_sink,
                       long long n_float4, int grid_size,
                       int block_size, int n_iters, int bypass_l1) {
    // Warmup
    for (int i = 0; i < 5; i++) {
        if (bypass_l1 == 1) {
            bw_kernel_bypass<<<grid_size, block_size>>>(d_data, d_sink, n_float4, 1);
        } else if (bypass_l1 == 0) {
            bw_kernel_cached<<<grid_size, block_size>>>(d_data, d_sink, n_float4, 1);
        } else {
            fprintf(stderr, "bypass_l1 must be 0 or 1, got %d\n", bypass_l1);
            exit(EXIT_FAILURE);
        }
    }
    CHECK(gpuDeviceSynchronize());

    gpuEvent_t start, end;
    CHECK(gpuEventCreate(&start));
    CHECK(gpuEventCreate(&end));

    CHECK(gpuEventRecord(start));
    if (bypass_l1 == 1) {
        bw_kernel_bypass<<<grid_size, block_size>>>(d_data, d_sink, n_float4, n_iters);
    } else if (bypass_l1 == 0) {
        bw_kernel_cached<<<grid_size, block_size>>>(d_data, d_sink, n_float4, n_iters);
    } else {
        fprintf(stderr, "bypass_l1 must be 0 or 1, got %d\n", bypass_l1);
        exit(EXIT_FAILURE);
    }
    CHECK(gpuEventRecord(end));
    CHECK(gpuDeviceSynchronize());

    float elapsed_ms;
    CHECK(gpuEventElapsedTime(&elapsed_ms, start, end));
    CHECK(gpuEventDestroy(start));
    CHECK(gpuEventDestroy(end));

    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0) {
        fprintf(stderr, "invalid elapsed time: %f ms\n", elapsed_ms);
        exit(EXIT_FAILURE);
    }

    double elapsed_s = elapsed_ms / 1000.0;
    double total_bytes = (double)n_float4 * sizeof(float4) * n_iters;
    double total_bw_gbs = total_bytes / elapsed_s / 1e9;

    BenchResult r;
    r.active_sms = grid_size;
    r.total_bw_gbs = total_bw_gbs;
    r.per_sm_bw_gbs = total_bw_gbs / grid_size;
    return r;
}

void run_sweep(const char* label, const float4* d_data, float* d_sink,
               long long n_float4, int block_size, int n_sm, int bypass_l1) {
    printf("\n");
    printf("================================================================\n");
    printf("  %s (block_size=%d, buffer=%.0f MB)\n", label, block_size,
           (double)n_float4 * sizeof(float4) / 1e6);
    printf("================================================================\n");
    printf("  %4s  %12s  %12s  %8s\n",
           "SMs", "Total(GB/s)", "PerSM(GB/s)", "Sat%");
    printf("  ----  ------------  ------------  --------\n");

    // SM count sweep points
    int points[] = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 104, 108, 128, 132};
    int n_points = sizeof(points) / sizeof(points[0]);

    double peak_total = 0;
    double peak_per_sm = 0;

    for (int i = 0; i < n_points; i++) {
        int k = points[i];
        if (k > n_sm) continue;

        // Adjust iterations: fewer SMs → need fewer iters for stable timing
        int iters = (k <= 4) ? 3 : (k <= 16) ? 5 : 10;

        BenchResult r = measure_bw(d_data, d_sink, n_float4, k, block_size,
                                   iters, bypass_l1);

        if (r.total_bw_gbs > peak_total) peak_total = r.total_bw_gbs;
        if (r.per_sm_bw_gbs > peak_per_sm) peak_per_sm = r.per_sm_bw_gbs;

        if (!std::isfinite(r.total_bw_gbs) || r.total_bw_gbs <= 0
                || !std::isfinite(r.per_sm_bw_gbs) || r.per_sm_bw_gbs <= 0) {
            fprintf(stderr, "bandwidth measurement must be finite and positive\n");
            exit(EXIT_FAILURE);
        }
        double sat_pct = r.total_bw_gbs / peak_total * 100.0;
        printf("  %4d  %10.1f    %10.2f    %6.1f%%\n",
               r.active_sms, r.total_bw_gbs, r.per_sm_bw_gbs, sat_pct);
    }

    if (peak_total <= 0 || peak_per_sm <= 0) {
        fprintf(stderr, "bandwidth sweep produced no valid measurements\n");
        exit(EXIT_FAILURE);
    }
    printf("\n  Summary:\n");
    printf("    Peak total BW:  %.1f GB/s\n", peak_total);
    printf("    Peak per-SM BW: %.2f GB/s (= bw_per_sm_cap)\n", peak_per_sm);
    printf("    SM_sat (total/per_sm): %.1f SMs needed to saturate\n",
           peak_total / peak_per_sm);
}

int main(int argc, char** argv) {
    if (argc > 3) {
        fprintf(stderr, "usage: %s [buffer_MB] [threads_per_block]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int buffer_mb = 2048;
    int block_size = 256;
    if (argc > 1) buffer_mb = parse_positive_integer(argv[1], "buffer_MB");
    if (argc > 2) block_size = parse_positive_integer(argv[2], "threads_per_block");

    gpuDeviceProp prop;
    CHECK(gpuGetDeviceProperties(&prop, 0));
    int n_sm = prop.multiProcessorCount;

    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", n_sm);
    printf("L2 cache: %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("Clock: %.2f GHz (SM), %.2f GHz (mem)\n",
           prop.clockRate / 1e6, prop.memoryClockRate / 1e6);
    printf("Buffer: %d MB (>> L2 to force DRAM)\n", buffer_mb);
    printf("Block size: %d threads\n", block_size);

    // Allocate buffer >> L2 to ensure DRAM traffic
    size_t buf_bytes = (size_t)buffer_mb * 1024 * 1024;
    long long n_float4 = buf_bytes / sizeof(float4);

    float4* d_data;
    float* d_sink;
    CHECK(gpuMalloc(&d_data, buf_bytes));
    CHECK(gpuMalloc(&d_sink, sizeof(float)));
    CHECK(gpuMemset(d_data, 0x42, buf_bytes));

    // Sweep 1: L1-bypass (models FA ldgsts path)
    run_sweep("L1-BYPASS (volatile) — models FA ldgsts path",
              d_data, d_sink, n_float4, block_size, n_sm, 1);

    // Sweep 2: L1-cached (models GEMM/MOE path)
    run_sweep("L1-CACHED (default load) — models GEMM/MOE path",
              d_data, d_sink, n_float4, block_size, n_sm, 0);

    // Sweep 3: Multi-block per SM (check if occupancy helps per-SM BW)
    printf("\n");
    printf("================================================================\n");
    printf("  Multi-block per SM sweep (L1-bypass, varying blocks/SM)\n");
    printf("  Tests whether higher occupancy increases per-SM BW cap.\n");
    printf("================================================================\n");
    printf("  %6s  %4s  %12s  %12s\n",
           "Blk/SM", "Grid", "Total(GB/s)", "PerSM(GB/s)");
    printf("  ------  ----  ------------  ------------\n");
    for (int blk_per_sm = 1; blk_per_sm <= 8; blk_per_sm *= 2) {
        int grid = n_sm * blk_per_sm;
        BenchResult r = measure_bw(d_data, d_sink, n_float4, grid, block_size,
                                   10, 1);
        printf("  %6d  %4d  %10.1f    %10.2f\n",
               blk_per_sm, grid, r.total_bw_gbs, r.total_bw_gbs / n_sm);
    }

    printf("\n");
    printf("================================================================\n");
    printf("  Multi-block per SM sweep (L1-cached, varying blocks/SM)\n");
    printf("================================================================\n");
    printf("  %6s  %4s  %12s  %12s\n",
           "Blk/SM", "Grid", "Total(GB/s)", "PerSM(GB/s)");
    printf("  ------  ----  ------------  ------------\n");
    for (int blk_per_sm = 1; blk_per_sm <= 8; blk_per_sm *= 2) {
        int grid = n_sm * blk_per_sm;
        BenchResult r = measure_bw(d_data, d_sink, n_float4, grid, block_size,
                                   10, 0);
        printf("  %6d  %4d  %10.1f    %10.2f\n",
               blk_per_sm, grid, r.total_bw_gbs, r.total_bw_gbs / n_sm);
    }

    // Sweep 5: 2D sweep (SM count × blk/SM) for L1-cached path
    // Produces the full BW(active_sms, warps_per_sm) surface.
    printf("\n");
    printf("================================================================\n");
    printf("  2D sweep: SM count x blocks/SM (L1-cached)\n");
    printf("  CSV format for plotting\n");
    printf("================================================================\n");
    printf("active_sms,blk_per_sm,warps_per_sm,grid,total_bw_gbs,per_sm_bw_gbs\n");
    int sm_points_2d[] = {1, 2, 4, 8, 16, 32, 52, 78, 104};
    int n_sm_points_2d = sizeof(sm_points_2d) / sizeof(sm_points_2d[0]);
    int blk_points[] = {1, 2, 4, 8};
    int n_blk_points = sizeof(blk_points) / sizeof(blk_points[0]);
    int warps_per_blk = block_size / 64;  // C500 warp_size=64

    for (int si = 0; si < n_sm_points_2d; si++) {
        int active = sm_points_2d[si];
        if (active > n_sm) continue;
        for (int bi = 0; bi < n_blk_points; bi++) {
            int bps = blk_points[bi];
            int grid = active * bps;
            int iters = (grid <= 16) ? 3 : (grid <= 104) ? 5 : 10;
            BenchResult r = measure_bw(d_data, d_sink, n_float4, grid,
                                       block_size, iters, 0);
            printf("%d,%d,%d,%d,%.2f,%.2f\n",
                   active, bps, bps * warps_per_blk, grid,
                   r.total_bw_gbs, r.total_bw_gbs / active);
        }
    }

    CHECK(gpuFree(d_data));
    CHECK(gpuFree(d_sink));

    printf("\n");
    printf("================================================================\n");
    printf("  USAGE: Copy bw_per_sm_cap values to hw YAML:\n");
    printf("    memory_subsystem:\n");
    printf("      bw_per_sm_bypass_gbs: <peak per-SM from L1-bypass sweep>\n");
    printf("      bw_per_sm_cached_gbs: <peak per-SM from L1-cached sweep>\n");
    printf("================================================================\n");

    return 0;
}
