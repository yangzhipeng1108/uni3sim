/*
 * L1 TEX MSHR concurrency microbenchmark (per-SM outstanding L1-cached loads).
 *
 * Variant of l2_outstanding.cu that uses ld.global (L1-CACHED path) instead
 * of ld.global.cg (L1-bypass). This measures the MSHR capacity of the L1 TEX
 * unit, which is the effective outstanding limit for GEMM/MOE kernels that
 * use standard global loads (not cp.async / ldgsts).
 *
 * Method: identical to l2_outstanding.cu — one CTA per SM, block_size =
 * OUTSTANDING, each thread carries one independent pointer-chase chain.
 * The difference is the load instruction: ld.global (default caching) goes
 * through L1 TEX → L2, while ld.global.cg bypasses L1 → L2 only.
 *
 * Working set: 8 MB fits in A800 L2 (40 MB). Each chain's stride is 128 B
 * (one sector) across random sectors, so the per-SM footprint exceeds L1
 * (192 KB when all 128 chains are active: 128 × 512 × 128 B = 8 MB).
 * This forces L1 misses on every access → measures L1 MSHR (not L1 hit BW).
 *
 * Build: nvcc -O2 -arch=sm_80 -o build/l1_tex_outstanding \
 *            microbench/nvidia_a800/l1_tex_outstanding.cu
 *
 * Output: sweep rows (outstanding, achieved_bw_GBps, p10_ns, p50_ns, p90_ns)
 *         then "l1_tex_mshr_per_sm = <int>" with the knee value.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <vector>

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(2); \
    } \
} while(0)

static const size_t WS_BYTES = 8ULL * 1024 * 1024;
static const size_t L2_HARD_CEIL_BYTES = 40ULL * 1024 * 1024;
static const int SECTOR_BYTES = 128;
static const int N_CHAINS = 128;
static const int CHAIN_LEN = 512;              // pool = 128 * 512 * 128 B = 8 MB
static const int N_TIMED_REPEATS = 32;
static const int N_WARMUP_REPEATS = 8;
// Extended sweep to catch potential knee below 128
static const int OUTSTANDING_SWEEP[] = {1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128};
static const int N_SWEEP = sizeof(OUTSTANDING_SWEEP) / sizeof(OUTSTANDING_SWEEP[0]);

// L1-cached global load (default caching policy). Goes through L1 TEX → L2.
// On L1 miss, the request is tracked by an L1 TEX MSHR entry until the L2
// response arrives. This is the path used by GEMM/MOE ld.global instructions.
__device__ __forceinline__ int load_l1_cached(const int* addr) {
    int v;
    asm volatile("ld.global.s32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

// L1-bypass load for warmup (fills L2 without polluting L1 with chain nodes)
__device__ __forceinline__ int load_l2_only(const int* addr) {
    int v;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

// Warmup kernel: walk every chain twice using L2-only loads to make all
// chain nodes L2-hot without filling L1.
__global__ void l1_warmup_kernel(const int* __restrict__ buf,
                                  const int* __restrict__ chain_heads) {
    int chain_id = (blockIdx.x + threadIdx.x) % N_CHAINS;
    int p = chain_heads[chain_id];
    for (int it = 0; it < 2 * CHAIN_LEN; it++) {
        p = load_l2_only(buf + p);
    }
    if (p == 0xDEADBEEF) ((int*)buf)[0] = p;
}

// Benchmark kernel: L1-cached loads. Each thread does a pointer-chase using
// ld.global (L1-cached). Since the working set (8 MB) >> L1 (192 KB),
// essentially all loads miss L1 and are tracked by L1 TEX MSHRs.
__global__ void l1_chains_kernel(const int* __restrict__ buf,
                                  const int* __restrict__ chain_heads,
                                  int n_steps,
                                  int* __restrict__ sink,
                                  unsigned int* __restrict__ clocks) {
    int chain_id = (blockIdx.x + threadIdx.x) % N_CHAINS;
    int p = chain_heads[chain_id];

    __syncthreads();
    __threadfence();
    unsigned int t0 = clock();

    for (int it = 0; it < n_steps; it++) {
        p = load_l1_cached(buf + p);
    }

    unsigned int t1 = clock();
    __syncthreads();
    if (threadIdx.x == 0) clocks[blockIdx.x] = t1 - t0;

    if (p == 0xDEADBEEF) sink[blockIdx.x * blockDim.x + threadIdx.x] = p;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    float sm_ghz = prop.clockRate / 1e6f;
    fprintf(stderr, "GPU: %s | L2 = %.1f MB | L1 = %d KB | SM = %d | clock = %.2f GHz\n",
            prop.name, prop.l2CacheSize / 1e6,
            prop.sharedMemPerMultiprocessor / 1024,
            prop.multiProcessorCount, sm_ghz);
    if (WS_BYTES >= L2_HARD_CEIL_BYTES) {
        fprintf(stderr, "ERROR: working set %.1f MB >= 40 MB hard ceiling\n",
                WS_BYTES / 1e6);
        return 3;
    }
    if ((size_t)prop.l2CacheSize < WS_BYTES) {
        fprintf(stderr, "ERROR: device L2 %.1f MB < working set %.1f MB.\n",
                prop.l2CacheSize / 1e6, WS_BYTES / 1e6);
        return 3;
    }

    int n_ints = (int)(WS_BYTES / sizeof(int));
    int sector_ints = SECTOR_BYTES / sizeof(int);
    int n_sectors = n_ints / sector_ints;
    int sectors_needed = N_CHAINS * CHAIN_LEN;
    if (n_sectors < sectors_needed) {
        fprintf(stderr, "ERROR: working set has %d sectors, need %d\n",
                n_sectors, sectors_needed);
        return 3;
    }

    // Random permutation of sector indices, carved into N_CHAINS disjoint
    // chains of CHAIN_LEN each.
    std::vector<int> perm(sectors_needed);
    for (int i = 0; i < sectors_needed; i++) perm[i] = i;
    std::mt19937 rng(42);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<int> h_buf(n_ints, 0);
    std::vector<int> h_chain_heads(N_CHAINS, 0);
    for (int c = 0; c < N_CHAINS; c++) {
        h_chain_heads[c] = perm[c * CHAIN_LEN] * sector_ints;
        for (int i = 0; i < CHAIN_LEN; i++) {
            int cur = perm[c * CHAIN_LEN + i];
            int nxt = perm[c * CHAIN_LEN + ((i + 1) % CHAIN_LEN)];
            h_buf[cur * sector_ints] = nxt * sector_ints;
        }
    }

    int* d_buf = nullptr;
    int* d_chain_heads = nullptr;
    int* d_sink = nullptr;
    unsigned int* d_clocks = nullptr;
    CHECK(cudaMalloc(&d_buf, WS_BYTES));
    CHECK(cudaMalloc(&d_chain_heads, sizeof(int) * N_CHAINS));
    CHECK(cudaMalloc(&d_sink, sizeof(int) * prop.multiProcessorCount * 128));
    CHECK(cudaMalloc(&d_clocks, sizeof(unsigned int) * prop.multiProcessorCount));
    CHECK(cudaMemcpy(d_buf, h_buf.data(), WS_BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_chain_heads, h_chain_heads.data(),
                     sizeof(int) * N_CHAINS, cudaMemcpyHostToDevice));

    const int grid = prop.multiProcessorCount;
    const int n_steps = CHAIN_LEN;

    printf("# l1_tex_outstanding sweep (1 CTA/SM, block_size=OUTSTANDING, one "
           "pointer-chase chain per thread, ld.global [L1-cached])\n");
    printf("# working_set = %.1f MB (>> L1 192 KB → L1 miss on every access)\n",
           WS_BYTES / 1e6);
    printf("# n_chains = %d, chain_len = %d, pool = %.1f MB\n",
           N_CHAINS, CHAIN_LEN,
           (double)sectors_needed * SECTOR_BYTES / 1e6);
    printf("# %-11s %-18s %-10s %-10s %-10s\n",
           "outstanding", "achieved_bw_GBps", "p10_ns", "p50_ns", "p90_ns");

    double bw_GBps[32] = {0};
    int    sweep_out[32] = {0};

    for (int s = 0; s < N_SWEEP; s++) {
        int outstanding = OUTSTANDING_SWEEP[s];
        int block = outstanding;

        // Warmup: fill L2 with chain nodes (L1-bypass so L1 is cold)
        for (int w = 0; w < N_WARMUP_REPEATS; w++) {
            l1_warmup_kernel<<<grid, block>>>(d_buf, d_chain_heads);
        }
        CHECK(cudaDeviceSynchronize());

        std::vector<double> ns_per_iter_runs(N_TIMED_REPEATS, 0.0);
        std::vector<unsigned int> h_clocks(prop.multiProcessorCount, 0);
        for (int r = 0; r < N_TIMED_REPEATS; r++) {
            l1_chains_kernel<<<grid, block>>>(d_buf, d_chain_heads, n_steps,
                                              d_sink, d_clocks);
            CHECK(cudaMemcpy(h_clocks.data(), d_clocks,
                             sizeof(unsigned int) * prop.multiProcessorCount,
                             cudaMemcpyDeviceToHost));
            std::vector<unsigned int> sorted = h_clocks;
            std::sort(sorted.begin(), sorted.end());
            unsigned int med_cycles = sorted[sorted.size() / 2];
            double med_ns = (double)med_cycles / sm_ghz;
            ns_per_iter_runs[r] = med_ns / n_steps;
        }
        std::sort(ns_per_iter_runs.begin(), ns_per_iter_runs.end());
        double p10_iter_ns = ns_per_iter_runs[N_TIMED_REPEATS / 10];
        double p50_iter_ns = ns_per_iter_runs[N_TIMED_REPEATS / 2];
        double p90_iter_ns = ns_per_iter_runs[(N_TIMED_REPEATS * 9) / 10];

        double per_cta_GBps =
            (double)outstanding * SECTOR_BYTES / p50_iter_ns;
        double bw_p50_GBps = per_cta_GBps * grid;

        printf("  %-11d %-18.2f %-10.2f %-10.2f %-10.2f\n",
               outstanding, bw_p50_GBps,
               p10_iter_ns, p50_iter_ns, p90_iter_ns);
        fflush(stdout);

        bw_GBps[s] = bw_p50_GBps;
        sweep_out[s] = outstanding;
    }

    // Knee detection: same 95% two-point hold rule as l2_outstanding.cu
    double bw_max = 0.0;
    for (int s = 0; s < N_SWEEP; s++) if (bw_GBps[s] > bw_max) bw_max = bw_GBps[s];
    const double hold_thresh = 0.95 * bw_max;
    int knee_idx = -1;
    for (int s = 0; s < N_SWEEP; s++) {
        if (bw_GBps[s] < hold_thresh) continue;
        bool second_in =
            (s + 1 < N_SWEEP) ? (bw_GBps[s + 1] >= hold_thresh) : true;
        if (second_in) { knee_idx = s; break; }
    }
    if (knee_idx < 0) {
        fprintf(stderr, "ERROR: no knee found; extend the L1 TEX MSHR sweep range.\n");
        printf("l1_tex_mshr_per_sm = >%d (knee not found)\n", OUTSTANDING_SWEEP[N_SWEEP - 1]);
    } else {
        int l1_mshr = sweep_out[knee_idx];
        printf("l1_tex_mshr_per_sm = %d\n", l1_mshr);
    }

    CHECK(cudaFree(d_buf));
    CHECK(cudaFree(d_chain_heads));
    CHECK(cudaFree(d_sink));
    CHECK(cudaFree(d_clocks));
    return knee_idx < 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
