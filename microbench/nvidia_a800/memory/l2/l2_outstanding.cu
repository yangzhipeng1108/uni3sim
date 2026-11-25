/*
 * L2 MSHR concurrency microbenchmark (per-CTA outstanding L2 loads).
 *
 * Method: one CTA per SM, block_size = OUTSTANDING (each thread carries one
 * independent pointer-chase chain). At each step, the CTA issues OUTSTANDING
 * mutually-independent ld.global.cg loads — one per thread, one per chain —
 * so the steady-state in-flight L2 sectors per CTA equals OUTSTANDING when
 * the per-SM L2 MSHR pool permits.
 *
 * Working set 16 MB fits comfortably in A800 L2 (40 MB) and exceeds the
 * per-SM L1 (192 KB); ld.global.cg additionally bypasses L1 so every load
 * is a guaranteed L2 hit.
 *
 * Timing uses the device-side `clock()` register inside the kernel so
 * cudaEvent + kernel-launch overhead is excluded from the per-iteration
 * latency. Per-iteration latency converts cycles -> nanoseconds at the
 * reported SM clock.
 *
 * Knee rule (deterministic, noise-tolerant via two-point hold):
 *   smallest OUTSTANDING value bw[i] such that bw[i]   >= 0.95 * max(bw) AND
 *                                              bw[i+1] >= 0.95 * max(bw).
 * For the last swept point we fall back to the one-point rule.
 *
 * Build: nvcc -O2 -arch=sm_80 -o build/l2_outstanding microbench/nvidia_a800/memory/l2/l2_outstanding.cu
 *
 * Output: 11 sweep rows (outstanding, achieved_bw_GBps, p10_ns, p50_ns, p90_ns)
 *         then a single line "mshr_concurrency = <int>" with the knee value.
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
static const int OUTSTANDING_SWEEP[] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256};
static const int N_SWEEP = sizeof(OUTSTANDING_SWEEP) / sizeof(OUTSTANDING_SWEEP[0]);

// Bypass L1, cache at L2 only. .cg = "Cache at Global level".
__device__ __forceinline__ int load_l2_only(const int* addr) {
    int v;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

// Warmup kernel: walks every chain twice to make all chain nodes L2-hot
// before the bench kernel runs. Splitting warmup into its own launch lets
// ncu measure the bench kernel in a fully L2-hot state.
__global__ void l2_warmup_kernel(const int* __restrict__ buf,
                                  const int* __restrict__ chain_heads) {
    int chain_id = (blockIdx.x + threadIdx.x) % N_CHAINS;
    int p = chain_heads[chain_id];
    for (int it = 0; it < 2 * CHAIN_LEN; it++) {
        p = load_l2_only(buf + p);
    }
    if (p == 0xDEADBEEF) ((int*)buf)[0] = p;  // never taken; keep loads live
}

// One chain per thread; block_size = OUTSTANDING. Each timed iteration issues
// OUTSTANDING mutually-independent loads (one per thread). The bench kernel
// itself does NOT walk the chain cold — l2_warmup_kernel is responsible for
// priming L2 before this kernel launches.
__global__ void l2_chains_kernel(const int* __restrict__ buf,
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
        p = load_l2_only(buf + p);
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
    fprintf(stderr, "GPU: %s | L2 = %.1f MB | SM = %d | clock = %.2f GHz\n",
            prop.name, prop.l2CacheSize / 1e6,
            prop.multiProcessorCount, sm_ghz);
    if (WS_BYTES >= L2_HARD_CEIL_BYTES) {
        fprintf(stderr, "ERROR: working set %.1f MB >= 40 MB hard ceiling; "
                        "DRAM-limited curve would mislead.\n", WS_BYTES / 1e6);
        return 3;
    }
    if ((size_t)prop.l2CacheSize < WS_BYTES) {
        fprintf(stderr, "ERROR: device L2 %.1f MB < working set %.1f MB.\n",
                prop.l2CacheSize / 1e6, WS_BYTES / 1e6);
        return 3;
    }

    int n_ints = (int)(WS_BYTES / sizeof(int));
    int sector_ints = SECTOR_BYTES / sizeof(int);     // 32
    int n_sectors = n_ints / sector_ints;
    int sectors_needed = N_CHAINS * CHAIN_LEN;
    if (n_sectors < sectors_needed) {
        fprintf(stderr, "ERROR: working set has %d sectors, need %d\n",
                n_sectors, sectors_needed);
        return 3;
    }
    if ((size_t)sectors_needed * SECTOR_BYTES > WS_BYTES) {
        fprintf(stderr, "ERROR: chain pool > working set; shrink CHAIN_LEN/N_CHAINS\n");
        return 3;
    }

    // Random permutation of sector indices, carved into N_CHAINS disjoint
    // chains of CHAIN_LEN each. Each chain is a cyclic singly-linked list
    // stored at sector starts in h_buf.
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
    // sink: up to OUTSTANDING_MAX (128) ints per CTA
    CHECK(cudaMalloc(&d_sink, sizeof(int) * prop.multiProcessorCount * 128));
    CHECK(cudaMalloc(&d_clocks, sizeof(unsigned int) * prop.multiProcessorCount));
    CHECK(cudaMemcpy(d_buf, h_buf.data(), WS_BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_chain_heads, h_chain_heads.data(),
                     sizeof(int) * N_CHAINS, cudaMemcpyHostToDevice));

    const int grid = prop.multiProcessorCount;
    const int n_steps = CHAIN_LEN;

    printf("# l2_outstanding sweep (1 CTA/SM, block_size=OUTSTANDING, one "
           "pointer-chase chain per thread, ld.global.cg)\n");
    printf("# working_set = %.1f MB, n_chains = %d, chain_len = %d, "
           "pool = %.1f MB\n",
           WS_BYTES / 1e6, N_CHAINS, CHAIN_LEN,
           (double)sectors_needed * SECTOR_BYTES / 1e6);
    printf("# %-11s %-18s %-10s %-10s %-10s\n",
           "outstanding", "achieved_bw_GBps", "p10_ns", "p50_ns", "p90_ns");

    double bw_GBps[32] = {0};
    int    sweep_out[32] = {0};

    for (int s = 0; s < N_SWEEP; s++) {
        int outstanding = OUTSTANDING_SWEEP[s];
        int block = outstanding;       // 1 chain per thread

        // Independent warmup launches: pull all chain nodes into L2.
        for (int w = 0; w < N_WARMUP_REPEATS; w++) {
            l2_warmup_kernel<<<grid, block>>>(d_buf, d_chain_heads);
        }
        CHECK(cudaDeviceSynchronize());

        std::vector<double> ns_per_iter_runs(N_TIMED_REPEATS, 0.0);
        std::vector<unsigned int> h_clocks(prop.multiProcessorCount, 0);
        for (int r = 0; r < N_TIMED_REPEATS; r++) {
            l2_chains_kernel<<<grid, block>>>(d_buf, d_chain_heads, n_steps,
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
        fprintf(stderr, "ERROR: no knee found (curve never reaches 95%% of max)\n");
        return 4;
    }
    int mshr_concurrency = sweep_out[knee_idx];
    printf("mshr_concurrency = %d\n", mshr_concurrency);

    CHECK(cudaFree(d_buf));
    CHECK(cudaFree(d_chain_heads));
    CHECK(cudaFree(d_sink));
    CHECK(cudaFree(d_clocks));
    return 0;
}
