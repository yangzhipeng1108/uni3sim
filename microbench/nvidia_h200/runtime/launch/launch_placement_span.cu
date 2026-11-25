/*
 * H200 diagnostic: does a small launch's span depend on where its CTAs land?
 *
 * Launches a one-CTA (or few-CTA) kernel back to back; every CTA records its
 * %smid and the %globaltimer at entry and exit.  Paired with the nsys kernel
 * duration of the same launches, this tests whether the period-4 two-level
 * duration pattern seen at small grids follows the SM (GPC) the CTA is placed
 * on.  Mechanism diagnostic only; no production kernel or duration.
 *
 * Build: nvcc -O3 -arch=sm_90 -cudart static -o launch_placement_span launch_placement_span.cu
 * Run:   ./launch_placement_span <grid> <launches>
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void placement(unsigned long long* rec, int launch) {
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (threadIdx.x == 0) {
        unsigned long long t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        const size_t base = ((size_t)launch * gridDim.x + blockIdx.x) * 3;
        rec[base] = smid;
        rec[base + 1] = t0;
        rec[base + 2] = t1;
    }
}

int main(int argc, char** argv) {
    const int grid = std::atoi(argv[1]);
    const int launches = std::atoi(argv[2]);
    unsigned long long* rec = nullptr;
    cudaMalloc(&rec, (size_t)grid * launches * 3 * sizeof(unsigned long long));
    for (int i = 0; i < 2000; ++i) placement<<<grid, 128>>>(rec, 0);
    cudaDeviceSynchronize();
    for (int i = 0; i < launches; ++i) placement<<<grid, 128>>>(rec, i);
    cudaDeviceSynchronize();
    std::vector<unsigned long long> host((size_t)grid * launches * 3);
    cudaMemcpy(host.data(), rec, host.size() * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    std::printf("launch,cta,smid,t_entry_ns\n");
    for (int i = 0; i < launches; ++i)
        for (int b = 0; b < grid; ++b) {
            const size_t k = ((size_t)i * grid + b) * 3;
            std::printf("%d,%d,%llu,%llu\n", i, b, host[k], host[k + 1]);
        }
    return 0;
}
