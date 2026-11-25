/*
 * SM-count → total DRAM BW microbench (controlled: fixed data, sweep SM count).
 * Fixed 256MB DRAM-resident working set, launch N CTAs (N=1..108), each streams
 * DRAM→reg→DRAM. Measures total achieved DRAM BW vs active SM count.
 * Shows BW grows with SMs then saturates near peak DRAM BW.
 * Build: nvcc -O2 -arch=sm_80 -o build/sm_count_bw microbench/nvidia_a800/memory/dram/sm_count_bw.cu
 */
#include <cuda_runtime.h>
#include <stdio.h>
#define CHECK(c) do{cudaError_t e=c; if(e!=cudaSuccess){fprintf(stderr,"ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
// each thread reads an int4, writes it back — pure DRAM streaming, no compute
__global__ void stream_kernel(const int4* __restrict__ src, int4* __restrict__ dst, long long n_per_cta) {
    long long base = (long long)blockIdx.x * n_per_cta;
    for (long long i = 0; i < n_per_cta; i += blockDim.x) {
        int idx = base + i + threadIdx.x;
        if (idx < base + n_per_cta) {
            int4 v = src[idx];
            dst[idx] = v;
        }
    }
}
int main() {
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,0));
    int nsm = prop.multiProcessorCount;
    fprintf(stderr,"GPU: %s | %d SMs | L2 %.1fMB\n", prop.name, nsm, prop.l2CacheSize/1e6);
    size_t total_bytes = 256ULL*1024*1024;
    long long n_int4 = total_bytes/16;
    int4 *d_src, *d_dst; CHECK(cudaMalloc(&d_src, total_bytes)); CHECK(cudaMalloc(&d_dst, total_bytes));
    CHECK(cudaMemset(d_src, 1, total_bytes));
    printf("# SM-count → total DRAM BW sweep (fixed 256MB, DRAM-resident, read+write)\n");
    printf("# %-6s %-12s %-14s\n","SMs","time_ms","total_BW_GBs");
    int sm_sweep[] = {1,2,4,8,16,24,32,48,64,80,96,108};
    int nsw = sizeof(sm_sweep)/sizeof(int);
    int block = 256;
    for (int s=0;s<nsw;s++){
        int grid = sm_sweep[s]; if (grid > nsm) grid = nsm;
        long long per_cta = (n_int4/grid);  // int4s per CTA
        long long n_per_cta = (per_cta/block)*block;  // round to block
        for(int w=0;w<3;w++) stream_kernel<<<grid,block>>>(d_src,d_dst,n_per_cta);
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        int reps=5; CHECK(cudaEventRecord(t0));
        for(int r=0;r<reps;r++) stream_kernel<<<grid,block>>>(d_src,d_dst,n_per_cta);
        CHECK(cudaEventRecord(t1)); CHECK(cudaDeviceSynchronize());
        float ms; CHECK(cudaEventElapsedTime(&ms,t0,t1)); ms/=reps;
        double bw = (double)total_bytes*2 / (ms*1e-3) / 1e9;  // read+write
        printf("  %-6d %-12.3f %-14.1f\n", grid, ms, bw);
        fflush(stdout);
        CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    }
    CHECK(cudaFree(d_src)); CHECK(cudaFree(d_dst));
    return 0;
}
