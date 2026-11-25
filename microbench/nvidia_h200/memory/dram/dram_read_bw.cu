/*
 * NVIDIA H200 achievable DRAM read bandwidth.
 *
 * Every CTA streams a disjoint slice of a DRAM-resident buffer far larger than
 * L2 with 16-byte vector loads and folds it into one register, so the only
 * global traffic is the compulsory read stream plus one 16-byte store per
 * thread.  The grid sweeps resident CTAs per SM so the saturated read rate is
 * reached independent of occupancy.  Timed with CUDA events around repeated
 * full passes; each pass reads the whole buffer once.
 *
 * It is a memory mechanism probe: no operator, no production shape and no
 * production or evaluation duration.
 *
 * Build: nvcc -O3 -arch=sm_90 -cudart static -o dram_read_bw dram_read_bw.cu
 * Run:   ./dram_read_bw <buffer_gib> <passes>
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__,       \
                 cudaGetErrorString(e)); std::exit(2); } } while (0)

__global__ void read_stream(const int4* __restrict__ src, long long n,
                            int4* __restrict__ sink) {
    int4 acc = make_int4(0, 0, 0, 0);
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const int4 v = __ldcs(src + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    sink[(long long)blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <buffer_gib> <passes>\n", argv[0]);
        return 1;
    }
    const size_t bytes = (size_t)std::atof(argv[1]) * (1ull << 30);
    const int passes = std::atoi(argv[2]);
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const long long n = (long long)(bytes / sizeof(int4));
    int4* src = nullptr;
    CHECK(cudaMalloc(&src, bytes));
    CHECK(cudaMemset(src, 1, bytes));
    const int block = 512;
    const int ctas_per_sm[] = {1, 2, 3, 4, 6, 8};
    int4* sink = nullptr;
    CHECK(cudaMalloc(&sink, (size_t)prop.multiProcessorCount * 8 * block *
                                sizeof(int4)));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    std::printf("device,%s\nsms,%d\nl2_bytes,%d\nbuffer_bytes,%zu\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize, bytes);
    std::printf("ctas_per_sm,grid,block,passes,ms,read_GBps\n");
    for (int k : ctas_per_sm) {
        const int grid = prop.multiProcessorCount * k;
        for (int w = 0; w < 3; ++w) read_stream<<<grid, block>>>(src, n, sink);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(start));
        for (int p = 0; p < passes; ++p) read_stream<<<grid, block>>>(src, n, sink);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double gbps = (double)bytes * passes / (ms * 1e-3) / 1e9;
        std::printf("%d,%d,%d,%d,%.4f,%.1f\n", k, grid, block, passes, ms, gbps);
    }
    return 0;
}
