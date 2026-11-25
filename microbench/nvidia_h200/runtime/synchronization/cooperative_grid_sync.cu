// NVIDIA H200 cooperative grid-barrier service.
//
// flashinfer's Hopper absorbed-MLA kernel (include/flashinfer/attention/
// mla_hopper.cuh, BatchMLAPageAttentionHopper) is launched with
// cudaLaunchCooperativeKernel on one 256-thread CTA per SM and, after its KV
// tiles, crosses cg::this_grid().sync() before the in-kernel split-KV merge.
// This probe isolates that mechanism: a cooperative kernel of GRID CTAs x 256
// threads with the same one-CTA-per-SM shared-memory envelope executes
// REPEATS back-to-back grid.sync() calls and nothing else.  Timing is
// %globaltimer read by CTA 0 thread 0 around the barrier loop (after one
// warm-up barrier), so launch latency is excluded; the per-barrier service
// is (t_end - t_start) / REPEATS.  No production operator and no production
// or evaluation duration is used.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -cudart static -o cooperative_grid_sync cooperative_grid_sync.cu
// Usage: cooperative_grid_sync <grid> <repeats> <samples> <dynamic_smem_bytes>
// Output CSV: sample,grid,repeats,elapsed_ns,ns_per_barrier,
//             cooperative_span_ns,regular_span_ns
// The last two columns are cudaEvent spans of one launch of the same kernel
// with zero barriers, launched cooperatively and as an ordinary launch, i.e.
// the fixed cost the cooperative launch adds to a kernel that does no work.

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

namespace {

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda((call), #call)

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
    return t;
}

__global__ void __launch_bounds__(256) empty_node(std::uint64_t* sink) {
    extern __shared__ unsigned char envelope[];
    (void)envelope;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *sink += 1;
    }
}

__global__ void __launch_bounds__(256) grid_sync_loop(
        std::uint64_t* elapsed, int repeats) {
    extern __shared__ unsigned char envelope[];
    (void)envelope;
    cg::grid_group grid = cg::this_grid();
    grid.sync();
    std::uint64_t start = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        start = global_ns();
    }
    for (int i = 0; i < repeats; ++i) {
        grid.sync();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *elapsed = global_ns() - start;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s <grid> <repeats> <samples> <dynamic_smem_bytes>\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    int grid = std::atoi(argv[1]);
    int repeats = std::atoi(argv[2]);
    int samples = std::atoi(argv[3]);
    int smem = std::atoi(argv[4]);
    if (grid <= 0 || repeats <= 0 || samples <= 0 || smem < 0) {
        std::fprintf(stderr, "arguments must be positive\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        grid_sync_loop, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CHECK_CUDA(cudaFuncSetAttribute(
        empty_node, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cudaEvent_t begin, end;
    CHECK_CUDA(cudaEventCreate(&begin));
    CHECK_CUDA(cudaEventCreate(&end));
    std::uint64_t* elapsed = nullptr;
    CHECK_CUDA(cudaMalloc(&elapsed, sizeof(std::uint64_t)));
    void* args[] = {&elapsed, &repeats};
    std::printf("sample,grid,repeats,elapsed_ns,ns_per_barrier,"
                "cooperative_span_ns,regular_span_ns\n");
    void* empty_args[] = {&elapsed};
    for (int sample = -3; sample < samples; ++sample) {
        CHECK_CUDA(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(grid_sync_loop), dim3(grid), dim3(256),
            args, static_cast<std::size_t>(smem), 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        std::uint64_t host = 0;
        CHECK_CUDA(cudaMemcpy(&host, elapsed, sizeof(host),
                              cudaMemcpyDeviceToHost));
        float cooperative_ms = 0.0f;
        float regular_ms = 0.0f;
        CHECK_CUDA(cudaEventRecord(begin));
        CHECK_CUDA(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(empty_node), dim3(grid), dim3(256),
            empty_args, static_cast<std::size_t>(smem), 0));
        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaEventSynchronize(end));
        CHECK_CUDA(cudaEventElapsedTime(&cooperative_ms, begin, end));
        CHECK_CUDA(cudaEventRecord(begin));
        empty_node<<<grid, 256, static_cast<std::size_t>(smem)>>>(elapsed);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaEventSynchronize(end));
        CHECK_CUDA(cudaEventElapsedTime(&regular_ms, begin, end));
        if (sample >= 0) {
            std::printf("%d,%d,%d,%llu,%.3f,%.1f,%.1f\n", sample, grid,
                        repeats, static_cast<unsigned long long>(host),
                        static_cast<double>(host) / repeats,
                        cooperative_ms * 1e6, regular_ms * 1e6);
        }
    }
    CHECK_CUDA(cudaFree(elapsed));
    return EXIT_SUCCESS;
}
