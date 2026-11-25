// NVIDIA H200 cooperative-kernel graph-node span.
//
// flashinfer's Hopper absorbed-MLA kernel (mla_hopper.cuh,
// BatchMLAPageAttentionHopper) is launched with cudaLaunchCooperativeKernel,
// one 256-thread CTA per SM with ~227 KiB of dynamic shared memory, and SGLang
// replays it inside its decode CUDA graph.  This probe measures what such a
// node costs by itself: a chain of NODES dependent launches of a kernel that
// touches one word per CTA, captured once into a CUDA graph and replayed, in
// four modes that differ only in how the node is launched and whether it
// crosses one grid.sync():
//   regular      <<<grid, 256, smem>>>, no barrier
//   coop         cudaLaunchCooperativeKernel, no barrier
//   coop_sync    cudaLaunchCooperativeKernel, one cg::this_grid().sync()
// Per-node span = graph replay time (cudaEvent) / NODES, median of TRIALS
// replays after WARMUP replays.  No production operator and no production or
// evaluation duration is used.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -cudart static -o cooperative_graph_node_span cooperative_graph_node_span.cu
// Usage: cooperative_graph_node_span <grid> <smem_bytes> <nodes> <trials>
// Output CSV: mode,grid,smem_bytes,nodes,trial,ns_per_node

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

__global__ void __launch_bounds__(256) touch_node(std::uint32_t* sink) {
    extern __shared__ unsigned char envelope[];
    (void)envelope;
    if (threadIdx.x == 0) {
        sink[blockIdx.x] += 1U;
    }
}

__global__ void __launch_bounds__(256) touch_sync_node(std::uint32_t* sink) {
    extern __shared__ unsigned char envelope[];
    (void)envelope;
    if (threadIdx.x == 0) {
        sink[blockIdx.x] += 1U;
    }
    cg::this_grid().sync();
}

enum class Mode { kRegular, kCoop, kCoopSync };

const char* name(Mode mode) {
    switch (mode) {
        case Mode::kRegular: return "regular";
        case Mode::kCoop: return "coop";
        case Mode::kCoopSync: return "coop_sync";
    }
    return "?";
}

void launch(Mode mode, int grid, int smem, std::uint32_t* sink,
            cudaStream_t stream) {
    void* args[] = {&sink};
    switch (mode) {
        case Mode::kRegular:
            touch_node<<<grid, 256, smem, stream>>>(sink);
            CHECK_CUDA(cudaGetLastError());
            return;
        case Mode::kCoop:
            CHECK_CUDA(cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(touch_node), dim3(grid), dim3(256),
                args, static_cast<std::size_t>(smem), stream));
            return;
        case Mode::kCoopSync:
            CHECK_CUDA(cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(touch_sync_node), dim3(grid),
                dim3(256), args, static_cast<std::size_t>(smem), stream));
            return;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <grid> <smem_bytes> <nodes> <trials>\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    const int grid = std::atoi(argv[1]);
    const int smem = std::atoi(argv[2]);
    const int nodes = std::atoi(argv[3]);
    const int trials = std::atoi(argv[4]);
    if (grid <= 0 || smem < 0 || nodes <= 0 || trials <= 0) {
        std::fprintf(stderr, "arguments must be positive\n");
        return EXIT_FAILURE;
    }
    for (auto kernel : {touch_node, touch_sync_node}) {
        CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    std::uint32_t* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&sink, sizeof(std::uint32_t) * grid));
    CHECK_CUDA(cudaMemset(sink, 0, sizeof(std::uint32_t) * grid));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t begin, end;
    CHECK_CUDA(cudaEventCreate(&begin));
    CHECK_CUDA(cudaEventCreate(&end));
    std::printf("mode,grid,smem_bytes,nodes,trial,ns_per_node\n");
    for (Mode mode : {Mode::kRegular, Mode::kCoop, Mode::kCoopSync}) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < nodes; ++i) {
            launch(mode, grid, smem, sink, stream);
        }
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        for (int i = 0; i < 20; ++i) {
            CHECK_CUDA(cudaGraphLaunch(exec, stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        for (int trial = 0; trial < trials; ++trial) {
            CHECK_CUDA(cudaEventRecord(begin, stream));
            CHECK_CUDA(cudaGraphLaunch(exec, stream));
            CHECK_CUDA(cudaEventRecord(end, stream));
            CHECK_CUDA(cudaEventSynchronize(end));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, begin, end));
            std::printf("%s,%d,%d,%d,%d,%.1f\n", name(mode), grid, smem, nodes,
                        trial, ms * 1e6 / nodes);
        }
        CHECK_CUDA(cudaGraphExecDestroy(exec));
        CHECK_CUDA(cudaGraphDestroy(graph));
    }
    CHECK_CUDA(cudaFree(sink));
    return EXIT_SUCCESS;
}
