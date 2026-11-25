/*
 * NVIDIA H200 CUDA-graph dependent-node gap benchmark.
 *
 * A vLLM decode step replays one CUDA graph of ~1000-2000 dependent kernel
 * nodes.  Between the end of one node and the start of the next the GPU is
 * idle for a short, launch-dependent interval (node dependency resolution,
 * grid launch and CTA rasterization of the successor).  The small-node span
 * probe (cuda_graph_small_node_span.cu) measures the span of a trivial node,
 * which mixes that gap with the node's own execution.  This probe isolates
 * the gap: every node of a replayed graph busy-waits a controlled time on the
 * global timer, so the node's own execution is known and long compared to
 * the gap, and the gap is read directly from the nsys node trace
 * (--cuda-graph-trace=node) as start(next) - end(previous).
 *
 * Swept: CTAs per node (grid), threads per CTA, and the busy-wait time per
 * CTA.  Output: per replay batch, nothing but the kernel launches; all timing
 * comes from the nsys trace (cuda_graph_node_gap_runner.py).  No production
 * operator, no production or evaluation duration is consumed.
 *
 * Build:
 *   nvcc -O3 -arch=sm_90 -cudart static -o cuda_graph_node_gap cuda_graph_node_gap.cu
 * Usage:
 *   cuda_graph_node_gap <nodes> <replays> <grid_csv> <threads_csv> <wait_ns_csv> [alt_smem_bytes_csv]
 * With the optional list, every configuration is repeated with the odd nodes
 * replaced by a distinct kernel function using each dynamic shared-memory
 * size (the kernel-switch / carveout-switch cost between dependent nodes).
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) do {                                              \
    const cudaError_t error = (call);                                      \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(error));       \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

__device__ __forceinline__ uint64_t global_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Every CTA waits wait_ns on the global timer, then writes one word so the
// node has a visible side effect.  The chain dependency is the graph edge.
__global__ void timed_node(uint32_t* sink, uint64_t wait_ns) {
    if (wait_ns > 0) {
        const uint64_t start = global_ns();
        while (global_ns() - start < wait_ns) {
        }
    }
    if (threadIdx.x == 0) {
        sink[blockIdx.x] += 1U;
    }
}

// A second, distinct kernel function with a dynamic shared-memory footprint:
// alternating timed_node and this kernel isolates the cost of switching the
// kernel (function, and optionally the shared-memory carveout) between two
// dependent graph nodes.
__global__ void timed_node_smem(uint32_t* sink, uint64_t wait_ns) {
    extern __shared__ uint32_t scratch[];
    if (wait_ns > 0) {
        const uint64_t start = global_ns();
        while (global_ns() - start < wait_ns) {
        }
    }
    scratch[threadIdx.x] = threadIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) {
        sink[blockIdx.x] += scratch[blockDim.x - 1];
    }
}

static std::vector<long> parse_csv(const char* text) {
    std::vector<long> values;
    for (const char* cursor = text; *cursor != '\0';) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value < 0) {
            std::fprintf(stderr, "bad list %s\n", text);
            std::exit(EXIT_FAILURE);
        }
        values.push_back(value);
        cursor = (*end == ',') ? end + 1 : end;
    }
    return values;
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::fprintf(stderr,
                     "usage: %s <nodes> <replays> <grid_csv> <threads_csv> <wait_ns_csv> [alt_smem_bytes_csv]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    // Optional alternation: odd nodes run timed_node_smem with each listed
    // dynamic shared-memory size (-1 = identical nodes only).
    const std::vector<long> alt_smem = argc == 7 ? parse_csv(argv[6]) : std::vector<long>{};
    const int nodes = std::atoi(argv[1]);
    const int replays = std::atoi(argv[2]);
    const std::vector<long> grids = parse_csv(argv[3]);
    const std::vector<long> threads = parse_csv(argv[4]);
    const std::vector<long> waits = parse_csv(argv[5]);
    if (nodes < 2 || nodes > 4096 || replays < 1 || replays > 1000) {
        std::fprintf(stderr, "nodes 2..4096, replays 1..1000\n");
        return EXIT_FAILURE;
    }
    uint32_t* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&sink, sizeof(uint32_t) * 65536));
    CHECK_CUDA(cudaMemset(sink, 0, sizeof(uint32_t) * 65536));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    std::printf("config_index,grid_ctas,threads_per_cta,wait_ns,nodes,replays,alt_smem_bytes\n");
    CHECK_CUDA(cudaFuncSetAttribute(timed_node_smem,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024));
    std::vector<long> modes{-1};
    modes.insert(modes.end(), alt_smem.begin(), alt_smem.end());
    int config_index = 0;
    for (const long mode : modes) {
    for (const long grid : grids) {
        for (const long block : threads) {
            for (const long wait_ns : waits) {
                cudaGraph_t graph;
                cudaGraphExec_t exec;
                CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                for (int node = 0; node < nodes; ++node) {
                    if (mode >= 0 && node % 2 == 1) {
                        const size_t smem = static_cast<size_t>(
                            mode > block * 4 ? mode : block * 4);
                        timed_node_smem<<<static_cast<int>(grid), static_cast<int>(block), smem, stream>>>(
                            sink, static_cast<uint64_t>(wait_ns));
                    } else {
                        timed_node<<<static_cast<int>(grid), static_cast<int>(block), 0, stream>>>(
                            sink, static_cast<uint64_t>(wait_ns));
                    }
                }
                CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
                CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                // Warm replays, then the traced ones; the runner keeps only
                // replays after the first traced one of each configuration.
                for (int r = 0; r < 3 + replays; ++r) {
                    CHECK_CUDA(cudaGraphLaunch(exec, stream));
                }
                CHECK_CUDA(cudaStreamSynchronize(stream));
                std::printf("%d,%ld,%ld,%ld,%d,%d,%ld\n", config_index, grid, block, wait_ns,
                            nodes, 3 + replays, mode);
                CHECK_CUDA(cudaGraphExecDestroy(exec));
                CHECK_CUDA(cudaGraphDestroy(graph));
                ++config_index;
            }
        }
    }
    }
    CHECK_CUDA(cudaFree(sink));
    return EXIT_SUCCESS;
}
