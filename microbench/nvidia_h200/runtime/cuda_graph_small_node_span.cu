/*
 * H200 owner copy of microbench/nvidia_a800/runtime/cuda_graph_small_node_span.cu.
 * Kernel body, host timing loop, warm-up, trial/mode alternation and CSV
 * output are byte-identical to the A800 probe.  Hardware coupling changed:
 *   - build with -arch=sm_90 (and -cudart static so the binary runs on the
 *     H200 host); the build line below is updated accordingly.
 * The A800 numbers quoted in the original comment are that card's.
 */
/*
 * NVIDIA A800 CUDA-graph small-node span benchmark.
 *
 * The published small-kernel span floor was measured on launches issued one
 * at a time into a stream: at grid 48 a memory-touching CTA spans 2080 ns
 * and an empty one 1760 ns.  Forward composes that floor by maximum with
 * every memory-touching launch, including the dozens of small kernels a
 * vLLM decode step executes.  But a decode step is a captured CUDA graph
 * replay, not a sequence of standalone launches, and a graph node's span is
 * not obliged to match a standalone launch's.  This benchmark measures the
 * difference and claims nothing else.
 *
 * The kernel is deliberately trivial and identical in both modes: one
 * aligned 16-byte load, one dependent integer op, one aligned 16-byte store
 * per active thread, over a grid small enough that the launch, not the work,
 * dominates.  The same kernel, grid and block are then timed two ways:
 *
 *   standalone  N separate launches into a stream, timed as a batch, so the
 *               per-launch span is the batch time divided by N -- the
 *               regime the published floor was measured in
 *   graph       the same N launches captured once into a CUDA graph and
 *               replayed, timed per replay and divided by N -- the regime a
 *               decode step actually executes in
 *
 * Their ratio is the only reported quantity.  There is no production
 * operator, no attention or MoE arithmetic, and no production or evaluation
 * duration is consumed.
 *
 * Build:
 *   nvcc -O3 -arch=sm_90 -cudart static -o cuda_graph_small_node_span \
 *        cuda_graph_small_node_span.cu
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do {                                              \
    const cudaError_t error = (call);                                      \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(error));       \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

constexpr int kThreads = 128;
constexpr int kWarmupRounds = 200;

__global__ void small_memory_node(const uint4* __restrict__ input,
                                  uint4* __restrict__ output,
                                  long long packs) {
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= packs) {
        return;
    }
    uint4 value = input[index];
    value.x += 1U;
    output[index] = value;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s <trials> <nodes_per_batch> <grid_csv>\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    const int trials = std::atoi(argv[1]);
    if (trials < 1 || trials > 64) {
        std::fprintf(stderr, "trials must be within 1..64\n");
        return EXIT_FAILURE;
    }
    const int nodes = std::atoi(argv[2]);
    if (nodes < 1 || nodes > 4096) {
        std::fprintf(stderr, "nodes_per_batch must be within 1..4096\n");
        return EXIT_FAILURE;
    }
    std::vector<int> grids;
    for (const char* cursor = argv[3]; *cursor != '\0';) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value <= 0 || value > 65535) {
            std::fprintf(stderr, "grids must be within 1..65535\n");
            return EXIT_FAILURE;
        }
        grids.push_back(static_cast<int>(value));
        cursor = (*end == ',') ? end + 1 : end;
    }

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    std::printf("device,%s\n", properties.name);
    std::printf("threads_per_cta,%d\n", kThreads);
    std::printf("nodes_per_batch,%d\n", nodes);
    std::printf("grid_ctas,mode,trial_index,batch_duration_ms\n");

    const int max_grid = *std::max_element(grids.begin(), grids.end());
    const long long max_packs =
        static_cast<long long>(max_grid) * kThreads;
    uint4* input = nullptr;
    uint4* output = nullptr;
    CHECK_CUDA(cudaMalloc(&input, sizeof(uint4) * max_packs));
    CHECK_CUDA(cudaMalloc(&output, sizeof(uint4) * max_packs));
    CHECK_CUDA(cudaMemset(input, 0x5A, sizeof(uint4) * max_packs));
    CHECK_CUDA(cudaMemset(output, 0x00, sizeof(uint4) * max_packs));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (const int grid : grids) {
        const long long packs = static_cast<long long>(grid) * kThreads;

        /* Capture the identical batch once. */
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        CHECK_CUDA(cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeGlobal));
        for (int node = 0; node < nodes; ++node) {
            small_memory_node<<<grid, kThreads, 0, stream>>>(
                input, output, packs);
        }
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

        /* Warm both paths so neither pays first-touch cost. */
        for (int warm = 0; warm < kWarmupRounds; ++warm) {
            CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
            for (int node = 0; node < nodes; ++node) {
                small_memory_node<<<grid, kThreads, 0, stream>>>(
                    input, output, packs);
            }
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        /* Trial-outer over the two modes so both see the same clock state. */
        for (int trial = 0; trial < trials; ++trial) {
            CHECK_CUDA(cudaEventRecord(start, stream));
            for (int node = 0; node < nodes; ++node) {
                small_memory_node<<<grid, kThreads, 0, stream>>>(
                    input, output, packs);
            }
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float standalone_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&standalone_ms, start, stop));
            std::printf("%d,standalone,%d,%.6f\n", grid, trial,
                        static_cast<double>(standalone_ms));

            CHECK_CUDA(cudaEventRecord(start, stream));
            CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float graph_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&graph_ms, start, stop));
            std::printf("%d,graph,%d,%.6f\n", grid, trial,
                        static_cast<double>(graph_ms));
        }
        CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
        CHECK_CUDA(cudaGraphDestroy(graph));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(input));
    CHECK_CUDA(cudaFree(output));
    return EXIT_SUCCESS;
}
