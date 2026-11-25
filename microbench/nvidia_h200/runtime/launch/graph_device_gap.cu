// Independent CUDA launch-boundary mechanism, never a model/kernel replay.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(status));
        std::exit(2);
    }
}

__global__ void hold_gpu(unsigned long long ns) {
    unsigned long long first, current;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(current));
    } while (current - first < ns);
}

__global__ void touch(float* data) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    data[index] += 1.0f;
}

int main() {
    cudaDeviceProp properties;
    check(cudaGetDeviceProperties(&properties, 0));
    int runtime, driver;
    check(cudaRuntimeGetVersion(&runtime));
    check(cudaDriverGetVersion(&driver));
    std::printf("{\"kind\":\"identity\",\"name\":\"%s\",\"runtime\":%d,\"driver\":%d,\"sms\":%d}\n",
                properties.name, runtime, driver, properties.multiProcessorCount);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    float* data;
    check(cudaMalloc(&data, 132 * 128 * sizeof(float)));
    check(cudaMemsetAsync(data, 0, 132 * 128 * sizeof(float), stream));
    for (int warm = 0; warm < 100; ++warm) touch<<<132,128,0,stream>>>(data);
    check(cudaStreamSynchronize(stream));
    constexpr int nodes = 64;
    constexpr int samples = 8;
    constexpr unsigned long long gate_ns = 20000000;
    for (int grid : {1,32,132}) {
        cudaGraph_t graph;
        check(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t previous = nullptr;
        void* arguments[] = {&data};
        for (int index = 0; index < nodes; ++index) {
            cudaKernelNodeParams parameters{};
            parameters.func = reinterpret_cast<void*>(touch);
            parameters.gridDim = dim3(grid);
            parameters.blockDim = dim3(128);
            parameters.kernelParams = arguments;
            cudaGraphNode_t node;
            check(cudaGraphAddKernelNode(&node, graph,
                index == 0 ? nullptr : &previous, index == 0 ? 0 : 1, &parameters));
            previous = node;
        }
        cudaGraphExec_t executable;
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int warm = 0; warm < 5; ++warm) check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        cudaEvent_t gate_end, start, stop;
        check(cudaEventCreateWithFlags(&gate_end, cudaEventDisableTiming));
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&stop));
        for (int trial = 0; trial < samples; ++trial) {
            hold_gpu<<<1,1,0,stream>>>(gate_ns);
            check(cudaEventRecord(gate_end, stream));
            check(cudaEventRecord(start, stream));
            auto began = std::chrono::steady_clock::now();
            touch<<<grid,128,0,stream>>>(data);
            touch<<<grid,128,0,stream>>>(data);
            check(cudaGraphLaunch(executable, stream));
            touch<<<grid,128,0,stream>>>(data);
            touch<<<grid,128,0,stream>>>(data);
            check(cudaEventRecord(stop, stream));
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - began).count();
            const cudaError_t gate_status = cudaEventQuery(gate_end);
            const bool ahead = gate_status == cudaErrorNotReady;
            if (!ahead) {
                std::fprintf(stderr, "Host did not finish enqueue before gate retired\n");
                return 3;
            }
            check(cudaEventSynchronize(stop));
            float milliseconds;
            check(cudaEventElapsedTime(&milliseconds, start, stop));
            std::printf("{\"kind\":\"sample\",\"grid\":%d,\"block\":128,\"trial\":%d,\"graph_nodes\":%d,\"eager_nodes\":4,\"host_ahead\":true,\"enqueue_ns\":%lld,\"device_chain_ms\":%.9f}\n",
                grid, trial, nodes, static_cast<long long>(elapsed), milliseconds);
        }
        check(cudaEventDestroy(gate_end));
        check(cudaEventDestroy(start));
        check(cudaEventDestroy(stop));
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(graph));
    }
    check(cudaStreamDestroy(stream));
    check(cudaFree(data));
    return 0;
}
