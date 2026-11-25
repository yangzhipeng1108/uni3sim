// Independent read-stream capacity under duty-cycle and CTA residency changes.
// No GEMM, attention, model weights, production shapes, or production timings.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <thread>

#define CUDA_OK(call) do { auto code = (call); if (code != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(code)); std::exit(1); \
} } while (0)

struct Stamp { unsigned long long cycles, ns; };
__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__global__ void read_stream(const uint4* input, size_t count,
                           unsigned long long* output, Stamp* stamps) {
  const auto start_ns = global_ns();
  const auto start_cycles = static_cast<unsigned long long>(clock64());
  unsigned long long sum = 0;
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += gridDim.x * blockDim.x) {
    uint4 value = input[index];
    sum += value.x; sum += value.y; sum += value.z; sum += value.w;
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    stamps[blockIdx.x] = {static_cast<unsigned long long>(clock64()) - start_cycles,
                         global_ns() - start_ns};
  }
}

int main() {
  cudaDeviceProp properties;
  CUDA_OK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 9 || properties.multiProcessorCount != 132) return 2;
  constexpr size_t bytes = size_t(512) << 20;
  uint4* input; unsigned long long* output; Stamp* stamps;
  CUDA_OK(cudaMalloc(&input, bytes));
  CUDA_OK(cudaMalloc(&output, 528 * 384 * sizeof(*output)));
  CUDA_OK(cudaMalloc(&stamps, 528 * sizeof(*stamps)));
  CUDA_OK(cudaMemset(input, 1, bytes));
  cudaStream_t stream;
  CUDA_OK(cudaStreamCreate(&stream));
  cudaEvent_t start, stop;
  CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  struct Profile { int blocks, threads, shared_bytes; };
  const Profile profiles[] = {{528, 128, 0}, {132, 384, 0}, {132, 384, 200 * 1024}};
  for (auto profile : profiles) {
    CUDA_OK(cudaFuncSetAttribute(read_stream, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                profile.shared_bytes));
    int residency;
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residency, read_stream,
                  profile.threads, profile.shared_bytes));
    cudaGraph_t graph;
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    read_stream<<<profile.blocks, profile.threads, profile.shared_bytes, stream>>>(
        input, bytes / sizeof(uint4), output, stamps);
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t replay;
    CUDA_OK(cudaGraphInstantiate(&replay, graph, nullptr, nullptr, 0));
    for (int warmup = 0; warmup < 16; ++warmup) CUDA_OK(cudaGraphLaunch(replay, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    for (int gap_ms : {0, 2, 16, 128}) {
      for (int trial = 0; trial < 9; ++trial) {
        std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
        CUDA_OK(cudaEventRecord(start, stream));
        CUDA_OK(cudaGraphLaunch(replay, stream));
        CUDA_OK(cudaEventRecord(stop, stream));
        CUDA_OK(cudaEventSynchronize(stop));
        float elapsed_ms;
        CUDA_OK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        Stamp host_stamps[528];
        CUDA_OK(cudaMemcpy(host_stamps, stamps, profile.blocks * sizeof(Stamp), cudaMemcpyDeviceToHost));
        double ratio_sum = 0.0;
        for (int block = 0; block < profile.blocks; ++block) {
          if (!host_stamps[block].ns) return 3;
          ratio_sum += double(host_stamps[block].cycles) / host_stamps[block].ns;
        }
        std::printf("{\"blocks\":%d,\"threads\":%d,\"dynamic_shared_bytes\":%d,"
          "\"resident_blocks_per_sm_capacity\":%d,\"gap_ms\":%d,\"trial\":%d,"
          "\"logical_read_bytes\":%zu,\"graph_elapsed_ms\":%.9g,\"mean_sm_clock_ghz\":%.9g}\n",
          profile.blocks, profile.threads, profile.shared_bytes, residency,
          gap_ms, trial, bytes, elapsed_ms, ratio_sum / profile.blocks);
      }
    }
    CUDA_OK(cudaGraphExecDestroy(replay)); CUDA_OK(cudaGraphDestroy(graph));
  }
  CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
  CUDA_OK(cudaStreamDestroy(stream));
  CUDA_OK(cudaFree(stamps)); CUDA_OK(cudaFree(output)); CUDA_OK(cudaFree(input));
}
