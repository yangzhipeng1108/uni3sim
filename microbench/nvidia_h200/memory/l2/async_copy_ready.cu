// Isolated cp.async issue-to-ready latency. No attention, MMA, or softmax.
// Warm unique per-CTA source, one 128-thread producer, one resident CTA/SM.
// Each warp times its own copy+wait_all; CTA barrier is outside the clock.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(2); } } while (0)

__global__ void ready(const uint4* input, unsigned long long* result,
                      int vectors, int repeats) {
  extern __shared__ __align__(16) uint4 stage[];
  const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(stage));
  input += static_cast<long long>(blockIdx.x) * vectors;
  for (int r = 0; r < repeats; ++r) {
    __syncthreads();
    unsigned long long start = clock64();
    for (int i = threadIdx.x; i < vectors; i += blockDim.x) {
      unsigned target = base + i * 16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                   "r"(target), "l"(input + i) : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    unsigned long long elapsed = clock64() - start;
    if (threadIdx.x % 32 == 0)
      result[(static_cast<long long>(blockIdx.x) * repeats + r) * 4
             + threadIdx.x / 32] = elapsed;
  }
}

int main() {
  cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("gpu,%s,sms,%d\n", prop.name, prop.multiProcessorCount);
  CHECK(cudaFuncSetAttribute(ready, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
  printf("grid,bytes,trial,median_cycles,p10_cycles,p90_cycles\n");
  for (int grid : {1, prop.multiProcessorCount}) for (int bytes : {8192,16384,32768,65536}) {
    int vectors = bytes / 16, repeats = 64;
    uint4* input; unsigned long long* output;
    size_t count = static_cast<size_t>(grid) * repeats * 4;
    CHECK(cudaMalloc(&input, static_cast<size_t>(grid) * bytes));
    CHECK(cudaMalloc(&output, count * sizeof(*output)));
    CHECK(cudaMemset(input, 1, static_cast<size_t>(grid) * bytes));
    ready<<<grid,128,128*1024>>>(input,output,vectors,repeats);
    CHECK(cudaDeviceSynchronize());
    for (int trial = 0; trial < 3; ++trial) {
      ready<<<grid,128,128*1024>>>(input,output,vectors,repeats);
      CHECK(cudaDeviceSynchronize());
      std::vector<unsigned long long> host(count), times;
      CHECK(cudaMemcpy(host.data(),output,count*sizeof(*output),cudaMemcpyDeviceToHost));
      // Per-CTA completion is the latest of four producer warps. Exclude
      // the first four repeats rather than calling them an observed hit.
      for (int c=0;c<grid;++c) for (int r=4;r<repeats;++r) {
        auto begin=host.begin()+(c*repeats+r)*4;
        times.push_back(*std::max_element(begin,begin+4));
      }
      std::sort(times.begin(),times.end());
      printf("%d,%d,%d,%llu,%llu,%llu\n",grid,bytes,trial,
             times[times.size()/2],times[times.size()/10],times[times.size()*9/10]);
    }
    CHECK(cudaFree(input)); CHECK(cudaFree(output));
  }
}
