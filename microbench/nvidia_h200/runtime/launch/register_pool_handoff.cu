// Independent SM90a register-pool allocation/synchronization primitive.
// No matrix instructions, operands, normalization or operator program.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2); }} while(0)

// The generated helper keeps 160 values live at one compiler boundary.
// Its only purpose is ensuring the CTA initially owns a 168-register/thread
// allocation quantum, sufficient for the 40/232 producer/consumer partition.
#include "register_pool_pressure.cuh"
template<bool Handoff>
__global__ void __launch_bounds__(384,1) pool(unsigned long long* cycles,float* sink) {
  extern __shared__ volatile unsigned words[];
  if(threadIdx.x==0) words[0]=0;
  __syncthreads();
  auto start=clock64();
  if constexpr(Handoff) {
    if(threadIdx.x<128) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  }
  __syncthreads();
  auto stop=clock64();
  if(threadIdx.x==128) cycles[blockIdx.x]=stop-start;
  if(threadIdx.x>=128) sink[blockIdx.x*256+threadIdx.x-128]=pressure(float(threadIdx.x))+words[0];
}
template<bool Handoff> void run(int grid,unsigned long long* cycles,float* sink) {
  CHECK(cudaFuncSetAttribute(pool<Handoff>,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024));
  for(int i=0;i<8;++i) pool<Handoff><<<grid,384,128*1024>>>(cycles,sink);
  CHECK(cudaDeviceSynchronize());
  for(int trial=0;trial<5;++trial) {
    pool<Handoff><<<grid,384,128*1024>>>(cycles,sink);CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid);
    CHECK(cudaMemcpy(host.data(),cycles,host.size()*sizeof(*cycles),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) printf("%d,%d,%d,%d,%llu\n",Handoff,grid,trial,c,host[c]);
  }
}
int main() {
  unsigned long long* cycles;float* sink;
  CHECK(cudaMalloc(&cycles,132*sizeof(*cycles)));CHECK(cudaMalloc(&sink,132*256*sizeof(float)));
  printf("handoff,grid,trial,cta,cycles\n");
  for(int grid:{1,32,132}) {run<false>(grid,cycles,sink);run<true>(grid,cycles,sink);}
  CHECK(cudaFree(cycles));CHECK(cudaFree(sink));
}
