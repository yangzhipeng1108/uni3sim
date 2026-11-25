// Cold vs immediate-repeated cp.async readiness. Independent memory only.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2); } } while(0)
__global__ void evict(uint4* data,int count) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
    data[i]=make_uint4(i,i+1,i+2,i+3);
}
__global__ void ready(const uint4* input,unsigned long long* ticks,int vectors) {
  extern __shared__ __align__(16) uint4 shared[];
  unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  input+=static_cast<long long>(blockIdx.x)*vectors;
  __syncthreads();
  auto begin=clock64();
  for(int i=threadIdx.x;i<vectors;i+=128) {
    unsigned dst=base+i*16;
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::
                 "r"(dst),"l"(input+i):"memory");
  }
  asm volatile("cp.async.commit_group;cp.async.wait_group 0;" :::"memory");
  auto elapsed=clock64()-begin;
  if(threadIdx.x%32==0) ticks[blockIdx.x*4+threadIdx.x/32]=elapsed;
}
int main() {
  constexpr int flush_bytes=384*1024*1024;
  uint4 *input,*flush;unsigned long long* ticks;
  CHECK(cudaMalloc(&input,132*65536));CHECK(cudaMalloc(&flush,flush_bytes));
  CHECK(cudaMalloc(&ticks,132*4*sizeof(*ticks)));
  CHECK(cudaMemset(input,1,132*65536));
  CHECK(cudaFuncSetAttribute(ready,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024));
  printf("grid,bytes,trial,state,cta,cycles\n");
  for(int grid:{1,8,32,132}) for(int bytes:{128,512,2048,8192,16384,32768,65536})
    for(int trial=0;trial<5;++trial) {
      evict<<<132,256>>>(flush,flush_bytes/16);
      for(int state=0;state<2;++state) {
        ready<<<grid,128,128*1024>>>(input,ticks,bytes/16);
        CHECK(cudaDeviceSynchronize());
        std::vector<unsigned long long> host(grid*4);
        CHECK(cudaMemcpy(host.data(),ticks,host.size()*sizeof(*ticks),cudaMemcpyDeviceToHost));
        for(int c=0;c<grid;++c) printf("%d,%d,%d,%d,%d,%llu\n",grid,bytes,trial,state,c,
            *std::max_element(host.begin()+c*4,host.begin()+c*4+4));
      }
    }
  CHECK(cudaFree(input));CHECK(cudaFree(flush));CHECK(cudaFree(ticks));
}
