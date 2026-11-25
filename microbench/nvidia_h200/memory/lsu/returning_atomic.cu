// Isolated global returning atomicAdd on one shared counter. No operator.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2); } } while(0)

__global__ void returning_atomic(int* counter, unsigned long long* clocks,
                                 int* sink, int repeats, int hold) {
  __syncthreads();
  if(threadIdx.x==0) {
    int checksum=0;
    unsigned long long begin=clock64(), first=0;
    for(int i=0;i<repeats;++i) {
      int ticket=atomicAdd(counter,1);
      // Consuming the ticket before the timestamp forces a returning atomic.
      asm volatile("add.s32 %0,%0,%1;" : "+r"(checksum) : "r"(ticket));
      unsigned long long ready=clock64();
      if(i==0) first=ready-begin;
      if(hold) while(clock64()-ready<static_cast<unsigned>(hold)) {}
    }
    clocks[blockIdx.x*2]=first;
    clocks[blockIdx.x*2+1]=clock64()-begin;
    sink[blockIdx.x]=checksum;
  }
}

int main() {
  int* counter; int* sink; unsigned long long* clocks;
  CHECK(cudaMalloc(&counter,sizeof(int))); CHECK(cudaMalloc(&sink,132*sizeof(int)));
  CHECK(cudaMalloc(&clocks,264*sizeof(*clocks)));
  CHECK(cudaFuncSetAttribute(returning_atomic,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024));
  CHECK(cudaMemset(counter,0,sizeof(int)));
  for(int i=0;i<32;++i) returning_atomic<<<132,128,128*1024>>>(counter,clocks,sink,512,0);
  CHECK(cudaDeviceSynchronize());
  printf("grid,repeats,hold,trial,cta,first_cycles,total_cycles\n");
  for(int grid:{1,2,8,32,64,132}) for(int repeats:{1,64,512})
    for(int hold:{0,512}) for(int trial=0;trial<3;++trial) {
      CHECK(cudaMemset(counter,0,sizeof(int)));
      returning_atomic<<<grid,128,128*1024>>>(counter,clocks,sink,repeats,hold);
      CHECK(cudaDeviceSynchronize());
      std::vector<unsigned long long> host(grid*2);
      CHECK(cudaMemcpy(host.data(),clocks,host.size()*sizeof(*clocks),cudaMemcpyDeviceToHost));
      int final_count;CHECK(cudaMemcpy(&final_count,counter,sizeof(int),cudaMemcpyDeviceToHost));
      if(final_count!=grid*repeats) return 3;
      for(int c=0;c<grid;++c) printf("%d,%d,%d,%d,%d,%llu,%llu\n",
        grid,repeats,hold,trial,c,host[c*2],host[c*2+1]);
    }
  CHECK(cudaFree(counter));CHECK(cudaFree(sink));CHECK(cudaFree(clocks));
}
