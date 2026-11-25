// Independent parameter-region read parallelism. Warp-sync markers retain
// each group of loaded values before reduction; this is not an operator price.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
template<int Words> struct Payload {unsigned long long words[Words];};
#include "parameter_read_groups.cuh"
constexpr int Threads=128,SharedBytes=128*1024;
int Mutation=2;
__device__ unsigned long long global_ns() {
  unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t)::"memory");return t;
}
template<int Words,int Group>
__global__ void __launch_bounds__(Threads,1)
parameter_parallel(const __grid_constant__ Payload<Words> payload,
                   unsigned long long* trace,unsigned long long* sink,int hold) {
  auto n0=global_ns();auto c0=clock64();extern __shared__ volatile unsigned scratch[];
  if(threadIdx.x==0) scratch[0]=1;
  unsigned long long value;
  if constexpr(Group==0) value=payload.words[0]^payload.words[Words-1];
  else value=grouped_parameter_reads<Words,Group>(payload);
  auto begin=clock64();
  while(clock64()-begin<static_cast<unsigned>(hold)) asm volatile("add.u64 %0,%0,0;":"+l"(value));
  sink[blockIdx.x*Threads+threadIdx.x]=value^threadIdx.x;
  __syncthreads();auto c1=clock64();auto n1=global_ns();
  if(threadIdx.x==0) {trace[blockIdx.x*3]=n0;trace[blockIdx.x*3+1]=n1;trace[blockIdx.x*3+2]=c1-c0;}
}
template<int Words,int Group>
void run(int grid,int hold,int& id,unsigned long long* trace,unsigned long long* sink) {
  auto kernel=parameter_parallel<Words,Group>;cudaFuncAttributes attributes;
  CHECK(cudaFuncGetAttributes(&attributes,kernel));
  CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  for(int trial=-4;trial<8;++trial) {
    Payload<Words> payload{};for(int i=0;i<Words;++i) payload.words[i]=i*9973ull+17
        +(Mutation==0?0:(Mutation==1?(i==0?trial+4:0):trial+4));
    unsigned long long expected=0;
    if constexpr(Group==0) expected=payload.words[0]^payload.words[Words-1];
    else for(int i=0;i<Words;i+=8) expected^=payload.words[i];
    kernel<<<grid,Threads,SharedBytes>>>(payload,trace,sink,hold);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid*3),values(grid*Threads);
    CHECK(cudaMemcpy(host.data(),trace,host.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values.data(),sink,values.size()*sizeof(*sink),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) {
      for(int t=0;t<Threads;++t) if(values[c*Threads+t]!=(expected^t)) {fprintf(stderr,"bad checksum\n");exit(3);}
      printf("%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%d\n",id,grid,Words*8,Group,hold,trial,attributes.numRegs,
        host[c*3],host[c*3+1],host[c*3+2],Mutation);
    }
    ++id;
  }
}
template<int Words> void groups(int grid,int hold,int& id,unsigned long long* trace,unsigned long long* sink) {
  run<Words,0>(grid,hold,id,trace,sink);run<Words,1>(grid,hold,id,trace,sink);
  run<Words,8>(grid,hold,id,trace,sink);run<Words,Words/8>(grid,hold,id,trace,sink);
}
int main(int argc,char**) {
  unsigned long long *trace,*sink;CHECK(cudaMalloc(&trace,132*3*sizeof(*trace)));
  CHECK(cudaMalloc(&sink,132*Threads*sizeof(*sink)));int id=0;
  printf("launch_id,grid,payload_bytes,group,hold_cycles,trial,registers,begin_ns,end_ns,body_cycles,mutation\n");
  if(argc==2) {
    for(int mutation:{0,1,2}) for(int grid:{1,132}) {
      Mutation=mutation;
      run<128,0>(grid,0,id,trace,sink);run<128,16>(grid,0,id,trace,sink);
      run<480,0>(grid,0,id,trace,sink);run<480,60>(grid,0,id,trace,sink);
    }
  } else for(int grid:{1,8,32,132}) for(int hold:{0,4096}) {
    groups<128>(grid,hold,id,trace,sink);groups<480>(grid,hold,id,trace,sink);
  }
  CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
