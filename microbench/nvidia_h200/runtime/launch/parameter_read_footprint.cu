// Independent kernel-parameter read footprint. No operator, matrix or operand
// stream. Whole-kernel NSYS timing also includes any compiler-hoisted loads.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int SharedBytes=128*1024;
template<int Words> struct Payload {unsigned long long words[Words];};
__device__ unsigned long long global_ns() {
  unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t)::"memory");return t;
}
template<int Threads,int Words,int Mode>
__global__ void __launch_bounds__(Threads,1)
parameter_reads(const __grid_constant__ Payload<Words> payload,unsigned long long* trace,
                unsigned long long* sink,int hold) {
  auto n0=global_ns();auto c0=clock64();
  extern __shared__ volatile unsigned scratch[];
  if(threadIdx.x==0) scratch[0]=1;
  unsigned long long value=0;
  if constexpr(Mode==0) value=payload.words[0]^payload.words[Words-1];
  if constexpr(Mode==1) {
    #pragma unroll
    for(int i=0;i<Words;i+=8) value^=payload.words[i];
  }
  if constexpr(Mode==2) {
    #pragma unroll
    for(int i=0;i<Words;++i) value^=payload.words[i];
  }
  auto begin=clock64();
  while(clock64()-begin<static_cast<unsigned>(hold)) asm volatile("add.u64 %0,%0,0;":"+l"(value));
  sink[blockIdx.x*Threads+threadIdx.x]=value^threadIdx.x;
  __syncthreads();auto c1=clock64();auto n1=global_ns();
  if(threadIdx.x==0) {
    trace[blockIdx.x*3]=n0;trace[blockIdx.x*3+1]=n1;trace[blockIdx.x*3+2]=c1-c0;
  }
}
template<int Threads,int Words,int Mode>
void run(int grid,int hold,int& id,unsigned long long* trace,unsigned long long* sink) {
  auto kernel=parameter_reads<Threads,Words,Mode>;cudaFuncAttributes attributes;
  CHECK(cudaFuncGetAttributes(&attributes,kernel));
  CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  for(int trial=-4;trial<8;++trial) {
    Payload<Words> payload{};for(int i=0;i<Words;++i) payload.words[i]=i*9973ull+17+trial+4;
    unsigned long long expected=0;
    if constexpr(Mode==0) expected=payload.words[0]^payload.words[Words-1];
    if constexpr(Mode==1) for(int i=0;i<Words;i+=8) expected^=payload.words[i];
    if constexpr(Mode==2) for(int i=0;i<Words;++i) expected^=payload.words[i];
    kernel<<<grid,Threads,SharedBytes>>>(payload,trace,sink,hold);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid*3),values(grid*Threads);
    CHECK(cudaMemcpy(host.data(),trace,host.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values.data(),sink,values.size()*sizeof(*sink),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) {
      for(int thread=0;thread<Threads;++thread) if(values[c*Threads+thread]!=(expected^thread)) {
        fprintf(stderr,"parameter checksum mismatch\n");exit(3);
      }
      if(host[c*3+1]<=host[c*3]) {fprintf(stderr,"invalid timing\n");exit(4);}
      printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu\n",id,grid,Threads,Words*8,Mode,hold,trial,
        attributes.numRegs,host[c*3],host[c*3+1],host[c*3+2]);
    }
    ++id;
  }
}
template<int Threads,int Words> void modes(int grid,int hold,int& id,unsigned long long* trace,unsigned long long* sink) {
  run<Threads,Words,0>(grid,hold,id,trace,sink);run<Threads,Words,1>(grid,hold,id,trace,sink);
  run<Threads,Words,2>(grid,hold,id,trace,sink);
}
template<int Threads> void sizes(int grid,int hold,int& id,unsigned long long* trace,unsigned long long* sink) {
  modes<Threads,8>(grid,hold,id,trace,sink);modes<Threads,128>(grid,hold,id,trace,sink);
  modes<Threads,480>(grid,hold,id,trace,sink);
}
int main() {
  unsigned long long *trace,*sink;CHECK(cudaMalloc(&trace,132*3*sizeof(*trace)));
  CHECK(cudaMalloc(&sink,132*384*sizeof(*sink)));int id=0;
  printf("launch_id,grid,threads,payload_bytes,mode,hold_cycles,trial,registers,begin_ns,end_ns,body_cycles\n");
  for(int grid:{1,132}) for(int hold:{0,4096}) {
    sizes<128>(grid,hold,id,trace,sink);sizes<256>(grid,hold,id,trace,sink);sizes<384>(grid,hold,id,trace,sink);
  }
  CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
