// Independent uniform/regular constant-parameter read paths and joint demand.
// One word per64B address region; no Attention, matrix or production input.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}while(0)
struct Payload { unsigned long long words[480]; };
#include "pipe_read_groups.cuh"
constexpr int Threads=128,SharedBytes=128*1024;
__device__ unsigned long long global_ns() {
  unsigned long long value;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(value)::"memory");return value;
}
// Modes0/1:uniform/regular,2:both per warp,3:two warps per path;
// modes4/5:matching single-path two-warp controls.
template<int Mode,int Regions>
__global__ void __launch_bounds__(Threads,1)
parameter_pipes(const __grid_constant__ Payload payload,unsigned long long* trace,
                unsigned long long* sink,unsigned hold_cycles) {
  static_assert(Mode>=0 && Mode<=5);
  extern __shared__ volatile unsigned offsets[];
  offsets[threadIdx.x]=0;__syncthreads();
  auto n0=global_ns();auto c0=clock64();
  unsigned idx=offsets[threadIdx.x];
  unsigned long long u=0,r=0;
  if constexpr(Mode==0) u=uniform_reads<Regions>(payload);
  else if constexpr(Mode==1) r=regular_reads<Regions>(payload,idx);
  else if constexpr(Mode==2) {
    u=uniform_reads<Regions>(payload);
    // Force the uniform result live before producing the opaque,physically
    // unchanged zero offset for the regular path.
    asm volatile("":"+r"(idx):"l"(u):"memory");
    r=regular_reads<Regions>(payload,idx);
  } else if constexpr(Mode==3) {
    if(threadIdx.x<64) u=uniform_reads<Regions>(payload);
    else r=regular_reads<Regions>(payload,idx);
  } else if constexpr(Mode==4) {
    if(threadIdx.x<64) u=uniform_reads<Regions>(payload);
  } else if constexpr(Mode==5) {
    if(threadIdx.x>=64) r=regular_reads<Regions>(payload,idx);
  }
  auto h0=clock64();
  while(clock64()-h0<hold_cycles) asm volatile("add.u64 %0,%0,0;":"+l"(r));
  int i=(blockIdx.x*Threads+threadIdx.x)*2;
  sink[i]=u;sink[i+1]=r;
  __syncthreads();auto c1=clock64();auto n1=global_ns();
  if(threadIdx.x==0) {trace[blockIdx.x*3]=n0;trace[blockIdx.x*3+1]=n1;trace[blockIdx.x*3+2]=c1-c0;}
}
template<int Mode,int Regions>
void run(int grid,unsigned hold,int& id,unsigned long long* trace,unsigned long long* sink) {
  auto kernel=parameter_pipes<Mode,Regions>;
  CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  for(int trial=-4;trial<8;++trial) {
    Payload payload{};for(int i=0;i<480;++i) payload.words[i]=i*9973ull+17+trial+4;
    unsigned long long expected=0;
    for(int i=0;i<Regions;++i) expected^=payload.words[(i==Regions-1?59:i)*8];
    kernel<<<grid,Threads,SharedBytes>>>(payload,trace,sink,hold);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> times(grid*3),values(grid*Threads*2);
    CHECK(cudaMemcpy(times.data(),trace,times.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values.data(),sink,values.size()*sizeof(*sink),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) {
      for(int t=0;t<Threads;++t) {
        unsigned long long eu=(Mode==0||Mode==2||((Mode==3||Mode==4)&&t<64))?expected:0;
        unsigned long long er=(Mode==1||Mode==2||((Mode==3||Mode==5)&&t>=64))?expected:0;
        int i=(c*Threads+t)*2;
        if(values[i]!=eu||values[i+1]!=er) {std::fprintf(stderr,"bad pipe checksum\n");std::exit(3);}
      }
      std::printf("%d,%d,%d,%d,%u,%d,%d,%llu,%llu,%llu\n",id,Mode,Regions,grid,hold,trial,c,times[c*3],times[c*3+1],times[c*3+2]);
    }
    ++id;
  }
}
int main() {
  unsigned long long *trace,*sink;CHECK(cudaMalloc(&trace,132*3*sizeof(*trace)));
  CHECK(cudaMalloc(&sink,132*Threads*2*sizeof(*sink)));int id=0;
  std::printf("launch_id,mode,regions,grid,hold_cycles,trial,cta,begin_ns,end_ns,cycles\n");
  for(int grid:{1,8,32,132}) for(unsigned hold:{0u,4096u}) {
    run<0,2>(grid,hold,id,trace,sink);run<0,16>(grid,hold,id,trace,sink);run<0,60>(grid,hold,id,trace,sink);
    run<1,2>(grid,hold,id,trace,sink);run<1,16>(grid,hold,id,trace,sink);run<1,60>(grid,hold,id,trace,sink);
    run<2,2>(grid,hold,id,trace,sink);run<2,16>(grid,hold,id,trace,sink);run<2,60>(grid,hold,id,trace,sink);
    run<3,2>(grid,hold,id,trace,sink);run<3,16>(grid,hold,id,trace,sink);run<3,60>(grid,hold,id,trace,sink);
    run<4,2>(grid,hold,id,trace,sink);run<4,16>(grid,hold,id,trace,sink);run<4,60>(grid,hold,id,trace,sink);
    run<5,2>(grid,hold,id,trace,sink);run<5,16>(grid,hold,id,trace,sink);run<5,60>(grid,hold,id,trace,sink);
  }
  CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
