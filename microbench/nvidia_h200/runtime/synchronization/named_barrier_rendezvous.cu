// Independent ready named-barrier pairs and2/3-actor token rings.
// No Tensor, Attention, descriptors, copy pipeline or production input.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
constexpr int SharedBytes=196608;
struct Stamp{unsigned long long c0,c1,n0,n1;};
struct Prelude{unsigned long long cycles,ns;};
__device__ __forceinline__ unsigned long long nanos(){unsigned long long v;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(v));return v;}
__device__ __forceinline__ void arrive(int id){asm volatile("bar.arrive %0,256;"::"r"(id):"memory");}
__device__ __forceinline__ void sync(int id){asm volatile("bar.sync %0,256;"::"r"(id):"memory");}
__host__ __device__ unsigned feedback(unsigned x){for(int i=0;i<4;++i)x=x*1664525u+1013904223u;return x;}
template<int Threads,int Groups,int Mode>
__global__ __launch_bounds__(Threads,1) void rendezvous(int repeats,unsigned* values,Stamp* traces,Prelude* preludes){
 if(threadIdx.x==0){
  unsigned state=blockIdx.x+1;auto n0=nanos();auto c0=clock64();
  do{for(int i=0;i<128;++i)asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;":"+r"(state));}while(nanos()-n0<3000000);
  asm volatile(""::"r"(state):"memory");
  preludes[blockIdx.x]={static_cast<unsigned long long>(clock64()-c0),nanos()-n0};
 }
 __syncthreads();
 if(threadIdx.x<128||threadIdx.x>=(Groups+1)*128)return;
 int actor=(threadIdx.x-128)/128,id=actor+1,next=(actor+1)%Groups+1;
 unsigned state=threadIdx.x+1;
 // The first actor provides the initial128 arrivals. Each phase consumes
 // 128 arrivals from its predecessor and128 from its own blocking sync.
 if constexpr(Mode==2)if(actor==0)arrive(id);
 Stamp s;s.n0=nanos();s.c0=clock64();
 #pragma unroll 1
 for(int r=0;r<repeats;++r){
  if constexpr(Mode==2)sync(id);
  #pragma unroll
  for(int i=0;i<4;++i)asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;":"+r"(state));
  if constexpr(Mode==1){arrive(id);sync(id);}
  if constexpr(Mode==2)arrive(next);
 }
 s.c1=clock64();s.n1=nanos();
 values[blockIdx.x*512+threadIdx.x]=state;
 if(threadIdx.x%128==0)traces[blockIdx.x*3+actor]=s;
}
template<int Threads,int Groups,int Mode>
void run(int grid,int repeats,unsigned* values,Stamp* traces,Prelude* preludes){
 auto kernel=rendezvous<Threads,Groups,Mode>;
 CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
 for(int trial=-1;trial<3;++trial){
  kernel<<<grid,Threads,SharedBytes>>>(repeats,values,traces,preludes);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  std::vector<unsigned> result(grid*512);std::vector<Stamp> clocks(grid*3);std::vector<Prelude> p(grid);
  CHECK(cudaMemcpy(result.data(),values,result.size()*4,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(clocks.data(),traces,clocks.size()*sizeof(Stamp),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(p.data(),preludes,p.size()*sizeof(Prelude),cudaMemcpyDeviceToHost));
  for(int c=0;c<grid;++c){
   for(int t=128;t<(Groups+1)*128;++t){unsigned expected=t+1;for(int r=0;r<repeats;++r)expected=feedback(expected);
    if(result[c*512+t]!=expected){std::fprintf(stderr,"numeric mismatch threads%d groups%d mode%d grid%d repeats%d trial%d cta%d thread%d\n",Threads,Groups,Mode,grid,repeats,trial,c,t);std::exit(3);}}
   if(trial>=0)for(int actor=0;actor<Groups;++actor){auto s=clocks[c*3+actor];if(s.c1<=s.c0||s.n1<s.n0||p[c].ns<3000000||!p[c].cycles)std::exit(4);
    std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,128,0\n",Threads,Groups,Mode,grid,repeats,trial,c,actor,s.c0,s.c1,s.n0,s.n1,p[c].cycles,p[c].ns);}
  }
  std::fflush(stdout);
 }
}
int main(){
 int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;
 unsigned* values;Stamp* traces;Prelude* preludes;CHECK(cudaMalloc(&values,132*512*4));CHECK(cudaMalloc(&traces,132*3*sizeof(Stamp)));CHECK(cudaMalloc(&preludes,132*sizeof(Prelude)));
 std::puts("threads,groups,mode,grid,repeats,trial,cta,actor,c0,c1,n0,n1,prelude_cycles,prelude_ns,checked_values,bad_values");
 for(int grid:{1,132})for(int repeats:{1,2,4,16,64,128,256}){
  run<384,1,0>(grid,repeats,values,traces,preludes);run<384,1,1>(grid,repeats,values,traces,preludes);
  run<384,2,0>(grid,repeats,values,traces,preludes);run<384,2,1>(grid,repeats,values,traces,preludes);run<384,2,2>(grid,repeats,values,traces,preludes);
  run<512,1,0>(grid,repeats,values,traces,preludes);run<512,1,1>(grid,repeats,values,traces,preludes);
  run<512,3,0>(grid,repeats,values,traces,preludes);run<512,3,1>(grid,repeats,values,traces,preludes);run<512,3,2>(grid,repeats,values,traces,preludes);
 }
 CHECK(cudaFree(values));CHECK(cudaFree(traces));CHECK(cudaFree(preludes));
}
