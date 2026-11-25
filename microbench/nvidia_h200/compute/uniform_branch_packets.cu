// Independent finite conditional forward-guard packets. No Attention, Tensor,
// softmax, descriptor, copy pipeline or production inputs. CP=1 is a generic
// uniform-taken path,CP=2 validates the skipped integer body independently.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
constexpr int Threads=384,SharedBytes=196608,MaxGuards=64;
struct Stamp{unsigned long long c0,c1,n0,n1;};
struct Prelude{unsigned long long cycles,ns;};
__device__ __forceinline__ unsigned long long nanos(){unsigned long long v;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(v));return v;}
static unsigned hash_word(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
#include "uniform_branch_packets.cuh"
template<int Guards,int Skip,bool Guarded>
__global__ __launch_bounds__(Threads,1) void guard_packet_kernel(unsigned mode,int warps,
    unsigned opaque_zero,const unsigned* input,unsigned* values,Stamp* trace,Prelude* preludes){
 if(threadIdx.x==0){
  unsigned state=blockIdx.x+1;auto n0=nanos();auto c0=clock64();
  do{for(int i=0;i<128;++i)asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;":"+r"(state));}while(nanos()-n0<3000000);
  asm volatile(""::"r"(state):"memory");
  preludes[blockIdx.x]={static_cast<unsigned long long>(clock64()-c0),nanos()-n0};
 }
 __syncthreads();
 if(threadIdx.x>=warps*32)return;
 size_t tid=size_t(blockIdx.x)*Threads+threadIdx.x;
 auto s=packet<Guards,Skip,Guarded>(mode,threadIdx.x,input+tid*MaxGuards,values+tid*(2*Guards),opaque_zero,warps*32);
 if(threadIdx.x%32==0)trace[blockIdx.x*12+threadIdx.x/32]=s;
}
static unsigned bits(float v){unsigned result;std::memcpy(&result,&v,4);return result;}
template<int Guards,int Skip,bool Guarded>
void run(int grid,int warps,unsigned mode,const unsigned* input,unsigned* values,Stamp* trace,Prelude* prelude,const std::vector<unsigned>& host_input){
 auto kernel=guard_packet_kernel<Guards,Skip,Guarded>;
 CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
 for(int trial=-1;trial<3;++trial){
  kernel<<<grid,Threads,SharedBytes>>>(mode,warps,0,input,values,trace,prelude);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  std::vector<unsigned> results(size_t(grid)*Threads*2*Guards);std::vector<Stamp> clocks(grid*12);std::vector<Prelude> p(grid);
  CHECK(cudaMemcpy(results.data(),values,results.size()*4,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(clocks.data(),trace,clocks.size()*sizeof(Stamp),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(p.data(),prelude,p.size()*sizeof(Prelude),cudaMemcpyDeviceToHost));
  for(int c=0;c<grid;++c){
   for(int t=0;t<warps*32;++t)for(int i=0;i<Guards;++i){
    size_t tid=size_t(c)*Threads+t;unsigned u=host_input[tid*MaxGuards+i];
    unsigned expected=bits(float(t+1)+float(i)*.25f);
    if(Guarded&&mode!=1){for(int k=0;k<Skip;++k)u=u*u+1013904223u;if(u>127)expected=0xff800000u;}
    else if((t%32)>127-4*i)expected=0xff800000u;
    if(results[tid*2*Guards+i]!=expected||results[tid*2*Guards+Guards+i]!=u){std::fprintf(stderr,"value mismatch guards%d skip%d guard%d mode%u cta%d thread%d i%d\n",Guards,Skip,int(Guarded),mode,c,t,i);std::exit(3);}
   }
   // globaltimer has a 32ns quantum: a short control may have zero ns delta.
   if(trial>=0)for(int w=0;w<warps;++w){auto s=clocks[c*12+w];if(s.c1<=s.c0||s.n1<s.n0||!p[c].ns||!p[c].cycles){
    std::fprintf(stderr,"invalid clock guards%d skip%d guard%d mode%u grid%d warps%d trial%d cta%d warp%d c%llu:%llu n%llu:%llu prelude%llu:%llu\n",Guards,Skip,int(Guarded),mode,grid,warps,trial,c,w,s.c0,s.c1,s.n0,s.n1,p[c].cycles,p[c].ns);std::exit(4);}
    std::printf("%d,%d,%d,%d,%u,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%d,0\n",Guards,Skip,int(Guarded),warps,mode,grid,trial,c,w,s.c0,s.c1,s.n0,s.n1,p[c].cycles,p[c].ns,2*Guards*32);
   }
  }
  std::fflush(stdout);
 }
}
template<int Guards>void packets(int grid,int warps,const unsigned* in,unsigned* values,Stamp* trace,Prelude* prelude,const std::vector<unsigned>& host){
 run<Guards,0,false>(grid,warps,1,in,values,trace,prelude,host);
 for(unsigned mode:{1u,2u}){
  run<Guards,8,true>(grid,warps,mode,in,values,trace,prelude,host);
  run<Guards,12,true>(grid,warps,mode,in,values,trace,prelude,host);
  run<Guards,16,true>(grid,warps,mode,in,values,trace,prelude,host);
 }
}
int main(){
 int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;
 std::vector<unsigned> host(size_t(132)*Threads*MaxGuards);for(size_t i=0;i<host.size();++i)host[i]=hash_word(unsigned(i)+17);
 unsigned *input,*values;Stamp* trace;Prelude* prelude;CHECK(cudaMalloc(&input,host.size()*4));CHECK(cudaMemcpy(input,host.data(),host.size()*4,cudaMemcpyHostToDevice));
 CHECK(cudaMalloc(&values,host.size()*8));CHECK(cudaMalloc(&trace,132*12*sizeof(Stamp)));CHECK(cudaMalloc(&prelude,132*sizeof(Prelude)));
 std::puts("guards,skip_ops,guarded,warps,mode,grid,trial,cta,warp,c0,c1,n0,n1,prelude_cycles,prelude_ns,checked_values,bad_values");
 for(int grid:{1,132})for(int warps:{1,4,8,12}){
  packets<16>(grid,warps,input,values,trace,prelude,host);packets<32>(grid,warps,input,values,trace,prelude,host);
  packets<48>(grid,warps,input,values,trace,prelude,host);packets<64>(grid,warps,input,values,trace,prelude,host);
 }
 CHECK(cudaFree(input));CHECK(cudaFree(values));CHECK(cudaFree(trace));CHECK(cudaFree(prelude));
}
