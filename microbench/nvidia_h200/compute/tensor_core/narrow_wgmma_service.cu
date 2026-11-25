// Instruction-window service only: no TMA, GEMM matrices, or matrix output.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "shared_tensor_activity_helper.cuh"
#ifdef SINGLE_ASM_WINDOW
#include "narrow_wgmma_service_helper.cuh"
#endif
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
 fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2); } } while(0)
constexpr int kGrid=132,kShared=196608;
__device__ uint64_t timer_ns() { uint64_t v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v));return v; }
template<int Pending> __device__ void drain() {
 if constexpr(Pending==0) asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
 if constexpr(Pending==1) asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");
 if constexpr(Pending==3) asm volatile("wgmma.wait_group.sync.aligned 3;":::"memory");
}
template<int N,int Acc,int Wgs,int Pending>
__global__ __launch_bounds__(Wgs*128,1) void instruction_window(int seed,int rounds,uint64_t* times,float* sums) {
 extern __shared__ __align__(1024) unsigned char storage[];
 for(int i=threadIdx.x*4;i<kShared;i+=blockDim.x*4) {
  unsigned bits=0x3c003c00u;
  if(seed) {
   unsigned state=unsigned(i)*747796405u+unsigned(seed)*2891336453u;
   state=((state>>((state>>28)+4))^state)*277803737u;state=(state>>22)^state;
   unsigned low=(0x3000u+((state&3u)<<10))|((state&4u)<<13)|((state>>6)&1023u);
   unsigned high=(0x3000u+(((state>>3)&3u)<<10))|(((state>>3)&4u)<<13)|((state>>16)&1023u);
   bits=low|(high<<16);
  }
  *reinterpret_cast<unsigned*>(storage+i)=bits;
 }
 __syncthreads();
 float accum[Acc][N/2]={};
 // Read-only operands are shared by both WGs. A CTA-uniform descriptor also
 // avoids compiler-inserted drains for general-register descriptor updates.
 unsigned base=unsigned(__cvta_generic_to_shared(storage));
 uint64_t c0=clock64(),t0=timer_ns();wg_fence();
#ifdef SINGLE_ASM_WINDOW
 MatrixWindowRequest<N,Acc,Pending>::issue(accum,make_desc(base,16,1024),
     make_desc(base+65536,16,1024),rounds);
#else
 for(int r=0;r<rounds;++r) {
#pragma unroll
  for(int step=0;step<4;++step) {
#pragma unroll
   for(int chain=0;chain<Acc;++chain) {
    uint64_t a=make_desc(base+chain*8192+step*32,16,1024);
    uint64_t b=make_desc(base+65536+chain*8192+step*32,16,1024);
    MatrixRequest<N>::issue(accum[chain],a,b,(r|step)!=0);
   }
  }
  wg_commit();drain<Pending>();
 }
#endif
 drain<0>();uint64_t t1=timer_ns(),c1=clock64();
 float sum=0;
#pragma unroll
 for(int chain=0;chain<Acc;++chain) {
#pragma unroll
  for(int i=0;i<N/2;++i)sum+=accum[chain][i];
 }
 sums[blockIdx.x*Wgs*128+threadIdx.x]=sum;
 if(threadIdx.x%128==0) {
  int index=(blockIdx.x*Wgs+threadIdx.x/128)*2;
  times[index]=c1-c0;times[index+1]=t1-t0;
 }
}
template<int N,int Acc,int Wgs,int Pending> void measure(uint64_t* times,float* sums) {
 auto function=instruction_window<N,Acc,Wgs,Pending>;
 CHECK(cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
 cudaFuncAttributes attr;CHECK(cudaFuncGetAttributes(&attr,function));
 if(attr.localSizeBytes) {fprintf(stderr,"spill\n");exit(3);}
 std::vector<uint64_t> host(kGrid*Wgs*2);std::vector<float> check(kGrid*Wgs*128);
 for(int seed:{0,1,2,3}) for(int rounds:{128,512}) {
  std::vector<double> cycles,nanos,clocks;
  for(int trial=0;trial<9;++trial) {
   function<<<kGrid,Wgs*128,kShared>>>(seed,rounds,times,sums);CHECK(cudaDeviceSynchronize());
   CHECK(cudaMemcpy(host.data(),times,host.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost));
   CHECK(cudaMemcpy(check.data(),sums,check.size()*sizeof(float),cudaMemcpyDeviceToHost));
   for(float value:check) {
    if(!std::isfinite(value) || std::fabs(value)>float(4*rounds*64*Acc*(N/2)))exit(4);
    if(seed==0 && value!=float(rounds*64*Acc*(N/2)))exit(5);
   }
   if(trial>=2) for(int i=0;i<kGrid*Wgs;++i) {
    if(!host[2*i] || !host[2*i+1])exit(6);
    cycles.push_back(host[2*i]);nanos.push_back(host[2*i+1]);clocks.push_back(1000.0*host[2*i]/host[2*i+1]);
   }
  }
  auto median=[](std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];};
  printf("{\"instruction_n\":%d,\"chains\":%d,\"math_warpgroups\":%d,\"pending_groups\":%d,\"rounds\":%d,\"seed\":%d,\"flops_per_round_per_sm\":%d,\"cycles_p50\":%.3f,\"ns_p50\":%.3f,\"clock_mhz\":%.3f,\"registers\":%d,\"numerical_check_passed\":true}\n",N,Acc,Wgs,Pending,rounds,seed,Wgs*4*Acc*2*64*N*16,median(cycles),median(nanos),median(clocks),attr.numRegs);
  fflush(stdout);
 }
}
template<int N,int Acc,int Wgs> void windows(uint64_t* t,float* s) {
 measure<N,Acc,Wgs,0>(t,s);measure<N,Acc,Wgs,1>(t,s);measure<N,Acc,Wgs,3>(t,s);
}
template<int N,int Acc> void warpgroups(uint64_t* t,float* s) {
 windows<N,Acc,1>(t,s);windows<N,Acc,2>(t,s);
}
template<int N> void chains(uint64_t* t,float* s) {
 warpgroups<N,1>(t,s);warpgroups<N,2>(t,s);warpgroups<N,4>(t,s);
}
int main() {
 uint64_t* t=nullptr;float* s=nullptr;
 CHECK(cudaMalloc(&t,kGrid*4*sizeof(uint64_t)));CHECK(cudaMalloc(&s,kGrid*256*sizeof(float)));
 CHECK(cudaFuncSetAttribute(instruction_window<64,4,2,3>,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
 auto start=std::chrono::steady_clock::now();
 do {instruction_window<64,4,2,3><<<kGrid,256,kShared>>>(1,512,t,s);CHECK(cudaDeviceSynchronize());}
 while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<2.0);
 chains<8>(t,s);chains<16>(t,s);chains<32>(t,s);chains<64>(t,s);
 CHECK(cudaFree(s));CHECK(cudaFree(t));
}
