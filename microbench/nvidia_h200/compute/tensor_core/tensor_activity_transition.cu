// Independent finite Tensor activity after an ALU-only high-clock prelude.
// Shared/register operands only; no GEMM, TMA, output matrix or library.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "shared_tensor_activity_helper.cuh"
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int kGrid=132,kThreads=256,kShared=196608,kWindows=256,kRounds=32;
struct Interval {uint64_t cycles,ns;};
__device__ __forceinline__ uint64_t timestamp(){uint64_t x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
template<int N,int Acc> __global__ __launch_bounds__(256,1) void transition(
    int seed,Interval* output,Interval* prelude,float* sums) {
  extern __shared__ __align__(1024) unsigned char shared[];
  for(int i=threadIdx.x*4;i<kShared;i+=blockDim.x*4) {
    unsigned bits=0x3c003c00u;
    if(seed){unsigned state=unsigned(i)*747796405u+unsigned(seed)*2891336453u;state=((state>>((state>>28)+4))^state)*277803737u;state=(state>>22)^state;
      unsigned lo=(0x3000u+((state&3u)<<10))|((state&4u)<<13)|((state>>6)&1023u);
      unsigned hi=(0x3000u+(((state>>3)&3u)<<10))|(((state>>3)&4u)<<13)|((state>>16)&1023u);bits=lo|(hi<<16);}
    *reinterpret_cast<unsigned*>(shared+i)=bits;
  }
  __syncthreads();
  if(threadIdx.x==0) {
    uint64_t c0=clock64(),t0=timestamp();unsigned state=blockIdx.x+1;
    do{asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;":"+r"(state));}while(timestamp()-t0<3000000);
    asm volatile(""::"r"(state):"memory");uint64_t t1=timestamp(),c1=clock64();prelude[blockIdx.x]={c1-c0,t1-t0};
  }
  cooperative_groups::this_grid().sync();
  float accum[Acc][N/2]={};
  unsigned base=unsigned(__cvta_generic_to_shared(shared))+(threadIdx.x/128)*32768;
  for(int window=0;window<kWindows;++window) {
    uint64_t c0=clock64(),t0=timestamp();
    wg_fence();
    for(int round=0;round<kRounds;++round) {
#pragma unroll
      for(int step=0;step<4;++step) {
#pragma unroll
        for(int chain=0;chain<Acc;++chain) {
          uint64_t a=make_desc(base+chain*8192+step*32,16,1024);
          uint64_t b=make_desc(base+65536+chain*8192+step*32,16,1024);
          MatrixRequest<N>::issue(accum[chain],a,b,(round|step)!=0);
        }
      }
      wg_commit();wg_wait0();
    }
    uint64_t t1=timestamp(),c1=clock64();
    if(threadIdx.x%128==0)output[(blockIdx.x*2+threadIdx.x/128)*kWindows+window]={c1-c0,t1-t0};
  }
  float sum=0;
#pragma unroll
  for(int chain=0;chain<Acc;++chain)
#pragma unroll
    for(int i=0;i<N/2;++i)sum+=accum[chain][i];
  sums[blockIdx.x*kThreads+threadIdx.x]=sum;
}
template<int N,int Acc> void measure(Interval* output,Interval* prelude,float* sums) {
  CHECK(cudaFuncSetAttribute(transition<N,Acc>,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  cudaFuncAttributes attr;CHECK(cudaFuncGetAttributes(&attr,transition<N,Acc>));if(attr.localSizeBytes)return exit(3);
  for(int seed:{0,1,2,3}) {
    std::vector<std::vector<double>> ns(kWindows),cycles(kWindows),clocks(kWindows);
    std::vector<double> prelude_clocks;
    std::vector<Interval> host(kGrid*2*kWindows),warm(kGrid);std::vector<float> check(kGrid*kThreads);
    for(int trial=0;trial<4;++trial) {
      void* args[]={&seed,&output,&prelude,&sums};CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(transition<N,Acc>),dim3(kGrid),dim3(kThreads),args,kShared));CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Interval),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(warm.data(),prelude,warm.size()*sizeof(Interval),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(check.data(),sums,check.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(float v:check)if(!std::isfinite(v) || (seed==0 && v!=float(kRounds*64*(N/2)*Acc)) || std::fabs(v)>float(4*kRounds*64*(N/2)*Acc))exit(4);
      for(auto t:host)if(!t.ns || !t.cycles || 1000.*t.cycles/t.ns>3000 || 1000.*t.cycles/t.ns<100)exit(5);
      if(trial==0)continue;
      for(auto t:warm)prelude_clocks.push_back(1000.*t.cycles/t.ns);
      for(int wg=0;wg<kGrid*2;++wg)for(int w=0;w<kWindows;++w) {
        auto t=host[wg*kWindows+w];ns[w].push_back(t.ns);cycles[w].push_back(t.cycles);clocks[w].push_back(1000.*t.cycles/t.ns);
      }
    }
    auto median=[](std::vector<double>& v){std::sort(v.begin(),v.end());return v[v.size()/2];};
    printf("{\"kind\":\"prelude\",\"instruction_n\":%d,\"chains\":%d,\"seed\":%d,\"clock_mhz\":%.6f,\"registers\":%d,\"local_bytes\":%zu,\"checks_passed\":true}\n",N,Acc,seed,median(prelude_clocks),attr.numRegs,attr.localSizeBytes);
    double elapsed=0;
    for(int w=0;w<kWindows;++w) {double dt=median(ns[w]),dc=median(cycles[w]);
      printf("{\"kind\":\"window\",\"instruction_n\":%d,\"chains\":%d,\"seed\":%d,\"window\":%d,\"rounds\":%d,\"start_ns\":%.3f,\"ns_p50\":%.3f,\"cycles_p50\":%.3f,\"clock_mhz_p50\":%.6f,\"checks_passed\":true}\n",N,Acc,seed,w,kRounds,elapsed,dt,dc,median(clocks[w]));elapsed+=dt;
    }
    fflush(stdout);
  }
}
int main(){Interval *output=nullptr,*prelude=nullptr;float* sums=nullptr;CHECK(cudaMalloc(&output,kGrid*2*kWindows*sizeof(Interval)));CHECK(cudaMalloc(&prelude,kGrid*sizeof(Interval)));CHECK(cudaMalloc(&sums,kGrid*kThreads*sizeof(float)));
  measure<64,3>(output,prelude,sums);measure<64,4>(output,prelude,sums);measure<96,3>(output,prelude,sums);measure<64,5>(output,prelude,sums);
  CHECK(cudaFree(output));CHECK(cudaFree(prelude));CHECK(cudaFree(sums));return 0;}
