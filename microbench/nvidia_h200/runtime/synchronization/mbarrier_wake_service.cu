// Independent ready-query and producer-notify/consumer-resume latencies.
// No TMA, arithmetic workload or matrix output.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int kShared=196608;
[[noreturn]] static void fail_reading(){fprintf(stderr,"invalid notification/reading\n");exit(3);}
struct Reading {uint64_t before,after,c0,c1,notified;};
__device__ uint32_t addr(const void* p){return unsigned(__cvta_generic_to_shared(p));}
__device__ uint64_t timer(){uint64_t v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}
template<int Mode> __device__ __forceinline__ void wait_ready(uint64_t* barrier) {
  if constexpr(Mode==0)
    asm volatile("{.reg .pred p; L_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0; @!p bra L_%=;}"::"r"(addr(barrier)):"memory");
  if constexpr(Mode==1)
    asm volatile("{.reg .pred p; L_%=: mbarrier.test_wait.parity.shared::cta.b64 p,[%0],0; @!p bra L_%=;}"::"r"(addr(barrier)):"memory");
  if constexpr(Mode==3)
    // Approximate the retained 0xc350 timeout hint. Record actual lowering:
    // local ptxas rounds this to0xc351, so do not claim bit-identical code.
    asm volatile("{.reg .pred p; L_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0,50000; @!p bra L_%=;}"::"r"(addr(barrier)):"memory");
  if constexpr(Mode==2)asm volatile("":::"memory");
}
template<int Mode> __global__ __launch_bounds__(384,1) void notify_resume(
    Reading* readings,int warpgroups,int delay_ns,int rounds) {
  __shared__ __align__(8) uint64_t barrier;
  __shared__ uint64_t notified;
  for(int round=0;round<rounds;++round) {
    if(threadIdx.x==256) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(addr(&barrier)):"memory");
      asm volatile("fence.mbarrier_init.release.cluster;":::"memory");notified=0;
      if(!delay_ns) {
        notified=timer();
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"::"r"(addr(&barrier)):"memory");
      }
    }
    __syncthreads();
    uint64_t before=0,after=0,c0=0,c1=0;
    if(threadIdx.x==256 && delay_ns) {
      uint64_t begin=timer();while(timer()-begin<uint64_t(delay_ns)) {}
      notified=timer();
      asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"::"r"(addr(&barrier)):"memory");
    } else if(threadIdx.x<warpgroups*128) {
      before=timer();c0=clock64();wait_ready<Mode>(&barrier);c1=clock64();after=timer();
    }
    __syncthreads();
    if(threadIdx.x<warpgroups*128 && threadIdx.x%32==0) {
      int index=(blockIdx.x*rounds+round)*8+threadIdx.x/32;
      readings[index]={before,after,c0,c1,notified};
    }
    __syncthreads();
  }
}
template<int Mode> void measure(Reading* output) {
  auto kernel=notify_resume<Mode>;CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  for(int grid:{32,132})for(int wgs:{1,2})for(int delay:{0,256,1024})for(int rounds:{128,512}) {
    if constexpr(Mode==2) {if(delay)continue;}
    std::vector<Reading> host(size_t(grid)*rounds*8);
    std::vector<double> query_cycles,wake_ns,blocked_fraction,clock_mhz;
    for(int trial=0;trial<9;++trial) {
      kernel<<<grid,384,kShared>>>(output,wgs,delay,rounds);CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Reading),cudaMemcpyDeviceToHost));
      int waiting=0,total=0;
      for(int block=0;block<grid;++block)for(int round=rounds-64;round<rounds;++round) {
        for(int group=0;group<wgs;++group) {
          uint64_t last=0,first=UINT64_MAX,first_cycle=UINT64_MAX,last_cycle=0,notification=0;
          for(int warp=0;warp<4;++warp) {
            auto r=host[(block*rounds+round)*8+group*4+warp];
            if(!r.before || r.after<r.before || r.c1<r.c0 || !r.notified)fail_reading();
            if constexpr(Mode!=2)if(r.after<r.notified)fail_reading();
            first=std::min(first,r.before);last=std::max(last,r.after);
            first_cycle=std::min(first_cycle,r.c0);last_cycle=std::max(last_cycle,r.c1);notification=r.notified;
          }
          bool started_waiting=first<notification;waiting+=started_waiting;total+=1;
          if(trial>=2) {
            query_cycles.push_back(double(last_cycle-first_cycle));
            if(delay && started_waiting)wake_ns.push_back(double(last-notification));
            if(delay && last>first)clock_mhz.push_back(1000.0*(last_cycle-first_cycle)/(last-first));
          }
        }
      }
      if(trial>=2)blocked_fraction.push_back(double(waiting)/total);
    }
    auto median=[](std::vector<double>& v){std::sort(v.begin(),v.end());return v[v.size()/2];};
    printf("{\"mode\":%d,\"grid\":%d,\"math_warpgroups\":%d,\"delay_ns\":%d,\"rounds\":%d,"
      "\"query_cycles_p50\":%.3f,\"started_before_notify_fraction\":%.6f,",
      Mode,grid,wgs,delay,rounds,median(query_cycles),median(blocked_fraction));
    if(delay) {
      if(wake_ns.empty() || clock_mhz.empty())fail_reading();
      printf("\"notify_to_group_resume_ns_p50\":%.3f,\"clock_mhz_p50\":%.3f,",median(wake_ns),median(clock_mhz));
    } else printf("\"notify_to_group_resume_ns_p50\":null,\"clock_mhz_p50\":null,");
    printf("\"checked_rounds\":64,\"checks_passed\":true}\n");fflush(stdout);
  }
}
int main() {
  Reading* output=nullptr;CHECK(cudaMalloc(&output,size_t(132)*512*8*sizeof(Reading)));
  measure<0>(output);measure<1>(output);measure<2>(output);measure<3>(output);CHECK(cudaFree(output));
}
