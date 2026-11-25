// Independent shared mbarrier notification latency. No global operands,
// copy payload, matrix, reduction, or Attention program.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2); } } while(0)
struct Reading { unsigned long long begin,end,ns,send,ready,release; unsigned count; };
__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long value;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(value)::"memory");return value;
}
__device__ __forceinline__ void init_bar(unsigned a,unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(a),"r"(n):"memory");
}
__device__ __forceinline__ void arrive(unsigned a) {
  asm volatile("{.reg .b64 state;mbarrier.arrive.shared::cta.b64 state,[%0];}"::"r"(a):"memory");
}
__device__ __forceinline__ void wait(unsigned a,unsigned phase) {
  asm volatile("{.reg .pred p;WAIT_%=:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;@!p bra WAIT_%=;}"::"r"(a),"r"(phase):"memory");
}
__device__ __forceinline__ void hold(unsigned cycles) {
  auto start=clock64();while(clock64()-start<cycles) {}
}
// Signal=0: one elected full-barrier arrival; Signal=1:128 arrivals;
// Signal=2:128 cp.async.mbarrier arrivals with an empty async group.
// Empty notification always has one arrival per consumer thread.
template<int Consumers,int Signal>
__global__ void __launch_bounds__(128+Consumers,1)
notification(int rounds,int producer_hold,int consumer_hold,Reading* result) {
  extern __shared__ unsigned char residency[];
  __shared__ __align__(8) unsigned long long bars[2];
  unsigned full=static_cast<unsigned>(__cvta_generic_to_shared(&bars[0]));
  unsigned empty=static_cast<unsigned>(__cvta_generic_to_shared(&bars[1]));
  if(threadIdx.x==0) {
    residency[0]=0;init_bar(full,Signal==0?1:128);init_bar(empty,Consumers);
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncthreads();
  bool producer=threadIdx.x<128;
  unsigned long long send=0,ready=0,release=0;
  unsigned count=0;
  auto n0=global_ns();unsigned long long begin=clock64();
  bool actor=!(producer && Signal==0 && threadIdx.x!=0);
  for(int i=0;actor && i<rounds;++i) {
    if(producer) {
      hold(producer_hold);
      if constexpr(Signal==0) {
        if(threadIdx.x==0) { if(i==0) send=clock64();arrive(full); }
      } else {
        if(i==0) send=clock64();
        if constexpr(Signal==1) arrive(full);
        else asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(full):"memory");
      }
      wait(empty,i&1);
      if(i==0) ready=clock64();
    } else {
      wait(full,i&1);
      if(i==0) ready=clock64();
      hold(consumer_hold);
      if(i==0) release=clock64();
      arrive(empty);
    }
    ++count;
  }
  unsigned long long end=clock64();auto n1=global_ns();
  __syncthreads();
  if(threadIdx.x%32==0) {
    int warps=(128+Consumers)/32;
    result[blockIdx.x*warps+threadIdx.x/32]={begin,end,n1-n0,send,ready,release,count};
  }
}
template<int Consumers,int Signal>
void run(int grid,int rounds,int ph,int ch,Reading* device) {
  auto kernel=notification<Consumers,Signal>;
  CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024));
  int warps=(128+Consumers)/32;
  for(int trial=-2;trial<7;++trial) {
    kernel<<<grid,128+Consumers,128*1024>>>(rounds,ph,ch,device);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<Reading> host(grid*warps);
    CHECK(cudaMemcpy(host.data(),device,host.size()*sizeof(Reading),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) {
      unsigned long long begin=~0ull,end=0,ns=0,send=0,full_ready=0,release=0,empty_ready=0;
      for(int w=0;w<warps;++w) {
        auto r=host[c*warps+w];
        bool actor=!(Signal==0 && w>0 && w<4);
        if(r.end<=r.begin || !r.ns || r.count!=static_cast<unsigned>(actor?rounds:0)) {std::fprintf(stderr,"invalid window\n");std::exit(3);}
        if(actor) {begin=std::min(begin,r.begin);end=std::max(end,r.end);ns=std::max(ns,r.ns);}
        if(w<4) {send=std::max(send,r.send);empty_ready=std::max(empty_ready,r.ready);}
        else {full_ready=std::max(full_ready,r.ready);release=std::max(release,r.release);}
      }
      if(full_ready<=send || empty_ready<=release) {std::fprintf(stderr,"invalid paired notification\n");std::exit(4);}
      if(trial>=0) std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",Consumers,Signal,grid,rounds,ph,ch,trial,c,begin,end,ns,send,full_ready,release,empty_ready);
    }
  }
}
int main() {
  Reading* device;CHECK(cudaMalloc(&device,132*12*sizeof(Reading)));
  std::printf("consumers,signal,grid,rounds,producer_hold,consumer_hold,trial,cta,begin,end,ns,send,full_ready,release,empty_ready\n");
  for(int grid:{1,32,132}) for(int rounds:{128,256,512}) for(int ph:{0,512}) for(int ch:{0,512}) {
    run<128,0>(grid,rounds,ph,ch,device);run<256,0>(grid,rounds,ph,ch,device);
    run<128,1>(grid,rounds,ph,ch,device);run<256,1>(grid,rounds,ph,ch,device);
    run<128,2>(grid,rounds,ph,ch,device);run<256,2>(grid,rounds,ph,ch,device);
  }
  CHECK(cudaFree(device));
}
