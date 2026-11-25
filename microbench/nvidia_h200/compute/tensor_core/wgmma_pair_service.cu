// Independent SS/RS mixed asynchronous queue and enqueue timing.
// Distinct accumulator sets and four A-register words per RS K16 slice.
// No scalar math in the timed region, global operand flow or Attention.
#include "pair_wgmma_helper.cuh"
#include <cmath>
struct Reading {
  unsigned long long cycles,ns,begin,ss_enqueued,rs_enqueued,wait1,wait0;
  float checksum;
};
__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long x;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(x)::"memory");return x;
}
__device__ __forceinline__ void wait_one() {
  asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");
}
template<int Mode,int SN,int SK,int RN,int RK>
__device__ __forceinline__ Reading timed_pair(uint64_t a,uint64_t b,int repeats,int active) {
  float s[SN/2],o[RN/2];uint32_t fragments[RK][4];
  #pragma unroll
  for(int i=0;i<SN/2;++i) {s[i]=0;asm volatile("":"+f"(s[i]));}
  #pragma unroll
  for(int i=0;i<RN/2;++i) {o[i]=0;asm volatile("":"+f"(o[i]));}
  #pragma unroll
  for(int k=0;k<RK;++k) {
    #pragma unroll
    for(int i=0;i<4;++i) {
      fragments[k][i]=0x3e003e00u+((threadIdx.x*17+k*4+i)&127);
      asm volatile("":"+r"(fragments[k][i]));
    }
  }
  auto n0=global_ns();auto c0=clock64();
  #pragma unroll 1
  for(int r=0;r<repeats;++r) {
    wg_fence();
    if constexpr(Mode!=1) {
      #pragma unroll
      for(int k=0;k<SK;++k) {
        if constexpr(SN==176) wgmma_ss_n176_kmajor(s,a,b,1);
        if constexpr(SN==192) wgmma_ss_n192_kmajor(s,a,b,1);
        if constexpr(SN==128) wgmma_ss_n128_kmajor(s,a,b,1);
      }
      wg_commit();
    }
    if constexpr(Mode!=0) {
      wg_fence();
      #pragma unroll
      for(int k=0;k<RK;++k) {
        if constexpr(RN==128) wgmma_rs_n128_mnmajor(o,fragments[k],b,1);
        if constexpr(RN==64) wgmma_rs_n64_mnmajor(o,fragments[k],b,1);
      }
      wg_commit();
    }
    if constexpr(Mode==2) wait_one();
    wg_wait0();
  }
  auto c1=clock64();auto n1=global_ns();
  // A peer cannot start its instrumented burst while another WG is still
  // in the slope region. Inactive WGs are outside this named barrier.
  asm volatile("bar.sync 1,%0;"::"r"(active*128):"memory");
  Reading result{};result.cycles=c1-c0;result.ns=n1-n0;
  wg_fence();result.begin=clock64();
  if constexpr(Mode!=1) {
    #pragma unroll
    for(int k=0;k<SK;++k) {
      if constexpr(SN==176) wgmma_ss_n176_kmajor(s,a,b,1);
      if constexpr(SN==192) wgmma_ss_n192_kmajor(s,a,b,1);
      if constexpr(SN==128) wgmma_ss_n128_kmajor(s,a,b,1);
    }
    wg_commit();
  }
  result.ss_enqueued=clock64();
  if constexpr(Mode!=0) {
    wg_fence();
    #pragma unroll
    for(int k=0;k<RK;++k) {
      if constexpr(RN==128) wgmma_rs_n128_mnmajor(o,fragments[k],b,1);
      if constexpr(RN==64) wgmma_rs_n64_mnmajor(o,fragments[k],b,1);
    }
    wg_commit();
  }
  result.rs_enqueued=clock64();
  if constexpr(Mode==2) wait_one();
  result.wait1=clock64();wg_wait0();result.wait0=clock64();
  asm volatile("bar.sync 1,%0;"::"r"(active*128):"memory");
  float sum=0;
  #pragma unroll
  for(int i=0;i<SN/2;++i) sum+=s[i];
  #pragma unroll
  for(int i=0;i<RN/2;++i) sum+=o[i];
  #pragma unroll
  for(int k=0;k<RK;++k) {
    #pragma unroll
    for(int i=0;i<4;++i) sum+=__uint_as_float(fragments[k][i]);
  }
  result.checksum=sum;return result;
}
template<int SN,int SK,int RN,int RK>
__global__ void __launch_bounds__(256,1)
pair_service(int mode,int active,int repeats,Reading* output) {
  extern __shared__ __align__(1024) unsigned char shared[];
  for(int i=threadIdx.x;i<kSmemBytes/2;i+=blockDim.x)
    reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u+((i*17)&127);
  unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  uint64_t a=make_desc(base,16,1024),b=make_desc(base+65536,16,1024);
  __syncthreads();Reading result{};int wg=threadIdx.x/128;
  if(wg<active) {
    if(mode==0) result=timed_pair<0,SN,SK,RN,RK>(a,b,repeats,active);
    else if(mode==1) result=timed_pair<1,SN,SK,RN,RK>(a,b,repeats,active);
    else result=timed_pair<2,SN,SK,RN,RK>(a,b,repeats,active);
  }
  __syncthreads();
  if(wg<active && threadIdx.x%128==0) output[blockIdx.x*2+wg]=result;
}
template<int SN,int SK,int RN,int RK>
void run(int grid,int active,int repeats,Reading* output) {
  auto kernel=pair_service<SN,SK,RN,RK>;
  CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmemBytes));
  for(int mode=0;mode<3;++mode) for(int trial=-1;trial<5;++trial) {
    kernel<<<grid,256,kSmemBytes>>>(mode,active,repeats,output);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<Reading> host(grid*2);CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Reading),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) for(int wg=0;wg<active;++wg) {
      auto x=host[c*2+wg];unsigned instructions=(mode==0?SK:(mode==1?RK:SK+RK));
      if(!x.cycles || !x.ns || !std::isfinite(x.checksum) || x.cycles<repeats*instructions*16ull || !(x.begin<=x.ss_enqueued && x.ss_enqueued<=x.rs_enqueued && x.rs_enqueued<=x.wait1 && x.wait1<=x.wait0)) {std::fprintf(stderr,"invalid primitive window\n");std::exit(3);}
      if(trial>=0) std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",SN,SK,RN,RK,grid,active,repeats,mode,trial,c,wg,x.cycles,x.ns,x.begin,x.ss_enqueued,x.rs_enqueued,x.wait1,x.wait0);
    }
  }
}
int main() {
  Reading* output;CHECK(cudaMalloc(&output,132*2*sizeof(Reading)));
  std::printf("ss_n,ss_k16,rs_n,rs_k16,grid,active_wgs,repeats,mode,trial,cta,wg,cycles,ns,begin,ss_enqueued,rs_enqueued,wait1,wait0\n");
  for(int grid:{1,32,132}) for(int active:{1,2}) for(int repeats:{128,256,512}) {
    run<176,8,128,11>(grid,active,repeats,output);
    run<192,4,64,12>(grid,active,repeats,output);
    run<128,8,128,8>(grid,active,repeats,output);
  }
  CHECK(cudaFree(output));
}
