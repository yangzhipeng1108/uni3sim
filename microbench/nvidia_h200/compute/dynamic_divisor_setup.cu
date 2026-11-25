// Independent register-only runtime reciprocal-integer setup, not Attention.
// The quotient expression is the generic unsigned64 / unsigned32 reciprocal
// multiplier construction. No tensors, reductions, normalization or KV.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2); } } while(0)

struct Stamp { unsigned long long c0,c1,n0,n1; };
__device__ __forceinline__ unsigned long long ns() {
  unsigned long long v;
  asm volatile("mov.u64 %0,%%globaltimer;":"=l"(v));return v;
}

template<int Mode,int Warps>
__global__ void __launch_bounds__(256,1) setup(
    unsigned base,unsigned opaque_zero,int repeats,Stamp* times,unsigned* checksums) {
  if (threadIdx.x >= Warps*32) return;
  unsigned sum=threadIdx.x+3;
  __syncwarp();
  unsigned long long n0=ns(),c0=clock64();
  #pragma unroll 1
  for(int i=0;i<repeats;++i) {
    // opaque_zero is a runtime kernel argument, always0 in the experiment.
    // This actual data dependency prevents reciprocal construction hoisting
    // and inter-iteration overlap, without changing the input divisor.
    unsigned d=base+(sum & opaque_zero);
    unsigned multiplier=0,shift=0;
    if(d!=1) {
      unsigned ceil_log2=32-__clz(d-1);
      unsigned p=31+ceil_log2;
      unsigned long long numerator=(1ull<<p)+d-1;
      if constexpr(Mode==1) multiplier=unsigned(numerator/d);
      else multiplier=unsigned(numerator)^d;
      shift=p-32;
    }
    sum=(sum ^ multiplier)+shift+1;
  }
  unsigned long long c1=clock64(),n1=ns();
  checksums[blockIdx.x*256+threadIdx.x]=sum;
  if(threadIdx.x%32==0) times[blockIdx.x*8+threadIdx.x/32]={c0,c1,n0,n1};
}

static unsigned reference(int mode,unsigned d,int repeats,unsigned seed) {
  unsigned m=0,shift=0;
  if(d!=1) {
    unsigned log=0;for(unsigned v=d-1;v;v>>=1) ++log;
    unsigned p=31+log;
    unsigned long long n=(1ull<<p)+d-1;
    m=mode ? unsigned(n/d) : unsigned(n)^d;shift=p-32;
  }
  for(int i=0;i<repeats;++i) seed=(seed ^ m)+shift+1;
  return seed;
}

template<int Mode,int Warps>
static void run(unsigned divisor,int grid,int shared,int repeats,int trial) {
  Stamp* times;unsigned* checksums;
  CHECK(cudaMalloc(&times,grid*8*sizeof(Stamp)));
  CHECK(cudaMalloc(&checksums,grid*256*sizeof(unsigned)));
  CHECK(cudaFuncSetAttribute(setup<Mode,Warps>,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
  for(int warm=0;warm<2;++warm) setup<Mode,Warps><<<grid,256,shared>>>(divisor,0,repeats,times,checksums);
  CHECK(cudaDeviceSynchronize());
  setup<Mode,Warps><<<grid,256,shared>>>(divisor,0,repeats,times,checksums);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  std::vector<Stamp> stamps(grid*8);std::vector<unsigned> sums(grid*256);
  CHECK(cudaMemcpy(stamps.data(),times,stamps.size()*sizeof(Stamp),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(sums.data(),checksums,sums.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
  std::vector<unsigned> expected(Warps*32);
  for(int t=0;t<Warps*32;++t) expected[t]=reference(Mode,divisor,repeats,t+3);
  for(int cta=0;cta<grid;++cta) {
    for(int t=0;t<Warps*32;++t) if(sums[cta*256+t]!=expected[t]) {
      std::fprintf(stderr,"checksum mismatch %d %d\n",cta,t);std::exit(3);
    }
    for(int warp=0;warp<Warps;++warp) {
      auto s=stamps[cta*8+warp];
      std::printf("%d,%d,%u,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu\n",
        Mode,Warps,divisor,grid,shared,repeats,trial,cta,warp,s.c0,s.c1,s.n0,s.n1);
    }
  }
  CHECK(cudaFree(times));CHECK(cudaFree(checksums));
}

int main() {
  std::puts("mode,warps,divisor,grid,shared,repeats,trial,cta,warp,c0,c1,n0,n1");
  for(unsigned d:{1u,3u,7u,17u,257u,4093u,65521u})
  for(int grid:{1,132}) for(int r:{64,128,256}) for(int trial=0;trial<3;++trial) {
    run<0,1>(d,grid,196608,r,trial);run<1,1>(d,grid,196608,r,trial);
    run<0,8>(d,grid,196608,r,trial);run<1,8>(d,grid,196608,r,trial);
  }
}
