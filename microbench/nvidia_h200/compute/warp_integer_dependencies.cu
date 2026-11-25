// Independent warp integer dependencies. No scheduler, Attention or data loads.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
struct Stamp{unsigned long long c0,c1,n0,n1;};
__device__ __forceinline__ unsigned long long nanos(){unsigned long long x;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(x)::"memory");return x;}
__host__ __device__ unsigned feedback(unsigned x){return (x*1664525u+1013904223u)&0x3fffffffu;}
template<int Mode,int Warps>
__global__ void __launch_bounds__(128,1) dependency(int divisor,unsigned opaque_zero,int repeats,Stamp* output,unsigned* values){
 if(threadIdx.x>=Warps*32)return;
 unsigned lane=threadIdx.x%32,state=lane+3;
 __syncwarp();
 unsigned long long n0=nanos(),c0=clock64();
 #pragma unroll 1
 for(int r=0;r<repeats;++r){
  state=feedback(state);
  if constexpr(Mode==1){
   // Runtime dependency defeats loop-invariant reciprocal hoisting even
   // though opaque_zero=0 for reproducible positive integer division.
   int d=divisor+int(state&opaque_zero);
   state=unsigned(int(state)/d);
  }
  if constexpr(Mode==2){
   #pragma unroll
   for(int step=1;step<32;step*=2){unsigned peer=__shfl_up_sync(0xffffffff,state,step);if(lane>=step)state+=peer;}
  }
 }
 unsigned long long c1=clock64(),n1=nanos();
 values[blockIdx.x*128+threadIdx.x]=state;
 if(lane==0)output[blockIdx.x*4+threadIdx.x/32]={c0,c1,n0,n1};
}
static std::vector<unsigned> reference(int mode,int d,int repeats){
 std::vector<unsigned> values(32);for(int lane=0;lane<32;++lane)values[lane]=lane+3;
 for(int r=0;r<repeats;++r){
  for(auto& x:values){x=feedback(x);if(mode==1)x=unsigned(int(x)/d);}
  if(mode==2)for(int step=1;step<32;step*=2){auto previous=values;for(int lane=step;lane<32;++lane)values[lane]+=previous[lane-step];}
 }
 return values;
}
template<int Mode,int Warps>
void run(int grid,int divisor,int repeats,int trial,Stamp* output,unsigned* values){
 auto kernel=dependency<Mode,Warps>;
 CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
 for(int warm=0;warm<2;++warm)kernel<<<grid,128,196608>>>(divisor,0,repeats,output,values);
 CHECK(cudaDeviceSynchronize());
 kernel<<<grid,128,196608>>>(divisor,0,repeats,output,values);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 std::vector<Stamp> clocks(grid*4);std::vector<unsigned> sums(grid*128);
 CHECK(cudaMemcpy(clocks.data(),output,clocks.size()*sizeof(Stamp),cudaMemcpyDeviceToHost));
 CHECK(cudaMemcpy(sums.data(),values,sums.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
 auto expected=reference(Mode,divisor,repeats);
 for(int cta=0;cta<grid;++cta){
  for(int t=0;t<Warps*32;++t)if(sums[cta*128+t]!=expected[t%32]){std::fprintf(stderr,"integer mismatch mode%d warp%d divisor%d repeats%d cta%d thread%d\n",Mode,Warps,divisor,repeats,cta,t);std::exit(3);}
  for(int w=0;w<Warps;++w){auto s=clocks[cta*4+w];if(!(s.c0<s.c1&&s.n0<s.n1))std::exit(4);
   std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%u\n",Mode,Warps,divisor,grid,repeats,trial,cta,w,s.c0,s.c1,s.n0,s.n1,sums[cta*128+w*32]);}
 }
}
int main(){
 int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;
 Stamp* output;unsigned* values;CHECK(cudaMalloc(&output,132*4*sizeof(Stamp)));CHECK(cudaMalloc(&values,132*128*sizeof(unsigned)));
 std::puts("mode,warps,divisor,grid,repeats,trial,cta,warp,c0,c1,n0,n1,checksum");
 for(int grid:{1,132})for(int n:{64,128,256})for(int trial=0;trial<3;++trial){
  for(int d:{1,7,257}){run<0,1>(grid,d,n,trial,output,values);run<1,1>(grid,d,n,trial,output,values);run<0,4>(grid,d,n,trial,output,values);run<1,4>(grid,d,n,trial,output,values);}
  run<2,1>(grid,1,n,trial,output,values);run<2,4>(grid,1,n,trial,output,values);
 }
 CHECK(cudaFree(output));CHECK(cudaFree(values));
}
