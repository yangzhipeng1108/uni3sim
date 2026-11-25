// Independent empty timestamp window: clock64 outside/inside globaltimer.
// No Tensor, Attention or numerical operator in the measured window.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}while(0)
struct Trace {unsigned long long start,end,bg_start,bg_end,cycle_sum,ns_sum;unsigned wrong,windows;};
__device__ __forceinline__ unsigned long long cycles(){unsigned long long t;asm volatile("mov.u64 %0,%%clock64;":"=l"(t)::"memory");return t;}
__device__ __forceinline__ unsigned long long nanos(){unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t)::"memory");return t;}
template<bool Inner,int Kind>
__global__ void __launch_bounds__(256,1) marker_order(int background,int repeats,Trace* output){
 __shared__ volatile unsigned ready;
 __shared__ unsigned long long shared[6];
 __shared__ unsigned errors[256],counts[128];
 float values[128];
 #pragma unroll
 for(int i=0;i<128;++i){values[i]=1.f+(i+threadIdx.x)*(1.f/1024);asm volatile("":"+f"(values[i]));}
 if(threadIdx.x==0)ready=0;
 __syncthreads();unsigned wrong=0,count=0;
 if(threadIdx.x<128){
  while(!ready){}
  auto start=cycles();unsigned long long cycle_sum=0,ns_sum=0;
  #pragma unroll 1
  for(int i=0;i<repeats;++i){
   unsigned long long c0,c1,n0,n1;
   if constexpr(Inner){n0=nanos();c0=cycles();c1=cycles();n1=nanos();}
   else{c0=cycles();n0=nanos();n1=nanos();c1=cycles();}
   wrong+=c0>=c1||n0>n1;cycle_sum+=c1-c0;ns_sum+=n1-n0;++count;
  }
  auto end=cycles();
  if(threadIdx.x==0){shared[0]=start;shared[1]=end;shared[4]=cycle_sum;shared[5]=ns_sum;}
  counts[threadIdx.x]=count;
 }else{
  auto start=cycles();
  if(background){
   #pragma unroll 1
   for(int rep=0;rep<4104;++rep){
    #pragma unroll
    for(int i=0;i<128;++i){
     if constexpr(Kind==0)asm volatile("fma.rn.f32 %0,%0,%1,%2;":"+f"(values[i]):"f"(.5f),"f"(.25f));
     if constexpr(Kind==1)asm volatile("{.reg .f32 x;neg.f32 x,%0;ex2.approx.ftz.f32 %0,x;}":"+f"(values[i]));
     if constexpr(Kind==2){float x;asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;":"=f"(x):"f"(values[i]),"f"(.5f),"f"(-.75f));asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(values[i]):"f"(x));}
    }
    if(rep==7&&threadIdx.x==128)ready=1;
   }
  }else if(threadIdx.x==128)ready=1;
  auto end=cycles();if(threadIdx.x==128){shared[2]=start;shared[3]=end;}
  if(background){
   constexpr float expected=Kind==0?.5f:Kind==1?.641185744504986f:.7788556131530093f;
   #pragma unroll
   for(int i=0;i<128;++i)wrong+=!std::isfinite(values[i])||fabsf(values[i]-expected)>2e-6f;
  }
 }
 errors[threadIdx.x]=wrong;__syncthreads();
 if(threadIdx.x==0){unsigned error=0,windows=0;for(int i=0;i<256;++i)error+=errors[i];for(int i=0;i<128;++i)windows+=counts[i];output[blockIdx.x]={shared[0],shared[1],shared[2],shared[3],shared[4],shared[5],error,windows};}
}
template<bool Inner,int Kind>void run(int grid,int n,int bg,int trial,Trace* output){
 auto kernel=marker_order<Inner,Kind>;CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,204800));
 kernel<<<grid,256,204800>>>(bg,n,output);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 std::vector<Trace> trace(grid);CHECK(cudaMemcpy(trace.data(),output,grid*sizeof(Trace),cudaMemcpyDeviceToHost));
 for(int c=0;c<grid;++c){auto t=trace[c];if(t.wrong||t.windows!=128u*n||t.start>=t.end||(bg&&!(t.bg_start<=t.start&&t.end<=t.bg_end))){std::fprintf(stderr,"marker-order numeric/coverage gate failed:%u\n",t.wrong);std::exit(3);}
  std::printf("%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%u,%u\n",int(Inner),Kind,grid,n,bg,trial,c,t.start,t.end,t.bg_start,t.bg_end,t.cycle_sum,t.ns_sum,t.wrong,t.windows);
 }
}
template<bool Inner>void kinds(int grid,int n,int bg,int trial,Trace* output){run<Inner,0>(grid,n,bg,trial,output);run<Inner,1>(grid,n,bg,trial,output);run<Inner,2>(grid,n,bg,trial,output);}
int main(){int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;Trace* out;CHECK(cudaMalloc(&out,132*sizeof(Trace)));
 std::puts("inner,kind,grid,repeats,background,trial,cta,c0,c1,bg0,bg1,cycle_sum,ns_sum,wrong,validated_windows");
 for(int grid:{1,132})for(int n:{64,128,256})for(int bg:{0,1})for(int trial=0;trial<3;++trial){kinds<false>(grid,n,bg,trial,out);kinds<true>(grid,n,bg,trial,out);}CHECK(cudaFree(out));}
