// Independent one-actor fresh-barrier batch;release fences coalesce to one.
// This measures minimum initialization work,not all native constructor fences.
// No CTA sync,arguments,
// matrix,payload or loop controller inside the measured actor window.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
constexpr int Shared=128*1024;
struct Reading{unsigned long long c0,c1;unsigned complete,errors;};
template<int Arrivals>__device__ __forceinline__ void init(unsigned p){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(p),"n"(Arrivals):"memory");}
__device__ __forceinline__ void arrive(unsigned p){asm volatile("{.reg .b64 s;mbarrier.arrive.shared::cta.b64 s,[%0];}"::"r"(p):"memory");}
__device__ __forceinline__ unsigned complete(unsigned p){unsigned v;asm volatile("{.reg .pred q;mbarrier.test_wait.parity.shared::cta.b64 q,[%1],0;selp.b32 %0,1,0,q;}":"=r"(v):"r"(p):"memory");return v;}
template<int Threads,int Size,int Groups,int Arrivals>
__global__ void __launch_bounds__(Threads,1) actor_groups(Reading* output){
 extern __shared__ __align__(16) unsigned char shared[];
 constexpr bool Hetero=Size==-1;constexpr int Count=Hetero?22:Size*Groups;
 unsigned base=unsigned(__cvta_generic_to_shared(shared));
 unsigned long long c0=0,c1=0;__syncthreads();
 if(threadIdx.x==0){
  c0=clock64();
  #pragma unroll
  for(int g=0;g<Groups;++g){
   #pragma unroll
   for(int b=0;b<(Hetero?(g==0?2:4):Size);++b){
    unsigned offset=Hetero?(g==0?b:2+4*(g-1)+b):g*Size+b;
    if constexpr(Hetero){if(g==0)init<128>(base+offset*8);else init<1>(base+offset*8);}
    else init<Arrivals>(base+offset*8);
   }
   asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  c1=clock64();
 }
 __syncthreads();unsigned ready=0;
 for(int i=0;i<Count;++i){unsigned arrivals=Hetero?(i<2?128:1):Arrivals;unsigned p=base+unsigned(i*8);
  if(threadIdx.x<arrivals)arrive(p);__syncthreads();if(threadIdx.x==0)ready+=complete(p);
 }
 if(threadIdx.x==0)output[blockIdx.x]={c0,c1,ready,ready!=Count};
}
template<int T,int N,int G,int A>void capture(int grid,int trial){
 auto fn=actor_groups<T,N,G,A>;CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));int resident;
 CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,T,Shared));if(resident!=1)exit(3);
 Reading* device;CHECK(cudaMalloc(&device,grid*sizeof(Reading)));fn<<<grid,T,Shared>>>(device);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());std::vector<Reading> host(grid);CHECK(cudaMemcpy(host.data(),device,grid*sizeof(Reading),cudaMemcpyDeviceToHost));CHECK(cudaFree(device));
 for(int b=0;b<grid;++b){auto r=host[b];if(r.errors||r.c1<=r.c0){fprintf(stderr,"Numeric/timer failure\n");exit(4);}printf("%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%u\n",T,N,G,A,grid,trial,b,r.c0,r.c1,r.complete);}
}
template<int T,int N,int G,int A>void runs(){capture<T,N,G,A>(132,-1);for(int grid:{1,132})for(int trial=0;trial<3;++trial){capture<T,N,G,A>(grid,-2-trial);capture<T,N,G,A>(grid,trial);}}
template<int T,int N,int A>void groups(){runs<T,N,1,A>();runs<T,N,2,A>();runs<T,N,4,A>();runs<T,N,8,A>();}
template<int T,int A>void sizes(){runs<T,0,0,A>();groups<T,2,A>();groups<T,4,A>();groups<T,8,A>();}
template<int T>void arrivals(){sizes<T,1>();sizes<T,4>();sizes<T,128>();runs<T,-1,6,1>();}
int main(){cudaDeviceProp p;CHECK(cudaGetDeviceProperties(&p,0));if(p.major!=9||p.multiProcessorCount!=132)return 5;puts("threads,size,groups,arrivals,grid,trial,cta,c0,c1,complete");arrivals<128>();arrivals<256>();return 0;}
