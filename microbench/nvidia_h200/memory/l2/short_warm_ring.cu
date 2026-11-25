// Independent byte ring. No matrix, routing, or operator output.
// Inline prewarm is outside the timed window and has a matched zero-round
// counter control; no duration is subtracted from the measured body.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace cg = cooperative_groups;
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(10);} } while(0)
constexpr int Grid=132,Threads=256,Panel=49152,Slots=4,Shared=Panel*Slots;
struct Record { unsigned sm,checksum; unsigned long long begin,end,cycles; };
__host__ __device__ unsigned pattern(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
__global__ void initialize(unsigned* x,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)x[i]=pattern(unsigned(i));}
__device__ void issue(unsigned char* dst,const unsigned char* src){
 #pragma unroll
 for(int v=threadIdx.x;v<Panel/16;v+=Threads){unsigned a=unsigned(__cvta_generic_to_shared(dst+v*16));asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(a),"l"(src+v*16):"memory");}
 asm volatile("cp.async.commit_group;":::"memory");
}
__global__ __launch_bounds__(Threads,1) void short_warm_ring(const unsigned char* source,Record* out,int rounds){
 extern __shared__ __align__(128) unsigned char shared[];
 const unsigned char* base=source+size_t(blockIdx.x)*Shared;
 for(int slot=0;slot<Slots;++slot)issue(shared+slot*Panel,base+slot*Panel);
 asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 cg::this_grid().sync();
 unsigned sum=0;unsigned long long begin=timer(),start=clock64();
 for(int slot=0;slot<Slots;++slot)if(slot<rounds)issue(shared+slot*Panel,base+slot*Panel);
 #pragma unroll 1
 for(int round=0;round<rounds;++round){
  asm volatile("cp.async.wait_group 3;":::"memory");__syncthreads();
  volatile unsigned* data=reinterpret_cast<volatile unsigned*>(shared+(round%Slots)*Panel);
  #pragma unroll
  for(int v=threadIdx.x;v<Panel/16;v+=Threads)sum+=data[v*4];
  __syncthreads();
  // Empty commits during drain preserve the four-group queue semantics.
  if(round+Slots<rounds)issue(shared+(round%Slots)*Panel,base+(round%Slots)*Panel);
  else asm volatile("cp.async.commit_group;":::"memory");
 }
 asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 unsigned long long cycles=clock64()-start,end=timer();
 if(threadIdx.x==0){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));out[blockIdx.x]={sm,sum,begin,end,cycles};}
}
int main(int argc,char**argv){
 if(argc!=4)return 1;int rounds=atoi(argv[1]),samples=atoi(argv[2]),process=atoi(argv[3]);
 if((rounds!=0&&rounds!=8&&rounds!=12&&rounds!=16&&rounds!=32&&rounds!=64)||samples<1||samples>63||process<0||process>1)return 2;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=Grid||!strstr(prop.name,"H200")||!prop.cooperativeLaunch)return 3;
 CHECK(cudaFuncSetAttribute(short_warm_ring,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,short_warm_ring,Threads,Shared));if(resident!=1)return 4;
 cudaFuncAttributes attr{};CHECK(cudaFuncGetAttributes(&attr,short_warm_ring));
 unsigned char* source;unsigned* eviction;Record* output;
 CHECK(cudaMalloc(&source,Grid*Shared));CHECK(cudaMalloc(&eviction,2*size_t(prop.l2CacheSize)));CHECK(cudaMalloc(&output,Grid*sizeof(Record)));
 initialize<<<Grid,Threads>>>(reinterpret_cast<unsigned*>(source),Grid*Shared/4);CHECK(cudaDeviceSynchronize());
 std::vector<Record> host(Grid);void* args[]={&source,&output,&rounds};
 for(int trial=-3;trial<samples;++trial){
  initialize<<<Grid,Threads>>>(eviction,2*size_t(prop.l2CacheSize)/4);CHECK(cudaDeviceSynchronize());
  CHECK(cudaLaunchCooperativeKernel((void*)short_warm_ring,Grid,Threads,args,Shared));CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(host.data(),output,Grid*sizeof(Record),cudaMemcpyDeviceToHost));
  std::vector<int> sms(Grid,0);
  for(int b=0;b<Grid;++b){unsigned expected=0;
   for(int r=0;r<rounds;++r)for(int v=0;v<Panel/16;v+=Threads)expected+=pattern(unsigned((size_t(b)*Shared+(r%Slots)*Panel+v*16)/4));
   const auto& x=host[b];if(x.sm>=Grid||x.checksum!=expected||x.end<=x.begin||x.cycles==0)return 5;++sms[x.sm];
  }
  if(std::any_of(sms.begin(),sms.end(),[](int n){return n!=1;}))return 6;
  if(trial<0)continue;
  printf("{\"rounds\":%d,\"trial\":%d,\"process\":%d,\"resident\":%d,\"registers\":%d,\"l2_bytes\":%d,\"ctas\":[",rounds,trial,process,resident,attr.numRegs,prop.l2CacheSize);
  for(int b=0;b<Grid;++b){const auto& x=host[b];printf("%s[%u,%llu,%llu,%llu]",b?",":"",x.sm,x.begin,x.end,x.cycles);}puts("]}");
 }
 CHECK(cudaFree(source));CHECK(cudaFree(eviction));CHECK(cudaFree(output));
}
