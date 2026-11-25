// Pure-byte all-return experiment: one cold segment must join hot segments.
// No matrix, routing, operator output or production K recurrence.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace cg=cooperative_groups;
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(10);}}while(0)
constexpr int SMs=132,Threads=256,Shared=98304,Grid=264,Cold=1024,Span=17408;
struct Record{unsigned sm,entry,checksum;unsigned long long total,minimum,maximum,begin,end,cycles,first_begin,first_end,last_begin,last_end;};
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
__host__ __device__ unsigned word(unsigned i){i^=i>>16;i*=0x7feb352du;i^=i>>15;i*=0x846ca68bu;return i^(i>>16);}
__global__ void initialize(unsigned* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=word(unsigned(i));}
__global__ void warm(const unsigned* p,unsigned* sink){unsigned sum=0;for(int i=threadIdx.x;i<Span/4;i+=Threads){unsigned v;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(v):"l"(p+size_t(blockIdx.x)*(Span/4)+i):"memory");sum^=v;}if(threadIdx.x==0)sink[blockIdx.x]=sum;}
__device__ void issue(unsigned char* dest,const unsigned char* from,unsigned bytes){
 for(unsigned v=threadIdx.x;v<bytes/16;v+=Threads){unsigned address=unsigned(__cvta_generic_to_shared(dest+v*16));asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(address),"l"(from+v*16):"memory");}
 asm volatile("cp.async.commit_group;":::"memory");
}
__global__ __launch_bounds__(Threads,2) void burst_join(const unsigned char* hot,const unsigned char* cold,unsigned* entries,unsigned long long* deadline,Record* output,int mode,int hot_bytes,int rounds){
 extern __shared__ __align__(128) unsigned char sm[];__shared__ unsigned smid,entry;
 auto grid=cg::this_grid();if(threadIdx.x==0){asm volatile("mov.u32 %0, %%smid;":"=r"(smid));entry=atomicAdd(entries+smid,1u);}__syncthreads();
 unsigned long long total=0,minimum=~0ull,maximum=0,first_begin=0,first_end=0,last_begin=0,last_end=0;
 if(blockIdx.x==0&&threadIdx.x==0)*deadline=timer()+4096;
 grid.sync();auto target=*reinterpret_cast<volatile unsigned long long*>(deadline);while(timer()<target){}
 auto begin=timer();auto cycles=clock64();
 for(int r=0;r<rounds;++r){
  const auto* h=hot+size_t(blockIdx.x)*Span;const auto* c=cold+(size_t(r)*Grid+blockIdx.x)*Span;
  auto a=timer();
  if(mode==4){while(timer()<a+512){}}else{
   if(mode!=3)issue(sm,mode==2?c:h,unsigned(hot_bytes));
   issue(sm+hot_bytes,(mode==0?h:c)+hot_bytes,Cold);
  }
  asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();auto b=timer();
  if(threadIdx.x==0){total+=b-a;minimum=minimum<b-a?minimum:b-a;maximum=maximum>b-a?maximum:b-a;if(r==0){first_begin=a;first_end=b;}last_begin=a;last_end=b;}
 }
 auto c1=clock64();auto end=timer();if(threadIdx.x==0){unsigned checksum=mode==4?0:*reinterpret_cast<volatile unsigned*>(sm+hot_bytes+Cold-4);output[blockIdx.x]={smid,entry,checksum,total,minimum,maximum,begin,end,(unsigned long long)(c1-cycles),first_begin,first_end,last_begin,last_end};}
}
int main(int argc,char** argv){
 if(argc!=6)return 2;int mode=atoi(argv[1]),hot_bytes=atoi(argv[2]),rounds=atoi(argv[3]),trials=atoi(argv[4]),process=atoi(argv[5]);
 if(mode<0||mode>4||(hot_bytes!=4096&&hot_bytes!=16384)||rounds!=256||trials<1||trials>63||process<0||process>1)return 2;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));if(!strstr(prop.name,"H200")||prop.multiProcessorCount!=SMs||!prop.cooperativeLaunch||prop.l2CacheSize<=0)return 3;
 CHECK(cudaFuncSetAttribute(burst_join,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));int occupancy;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,burst_join,Threads,Shared));if(occupancy!=2)return 3;
 cudaFuncAttributes attr{};CHECK(cudaFuncGetAttributes(&attr,burst_join));if(attr.localSizeBytes)return 3;
 unsigned char *hot,*cold;unsigned *entries,*eviction,*sink;unsigned long long* deadline;Record* output;
 const size_t cold_bytes=size_t(Grid)*rounds*Span,hot_size=size_t(Grid)*Span,evict_size=2*size_t(prop.l2CacheSize);
 CHECK(cudaMalloc(&hot,hot_size));CHECK(cudaMalloc(&cold,cold_bytes));CHECK(cudaMalloc(&entries,SMs*4));CHECK(cudaMalloc(&deadline,8));CHECK(cudaMalloc(&output,Grid*sizeof(Record)));CHECK(cudaMalloc(&eviction,evict_size));CHECK(cudaMalloc(&sink,Grid*4));
 initialize<<<132,256>>>(reinterpret_cast<unsigned*>(hot),hot_size/4);initialize<<<132,256>>>(reinterpret_cast<unsigned*>(cold),cold_bytes/4);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 void* args[]={&hot,&cold,&entries,&deadline,&output,&mode,&hot_bytes,&rounds};std::vector<Record> host(Grid);
 for(int trial=-2;trial<trials;++trial){
  initialize<<<132,256>>>(eviction,evict_size/4);warm<<<Grid,Threads>>>(reinterpret_cast<unsigned*>(hot),sink);CHECK(cudaGetLastError());CHECK(cudaMemset(entries,0,SMs*4));CHECK(cudaLaunchCooperativeKernel((void*)burst_join,Grid,Threads,args,Shared));CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(host.data(),output,Grid*sizeof(Record),cudaMemcpyDeviceToHost));
  int bysm[SMs][2]={};for(int block=0;block<Grid;++block){auto& v=host[block];if(v.sm>=SMs||v.entry>1||v.total==0||v.end<=v.begin)return 4;++bysm[v.sm][v.entry];unsigned expected=mode==4?0:word(unsigned(((mode==0?size_t(block):size_t((rounds-1)*Grid+block))*Span+hot_bytes+Cold-4)/4));if(v.checksum!=expected)return 4;}
  for(int sm=0;sm<SMs;++sm)if(bysm[sm][0]!=1||bysm[sm][1]!=1)return 5;
  if(trial<0)continue;printf("{\"mode\":%d,\"hot_bytes\":%d,\"rounds\":%d,\"trial\":%d,\"process\":%d,\"occupancy\":%d,\"registers_per_thread\":%d,\"l2_bytes\":%d,\"ctas\":[",mode,hot_bytes,rounds,trial,process,occupancy,attr.numRegs,prop.l2CacheSize);
  for(int block=0;block<Grid;++block){auto& v=host[block];printf("%s[%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu]",block?",":"",v.sm,v.entry,v.total,v.minimum,v.maximum,v.begin,v.end,v.cycles,v.first_begin,v.first_end,v.last_begin,v.last_end);}puts("]}");
 }
 CHECK(cudaFree(sink));CHECK(cudaFree(eviction));CHECK(cudaFree(output));CHECK(cudaFree(deadline));CHECK(cudaFree(entries));CHECK(cudaFree(cold));CHECK(cudaFree(hot));return 0;
}
