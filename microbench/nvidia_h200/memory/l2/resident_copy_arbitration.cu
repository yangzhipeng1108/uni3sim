// Independent synchronized byte bursts; no production operator or matrix work.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace cg=cooperative_groups;
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(10);}}while(0)
constexpr int SMs=132,Threads=256,Shared=98304,Grid=264,Group=16384;
struct Stamp{unsigned long long begin,end;};
struct Record{unsigned sm,role,bytes,checksum;unsigned long long latency_sum,latency_min,latency_max,start_skew_sum,finish_skew_sum,begin,end,cycles;Stamp first,last;};
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
__host__ __device__ unsigned word(unsigned i){i^=i>>16;i*=0x7feb352du;i^=i>>15;i*=0x846ca68bu;return i^(i>>16);}
__global__ void initialize(unsigned* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=word(unsigned(i));}
__global__ __launch_bounds__(Threads,2) void copy_arbitration(const unsigned char* input,unsigned* entries,unsigned* ids,Stamp* stamps,unsigned long long* deadline,Record* output,int mode,int rounds,int rotation){
 extern __shared__ __align__(128) unsigned char sm[];
 __shared__ unsigned smid,role,bytes;
 auto grid=cg::this_grid();
 if(threadIdx.x==0){asm volatile("mov.u32 %0, %%smid;":"=r"(smid));role=atomicAdd(entries+smid,1u);if(role<2)ids[smid*2+role]=blockIdx.x;
  unsigned active=(role+rotation)%2;bytes=mode==0?(active==0?32768:98304):mode==1?(active==0?32768:0):mode==2?(active==1?98304:0):mode==3?65536:mode==4?(active==0?65536:0):0;}
 __syncthreads();grid.sync();
 unsigned long long total=0,minimum=~0ull,maximum=0,starts=0,finishes=0;Stamp first{},last{};
 auto outer=timer();auto cycles=clock64();
 for(int r=0;r<rounds;++r){
  if(blockIdx.x==0&&threadIdx.x==0)*deadline=timer()+4096;
  grid.sync();
  auto target=*reinterpret_cast<volatile unsigned long long*>(deadline);
  while(timer()<target){}
  auto begin=timer();
  #pragma unroll 1
  for(unsigned g=0;g<bytes/Group;++g){
   #pragma unroll
   for(unsigned v=threadIdx.x;v<Group/16;v+=Threads){
    unsigned address=unsigned(__cvta_generic_to_shared(sm+g*Group+v*16));
    const auto* from=input+size_t(blockIdx.x)*Shared+g*Group+v*16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(from):"memory");
   }
   asm volatile("cp.async.commit_group;":::"memory");
  }
  asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();auto end=timer();
  if(threadIdx.x==0)stamps[blockIdx.x]={begin,end};
  grid.sync();
  if(threadIdx.x==0){auto own=stamps[blockIdx.x];auto peer=stamps[ids[smid*2+(1-role)]];auto latency=own.end-own.begin;
   total+=latency;minimum=minimum<latency?minimum:latency;maximum=maximum>latency?maximum:latency;
   starts+=own.begin>peer.begin?own.begin-peer.begin:peer.begin-own.begin;
   finishes+=own.end>peer.end?own.end-peer.end:peer.end-own.end;
   if(r==0)first=own;last=own;
  }
  grid.sync();
 }
 auto c1=clock64();auto end=timer();
 if(threadIdx.x==0){unsigned checksum=bytes?*reinterpret_cast<volatile unsigned*>(sm+bytes-4):0;
  output[blockIdx.x]={smid,role,bytes,checksum,total,minimum,maximum,starts,finishes,outer,end,(unsigned long long)(c1-cycles),first,last};}
}
int main(int argc,char** argv){
 if(argc!=5)return 2;int mode=atoi(argv[1]),rounds=atoi(argv[2]),trials=atoi(argv[3]),rotation=atoi(argv[4]);
 if(mode<0||mode>5||(rounds!=128&&rounds!=256)||trials<1||trials>63||rotation<0||rotation>1)return 2;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));if(!strstr(prop.name,"H200")||prop.multiProcessorCount!=SMs||!prop.cooperativeLaunch)return 3;
 CHECK(cudaFuncSetAttribute(copy_arbitration,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int occupancy;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,copy_arbitration,Threads,Shared));if(occupancy!=2)return 3;
 cudaFuncAttributes attr{};CHECK(cudaFuncGetAttributes(&attr,copy_arbitration));if(attr.localSizeBytes)return 3;
 unsigned char* input;unsigned* entries;unsigned* ids;Stamp* stamps;unsigned long long* deadline;Record* output;
 CHECK(cudaMalloc(&input,size_t(Grid)*Shared));CHECK(cudaMalloc(&entries,SMs*4));CHECK(cudaMalloc(&ids,Grid*4));CHECK(cudaMalloc(&stamps,Grid*sizeof(Stamp)));CHECK(cudaMalloc(&deadline,8));CHECK(cudaMalloc(&output,Grid*sizeof(Record)));
 initialize<<<132,256>>>(reinterpret_cast<unsigned*>(input),size_t(Grid)*Shared/4);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 void* args[]={&input,&entries,&ids,&stamps,&deadline,&output,&mode,&rounds,&rotation};std::vector<Record> host(Grid);
 for(int trial=-2;trial<trials;++trial){
  CHECK(cudaMemset(entries,0,SMs*4));CHECK(cudaLaunchCooperativeKernel((void*)copy_arbitration,Grid,Threads,args,Shared));CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(host.data(),output,Grid*sizeof(Record),cudaMemcpyDeviceToHost));
  int bysm[SMs][2]={};
  for(int block=0;block<Grid;++block){auto& v=host[block];if(v.sm>=SMs||v.role>1||v.end<=v.begin||v.latency_min==0)return 4;
   ++bysm[v.sm][v.role];if(v.bytes&&v.checksum!=word(unsigned((size_t(block)*Shared+v.bytes-4)/4)))return 4;
  }
  for(int sm=0;sm<SMs;++sm)if(bysm[sm][0]!=1||bysm[sm][1]!=1)return 5;
  if(trial<0)continue;
  printf("{\"mode\":%d,\"rounds\":%d,\"trial\":%d,\"rotation\":%d,\"occupancy\":%d,\"registers_per_thread\":%d,\"ctas\":[",mode,rounds,trial,rotation,occupancy,attr.numRegs);
  for(int block=0;block<Grid;++block){auto& v=host[block];printf("%s[%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu]",block?",":"",v.sm,v.role,v.bytes,v.latency_sum,v.latency_min,v.latency_max,v.start_skew_sum,v.finish_skew_sum,v.begin,v.end,v.cycles,v.first.begin,v.first.end,v.last.begin,v.last.end);}
  puts("]}");
 }
 CHECK(cudaFree(output));CHECK(cudaFree(deadline));CHECK(cudaFree(stamps));CHECK(cudaFree(ids));CHECK(cudaFree(entries));CHECK(cudaFree(input));return 0;
}
