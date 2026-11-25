// Independent row/column mbarrier token ring. No TMA, MMA, matrices or GEMM.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int kShared=212992,kThreads=384,kMaxDepth=16;
__device__ unsigned shared_address(const void* p){return unsigned(__cvta_generic_to_shared(p));}
__device__ void initialize_barrier(uint64_t* p,int count){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(shared_address(p)),"r"(count):"memory");}
__device__ void arrive_cluster(uint64_t* p){asm volatile("mbarrier.arrive.shared::cluster.b64 _,[%0];"::"r"(shared_address(p)):"memory");}
__device__ void wait_ready(uint64_t* p,int phase,int hint){
 if(hint)asm volatile("{.reg .pred ok; L_%=: mbarrier.try_wait.parity.shared::cta.b64 ok,[%0],%1,50000; @!ok bra L_%=;}"::"r"(shared_address(p)),"r"(phase):"memory");
 else asm volatile("{.reg .pred ok; L_%=: mbarrier.try_wait.parity.shared::cta.b64 ok,[%0],%1; @!ok bra L_%=;}"::"r"(shared_address(p)),"r"(phase):"memory");
}
struct Result {unsigned producer_rounds,consumer_rounds,sm;};
__global__ __launch_bounds__(384,1) void token_ring(Result* out,int cm,int cn,int depth,int rounds,int hint){
 extern __shared__ unsigned char storage[];
 __shared__ __align__(8) uint64_t full[kMaxDepth],empty[kMaxDepth];
 auto group=cooperative_groups::this_cluster();int rank=group.block_rank();
 int my_m=rank/cn,my_n=rank%cn,warp=threadIdx.x/32,lane=threadIdx.x%32;
 if(threadIdx.x==0){
  // Make the declared dynamic shared allocation part of the kernel resource.
  storage[0]=0;
  for(int s=0;s<depth;++s){initialize_barrier(full+s,cm+cn);initialize_barrier(empty+s,cm+cn);}
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 }
 group.sync();
 unsigned completed=0;int slot=0,phase=0;
 if(warp==10){
  for(int step=0;step<rounds;++step){
   if(lane==0){wait_ready(empty+slot,phase^1,hint);}
   __syncwarp();
   if(lane<cn)arrive_cluster(group.map_shared_rank(full+slot,my_m*cn+lane));
   else if(lane<cn+cm)arrive_cluster(group.map_shared_rank(full+slot,(lane-cn)*cn+my_n));
   ++completed;if(++slot==depth){slot=0;phase^=1;}
  }
 }else if(warp==0){
  for(int step=0;step<rounds;++step){
   if(lane==0)wait_ready(full+slot,phase,hint);
   __syncwarp();
   if(lane<cn)arrive_cluster(group.map_shared_rank(empty+slot,my_m*cn+lane));
   else if(lane<cn+cm)arrive_cluster(group.map_shared_rank(empty+slot,(lane-cn)*cn+my_n));
   ++completed;if(++slot==depth){slot=0;phase^=1;}
  }
 }
 // Final fanout completion is observed before any peer barrier storage dies.
 __syncthreads();group.sync();
 if(threadIdx.x==320)out[blockIdx.x].producer_rounds=completed;
 if(threadIdx.x==0){out[blockIdx.x].consumer_rounds=completed;unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));out[blockIdx.x].sm=sm;}
}
int main(){
 Result* out=nullptr;CHECK(cudaMalloc(&out,128*sizeof(Result)));
 CHECK(cudaFuncSetAttribute(token_ring,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
 auto launch=[&](int grid,int cm,int cn,int depth,int rounds,int hint){
  cudaLaunchAttribute attribute={};attribute.id=cudaLaunchAttributeClusterDimension;attribute.val.clusterDim.x=cm*cn;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
  cudaLaunchConfig_t config={};config.gridDim=dim3(grid);config.blockDim=dim3(kThreads);config.dynamicSmemBytes=kShared;config.stream=stream;config.attrs=&attribute;config.numAttrs=1;
  CHECK(cudaLaunchKernelEx(&config,token_ring,out,cm,cn,depth,rounds,hint));
 };
 auto warm=std::chrono::steady_clock::now();do{launch(128,1,1,4,128,0);CHECK(cudaStreamSynchronize(stream));}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
 for(auto axes:std::vector<std::pair<int,int>>{{1,1},{1,2},{1,4},{2,2},{2,4},{4,2}})for(int grid:{32,96})for(int depth:{1,4,16})for(int hint:{0,1})for(int rounds:{128,512}){
  int cm=axes.first,cn=axes.second;
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int node=0;node<8;++node)launch(grid,cm,cn,depth,rounds,hint);
  cudaGraph_t graph;CHECK(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t exec;CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  std::vector<double> samples;std::vector<Result> host(grid);
  for(int trial=0;trial<11;++trial){
   CHECK(cudaMemsetAsync(out,0,128*sizeof(Result),stream));CHECK(cudaEventRecord(begin,stream));CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));if(!(ms>0))return 4;
   CHECK(cudaMemcpy(host.data(),out,grid*sizeof(Result),cudaMemcpyDeviceToHost));std::vector<unsigned> sms;
   for(auto x:host){if(x.producer_rounds!=unsigned(rounds) || x.consumer_rounds!=unsigned(rounds))return 5;sms.push_back(x.sm);}
   std::sort(sms.begin(),sms.end());if(std::unique(sms.begin(),sms.end())!=sms.end())return 6;
   if(trial>=2)samples.push_back(double(ms)*1e6/8);
  }
  std::sort(samples.begin(),samples.end());printf("{\"cluster_m\":%d,\"cluster_n\":%d,\"grid\":%d,\"depth\":%d,\"hint\":%d,\"rounds\":%d,\"graph_nodes\":8,\"event_ns_p50\":%.9f,\"event_ns_p10\":%.9f,\"event_ns_p90\":%.9f,\"checks_passed\":true}\n",cm,cn,grid,depth,hint,rounds,samples[4],samples[0],samples[8]);fflush(stdout);
  CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
 }
 CHECK(cudaFree(out));CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaStreamDestroy(stream));return 0;
}
