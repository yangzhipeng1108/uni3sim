// Independent LSU store completion; no loads, checksum store, matrix or GEMM.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}while(0)
constexpr int kThreads=512,kNodes=128;
__global__ __launch_bounds__(kThreads,1) void store_completion(
 uint16_t* out,int enable,int active_warps,int fresh,int node){
 unsigned t=threadIdx.x,warp=t/32,lane=t%32;
 size_t index=size_t(fresh?node:0)*gridDim.x*kThreads+(size_t(warp)*gridDim.x+blockIdx.x)*32+lane;
 uint16_t value=uint16_t((blockIdx.x*3+t+node*17)&0x7fff);
 // Force address/value generation in both controls. This local-only prefix
 // prevents the compiler from moving all that work behind an early EXIT in
 // the enabled-store mode. It produces no global request in the baseline.
 __shared__ __align__(16) uint64_t common[kThreads*2];
 unsigned slot=unsigned(__cvta_generic_to_shared(common+t*2));
 asm volatile("st.volatile.shared.b64 [%0],%1; st.volatile.shared.b16 [%0+8],%2;"::"r"(slot),"l"(out+index),"h"(value):"memory");
 __syncthreads();
 asm volatile("{.reg .pred p,q; setp.ne.u32 p,%2,0; setp.lt.u32 q,%3,%4; and.pred p,p,q; @p st.global.u16 [%0],%1;}"::"l"(out+index),"h"(value),"r"(enable),"r"(t),"r"(active_warps*32):"memory");
}
static void verify(const std::vector<uint16_t>& output,int grid,int warps,int fresh,int enable){
 for(int n=0;n<(fresh?kNodes:1);++n)for(int warp=0;warp<16;++warp)for(int cta=0;cta<grid;++cta)for(int lane=0;lane<32;++lane){
  int last=fresh?n:kNodes-1;size_t index=size_t(n)*grid*kThreads+(size_t(warp)*grid+cta)*32+lane;
  uint16_t expected=enable && warp<warps?uint16_t((cta*3+warp*32+lane+last*17)&0x7fff):uint16_t(0xffff);
  if(output[index]!=expected){fprintf(stderr,"store output check grid%d warp%d node%d enable%d\n",grid,warp,n,enable);std::exit(3);}
 }
}
int main(){
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
 cudaFuncAttributes attributes;CHECK(cudaFuncGetAttributes(&attributes,store_completion));if(attributes.localSizeBytes)return 4;
 uint16_t* warm_out=nullptr;CHECK(cudaMalloc(&warm_out,132*kThreads*2));
 auto warm=std::chrono::steady_clock::now();
 do{for(int n=0;n<32;++n)store_completion<<<132,kThreads,0,stream>>>(warm_out,1,16,0,n);CHECK(cudaStreamSynchronize(stream));}
 while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
 CHECK(cudaFree(warm_out));
 for(int grid:{1,32,132,896})for(int warps:{1,4,16})for(int fresh:{0,1}){
  size_t elements=size_t(grid)*kThreads*(fresh?kNodes:1);uint16_t* output=nullptr;CHECK(cudaMalloc(&output,elements*2));
  cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
  for(int mode:{0,1}){
   CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
   for(int n=0;n<kNodes;++n)store_completion<<<grid,kThreads,0,stream>>>(output,mode,warps,fresh,n);
   CHECK(cudaStreamEndCapture(stream,&graphs[mode]));
   size_t count=0;CHECK(cudaGraphGetNodes(graphs[mode],nullptr,&count));if(count!=kNodes)return 5;
   std::vector<cudaGraphNode_t> nodes(count);CHECK(cudaGraphGetNodes(graphs[mode],nodes.data(),&count));
   for(auto node:nodes){cudaGraphNodeType type;CHECK(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)return 5;}
   CHECK(cudaGraphInstantiate(&execs[mode],graphs[mode],nullptr,nullptr,0));
  }
  std::vector<uint16_t> host(elements);std::vector<double> times[2],paired;
  for(int trial=0;trial<11;++trial){
   double row[2];
   for(int order=0;order<2;++order){int mode=(trial+order)&1;
    CHECK(cudaMemsetAsync(output,0xff,elements*2,stream));CHECK(cudaEventRecord(begin,stream));CHECK(cudaGraphLaunch(execs[mode],stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
    float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));if(!(ms>0))return 6;row[mode]=double(ms)*1e6/kNodes;
    CHECK(cudaMemcpy(host.data(),output,elements*2,cudaMemcpyDeviceToHost));verify(host,grid,warps,fresh,mode);
   }
   if(trial>=2){times[0].push_back(row[0]);times[1].push_back(row[1]);paired.push_back(row[1]-row[0]);}
  }
  auto median=[](std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];};
  printf("{\"grid\":%d,\"threads\":512,\"active_warps\":%d,\"fresh_node_addresses\":%d,\"graph_nodes\":128,\"registers_per_thread\":%d,\"static_shared_bytes\":%zu,\"no_store_ns_p50\":%.9f,\"store_ns_p50\":%.9f,\"paired_increment_ns_p50\":%.9f,\"all_output_and_baseline_sentinels_checked\":true,\"paired_samples_ns\":[",grid,warps,fresh,attributes.numRegs,attributes.sharedSizeBytes,median(times[0]),median(times[1]),median(paired));
  for(size_t i=0;i<paired.size();++i)printf("%s%.9f",i?",":"",paired[i]);printf("]}\n");fflush(stdout);
  for(int mode:{0,1}){CHECK(cudaGraphExecDestroy(execs[mode]));CHECK(cudaGraphDestroy(graphs[mode]));}CHECK(cudaFree(output));
 }
 CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaStreamDestroy(stream));return 0;
}
