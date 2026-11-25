// Independent output staging and bulk-store completion. No MMA or GEMM.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int kShared=212992;
__device__ uint32_t address(const void* p){return unsigned(__cvta_generic_to_shared(p));}
__global__ __launch_bounds__(384,1) void output_stage(uint32_t* out,uint32_t* sums,int bytes,int fragments,int mode,const __grid_constant__ CUtensorMap tensor_map) {
  extern __shared__ __align__(128) unsigned char raw[];
  const int group=threadIdx.x/128, local=threadIdx.x%128;
  volatile uint32_t* storage=reinterpret_cast<volatile uint32_t*>(raw+(group%2)*8192);
  uint32_t sum=0;
  for(int fragment=0;fragment<fragments;++fragment) {
    if(threadIdx.x<256)for(int index=local;index<bytes/8;index+=128) {
      int word=group*(bytes/8)+index;
      uint32_t value=unsigned(blockIdx.x)*100003u+fragment*unsigned(bytes/4)+word+1;
      sum+=value;
      if(mode==1 || mode>=3)storage[index]=value;
      if(mode==2)out[(blockIdx.x*fragments+fragment)*(bytes/4)+word]=value;
    }
    if(threadIdx.x<256)asm volatile("bar.sync %0,128;"::"r"(group+1):"memory");
    if(mode>=3 && threadIdx.x<256) {
      if(threadIdx.x<256)asm volatile("fence.proxy.async.shared::cta;":::"memory");
      if(threadIdx.x<256)asm volatile("bar.sync %0,128;"::"r"(group+1):"memory");
      if(local==0) {
        if(mode<5)asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0],[%1],%2;"::"l"(out+(blockIdx.x*fragments+fragment)*(bytes/4)+group*(bytes/8)),"r"(address(raw+group*8192)),"r"(bytes/2):"memory");
        else asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{0,0,%2}],[%1];"::"l"(&tensor_map),"r"(address(raw+group*8192)),"r"((int(blockIdx.x)*fragments+fragment)*2+group):"memory");
        asm volatile("cp.async.bulk.commit_group;":::"memory");
        if(mode==3 || mode==5)asm volatile("cp.async.bulk.wait_group.read 0;":::"memory");
        else asm volatile("cp.async.bulk.wait_group 0;":::"memory");
      }
      if(threadIdx.x<256)asm volatile("bar.sync %0,128;"::"r"(group+1):"memory");
    }
  }
  if((mode==3 || mode==5) && threadIdx.x<256 && local==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
  if(threadIdx.x<256) {
    for(int delta=16;delta;delta>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,delta);
    if(threadIdx.x%32==0)sums[blockIdx.x*8+threadIdx.x/32]=sum;
  }
}
static CUtensorMap make_output_map(void* pointer,int group_bytes) {
  void* function=nullptr;cudaDriverEntryPointQueryResult status;
  CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&function,cudaEnableDefault,&status));
  if(!function || status!=cudaDriverEntryPointSuccess)exit(5);
  auto encode=reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function);
  cuuint64_t dimensions[3]={128,cuuint64_t(group_bytes/128),132*4*2},strides[2]={128,cuuint64_t(group_bytes)};
  cuuint32_t box[3]={128,cuuint32_t(group_bytes/128),1},step[3]={1,1,1};CUtensorMap result;
  if(encode(&result,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,pointer,dimensions,strides,box,step,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)exit(6);
  return result;
}
int main() {
  uint32_t *output=nullptr,*sums=nullptr;CHECK(cudaMalloc(&output,132*4*8192));CHECK(cudaMalloc(&sums,132*8*4));
  CHECK(cudaFuncSetAttribute(output_stage,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
  auto warm_map=make_output_map(output,4096);
  auto warm=std::chrono::steady_clock::now();
  do{output_stage<<<132,384,kShared,stream>>>(output,sums,8192,4,6,warm_map);CHECK(cudaStreamSynchronize(stream));}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
  for(int grid:{32,132})for(int bytes:{512,2048,8192})for(int fragments:{1,2,4})for(int mode:{0,1,2,3,4,5,6}) {
    auto tensor_map=make_output_map(output,bytes/2);
    CHECK(cudaMemset(output,0xff,132*4*8192));CHECK(cudaMemset(sums,0,132*8*4));
    cudaGraph_t graph;CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int replay=0;replay<32;++replay)output_stage<<<grid,384,kShared,stream>>>(output,sums,bytes,fragments,mode,tensor_map);
    CHECK(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t executable;CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    std::vector<double> samples;
    for(int trial=0;trial<11;++trial) {
      CHECK(cudaEventRecord(begin,stream));CHECK(cudaGraphLaunch(executable,stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));if(trial>=2)samples.push_back(double(ms)*1e6/32);
    }
    std::vector<uint32_t> host(size_t(grid)*fragments*bytes/4),checks(grid*8);
    CHECK(cudaMemcpy(host.data(),output,host.size()*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(checks.data(),sums,checks.size()*4,cudaMemcpyDeviceToHost));
    for(int cta=0;cta<grid;++cta) {
      for(int warp=0;warp<8;++warp){uint32_t expected=0;for(int f=0;f<fragments;++f)for(int lane=0;lane<32;++lane)for(int index=(warp%4)*32+lane;index<bytes/8;index+=128){int word=(warp/4)*(bytes/8)+index;expected+=uint32_t(cta)*100003u+f*unsigned(bytes/4)+word+1;}
        if(checks[cta*8+warp]!=expected)return 3;}
      for(int f=0;f<fragments;++f)for(int word=0;word<bytes/4;++word){size_t index=(size_t(cta)*fragments+f)*(bytes/4)+word;uint32_t expected=mode<2?0xffffffffu:uint32_t(cta)*100003u+f*unsigned(bytes/4)+word+1;if(host[index]!=expected)return 4;}
    }
    std::sort(samples.begin(),samples.end());
    printf("{\"grid\":%d,\"threads\":384,\"dynamic_shared_bytes\":%d,\"fragment_bytes\":%d,\"fragments\":%d,\"mode\":%d,\"graph_nodes\":32,\"median_ns\":%.6f,\"min_ns\":%.6f,\"max_ns\":%.6f,\"all_output_and_warp_sums_checked\":true}\n",grid,kShared,bytes,fragments,mode,samples[samples.size()/2],samples.front(),samples.back());fflush(stdout);
    CHECK(cudaGraphExecDestroy(executable));CHECK(cudaGraphDestroy(graph));
  }
  CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaStreamDestroy(stream));CHECK(cudaFree(output));CHECK(cudaFree(sums));return 0;
}
