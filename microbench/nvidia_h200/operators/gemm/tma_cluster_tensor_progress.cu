// Independent cluster-broadcast -> Tensor consume -> peer-slot-release progress.
// Fixed instruction request and streams, no GEMM shape, scheduler or output matrix.
#define main unused_token_main
#include "../../runtime/launch/cluster_operand_token_ring.cu"
#undef main
#include "shared_tensor_activity_helper.cuh"
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cmath>
constexpr int kLarge=8192,kSmall=1024,kPlanes=128*512;
struct CoupledResult { unsigned producer_rounds,consumer_rounds,sm,failed; };
__global__ void fill_one(uint32_t* p,size_t count){for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x)p[i]=0x3c003c00;}
__device__ void expect_full(uint64_t* p,int bytes){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(shared_address(p)),"r"(bytes):"memory");}
__device__ void copy_large(void* p,const CUtensorMap* map,uint64_t* barrier,int plane){asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4,%5}],[%2];"::"r"(shared_address(p)),"l"(map),"r"(shared_address(barrier)),"r"(0),"r"(0),"r"(plane):"memory");}
__device__ void copy_small(void* p,const CUtensorMap* map,uint64_t* barrier,int plane,uint16_t mask){asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4,%5}],[%2],%6;"::"r"(shared_address(p)),"l"(map),"r"(shared_address(barrier)),"r"(0),"r"(0),"r"(plane),"h"(mask):"memory");}
__global__ __launch_bounds__(384,1) void cluster_tensor_progress(
 const __grid_constant__ CUtensorMap a,const __grid_constant__ CUtensorMap b,
 CoupledResult* out,int cluster_ctas,int depth,int rounds,int resident,int consumers,int math){
 extern __shared__ __align__(1024) unsigned char storage[];
 __shared__ __align__(8) uint64_t full[16],empty[16];
 __shared__ unsigned consumer_count[2];
 auto group=cooperative_groups::this_cluster();int rank=group.block_rank(),wg=threadIdx.x/128,lane=threadIdx.x%128;
 int large_bytes=kLarge*consumers,stride=large_bytes+kSmall;
 if(threadIdx.x==0){consumer_count[0]=consumer_count[1]=0;for(int s=0;s<depth;++s){initialize_barrier(full+s,1);initialize_barrier(empty+s,cluster_ctas+1);expect_full(full+s,stride);}asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
 group.sync();float d[4]={};unsigned completed=0;int slot=0,phase=0;
 if(threadIdx.x==320){
  for(int r=0;r<rounds;++r){
   wait_ready(empty+slot,phase^1,0);int index=resident?slot:r;
   copy_large(storage+slot*stride,&a,full+slot,blockIdx.x*512+index);
   if(rank==0)copy_small(storage+slot*stride+large_bytes,&b,full+slot,(blockIdx.x/cluster_ctas)*512+index,uint16_t((1u<<cluster_ctas)-1));
   ++completed;if(++slot==depth){slot=0;phase^=1;}
  }
 }else if(wg<consumers){
  for(int r=0;r<rounds;++r){
   {
    if(lane%32==0)wait_ready(full+slot,phase,0);
    __syncwarp();if(wg==0)asm volatile("bar.sync 1,128;":::"memory");else asm volatile("bar.sync 2,128;":::"memory");
    if(wg==0 && lane==0 && r+depth<rounds)expect_full(full+slot,stride);
    if(math){unsigned base=shared_address(storage+slot*stride);wg_fence();
#pragma unroll
     for(int k=0;k<4;++k)MatrixRequest<8>::issue(d,make_desc(base+wg*kLarge+k*32,16,1024),make_desc(base+large_bytes+k*32,16,1024),1);
     wg_commit();wg_wait0();}
    asm volatile("bar.sync 3,%0;"::"r"(consumers*128):"memory");
    // The shared small input can be overwritten only after every peer
    // consumer finishes. One additional self arrival covers the private input.
    if(wg==0 && lane<cluster_ctas)arrive_cluster(group.map_shared_rank(empty+slot,lane));
    else if(wg==0 && lane==cluster_ctas)arrive_cluster(group.map_shared_rank(empty+slot,rank));
    ++completed;
   }
   if(++slot==depth){slot=0;phase^=1;}
  }
  float sum=d[0]+d[1]+d[2]+d[3];unsigned count=rounds;
  if(!isfinite(sum)||sum!=float(math?count*64*4:0))atomicAdd(&out[blockIdx.x].failed,1u);
  if(lane==0)consumer_count[wg]=completed;
 }
 __syncthreads();group.sync();
 for(int s=0;s<depth;++s)for(int word=threadIdx.x;word<stride/4;word+=blockDim.x)
  if(reinterpret_cast<uint32_t*>(storage+s*stride)[word]!=0x3c003c00)atomicAdd(&out[blockIdx.x].failed,1u);
 if(threadIdx.x==320)out[blockIdx.x].producer_rounds=completed;
 if(threadIdx.x==0){out[blockIdx.x].consumer_rounds=consumer_count[0]+consumer_count[1];unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));out[blockIdx.x].sm=sm;}
 group.sync();
}
static CUtensorMap make_full_map(void* pointer,int bytes){
 void* function=nullptr;cudaDriverEntryPointQueryResult status;CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&function,cudaEnableDefault,&status));if(!function||status!=cudaDriverEntryPointSuccess)exit(3);
 auto encode=reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function);cuuint64_t dims[3]={128,cuuint64_t(bytes/128),kPlanes},strides[2]={128,cuuint64_t(bytes)};cuuint32_t box[3]={128,cuuint32_t(bytes/128),1},steps[3]={1,1,1};CUtensorMap result;
 if(encode(&result,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,pointer,dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)exit(4);return result;
}
int main(){
 uint32_t *a=nullptr,*b=nullptr;CoupledResult* out=nullptr;CHECK(cudaMalloc(&a,size_t(kPlanes)*kLarge*2));CHECK(cudaMalloc(&b,size_t(kPlanes)*kSmall));CHECK(cudaMalloc(&out,128*sizeof(CoupledResult)));
 fill_one<<<4096,256>>>(a,size_t(kPlanes)*kLarge*2/4);fill_one<<<4096,256>>>(b,size_t(kPlanes)*kSmall/4);CHECK(cudaDeviceSynchronize());auto bm=make_full_map(b,kSmall);CUtensorMap am[2]={make_full_map(a,kLarge),make_full_map(a,kLarge*2)};
 CHECK(cudaFuncSetAttribute(cluster_tensor_progress,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));cudaFuncAttributes attributes;CHECK(cudaFuncGetAttributes(&attributes,cluster_tensor_progress));if(attributes.localSizeBytes)return 8;
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
 auto launch=[&](int grid,int cluster,int depth,int rounds,int resident,int consumers,int math){cudaLaunchAttribute attribute={};attribute.id=cudaLaunchAttributeClusterDimension;attribute.val.clusterDim.x=cluster;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;cudaLaunchConfig_t config={};config.gridDim=dim3(grid);config.blockDim=dim3(kThreads);config.dynamicSmemBytes=kShared;config.stream=stream;config.attrs=&attribute;config.numAttrs=1;CHECK(cudaLaunchKernelEx(&config,cluster_tensor_progress,am[consumers-1],bm,out,cluster,depth,rounds,resident,consumers,math));};
 auto warm=std::chrono::steady_clock::now();do{CHECK(cudaMemsetAsync(out,0,128*sizeof(CoupledResult),stream));launch(96,4,12,128,0,2,1);CHECK(cudaStreamSynchronize(stream));}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
 for(int cluster:{1,4})for(int grid:{32,96})for(int depth:{4,12})for(int resident:{0,1})for(int consumers:{1,2})for(int math:{0,1})for(int rounds:{32,128,512}){
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int node=0;node<8;++node)launch(grid,cluster,depth,rounds,resident,consumers,math);cudaGraph_t graph;CHECK(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t exec;CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  std::vector<double> samples;std::vector<CoupledResult> host(grid);
  for(int trial=0;trial<11;++trial){CHECK(cudaMemsetAsync(out,0,128*sizeof(CoupledResult),stream));CHECK(cudaEventRecord(begin,stream));CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));if(!(ms>0))return 5;CHECK(cudaMemcpy(host.data(),out,grid*sizeof(CoupledResult),cudaMemcpyDeviceToHost));std::vector<unsigned> sms;
   for(auto x:host){if(x.failed||x.producer_rounds!=unsigned(rounds)||x.consumer_rounds!=unsigned(rounds*consumers)){fprintf(stderr,"validation cluster%d grid%d depth%d rounds%d resident%d consumers%d math%d failed%u prod%u cons%u\n",cluster,grid,depth,rounds,resident,consumers,math,x.failed,x.producer_rounds,x.consumer_rounds);return 6;}sms.push_back(x.sm);}std::sort(sms.begin(),sms.end());if(std::unique(sms.begin(),sms.end())!=sms.end())return 7;if(trial>=2)samples.push_back(double(ms)*1e6/8);
  }
  std::sort(samples.begin(),samples.end());printf("{\"cluster_ctas\":%d,\"grid\":%d,\"depth\":%d,\"resident\":%d,\"consumers\":%d,\"math\":%d,\"rounds\":%d,\"instruction_n\":8,\"large_private_bytes_per_consumer\":8192,\"small_shared_bytes\":1024,\"threads\":384,\"registers\":%d,\"static_smem_bytes\":%zu,\"event_ns_p50\":%.9f,\"event_ns_p10\":%.9f,\"event_ns_p90\":%.9f,\"checks_passed\":true}\n",cluster,grid,depth,resident,consumers,math,rounds,attributes.numRegs,attributes.sharedSizeBytes,samples[4],samples[0],samples[8]);fflush(stdout);CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
 }
 CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaStreamDestroy(stream));CHECK(cudaFree(out));CHECK(cudaFree(a));CHECK(cudaFree(b));return 0;
}
