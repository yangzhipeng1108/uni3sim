// Independent split-payload multicast queue. No MMA, GEMM, or output matrix.
#define main token_reference_main
#include "../../runtime/launch/cluster_operand_token_ring.cu"
#undef main
#include <cuda.h>
#include <cudaTypedefs.h>
#ifndef TMA_ROW_STRIDE_BYTES
#define TMA_ROW_STRIDE_BYTES 128
#endif
#ifndef TMA_PAYLOAD_A_BYTES
#define TMA_PAYLOAD_A_BYTES 8192
#endif
static_assert(TMA_ROW_STRIDE_BYTES>=128 && TMA_ROW_STRIDE_BYTES%128==0);
constexpr int kA=TMA_PAYLOAD_A_BYTES,kB=9216-kA;
constexpr int kPanelsPerPlane=TMA_ROW_STRIDE_BYTES/128;
constexpr int kPlanesPerGroup=(512+kPanelsPerPlane-1)/kPanelsPerPlane;
constexpr int kPlanes=128*kPlanesPerGroup;
__host__ __device__ int aligned_part(int bytes){return ((bytes+1023)/1024)*1024;}
__host__ __device__ uint32_t hash_word(uint32_t word,uint32_t seed){uint32_t v=word*747796405u+seed*2891336453u;v=((v>>((v>>28)+4))^v)*277803737u;return(v>>22)^v;}
__global__ void fill_random(uint32_t* p,size_t count,unsigned seed){for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x)p[i]=hash_word(unsigned(i),seed);}
__device__ void expect_copy(uint64_t* p){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(shared_address(p)),"r"(kA+kB):"memory");}
__device__ void load_multicast(void* p,const CUtensorMap* map,uint64_t* barrier,int column,int row,int plane,uint16_t mask){asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4,%5}],[%2],%6;"::"r"(shared_address(p)),"l"(map),"r"(shared_address(barrier)),"r"(column),"r"(row),"r"(plane),"h"(mask):"memory");}
struct CopyResult{unsigned producer_rounds,consumer_rounds,sm,failed;};
__global__ __launch_bounds__(384,1) void multicast_copy(const __grid_constant__ CUtensorMap a,const __grid_constant__ CUtensorMap b,CopyResult* out,int cm,int cn,int depth,int rounds,int hint,int resident){
 extern __shared__ __align__(128) unsigned char storage[];
 __shared__ __align__(8) uint64_t full[16],empty[16];
 __shared__ unsigned long long actual,expected;
 __shared__ unsigned ax,ex;
 auto group=cooperative_groups::this_cluster();int rank=group.block_rank(),my_m=rank/cn,my_n=rank%cn;
 int warp=threadIdx.x/32,lane=threadIdx.x%32,cluster_index=blockIdx.x/(cm*cn);
 int part_a=kA/cn,part_b=kB/cm,span_a=aligned_part(part_a)*cn,span_b=aligned_part(part_b)*cm,slot_bytes=span_a+span_b;
 if(threadIdx.x==0){for(int s=0;s<depth;++s){initialize_barrier(full+s,1);initialize_barrier(empty+s,cm+cn);expect_copy(full+s);}asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
 group.sync();int slot=0,phase=0;unsigned completed=0;
 if(warp==10 && lane==0){
  uint16_t mask_a=uint16_t(((1u<<cn)-1)<<(my_m*cn)),mask_b=0;for(int i=0;i<cm;++i)mask_b|=uint16_t(1u<<(i*cn+my_n));
  for(int step=0;step<rounds;++step){
   wait_ready(empty+slot,phase^1,hint);int index=resident?slot:(step&511);
   int pa=(cluster_index*cm+my_m)*kPlanesPerGroup+index/kPanelsPerPlane,pb=(cluster_index*cn+my_n)*kPlanesPerGroup+index/kPanelsPerPlane,column=(index%kPanelsPerPlane)*128;
   auto* dest=storage+slot*slot_bytes;
   load_multicast(dest+my_n*aligned_part(part_a),&a,full+slot,column,my_n*(part_a/128),pa,mask_a);
   load_multicast(dest+span_a+my_m*aligned_part(part_b),&b,full+slot,column,my_m*(part_b/128),pb,mask_b);
   ++completed;if(++slot==depth){slot=0;phase^=1;}
  }
 }else if(warp==0){
  for(int step=0;step<rounds;++step){
   if(lane==0){wait_ready(full+slot,phase,hint);if(step+depth<rounds)expect_copy(full+slot);}
   __syncwarp();
   if(lane<cn)arrive_cluster(group.map_shared_rank(empty+slot,my_m*cn+lane));
   else if(lane<cn+cm)arrive_cluster(group.map_shared_rank(empty+slot,(lane-cn)*cn+my_n));
   ++completed;if(++slot==depth){slot=0;phase^=1;}
  }
 }
 __syncthreads();group.sync();
 if(threadIdx.x==320)out[blockIdx.x].producer_rounds=completed;
 if(threadIdx.x==0){out[blockIdx.x].consumer_rounds=completed;unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));out[blockIdx.x].sm=sm;}
 for(int s=0;s<depth;++s){
  if(threadIdx.x==0){actual=expected=0;ax=ex=0;}__syncthreads();
  int last=rounds-1-(rounds-1-s)%depth,index=resident?s:(last&511);
  unsigned long long av=0,ev=0;unsigned a_xor=0,e_xor=0;
  for(int word=threadIdx.x;word<(kA+kB)/4;word+=blockDim.x){
   bool isa=word<kA/4;int local=isa?word:word-kA/4,part=isa?part_a:part_b,whole=isa?kA:kB;
   int shard=local/(part/4),within=local%(part/4),plane=(isa?(cluster_index*cm+my_m):(cluster_index*cn+my_n))*kPlanesPerGroup+index/kPanelsPerPlane,column=(index%kPanelsPerPlane)*128;
   auto* data=reinterpret_cast<uint32_t*>(storage+s*slot_bytes+(isa?0:span_a)+shard*aligned_part(part));
   unsigned value=data[within],reference=hash_word(unsigned((size_t(plane)*(whole/128)+local/32)*(TMA_ROW_STRIDE_BYTES/4)+column/4+local%32),isa?1:2);av+=value;ev+=reference;a_xor^=value;e_xor^=reference;
  }
  atomicAdd(&actual,av);atomicAdd(&expected,ev);atomicXor(&ax,a_xor);atomicXor(&ex,e_xor);__syncthreads();
  if(threadIdx.x==0 && (actual!=expected || ax!=ex))++out[blockIdx.x].failed;
  __syncthreads();
 }
 group.sync();
}
static CUtensorMap make_map(void* pointer,int bytes,int shards){
 void* function=nullptr;cudaDriverEntryPointQueryResult status;CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&function,cudaEnableDefault,&status));if(!function || status!=cudaDriverEntryPointSuccess)exit(3);
 auto encode=reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function);cuuint64_t dims[3]={TMA_ROW_STRIDE_BYTES,cuuint64_t(bytes/128),kPlanes},strides[2]={TMA_ROW_STRIDE_BYTES,cuuint64_t(bytes)*kPanelsPerPlane};cuuint32_t box[3]={128,cuuint32_t(bytes/shards/128),1},steps[3]={1,1,1};CUtensorMap result;
 if(encode(&result,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,pointer,dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)exit(4);return result;
}
int main(int argc,char**){
 if(argc>1)return token_reference_main();
 uint32_t *a=nullptr,*b=nullptr;CopyResult* out=nullptr;CHECK(cudaMalloc(&a,size_t(kPlanes)*kA*kPanelsPerPlane));CHECK(cudaMalloc(&b,size_t(kPlanes)*kB*kPanelsPerPlane));CHECK(cudaMalloc(&out,128*sizeof(CopyResult)));
 fill_random<<<4096,256>>>(a,size_t(kPlanes)*kA*kPanelsPerPlane/4,1);fill_random<<<4096,256>>>(b,size_t(kPlanes)*kB*kPanelsPerPlane/4,2);CHECK(cudaDeviceSynchronize());CHECK(cudaFuncSetAttribute(multicast_copy,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
 auto launch=[&](const CUtensorMap& am,const CUtensorMap& bm,int grid,int cm,int cn,int depth,int rounds,int hint,int resident){cudaLaunchAttribute attribute={};attribute.id=cudaLaunchAttributeClusterDimension;attribute.val.clusterDim.x=cm*cn;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;cudaLaunchConfig_t config={};config.gridDim=dim3(grid);config.blockDim=dim3(kThreads);config.dynamicSmemBytes=kShared;config.stream=stream;config.attrs=&attribute;config.numAttrs=1;CHECK(cudaLaunchKernelEx(&config,multicast_copy,am,bm,out,cm,cn,depth,rounds,hint,resident));};
 auto wa=make_map(a,kA,1),wb=make_map(b,kB,1);auto warm=std::chrono::steady_clock::now();do{CHECK(cudaMemsetAsync(out,0,128*sizeof(CopyResult),stream));launch(wa,wb,96,1,1,4,128,0,0);CHECK(cudaStreamSynchronize(stream));}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
 for(auto axes:std::vector<std::pair<int,int>>{{1,1},{1,2},{1,4},{2,2},{2,4},{4,2}}){int cm=axes.first,cn=axes.second;auto am=make_map(a,kA,cn),bm=make_map(b,kB,cm);
 for(int grid:{32,96})for(int depth:{4,12,16})for(int hint:{0,1})for(int resident:{0,1})for(int rounds:{32,128,512}){
  if(depth*(aligned_part(kA/cn)*cn+aligned_part(kB/cm)*cm)>kShared)return 9;
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int node=0;node<8;++node)launch(am,bm,grid,cm,cn,depth,rounds,hint,resident);cudaGraph_t graph;CHECK(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t exec;CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));std::vector<double> samples;std::vector<CopyResult> host(grid);
  for(int trial=0;trial<11;++trial){CHECK(cudaMemsetAsync(out,0,128*sizeof(CopyResult),stream));CHECK(cudaEventRecord(begin,stream));CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));if(!(ms>0))return 5;CHECK(cudaMemcpy(host.data(),out,grid*sizeof(CopyResult),cudaMemcpyDeviceToHost));std::vector<unsigned> sms;for(auto x:host){if(x.failed || x.producer_rounds!=unsigned(rounds) || x.consumer_rounds!=unsigned(rounds))return 6;sms.push_back(x.sm);}std::sort(sms.begin(),sms.end());if(std::unique(sms.begin(),sms.end())!=sms.end())return 7;if(trial>=2)samples.push_back(double(ms)*1e6/8);}
  std::sort(samples.begin(),samples.end());printf("{\"row_stride_bytes\":%d,\"a_bytes\":%d,\"b_bytes\":%d,\"cluster_m\":%d,\"cluster_n\":%d,\"grid\":%d,\"depth\":%d,\"hint\":%d,\"resident\":%d,\"rounds\":%d,\"event_ns_p50\":%.9f,\"event_ns_p10\":%.9f,\"event_ns_p90\":%.9f,\"checks_passed\":true}\n",TMA_ROW_STRIDE_BYTES,kA,kB,cm,cn,grid,depth,hint,resident,rounds,samples[4],samples[0],samples[8]);fflush(stdout);CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
 }}
 CHECK(cudaFree(out));CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaStreamDestroy(stream));return 0;
}
