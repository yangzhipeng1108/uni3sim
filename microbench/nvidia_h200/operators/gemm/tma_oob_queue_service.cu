// Paired valid payload, compact versus zero-filled TMA box. No matrix math.
#define main unused_queue_service_main
#include "tma_queue_service.cu"
#undef main

__global__ __launch_bounds__(384,1) void oob_queue_service(
    const __grid_constant__ CUtensorMap a,const __grid_constant__ CUtensorMap b,
    Timing* result,int valid_rows,int box_rows,int depth,int rounds,int resident) {
  extern __shared__ __align__(128) unsigned char storage[];
  __shared__ __align__(8) uint64_t full[16];
  __shared__ unsigned long long actual_sum,expected_sum;
  __shared__ unsigned actual_xor,expected_xor;
  int a_bytes=box_rows*128,b_bytes=8192;
  if(threadIdx.x==0){
    result[blockIdx.x].failed=0;
    uint64_t begin=timer();
    do{copy_rounds(&a,&b,full,storage,a_bytes,b_bytes,depth,512,resident);}
    while(timer()-begin<3000000);
  }
  cooperative_groups::this_grid().sync();
  if(threadIdx.x==0){
    uint64_t c0=clock64(),t0=timer();
    copy_rounds(&a,&b,full,storage,a_bytes,b_bytes,depth,rounds,resident);
    uint64_t t1=timer(),c1=clock64();unsigned sm;
    asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
    result[blockIdx.x].c0=c0;result[blockIdx.x].t0=t0;
    result[blockIdx.x].c1=c1;result[blockIdx.x].t1=t1;
    result[blockIdx.x].sm=sm;
  }
  __syncthreads();
  for(int slot=0;slot<depth;++slot){
    if(threadIdx.x==0){actual_sum=expected_sum=0;actual_xor=expected_xor=0;}
    __syncthreads();
    int last=rounds-1-(rounds-1-slot)%depth;
    int logical=resident?slot:(last&511);
    int plane=int(blockIdx.x)*kPlanesPerCTA+logical;
    unsigned long long actual=0,expected=0;unsigned ax=0,ex=0;
    for(int word=threadIdx.x;word<(a_bytes+b_bytes)/4;word+=blockDim.x){
      bool operand_a=word<a_bytes/4;
      int local_word=operand_a?word:word-a_bytes/4;
      int source_rows=operand_a?valid_rows:64;
      size_t global_word=size_t(plane)*source_rows*32+local_word;
      auto* words=reinterpret_cast<const uint32_t*>(storage+slot*slot_storage(a_bytes,b_bytes)
          +(operand_a?0:a_storage(a_bytes)));
      unsigned v=words[local_word];
      unsigned e=operand_a && local_word/32>=valid_rows ? 0u:
          value_at(unsigned(global_word),operand_a?1:2);
      actual+=v;expected+=e;ax^=v;ex^=e;
    }
    atomicAdd(&actual_sum,actual);atomicAdd(&expected_sum,expected);
    atomicXor(&actual_xor,ax);atomicXor(&expected_xor,ex);__syncthreads();
    if(threadIdx.x==0 && (actual_sum!=expected_sum || actual_xor!=expected_xor))
      result[blockIdx.x].failed+=1;
    __syncthreads();
  }
}

static CUtensorMap oob_map(void* pointer,int valid_rows,int box_rows){
  void* function=nullptr;cudaDriverEntryPointQueryResult status;
  CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&function,cudaEnableDefault,&status));
  if(!function||status!=cudaDriverEntryPointSuccess)exit(3);
  auto encode=reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function);
  cuuint64_t dims[3]={128,cuuint64_t(valid_rows),kPlanes};
  cuuint64_t strides[2]={128,cuuint64_t(valid_rows)*128};
  cuuint32_t box[3]={128,cuuint32_t(box_rows),1},steps[3]={1,1,1};CUtensorMap result;
  if(encode(&result,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,pointer,dims,strides,box,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)exit(4);
  return result;
}

int main(){
  int cooperative=0;CHECK(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,0));
  if(!cooperative)return 5;
  Timing* output=nullptr;uint32_t* b=nullptr;
  CHECK(cudaMalloc(&output,132*sizeof(Timing)));CHECK(cudaMalloc(&b,size_t(kPlanes)*8192));
  initialize<<<4096,256>>>(b,size_t(kPlanes)*8192/4,2);CHECK(cudaDeviceSynchronize());
  auto b_map=oob_map(b,64,64);
  CHECK(cudaFuncSetAttribute(oob_queue_service,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  for(int valid_rows:{1,2,4,8}){
    uint32_t* a=nullptr;size_t a_size=size_t(kPlanes)*valid_rows*128;
    CHECK(cudaMalloc(&a,a_size));initialize<<<4096,256>>>(a,a_size/4,1);CHECK(cudaDeviceSynchronize());
    for(int box_rows:std::vector<int>(valid_rows==8?std::vector<int>{8}:std::vector<int>{valid_rows,8})){
      auto a_map=oob_map(a,valid_rows,box_rows);
      for(int grid:{32,132})for(int depth:{4,12})for(int resident:{0,1})for(int rounds:{128,512}){
        std::vector<double> spans,intervals,cycles,clocks;std::vector<Timing> host(grid);
        for(int trial=0;trial<9;++trial){
          void* args[]={&a_map,&b_map,&output,&valid_rows,&box_rows,&depth,&rounds,&resident};
          CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(oob_queue_service),dim3(grid),dim3(384),args,kShared));
          CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(host.data(),output,grid*sizeof(Timing),cudaMemcpyDeviceToHost));
          uint64_t first=UINT64_MAX,last=0;std::vector<unsigned> sms;
          for(auto t:host){
            if(t.failed||t.t1<=t.t0||t.c1<=t.c0){fprintf(stderr,"payload/timing valid%d box%d failed%u\n",valid_rows,box_rows,t.failed);return 6;}
            double clock=1000.0*(t.c1-t.c0)/(t.t1-t.t0);
            if(!(100<clock&&clock<3000))return 8;
            first=std::min(first,t.t0);last=std::max(last,t.t1);sms.push_back(t.sm);
            if(trial>=2){intervals.push_back(t.t1-t.t0);cycles.push_back(t.c1-t.c0);clocks.push_back(clock);}
          }
          std::sort(sms.begin(),sms.end());if(std::unique(sms.begin(),sms.end())!=sms.end())return 7;
          if(trial>=2)spans.push_back(last-first);
        }
        auto median=[](std::vector<double>& v){std::sort(v.begin(),v.end());return v[v.size()/2];};
        printf("{\"valid_rows\":%d,\"box_rows\":%d,\"b_bytes\":8192,\"grid\":%d,\"depth\":%d,\"resident\":%d,\"rounds\":%d,\"span_ns_p50\":%.3f,\"cta_ns_p50\":%.3f,\"cycles_p50\":%.3f,\"clock_mhz_p50\":%.3f,\"validation\":\"full_box_sum_xor_including_zero_fill\",\"checks_passed\":true}\n",
          valid_rows,box_rows,grid,depth,resident,rounds,median(spans),median(intervals),median(cycles),median(clocks));fflush(stdout);
      }
    }
    CHECK(cudaFree(a));
  }
  CHECK(cudaFree(b));CHECK(cudaFree(output));return 0;
}
