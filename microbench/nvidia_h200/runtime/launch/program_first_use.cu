// Independent first/repeated program-use control. No matrices or TMA.
#include <cuda_runtime.h>
#include <cupti.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#define CUDA(call) do {auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)
#define CUPTI(call) do {auto e=(call);if(e!=CUPTI_SUCCESS){const char* s=nullptr;cuptiGetResultString(e,&s);fprintf(stderr,"CUPTI %s\n",s);exit(3);}}while(0)
struct DeviceResult { uint64_t begin,end;unsigned checksum,sm; };
struct Activity {uint64_t start,end;unsigned grid,block,shared;std::string name;};
static std::mutex activity_mutex;
static std::vector<Activity> activities;
static size_t dropped=0;
static void CUPTIAPI request(uint8_t** buffer,size_t* size,size_t* max_records) {
  *size=8*1024*1024;*max_records=0;void* data=nullptr;
  if(posix_memalign(&data,8,*size))exit(4);*buffer=static_cast<uint8_t*>(data);
}
static void CUPTIAPI complete(CUcontext context,uint32_t stream,uint8_t* buffer,size_t,size_t valid) {
  std::lock_guard<std::mutex> lock(activity_mutex);
  CUpti_Activity* record=nullptr;
  while(true) {
    auto status=cuptiActivityGetNextRecord(buffer,valid,&record);
    if(status==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
    CUPTI(status);
    if(record->kind!=CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && record->kind!=CUPTI_ACTIVITY_KIND_KERNEL)exit(5);
    auto* kernel=reinterpret_cast<CUpti_ActivityKernel9*>(record);
    if(kernel->gridY!=1 || kernel->gridZ!=1 || kernel->blockY!=1 || kernel->blockZ!=1)exit(6);
    activities.push_back({kernel->start,kernel->end,unsigned(kernel->gridX),unsigned(kernel->blockX),
        unsigned(kernel->dynamicSharedMemory),kernel->name});
  }
  size_t lost=0;CUPTI(cuptiActivityGetNumDroppedRecords(context,stream,&lost));dropped+=lost;free(buffer);
}
__device__ __forceinline__ uint64_t timer_ns() {
  uint64_t result;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(result));return result;
}
template<int Ops,int Tag>
__global__ __launch_bounds__(384,1) void program_first_use(DeviceResult* out) {
  uint64_t begin=timer_ns();
  unsigned value=threadIdx.x+blockIdx.x+Tag;
#pragma unroll
  for(int i=0;i<Ops;++i)
    asm volatile("mad.lo.u32 %0, %0, 3, %1;":"+r"(value):"n"(2*Tag+1));
  uint64_t end=timer_ns();
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  if(threadIdx.x==0)out[blockIdx.x]={begin,end,value,sm};
}
struct Launch {int ops,tag,repeat,grid,block,shared,slot;};
static std::vector<Launch> launches;
template<int Ops,int Tag> void pair(DeviceResult* output) {
  constexpr int cell=Tag/16;
  constexpr int grid=cell%2?132:32;
  constexpr int block=cell/2?384:128;
  constexpr int shared=cell/2?196608:0;
  auto kernel=program_first_use<Ops,Tag>;
  CUDA(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
  for(int repeat=0;repeat<2;++repeat) {
    int slot=int(launches.size());launches.push_back({Ops,Tag,repeat,grid,block,shared,slot});
    kernel<<<grid,block,shared>>>(output+slot*132);
    CUDA(cudaGetLastError());
  }
}
template<int Ops,int... Tags> void variants(DeviceResult* output,std::integer_sequence<int,Tags...>) {
  (pair<Ops,Tags>(output),...);
}
int main() {
  DeviceResult* output=nullptr;constexpr int count=3*64*2;
  CUDA(cudaMalloc(&output,count*132*sizeof(DeviceResult)));
  CUDA(cudaMemset(output,0,count*132*sizeof(DeviceResult)));CUDA(cudaDeviceSynchronize());
  CUPTI(cuptiActivityRegisterCallbacks(request,complete));
  CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  variants<64>(output,std::make_integer_sequence<int,64>{});
  variants<256>(output,std::make_integer_sequence<int,64>{});
  variants<1024>(output,std::make_integer_sequence<int,64>{});
  CUDA(cudaDeviceSynchronize());CUPTI(cuptiActivityFlushAll(0));
  CUPTI(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  std::vector<DeviceResult> host(count*132);
  CUDA(cudaMemcpy(host.data(),output,host.size()*sizeof(DeviceResult),cudaMemcpyDeviceToHost));
  CUDA(cudaFree(output));
  if(dropped || activities.size()!=launches.size()) {fprintf(stderr,"activity count=%zu dropped=%zu\n",activities.size(),dropped);return 7;}
  std::sort(activities.begin(),activities.end(),[](auto& a,auto& b){return a.start<b.start;});
  for(size_t i=0;i<launches.size();++i) {
    auto& launch=launches[i];auto& activity=activities[i];
    if(activity.grid!=unsigned(launch.grid) || activity.block!=unsigned(launch.block)
        || activity.shared!=unsigned(launch.shared) || activity.end<=activity.start)return 8;
    uint64_t first=UINT64_MAX,last=0;std::vector<unsigned> sms;
    for(int cta=0;cta<launch.grid;++cta) {
      auto& value=host[launch.slot*132+cta];unsigned expected=unsigned(cta+launch.tag);
      for(int op=0;op<launch.ops;++op)expected=3u*expected+unsigned(2*launch.tag+1);
      if(value.checksum!=expected || value.end<value.begin || !value.begin)return 9;
      first=std::min(first,value.begin);last=std::max(last,value.end);
      sms.push_back(value.sm);
    }
    printf("{\"ops\":%d,\"tag\":%d,\"repeat\":%d,\"grid\":%d,\"block\":%d,\"shared\":%d,"
        "\"kernel_ns\":%llu,\"device_body_union_ns\":%llu,\"kernel\":\"%s\",\"checks_passed\":true,\"sm_ids\":[",
        launch.ops,launch.tag,launch.repeat,launch.grid,launch.block,launch.shared,
        static_cast<unsigned long long>(activity.end-activity.start),
        static_cast<unsigned long long>(last-first),activity.name.c_str());
    for(size_t index=0;index<sms.size();++index)printf("%s%u",index?",":"",sms[index]);
    printf("]}\n");
  }
}
