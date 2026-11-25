// Copy-only launch-boundary diagnostic. Warm ALU, never warm TMA inside
// the timed launch; compare eager and graph using identical constant maps.
#define main queue_capacity_main
#include "tma_queue_service.cu"
#undef main
struct BoundaryTiming { uint64_t begin,ready,first,last; unsigned sm,failed; };
__global__ __launch_bounds__(384,1) void launch_boundary(
    const __grid_constant__ CUtensorMap a,const __grid_constant__ CUtensorMap b,
    BoundaryTiming* out,int a_bytes,int b_bytes,int requests) {
  extern __shared__ __align__(128) unsigned char buffer[];
  __shared__ __align__(8) uint64_t bar;
  // No descriptor instruction in this warmup. Keep the device active through
  // host enqueue gaps while leaving the TMA descriptor first-use measurable.
  if(threadIdx.x==0) {
    uint64_t start=timer();unsigned x=blockIdx.x+1;
    do {asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;":"+r"(x));}while(timer()-start<2000000);
    asm volatile(""::"r"(x):"memory");
  }
  cooperative_groups::this_grid().sync();
  if(threadIdx.x==0) {
    out[blockIdx.x].begin=timer();
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(addr(&bar)):"memory");
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
    out[blockIdx.x].ready=timer();
    for(int i=0;i<requests;++i) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(addr(&bar)),"r"(a_bytes+b_bytes):"memory");
      load(buffer,&a,&bar,int(blockIdx.x)*kPlanesPerCTA,0);
      load(buffer+a_storage(a_bytes),&b,&bar,int(blockIdx.x)*kPlanesPerCTA,0);
      wait(&bar,i&1);
      if(i==0)out[blockIdx.x].first=timer();
    }
    if(requests==0)out[blockIdx.x].first=timer();
    out[blockIdx.x].last=timer();
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
    out[blockIdx.x].sm=sm;out[blockIdx.x].failed=0;
  }
  __syncthreads();
  // The descriptor is 128B-swizzled, so validate complete copied sums+xors.
  if(threadIdx.x==0 && requests>0) {
    for(int operand=0;operand<2;++operand) {
      unsigned long long actual=0,expected=0;unsigned ax=0,ex=0;
      int bytes=operand?b_bytes:a_bytes;
      const uint32_t* data=reinterpret_cast<const uint32_t*>(buffer+(operand?a_storage(a_bytes):0));
      unsigned base=blockIdx.x*kPlanesPerCTA*(bytes/4);
      for(int word=0;word<bytes/4;++word) {
        unsigned v=data[word],e=value_at(base+word,operand?2:1);
        actual+=v;expected+=e;ax^=v;ex^=e;
      }
      if(actual!=expected || ax!=ex)++out[blockIdx.x].failed;
    }
  }
}
int main() {
  BoundaryTiming* output=nullptr;CHECK(cudaMalloc(&output,132*sizeof(BoundaryTiming)));
  CHECK(cudaFuncSetAttribute(launch_boundary,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
  for(auto pair:std::vector<std::pair<int,int>>{{128,8192},{4096,4096}}) {
    int ab=pair.first,bb=pair.second;uint32_t *a=nullptr,*b=nullptr;
    size_t as=size_t(kPlanes)*ab,bs=size_t(kPlanes)*bb;
    CHECK(cudaMalloc(&a,as));CHECK(cudaMalloc(&b,bs));
    initialize<<<4096,256>>>(a,as/4,1);initialize<<<4096,256>>>(b,bs/4,2);CHECK(cudaDeviceSynchronize());
    auto am=map(a,ab),bm=map(b,bb);
    for(int grid:{32,132})for(int requests:{0,1,8}) {
      void* args[]={&am,&bm,&output,&ab,&bb,&requests};
      cudaGraph_t graph;cudaGraphExec_t executable;
      CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(launch_boundary),dim3(grid),dim3(384),args,kShared,stream));
      CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
      for(int trial=0;trial<13;++trial)for(int order=0;order<2;++order) {
        int graph_mode=(trial+order)&1;
        if(graph_mode)CHECK(cudaGraphLaunch(executable,stream));
        else CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(launch_boundary),dim3(grid),dim3(384),args,kShared,stream));
        CHECK(cudaStreamSynchronize(stream));std::vector<BoundaryTiming> host(grid);
        CHECK(cudaMemcpy(host.data(),output,grid*sizeof(BoundaryTiming),cudaMemcpyDeviceToHost));
        std::vector<unsigned> sms;
        for(int cta=0;cta<grid;++cta) {
          auto t=host[cta];
          if(t.failed || t.first<t.ready || t.ready<t.begin || t.last<t.first)return 8;
          sms.push_back(t.sm);
          if(trial>=3)printf("{\"a_bytes\":%d,\"b_bytes\":%d,\"grid\":%d,\"requests\":%d,\"graph\":%d,\"trial\":%d,\"cta\":%d,\"sm\":%u,\"init_ns\":%llu,\"first_ns\":%llu,\"tail_ns\":%llu,\"checks_passed\":true}\n",ab,bb,grid,requests,graph_mode,trial,cta,t.sm,(unsigned long long)(t.ready-t.begin),(unsigned long long)(t.first-t.ready),(unsigned long long)(t.last-t.first));
        }
        std::sort(sms.begin(),sms.end());if(std::unique(sms.begin(),sms.end())!=sms.end())return 9;
      }
      CHECK(cudaGraphExecDestroy(executable));CHECK(cudaGraphDestroy(graph));
    }
    CHECK(cudaFree(a));CHECK(cudaFree(b));
  }
  CHECK(cudaStreamDestroy(stream));CHECK(cudaFree(output));
}
