// Whole-Graph CUDA-event handoff controls. No cross-warp body timers, matrix math, or GEMM.
#define main unused_bare_queue_main
#include "tma_queue_service.cu"
#undef main
constexpr int kBoundaryShared=212992;
struct HandoffTiming {unsigned sm,failed;};
__device__ void handoff_wait(uint64_t* p,int phase,int hint) {
  if(hint)asm volatile("{.reg .pred done; L_%=: mbarrier.try_wait.parity.shared::cta.b64 done,[%0],%1,50000; @!done bra L_%=;}"::"r"(addr(p)),"r"(phase):"memory");
  else wait(p,phase);
}
__device__ void handoff_arrive(uint64_t* p) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(addr(p)):"memory");
}
__device__ void run_handoff(const CUtensorMap* a,const CUtensorMap* b,
    uint64_t* full,uint64_t* empty,unsigned char* storage,
    int ab,int bb,int depth,int rounds,int resident,int mode) {
  int warp=threadIdx.x/32,lane=threadIdx.x%32;
  int consumer_warps=mode>=4?4:8;
  int hint=(mode==3 || mode==5);
  if(threadIdx.x==0)for(int i=0;i<depth;++i) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(addr(full+i)):"memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(addr(empty+i)),"r"(consumer_warps):"memory");
  }
  if(threadIdx.x==0)asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  __syncthreads();
  if(mode==0) {
    if(threadIdx.x==320)copy_rounds(a,b,full,storage,ab,bb,depth,rounds,resident);
  } else if(warp==10 && lane==0) {
    int slot=0,phase=0;
    for(int step=0;step<rounds;++step) {
      handoff_wait(empty+slot,phase^1,hint);
      if(mode==1)handoff_arrive(full+slot);
      else {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(addr(full+slot)),"r"(ab+bb):"memory");
        int plane=blockIdx.x*kPlanesPerCTA+(resident?slot:(step&511));
        auto* destination=storage+slot*slot_storage(ab,bb);
        load(destination,a,full+slot,plane,0);load(destination+a_storage(ab),b,full+slot,plane,0);
      }
      if(++slot==depth){slot=0;phase^=1;}
    }
  } else if(warp<consumer_warps) {
    int slot=0,phase=0;
    for(int step=0;step<rounds;++step) {
      if(lane==0)handoff_wait(full+slot,phase,hint);
      __syncwarp();
      if(lane==0)handoff_arrive(empty+slot);
      if(++slot==depth){slot=0;phase^=1;}
    }
  }
  // Consumers have observed every full completion before reuse or exit.
  __syncthreads();
}
__global__ __launch_bounds__(384,1) void handoff_queue(
    const __grid_constant__ CUtensorMap a,const __grid_constant__ CUtensorMap b,
    HandoffTiming* out,int ab,int bb,int depth,int rounds,int resident,int mode) {
  extern __shared__ __align__(128) unsigned char storage[];
  __shared__ __align__(8) uint64_t full[16],empty[16];
  __shared__ unsigned again;
  __shared__ unsigned long long actual,expected;
  __shared__ unsigned ax,ex;
  run_handoff(&a,&b,full,empty,storage,ab,bb,depth,rounds,resident,mode);
  if(threadIdx.x==0) {
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));out[blockIdx.x].sm=sm;
  }
  __syncthreads();
  if(mode!=1)for(int slot=0;slot<depth;++slot) {
    if(threadIdx.x==0){actual=expected=0;ax=ex=0;}__syncthreads();
    int last=rounds-1-(rounds-1-slot)%depth;
    int plane=blockIdx.x*kPlanesPerCTA+(resident?slot:(last&511));
    unsigned long long av=0,ev=0;unsigned a_xor=0,e_xor=0;
    for(int word=threadIdx.x;word<(ab+bb)/4;word+=blockDim.x) {
      bool is_a=word<ab/4;int local=is_a?word:word-ab/4,bytes=is_a?ab:bb;
      auto* data=reinterpret_cast<uint32_t*>(storage+slot*slot_storage(ab,bb)+(is_a?0:a_storage(ab)));
      unsigned v=data[local],e=value_at(unsigned(size_t(plane)*(bytes/4)+local),is_a?1:2);
      av+=v;ev+=e;a_xor^=v;e_xor^=e;
    }
    atomicAdd(&actual,av);atomicAdd(&expected,ev);atomicXor(&ax,a_xor);atomicXor(&ex,e_xor);__syncthreads();
    if(threadIdx.x==0 && (actual!=expected || ax!=ex))++out[blockIdx.x].failed;
    __syncthreads();
  }
}
int main() {
  HandoffTiming* output=nullptr;CHECK(cudaMalloc(&output,132*sizeof(HandoffTiming)));
  CHECK(cudaFuncSetAttribute(handoff_queue,cudaFuncAttributeMaxDynamicSharedMemorySize,kBoundaryShared));
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
  cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
  for(auto pair:std::vector<std::pair<int,int>>{{128,8192},{1024,16384}}) {
    int ab=pair.first,bb=pair.second;uint32_t *a=nullptr,*b=nullptr;
    size_t as=size_t(kPlanes)*ab,bs=size_t(kPlanes)*bb;
    CHECK(cudaMalloc(&a,as));CHECK(cudaMalloc(&b,bs));
    initialize<<<4096,256>>>(a,as/4,1);initialize<<<4096,256>>>(b,bs/4,2);CHECK(cudaDeviceSynchronize());
    auto am=map(a,ab),bm=map(b,bb);
    auto warm=std::chrono::steady_clock::now();
    do{CHECK(cudaMemsetAsync(output,0,132*sizeof(HandoffTiming),stream));
      handoff_queue<<<132,384,kBoundaryShared,stream>>>(am,bm,output,ab,bb,12,128,0,0);
      CHECK(cudaStreamSynchronize(stream));
    }while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
    for(int depth:{4,12})for(int grid:{32,132})for(int resident:{0,1})for(int rounds:{32,128,512}) {
      if(depth*slot_storage(ab,bb)>kBoundaryShared)return 4;
      for(int mode:{0,2,3,4,5}) {
        CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int node=0;node<16;++node)
          handoff_queue<<<grid,384,kBoundaryShared,stream>>>(am,bm,output,ab,bb,depth,rounds,resident,mode);
        cudaGraph_t graph;CHECK(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        std::vector<double> samples;std::vector<HandoffTiming> host(grid);
        for(int trial=0;trial<11;++trial) {
          CHECK(cudaMemsetAsync(output,0,132*sizeof(HandoffTiming),stream));
          CHECK(cudaEventRecord(begin,stream));CHECK(cudaGraphLaunch(exec,stream));
          CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
          float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));if(!(ms>0))return 8;
          CHECK(cudaMemcpy(host.data(),output,grid*sizeof(HandoffTiming),cudaMemcpyDeviceToHost));
          std::vector<unsigned> sms;
          for(auto x:host){if(x.failed)return 6;sms.push_back(x.sm);}
          std::sort(sms.begin(),sms.end());if(std::unique(sms.begin(),sms.end())!=sms.end())return 7;
          if(trial>=2)samples.push_back(double(ms)*1e6/16);
        }
        std::sort(samples.begin(),samples.end());
        printf("{\"a_bytes\":%d,\"b_bytes\":%d,\"depth\":%d,\"grid\":%d,\"resident\":%d,\"mode\":%d,\"rounds\":%d,\"event_ns_p50\":%.9f,\"event_ns_p10\":%.9f,\"event_ns_p90\":%.9f,\"graph_nodes\":16,\"checks_passed\":true}\n",ab,bb,depth,grid,resident,mode,rounds,samples[4],samples[0],samples[8]);fflush(stdout);
        CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
      }
    }
    CHECK(cudaFree(a));CHECK(cudaFree(b));
  }
  CHECK(cudaFree(output));CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaStreamDestroy(stream));return 0;
}
