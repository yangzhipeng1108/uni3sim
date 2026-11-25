// Independent operand queue / Tensor-consumer progress. No GEMM or output matrix.
#define main unused_queue_reference_main
#include "tma_queue_service.cu"
#undef main
#include "shared_tensor_activity_helper.cuh"
#include <chrono>
#include <cmath>
struct CoupledResult { unsigned failed,sm; };
__global__ void initialize_one(uint32_t* p,size_t n) {
 for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)p[i]=0x3c003c00;
}
__device__ void arrive_local(uint64_t* p) {
 asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(addr(p)):"memory");
}
template<int N> __global__ __launch_bounds__(384,1) void coupled_progress(
 const __grid_constant__ CUtensorMap a,const __grid_constant__ CUtensorMap b,
 CoupledResult* out,int depth,int rounds,int resident,int consumers,int math) {
 extern __shared__ __align__(1024) unsigned char operand_storage[];
 __shared__ __align__(8) uint64_t full[16],empty[16];
 constexpr int ab=8192,bb=N*128,stride=ab+bb;
 int wg=threadIdx.x/128,lane=threadIdx.x%128;
 if(threadIdx.x==0){
  for(int s=0;s<depth;++s){
   asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(addr(full+s)):"memory");
   asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(addr(empty+s)):"memory");
  }
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 }
 __syncthreads();
 float d[N/2]={};
 if(threadIdx.x==320){
  int slot=0,phase=0;
  for(int r=0;r<rounds;++r){
   wait(empty+slot,phase^1);
   asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(addr(full+slot)),"r"(ab+bb):"memory");
   int plane=blockIdx.x*kPlanesPerCTA+(resident?slot:r);
   load(operand_storage+slot*stride,&a,full+slot,plane,0);
   load(operand_storage+slot*stride+ab,&b,full+slot,plane,0);
   if(++slot==depth){slot=0;phase^=1;}
  }
 }else if(wg<consumers){
  int slot=0,phase=0;
  for(int r=0;r<rounds;++r){
   if(r%consumers==wg){
    if(lane%32==0)wait(full+slot,phase);
    // Every participating lane must acquire the TMA completion before
    // issuing asynchronous proxy reads. The warp wait is followed by WG sync.
    __syncwarp();
    if(consumers==1)asm volatile("bar.sync 1,128;":::"memory");
    else if(wg==0)asm volatile("bar.sync 1,128;":::"memory");
    else asm volatile("bar.sync 2,128;":::"memory");
    if(math){
     unsigned base=addr(operand_storage+slot*stride);wg_fence();
#pragma unroll
     for(int k=0;k<4;++k){
      uint64_t ad=make_desc(base+k*32,16,1024),bd=make_desc(base+ab+k*32,16,1024);
      MatrixRequest<N>::issue(d,ad,bd,1);
     }
     wg_commit();wg_wait0();
    }
    if(lane==0)arrive_local(empty+slot);
   }
   if(++slot==depth){slot=0;phase^=1;}
  }
  float sum=0;
#pragma unroll
  for(int i=0;i<N/2;++i)sum+=d[i];
  int count=(rounds+consumers-1-wg)/consumers;
  if(!isfinite(sum) || sum!=float(math?count*64*(N/2):0))atomicAdd(&out[blockIdx.x].failed,1u);
 }
 __syncthreads();
 // Full final resident payload check, independent of a Tensor sum.
 for(int s=0;s<depth;++s)for(int word=threadIdx.x;word<stride/4;word+=blockDim.x)
  if(reinterpret_cast<uint32_t*>(operand_storage+s*stride)[word]!=0x3c003c00)atomicAdd(&out[blockIdx.x].failed,1u);
 if(threadIdx.x==0){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));out[blockIdx.x].sm=sm;}
}
template<int N> void measure_coupled(){
 constexpr int ab=8192,bb=N*128;
 uint32_t *a=nullptr,*b=nullptr;CoupledResult* out=nullptr;
 size_t as=size_t(kPlanes)*ab,bs=size_t(kPlanes)*bb;
 CHECK(cudaMalloc(&a,as));CHECK(cudaMalloc(&b,bs));CHECK(cudaMalloc(&out,132*sizeof(CoupledResult)));
 initialize_one<<<4096,256>>>(a,as/4);initialize_one<<<4096,256>>>(b,bs/4);CHECK(cudaDeviceSynchronize());
 auto am=map(a,ab),bm=map(b,bb);
 CHECK(cudaFuncSetAttribute(coupled_progress<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
 cudaFuncAttributes attributes;CHECK(cudaFuncGetAttributes(&attributes,coupled_progress<N>));if(attributes.localSizeBytes)exit(3);
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
 auto warm=std::chrono::steady_clock::now();
 do{CHECK(cudaMemsetAsync(out,0,132*sizeof(CoupledResult),stream));coupled_progress<N><<<96,384,kShared,stream>>>(am,bm,out,12,128,0,2,1);CHECK(cudaStreamSynchronize(stream));}
 while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
 for(int grid:{32,96})for(int depth:{4,12})for(int resident:{0,1})for(int consumers:{1,2})for(int math:{0,1})for(int rounds:{32,128,512}){
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  for(int node=0;node<8;++node)coupled_progress<N><<<grid,384,kShared,stream>>>(am,bm,out,depth,rounds,resident,consumers,math);
  cudaGraph_t graph;CHECK(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t exec;CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  std::vector<double> times;std::vector<CoupledResult> host(grid);
  for(int trial=0;trial<11;++trial){
   CHECK(cudaMemsetAsync(out,0,132*sizeof(CoupledResult),stream));CHECK(cudaEventRecord(begin,stream));CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
   float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));if(!(ms>0))exit(4);
   CHECK(cudaMemcpy(host.data(),out,grid*sizeof(CoupledResult),cudaMemcpyDeviceToHost));std::vector<unsigned> sms;
   for(auto x:host){if(x.failed)exit(5);sms.push_back(x.sm);}
   std::sort(sms.begin(),sms.end());if(std::unique(sms.begin(),sms.end())!=sms.end())exit(6);
   if(trial>=2)times.push_back(double(ms)*1e6/8);
  }
  std::sort(times.begin(),times.end());
  printf("{\"instruction_n\":%d,\"grid\":%d,\"depth\":%d,\"resident\":%d,\"consumers\":%d,\"math\":%d,\"rounds\":%d,\"event_ns_p50\":%.9f,\"checks_passed\":true}\n",N,grid,depth,resident,consumers,math,rounds,times[4]);fflush(stdout);
  CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
 }
 CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaStreamDestroy(stream));CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(out));
}
int main(){measure_coupled<8>();measure_coupled<32>();measure_coupled<64>();return 0;}
