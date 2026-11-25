// Independent 3D TMA ring: no MMA, matrix output, library or work queue.
// Remote empty-stage releases replace per-step cluster-wide synchronizations.
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace cg = cooperative_groups;
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
constexpr int kPrivateBytes=8192, kSharedBytes=2048, kDepth=16, kRounds=128;
constexpr int kPlanes=132*kRounds*4;
__device__ uint32_t addr(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ void init(uint64_t* p, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"(addr(p)),"r"(n):"memory");
}
__device__ void wait(uint64_t* p, int phase) {
  asm volatile("{ .reg .pred done; L_%=: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; @!done bra L_%=; }"::"r"(addr(p)),"r"(phase):"memory");
}
__device__ void arrive(uint64_t* p) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"::"r"(addr(p)):"memory");
}
__device__ void expect(uint64_t* p) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(addr(p)),"r"(kPrivateBytes+kSharedBytes):"memory");
}
__device__ void load(void* p, const CUtensorMap* map, uint64_t* bar, int plane) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {0, 0, %3}], [%2];"::"r"(addr(p)),"l"(map),"r"(addr(bar)),"r"(plane):"memory");
}
__device__ void multicast(void* p, const CUtensorMap* map, uint64_t* bar, int plane, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {0, 0, %3}], [%2], %4;"::"r"(addr(p)),"l"(map),"r"(addr(bar)),"r"(plane),"h"(mask):"memory");
}
__global__ __launch_bounds__(384,1) void ring(
    const __grid_constant__ CUtensorMap priv, const __grid_constant__ CUtensorMap common,
    uint32_t* sums, int depth, int rounds, int cast, int cached, int base, int prearmed) {
  extern __shared__ __align__(128) uint8_t storage[];
  __shared__ __align__(8) uint64_t full[kDepth], empty[kDepth], ready[kDepth];
  auto cluster=cg::this_cluster();
  const int rank=cluster.block_rank(), size=cluster.num_blocks();
  const int warp=threadIdx.x/32, lane=threadIdx.x%32;
  if (threadIdx.x==0) {
    for(int s=0;s<depth;++s) {
      init(&full[s],1); init(&empty[s],8*(rank==0?size:1)); init(&ready[s],size);
      if(prearmed && s<rounds) expect(&full[s]);
    }
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  cluster.sync();
  if(warp==10 && lane==0) {
    for(int step=0;step<rounds;++step) {
      const int slot=step%depth, phase=(step/depth)&1;
      wait(&empty[slot],phase^1);
      if(!prearmed) expect(&full[slot]);
      const int index=cached? step%8: step;
      load(storage+slot*(kPrivateBytes+kSharedBytes),&priv,&full[slot],base+blockIdx.x*kRounds+index);
      if(!prearmed) {
        arrive(cluster.map_shared_rank(&ready[slot],0));
        if(rank==0) wait(&ready[slot],phase);
      }
      uint8_t* destination=storage+slot*(kPrivateBytes+kSharedBytes)+kPrivateBytes;
      const int plane=base+(blockIdx.x/size)*kRounds+index;
      if(cast) { if(rank==0) multicast(destination,&common,&full[slot],plane,uint16_t((1u<<size)-1)); }
      else load(destination,&common,&full[slot],plane);
    }
  } else if(warp<8) {
    uint32_t sum=0;
    for(int step=0;step<rounds;++step) {
      const int slot=step%depth, phase=(step/depth)&1;
      wait(&full[slot],phase);
      const uint8_t* p=storage+slot*(kPrivateBytes+kSharedBytes);
      sum+=p[threadIdx.x]+p[kPrivateBytes+threadIdx.x];
      // The empty-stage release carries this arm to the producer. Rank zero
      // waits for all ranks, so every multicast destination is ready.
      if(prearmed && threadIdx.x==0 && step+depth<rounds) expect(&full[slot]);
      __syncwarp();
      if(lane==0) { arrive(&empty[slot]); if(rank!=0) arrive(cluster.map_shared_rank(&empty[slot],0)); }
    }
    if(threadIdx.x==0) sums[blockIdx.x]=sum;
  }
  // Keep all remote barrier storage alive through the final releases.
  cluster.sync();
}
static CUtensorMap map(void* p,int rows) {
  void* function=nullptr; cudaDriverEntryPointQueryResult status;
  CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&function,cudaEnableDefault,&status));
  if(function==nullptr || status!=cudaDriverEntryPointSuccess) exit(2);
  auto encode=reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function);
  cuuint64_t dims[3]={128,cuuint64_t(rows),kPlanes}, strides[2]={128,cuuint64_t(rows*128)};
  cuuint32_t box[3]={128,cuuint32_t(rows),1}, step[3]={1,1,1}; CUtensorMap result;
  if(encode(&result,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,p,dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS) exit(3);
  return result;
}
__device__ uint64_t timer() {
  uint64_t value; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory"); return value;
}
__global__ __launch_bounds__(384,1) void boundary(uint64_t* output,int depth,int mode) {
  extern __shared__ __align__(128) uint8_t storage[];
  __shared__ __align__(8) uint64_t full[kDepth], empty[kDepth];
  auto cluster=cg::this_cluster();
  uint64_t begin=0, cycles=0;
  if(threadIdx.x==0) {
    begin=timer(); cycles=clock64();
    if(mode&1) {
      for(int s=0;s<depth;++s) { init(&full[s],1); init(&empty[s],8); }
      asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
    }
  }
  if(mode&2) cluster.sync();
  if(threadIdx.x==0) {
    cycles=clock64()-cycles; const uint64_t end=timer();
    output[4*blockIdx.x]=end-begin; output[4*blockIdx.x+1]=cycles;
    output[4*blockIdx.x+2]=begin; output[4*blockIdx.x+3]=end;
  }
}
static int boundary_measure() {
  uint64_t* output=nullptr; CHECK(cudaMalloc(&output,132*4*sizeof(uint64_t)));
  CHECK(cudaFuncSetAttribute(boundary,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
  cudaStream_t stream; CHECK(cudaStreamCreate(&stream)); cudaEvent_t begin,end;
  CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
  auto launch=[&](int grid,int size,int depth,int mode) {
    cudaLaunchConfig_t config={}; config.gridDim=dim3(grid); config.blockDim=dim3(384);
    config.dynamicSmemBytes=196608; config.stream=stream;
    cudaLaunchAttribute a={}; a.id=cudaLaunchAttributeClusterDimension;
    a.val.clusterDim.x=size; a.val.clusterDim.y=1; a.val.clusterDim.z=1; config.attrs=&a; config.numAttrs=1;
    CHECK(cudaLaunchKernelEx(&config,boundary,output,depth,mode));
  };
  auto start=std::chrono::steady_clock::now();
  do { launch(132,2,16,3); CHECK(cudaStreamSynchronize(stream)); }
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<2.0);
  for(int grid:{32,72,132}) for(int size:{1,2,4,8}) for(int depth:{4,16}) for(int mode:{0,1,2,3}) {
    if(grid%size) continue;
    cudaGraph_t graph; CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int replay=0;replay<64;++replay) launch(grid,size,depth,mode);
    CHECK(cudaStreamEndCapture(stream,&graph)); cudaGraphExec_t executable;
    CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    std::vector<double> event,typical,maxima; std::vector<uint64_t> host(132*4);
    for(int trial=0;trial<11;++trial) {
      CHECK(cudaEventRecord(begin,stream)); CHECK(cudaGraphLaunch(executable,stream)); CHECK(cudaEventRecord(end,stream));
      CHECK(cudaEventSynchronize(end)); float ms; CHECK(cudaEventElapsedTime(&ms,begin,end));
      CHECK(cudaMemcpy(host.data(),output,132*4*sizeof(uint64_t),cudaMemcpyDeviceToHost));
      std::vector<uint64_t> intervals;
      for(int cta=0;cta<grid;++cta) {
        if(host[4*cta+3]<host[4*cta+2] || host[4*cta]!=host[4*cta+3]-host[4*cta+2]) return 5;
        intervals.push_back(host[4*cta]);
      }
      std::sort(intervals.begin(),intervals.end());
      if(trial>=2) { event.push_back(ms*1e6/64); typical.push_back(intervals[grid/2]); maxima.push_back(intervals.back()); }
    }
    std::sort(event.begin(),event.end()); std::sort(typical.begin(),typical.end()); std::sort(maxima.begin(),maxima.end());
    printf("{\"grid\":%d,\"cluster\":%d,\"depth\":%d,\"mode\":%d,\"threads\":384,\"dynamic_smem_bytes\":196608,\"median_event_ns\":%.3f,\"median_cta_interval_ns\":%.3f,\"median_max_cta_interval_ns\":%.3f,\"timestamp_check_ok\":true}\n",grid,size,depth,mode,event[4],typical[4],maxima[4]);
    fflush(stdout); CHECK(cudaGraphExecDestroy(executable)); CHECK(cudaGraphDestroy(graph));
  }
  CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end)); CHECK(cudaStreamDestroy(stream)); CHECK(cudaFree(output)); return 0;
}
int main(int argc,char**) {
  if(argc==2) return boundary_measure();
  uint8_t *a=nullptr,*b=nullptr,*evict=nullptr; uint32_t* sums=nullptr;
  CHECK(cudaMalloc(&a,size_t(kPlanes)*kPrivateBytes)); CHECK(cudaMalloc(&b,size_t(kPlanes)*kSharedBytes));
  CHECK(cudaMalloc(&evict,256ull*1024*1024)); CHECK(cudaMalloc(&sums,132*sizeof(uint32_t)));
  CHECK(cudaMemset(a,1,size_t(kPlanes)*kPrivateBytes)); CHECK(cudaMemset(b,2,size_t(kPlanes)*kSharedBytes));
  CUtensorMap priv=map(a,64), common=map(b,16);
  CHECK(cudaFuncSetAttribute(ring,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
  cudaStream_t stream; CHECK(cudaStreamCreate(&stream)); cudaEvent_t begin,end;
  CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
  auto launch=[&](int grid,int cluster,int depth,int rounds,int cast,int cached,int base,int prearmed) {
    cudaLaunchConfig_t config={}; config.gridDim=dim3(grid); config.blockDim=dim3(384);
    config.dynamicSmemBytes=196608; config.stream=stream;
    cudaLaunchAttribute attribute={}; attribute.id=cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x=cluster; attribute.val.clusterDim.y=1; attribute.val.clusterDim.z=1;
    config.attrs=&attribute; config.numAttrs=1;
    CHECK(cudaLaunchKernelEx(&config,ring,priv,common,sums,depth,rounds,cast,cached,base,prearmed));
  };
  auto warm=std::chrono::steady_clock::now();
  do { launch(132,2,4,128,1,0,0,1); CHECK(cudaStreamSynchronize(stream)); }
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
  for(int grid:{32,132}) for(int cluster:{1,2,4}) for(int depth:{4,16})
  for(int rounds:{0,8,32,128}) for(int cast:{0,1}) for(int cached:{0,1}) for(int prearmed:{0,1}) {
    CHECK(cudaMemset(sums,0xFF,132*sizeof(uint32_t)));
    cudaGraph_t graph; CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int replay=0;replay<32;++replay) launch(grid,cluster,depth,rounds,cast,cached,cached?0:(replay%4)*132*kRounds,prearmed);
    CHECK(cudaStreamEndCapture(stream,&graph)); cudaGraphExec_t executable;
    CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    std::vector<double> times;
    for(int trial=0;trial<9;++trial) {
      if(cached) { CHECK(cudaGraphLaunch(executable,stream)); }
      else { CHECK(cudaMemsetAsync(evict,trial,256ull*1024*1024,stream)); }
      CHECK(cudaEventRecord(begin,stream)); CHECK(cudaGraphLaunch(executable,stream)); CHECK(cudaEventRecord(end,stream));
      CHECK(cudaEventSynchronize(end)); float ms; CHECK(cudaEventElapsedTime(&ms,begin,end));
      if(trial>=2) times.push_back(ms*1e6/32);
    }
    std::vector<uint32_t> host(grid); CHECK(cudaMemcpy(host.data(),sums,grid*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    for(uint32_t value:host) if(value!=uint32_t(3*rounds)) { fprintf(stderr,"checksum %u expected %d\n",value,3*rounds); return 5; }
    std::sort(times.begin(),times.end());
    printf("{\"grid\":%d,\"cluster\":%d,\"depth\":%d,\"rounds\":%d,\"multicast\":%d,\"cached\":%d,\"prearmed\":%d,\"private_bytes\":8192,\"shared_bytes\":2048,\"median_ns\":%.3f,\"min_ns\":%.3f,\"max_ns\":%.3f,\"checksum_ok\":true}\n",grid,cluster,depth,rounds,cast,cached,prearmed,times[3],times.front(),times.back());
    fflush(stdout); CHECK(cudaGraphExecDestroy(executable)); CHECK(cudaGraphDestroy(graph));
  }
  CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end)); CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(sums)); CHECK(cudaFree(evict)); CHECK(cudaFree(b)); CHECK(cudaFree(a));
}
