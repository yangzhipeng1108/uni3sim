// Bare TMA queue capacity: one issuing lane, no matrix or consumer protocol.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef TMA_CTA_THREADS
#define TMA_CTA_THREADS 384
#endif
#ifndef TMA_BALANCED_PAIRS
#define TMA_BALANCED_PAIRS 0
#endif
#ifndef TMA_CP_ASYNC_PAIRS
#define TMA_CP_ASYNC_PAIRS 0
#endif
#ifndef TMA_BALANCED_ROW_GAP
#define TMA_BALANCED_ROW_GAP 1
#endif
#ifndef TMA_BALANCED_STRIDE_SCREEN
#define TMA_BALANCED_STRIDE_SCREEN 0
#endif
#ifndef TMA_STREAM_PACKETS
#define TMA_STREAM_PACKETS 512
#endif
#ifndef TMA_FINITE_PROGRESS
#define TMA_FINITE_PROGRESS 0
#endif
#ifndef TMA_MULTICAST_SELF
#define TMA_MULTICAST_SELF 0
#endif
#ifndef TMA_ROW_STRIDE_BYTES
#define TMA_ROW_STRIDE_BYTES 128
#endif
#ifndef TMA_SHARED_OPERAND_MASK
#define TMA_SHARED_OPERAND_MASK 0
#endif
#ifndef TMA_COUNTER_CAPTURE
#define TMA_COUNTER_CAPTURE 0
#endif
#ifndef TMA_ACTIVITY_PRELUDE
#define TMA_ACTIVITY_PRELUDE 0
#endif
static_assert(TMA_ACTIVITY_PRELUDE>=0 && TMA_ACTIVITY_PRELUDE<=2);
static_assert(!TMA_CP_ASYNC_PAIRS || (TMA_BALANCED_PAIRS && TMA_ACTIVITY_PRELUDE == 2));
static_assert(TMA_BALANCED_ROW_GAP>=1);
static_assert(TMA_BALANCED_ROW_GAP==1 || (TMA_BALANCED_PAIRS && !TMA_CP_ASYNC_PAIRS));
static_assert(!TMA_BALANCED_STRIDE_SCREEN || (TMA_BALANCED_PAIRS && TMA_ACTIVITY_PRELUDE==2));
static_assert(TMA_STREAM_PACKETS>0 && (TMA_STREAM_PACKETS&(TMA_STREAM_PACKETS-1))==0);
static_assert(TMA_SHARED_OPERAND_MASK>=0 && TMA_SHARED_OPERAND_MASK<=3);
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
static_assert(TMA_ROW_STRIDE_BYTES>=128 && TMA_ROW_STRIDE_BYTES%128==0);
constexpr int kShared=212992,kPanelsPerPlane=TMA_ROW_STRIDE_BYTES/128;
constexpr int kPlanesPerCTA=(TMA_STREAM_PACKETS+kPanelsPerPlane-1)/kPanelsPerPlane;
constexpr int kPlaneDomains=TMA_ACTIVITY_PRELUDE?2:1;
constexpr int kPlanes=132*kPlanesPerCTA*kPlaneDomains;
__host__ __device__ int a_storage(int a_bytes){return ((a_bytes+1023)/1024)*1024;}
__host__ __device__ int slot_storage(int a_bytes,int b_bytes){return a_storage(a_bytes)+b_bytes;}
struct Timing {uint64_t c0,t0,c1,t1,prelude_cycles,prelude_ns;unsigned failed,sm;};
__host__ __device__ uint32_t value_at(uint32_t word,uint32_t seed) {
  uint32_t state=word*747796405u+seed*2891336453u;
  state=((state>>((state>>28)+4))^state)*277803737u;return (state>>22)^state;
}
__global__ void initialize(uint32_t* data,size_t words,unsigned seed) {
  for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)
    data[i]=value_at(unsigned(i),seed);
}
__device__ uint32_t addr(const void* p){return unsigned(__cvta_generic_to_shared(p));}
__device__ uint64_t timer(){uint64_t v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v));return v;}
__device__ void wait(uint64_t* p,int phase) {
  asm volatile("{.reg .pred done; L_%=: mbarrier.try_wait.parity.shared::cta.b64 done,[%0],%1; @!done bra L_%=;}"::"r"(addr(p)),"r"(phase):"memory");
}
__device__ void load(void* p,const CUtensorMap* map,uint64_t* barrier,int plane,int column) {
#if TMA_BALANCED_PAIRS
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, 0, 0, %3}], [%4];"
      ::"r"(addr(p)),"l"(map),"r"(column),"r"(plane),"r"(addr(barrier)):"memory");
#else
#if TMA_MULTICAST_SELF
  // Cluster size one, mask bit zero: identical source and destination bytes.
  // This isolates opcode routing, not multi-CTA multicast fabric service.
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%4,0,%3}],[%2],%5;"::"r"(addr(p)),"l"(map),"r"(addr(barrier)),"r"(plane),"r"(column),"h"(uint16_t(1)):"memory");
#else
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%4,0,%3}],[%2];"::"r"(addr(p)),"l"(map),"r"(addr(barrier)),"r"(plane),"r"(column):"memory");
#endif
#endif
}
__device__ void copy_rounds(const CUtensorMap* a,const CUtensorMap* b,uint64_t* full,
    unsigned char* storage,int a_bytes,int b_bytes,int depth,int rounds,int resident,
    int plane_offset=0) {
  for(int i=0;i<depth;++i)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(addr(full+i)):"memory");
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  int slot=0,phase=0;
  for(int step=0;step<rounds;++step) {
    if(step>=depth)wait(full+slot,phase^1);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(addr(full+slot)),"r"(a_bytes+b_bytes):"memory");
    // Traverse consecutive 128B K panels across each strided row before
    // moving to a fresh row group. Adjacent-panel reuse remains possible.
    int logical=resident?slot:(step&(TMA_STREAM_PACKETS-1));
    int a_plane=((TMA_SHARED_OPERAND_MASK&1)?0:int(blockIdx.x))*kPlanesPerCTA*kPlaneDomains+plane_offset+logical/kPanelsPerPlane;
    int b_plane=((TMA_SHARED_OPERAND_MASK&2)?0:int(blockIdx.x))*kPlanesPerCTA*kPlaneDomains+plane_offset+logical/kPanelsPerPlane;
    int column=(logical%kPanelsPerPlane)*128;
    auto* destination=storage+slot*slot_storage(a_bytes,b_bytes);
    load(destination,a,full+slot,a_plane,column);load(destination+a_storage(a_bytes),b,full+slot,b_plane,column);
    if(++slot==depth){slot=0;phase^=1;}
  }
  for(int i=0;i<depth;++i)wait(full+i,i<slot?phase:phase^1);
}
#if TMA_CP_ASYNC_PAIRS
// Four producer warps issue two independent equal packets. Two commit groups
// per round and at most two round slots stay within the eight-group ISA limit.
// No Tensor, consumer handoff, page walk or operator is represented.
__device__ void cp_async_rounds(const uint32_t* a,const uint32_t* b,
    unsigned char* storage,int a_bytes,int b_bytes,int depth,int rounds,int resident) {
  if(threadIdx.x>=128)return;
  for(int step=0;step<rounds;++step) {
    if(step>=depth) {
      if(depth==1)asm volatile("cp.async.wait_group 0;":::"memory");
      else asm volatile("cp.async.wait_group 2;":::"memory");
    }
    int slot=step%depth,logical=resident?slot:(step&(TMA_STREAM_PACKETS-1));
    size_t packet=size_t(blockIdx.x)*kPlanesPerCTA*kPlaneDomains+logical;
    auto* destination=storage+slot*slot_storage(a_bytes,b_bytes);
    for(int offset=threadIdx.x*16;offset<a_bytes;offset+=128*16)
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::
        "r"(addr(destination+offset)),"l"(reinterpret_cast<const unsigned char*>(a)+packet*a_bytes+offset):"memory");
    asm volatile("cp.async.commit_group;":::"memory");
    for(int offset=threadIdx.x*16;offset<b_bytes;offset+=128*16)
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::
        "r"(addr(destination+a_storage(a_bytes)+offset)),
        "l"(reinterpret_cast<const unsigned char*>(b)+packet*b_bytes+offset):"memory");
    asm volatile("cp.async.commit_group;":::"memory");
  }
  asm volatile("cp.async.wait_group 0;":::"memory");
}
#endif
__global__ __launch_bounds__(TMA_CTA_THREADS,1) void queue_service(
    const __grid_constant__ CUtensorMap a,const __grid_constant__ CUtensorMap b,
    Timing* result,int a_bytes,int b_bytes,int depth,int rounds,int resident
#if TMA_CP_ASYNC_PAIRS
    ,const uint32_t* a_data,const uint32_t* b_data
#endif
    ) {
  extern __shared__ __align__(128) unsigned char storage[];
  __shared__ __align__(8) uint64_t full[16];
  __shared__ unsigned long long actual_sum,expected_sum;
  __shared__ unsigned actual_xor,expected_xor;
  uint64_t prelude_cycles=0,prelude_ns=0;
  if(threadIdx.x==0) {
    result[blockIdx.x].failed=0;
    if(!TMA_COUNTER_CAPTURE) {
      uint64_t begin=timer(),begin_cycles=clock64();
#if TMA_ACTIVITY_PRELUDE==2
      unsigned state=blockIdx.x+1;
      do {
        for(int i=0;i<128;++i)
          asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;":"+r"(state));
      } while(timer()-begin<3000000);
      asm volatile(""::"r"(state):"memory");
#else
      do{copy_rounds(&a,&b,full,storage,a_bytes,b_bytes,depth,512,
          TMA_ACTIVITY_PRELUDE?0:resident,
          TMA_ACTIVITY_PRELUDE?kPlanesPerCTA:0);}while(timer()-begin<3000000);
#endif
      prelude_ns=timer()-begin;prelude_cycles=clock64()-begin_cycles;
    }
  }
  cooperative_groups::this_grid().sync();
#if TMA_CP_ASYNC_PAIRS
  if(threadIdx.x==0) {
    result[blockIdx.x].c0=clock64();result[blockIdx.x].t0=timer();
  }
  cp_async_rounds(a_data,b_data,storage,a_bytes,b_bytes,depth,rounds,resident);
  __syncthreads();
  if(threadIdx.x==0) {
    result[blockIdx.x].t1=timer();result[blockIdx.x].c1=clock64();
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
    result[blockIdx.x].sm=sm;
    result[blockIdx.x].prelude_cycles=prelude_cycles;
    result[blockIdx.x].prelude_ns=prelude_ns;
  }
#else
  if(threadIdx.x==0) {
    uint64_t c0=clock64(),t0=timer();
    copy_rounds(&a,&b,full,storage,a_bytes,b_bytes,depth,rounds,resident);
    uint64_t t1=timer(),c1=clock64();unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
    result[blockIdx.x].c0=c0;result[blockIdx.x].t0=t0;result[blockIdx.x].c1=c1;result[blockIdx.x].t1=t1;
    result[blockIdx.x].sm=sm;
    result[blockIdx.x].prelude_cycles=prelude_cycles;
    result[blockIdx.x].prelude_ns=prelude_ns;
  }
#endif
  __syncthreads();
  // 128B swizzle permutes 16B chunks. Per-stage sum+xor validates the entire
  // copied payload without assuming a physical swizzle-address formula.
  for(int slot=0;slot<depth;++slot) {
    if(threadIdx.x==0){actual_sum=expected_sum=0;actual_xor=expected_xor=0;}
    __syncthreads();
    int last=rounds-1-(rounds-1-slot)%depth;
    int logical=resident?slot:(last&(TMA_STREAM_PACKETS-1));
    int column=(logical%kPanelsPerPlane)*128;
    unsigned long long actual=0,expected=0;unsigned ax=0,ex=0;
    for(int word=threadIdx.x;word<(a_bytes+b_bytes)/4;word+=blockDim.x) {
      bool operand_a=word<a_bytes/4;
      bool shared_operand=(TMA_SHARED_OPERAND_MASK&(operand_a?1:2))!=0;
      int plane=(shared_operand?0:int(blockIdx.x))*kPlanesPerCTA*kPlaneDomains+logical/kPanelsPerPlane;
      int local_word=operand_a?word:word-a_bytes/4;
      int operand_rows=(operand_a?a_bytes:b_bytes)/128;
      size_t global_word=(size_t(plane)*operand_rows+local_word/32)
          *(TMA_ROW_STRIDE_BYTES/4)+column/4+local_word%32;
#if TMA_BALANCED_PAIRS
      // Each logical row is256B (two128B halves). A row gap leaves
      // independent untouched bytes between rows, without extra demand.
      global_word=size_t(plane)*(operand_a?a_bytes:b_bytes)/4*TMA_BALANCED_ROW_GAP
          +size_t(local_word/64)*64*TMA_BALANCED_ROW_GAP+local_word%64;
#endif
      auto* words=reinterpret_cast<const uint32_t*>(storage+slot*slot_storage(a_bytes,b_bytes)
          +(operand_a?0:a_storage(a_bytes)));
      unsigned v=words[local_word],e=value_at(unsigned(global_word),operand_a?1:2);
      actual+=v;expected+=e;ax^=v;ex^=e;
    }
    atomicAdd(&actual_sum,actual);atomicAdd(&expected_sum,expected);
    atomicXor(&actual_xor,ax);atomicXor(&expected_xor,ex);__syncthreads();
    if(threadIdx.x==0 && (actual_sum!=expected_sum || actual_xor!=expected_xor))result[blockIdx.x].failed+=1;
    __syncthreads();
  }
}
static CUtensorMap map(void* pointer,int bytes) {
  void* function=nullptr;cudaDriverEntryPointQueryResult status;
  CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&function,cudaEnableDefault,&status));
  if(!function || status!=cudaDriverEntryPointSuccess)exit(3);
  auto encode=reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function);
#if TMA_BALANCED_PAIRS
  if(bytes%256 || TMA_ROW_STRIDE_BYTES!=128)exit(4);
  // 128B x2 xrows;only the global row pitch changes. Dense default is
  // identical. Every box axis stays within256,including44KiB packets.
  cuuint64_t dims[4]={128,2,cuuint64_t(bytes/256),kPlanes};
  cuuint64_t strides[3]={128,256*TMA_BALANCED_ROW_GAP,cuuint64_t(bytes)*TMA_BALANCED_ROW_GAP};
  cuuint32_t box[4]={128,2,cuuint32_t(bytes/256),1},steps[4]={1,1,1,1};CUtensorMap result;
  if(encode(&result,CU_TENSOR_MAP_DATA_TYPE_UINT8,4,pointer,dims,strides,box,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)exit(4);
#else
  cuuint64_t dims[3]={TMA_ROW_STRIDE_BYTES,cuuint64_t(bytes/128),kPlanes};
  cuuint64_t strides[2]={TMA_ROW_STRIDE_BYTES,cuuint64_t(bytes)*kPanelsPerPlane};
  cuuint32_t box[3]={128,cuuint32_t(bytes/128),1},steps[3]={1,1,1};CUtensorMap result;
  if(encode(&result,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,pointer,dims,strides,box,steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)exit(4);
#endif
  return result;
}
int main(int argc,char** argv) {
  // Optional single-cell selection is only for independent counter capture.
  // It does not change the default timing sweep or select any production case.
  if(argc!=1 && argc!=7)return 8;
  int selected_cells=0;
  int cooperative=0;CHECK(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,0));if(!cooperative)return 5;
  Timing* output=nullptr;CHECK(cudaMalloc(&output,132*sizeof(Timing)));
  unsigned char* eviction=nullptr;
  if(TMA_ACTIVITY_PRELUDE)CHECK(cudaMalloc(&eviction,256ull*1024*1024));
  CHECK(cudaFuncSetAttribute(queue_service,cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
#if TMA_BALANCED_PAIRS
#if TMA_BALANCED_STRIDE_SCREEN
  const std::vector<std::pair<int,int>> payloads={{32768,32768},{45056,45056}};
#else
  const std::vector<std::pair<int,int>> payloads={{16384,16384},{24576,24576},
      {32768,32768},{40960,40960},{45056,45056}};
#endif
#else
  const auto payloads=TMA_ACTIVITY_PRELUDE?
      std::vector<std::pair<int,int>>{{1024,4096},{1024,8192},{1024,16384},{1024,32768}}:
      std::vector<std::pair<int,int>>{{128,8192},{1024,8192},{4096,4096},{1024,16384},{1024,32768}};
#endif
  for(auto pair:payloads) {
    int a_bytes=pair.first,b_bytes=pair.second;uint32_t *a=nullptr,*b=nullptr;
    size_t a_size=size_t(kPlanes)*a_bytes*kPanelsPerPlane;
    size_t b_size=size_t(kPlanes)*b_bytes*kPanelsPerPlane;
#if TMA_BALANCED_PAIRS
    a_size*=TMA_BALANCED_ROW_GAP;b_size*=TMA_BALANCED_ROW_GAP;
#endif
    CHECK(cudaMalloc(&a,a_size));CHECK(cudaMalloc(&b,b_size));
    initialize<<<4096,256>>>(a,a_size/4,1);initialize<<<4096,256>>>(b,b_size/4,2);CHECK(cudaDeviceSynchronize());
    auto a_map=map(a,a_bytes),b_map=map(b,b_bytes);
#if TMA_CP_ASYNC_PAIRS
    const std::vector<int> depths={1,2};
#elif TMA_BALANCED_PAIRS
#if TMA_BALANCED_STRIDE_SCREEN
    const std::vector<int> depths={2};
#else
    const std::vector<int> depths={1,2,3,4,6};
#endif
#else
    const std::vector<int> depths={4,6,8,12,16};
#endif
    for(int depth:depths) {
      if(depth*slot_storage(a_bytes,b_bytes)>kShared)continue;
#if TMA_BALANCED_PAIRS
#if TMA_BALANCED_STRIDE_SCREEN
      const std::vector<int> grids={1,32,132};
      const std::vector<int> lengths={2,4,8,16,32,64};
#else
      const std::vector<int> grids={32,64,96,132};
      const std::vector<int> lengths={1,2,4,8,16,32,64,128,256};
#endif
#elif TMA_FINITE_PROGRESS
      const std::vector<int> grids={56,80,96,112,132};
      const std::vector<int> lengths={16,32,64,128,256,512};
#else
      const std::vector<int> grids=TMA_ACTIVITY_PRELUDE?
          std::vector<int>{32,64,96,132}:std::vector<int>{32,132};
      const std::vector<int> lengths=TMA_ACTIVITY_PRELUDE?
          std::vector<int>{16,32,64,128,512}:std::vector<int>{128,512};
#endif
      for(int grid:grids)for(int resident:{0,1})for(int rounds:lengths) {
        if(TMA_BALANCED_PAIRS && rounds<depth)continue;
        if(argc==7 && (a_bytes!=std::atoi(argv[1]) || b_bytes!=std::atoi(argv[2])
            || depth!=std::atoi(argv[3]) || grid!=std::atoi(argv[4])
            || resident!=std::atoi(argv[5]) || rounds!=std::atoi(argv[6])))continue;
        ++selected_cells;
        std::vector<double> spans,intervals,cycles,clocks,prelude_clocks;std::vector<Timing> host(grid);
        for(int trial=0;trial<9;++trial) {
          if(TMA_ACTIVITY_PRELUDE) {
            CHECK(cudaMemset(eviction,trial,256ull*1024*1024));
            CHECK(cudaDeviceSynchronize());
          }
          void* args[]={&a_map,&b_map,&output,&a_bytes,&b_bytes,&depth,&rounds,&resident
#if TMA_CP_ASYNC_PAIRS
              ,&a,&b
#endif
          };
          CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(queue_service),dim3(grid),dim3(TMA_CTA_THREADS),args,kShared));
          CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(host.data(),output,grid*sizeof(Timing),cudaMemcpyDeviceToHost));
          uint64_t first=UINT64_MAX,last=0;std::vector<unsigned> sms;
          for(auto t:host) {
            if(t.failed || t.t1<=t.t0 || t.c1<=t.c0)return 6;
            first=std::min(first,t.t0);last=std::max(last,t.t1);sms.push_back(t.sm);
            if(trial>=2){intervals.push_back(t.t1-t.t0);cycles.push_back(t.c1-t.c0);clocks.push_back(1000.0*(t.c1-t.c0)/(t.t1-t.t0));}
            if(trial>=2 && t.prelude_ns)prelude_clocks.push_back(1000.0*t.prelude_cycles/t.prelude_ns);
          }
          std::sort(sms.begin(),sms.end());if(std::unique(sms.begin(),sms.end())!=sms.end())return 7;
          if(trial>=2)spans.push_back(last-first);
        }
#if TMA_BALANCED_PAIRS
        const auto raw_spans=spans;
#endif
        auto median=[](std::vector<double>& v){std::sort(v.begin(),v.end());return v[v.size()/2];};
        printf("{\"activity_prelude\":%d,\"prelude_clock_mhz_p50\":%.3f,\"shared_operand_mask\":%d,\"row_stride_bytes\":%d,\"multicast_self\":%d,\"a_bytes\":%d,\"b_bytes\":%d,\"depth\":%d,\"grid\":%d,\"resident\":%d,\"rounds\":%d,"
          "\"span_ns_p50\":%.3f,\"cta_ns_p50\":%.3f,\"cycles_p50\":%.3f,\"clock_mhz_p50\":%.3f,"
          "\"stage_requests\":2,\"validation\":\"per_stage_sum_xor\",\"checks_passed\":true}\n",
          TMA_ACTIVITY_PRELUDE,prelude_clocks.empty()?0:median(prelude_clocks),TMA_SHARED_OPERAND_MASK,TMA_ROW_STRIDE_BYTES,TMA_MULTICAST_SELF,a_bytes,b_bytes,depth,grid,resident,rounds,median(spans),median(intervals),median(cycles),median(clocks));
#if TMA_BALANCED_PAIRS
        // Emit independent per-trial spans separately;raw does not discard
        // instability behind a median. They are outside every timed region.
        printf("{\"trial_spans\":true,\"threads\":%d,\"a_bytes\":%d,\"b_bytes\":%d,\"depth\":%d,\"grid\":%d,\"resident\":%d,\"rounds\":%d,",TMA_CTA_THREADS,a_bytes,b_bytes,depth,grid,resident,rounds);
#if TMA_BALANCED_STRIDE_SCREEN
        printf("\"balanced_row_bytes\":256,\"balanced_row_pitch_bytes\":%d,\"stream_packets\":%d,",256*TMA_BALANCED_ROW_GAP,TMA_STREAM_PACKETS);
#endif
        printf("\"spans_ns\":[");
        for(size_t trial=0;trial<raw_spans.size();++trial)printf("%s%.3f",trial?",":"",raw_spans[trial]);
        printf("]}\n");
#endif
        fflush(stdout);
      }
    }
    CHECK(cudaFree(a));CHECK(cudaFree(b));
  }
  CHECK(cudaFree(output));
  if(eviction)CHECK(cudaFree(eviction));
  if(argc==7 && selected_cells!=1)return 8;
}
