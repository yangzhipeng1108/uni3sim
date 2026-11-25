// Independent balanced TMA byte streams and static shared-operand WGMMA.
// No QK/PV pair, softmax, Attention inputs, output tile or operator duration.
#include <cuda.h>
#include <cudaTypedefs.h>
#include "tma_tensor_helper.cuh"
#include <algorithm>
#include <cmath>
constexpr int Shared=212992, CopyOffset=32768, MatrixBOffset=8192, Packets=256;
struct Stamp { unsigned long long c0,c1,n0,n1; };
struct Prelude { unsigned long long cycles,ns; };
__device__ __forceinline__ uint64_t nanos() { uint64_t v; asm volatile("mov.u64 %0,%%globaltimer;":"=l"(v)::"memory");return v; }
__device__ __forceinline__ unsigned addr(const void* p) { return unsigned(__cvta_generic_to_shared(p)); }
__host__ __device__ unsigned pattern(size_t i) { unsigned x=unsigned(i)*747796405u+2891336453u;x=((x>>((x>>28)+4))^x)*277803737u;return (x>>22)^x; }
__global__ void initialize(unsigned* data,size_t words) { for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)data[i]=pattern(i); }
__device__ __forceinline__ void wait_copy(uint64_t* b,int phase) { asm volatile("{.reg .pred p; L_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra L_%=;}"::"r"(addr(b)),"r"(phase):"memory"); }
__device__ __forceinline__ void tma(void* dest,const CUtensorMap* map,uint64_t* barrier,int plane) {
 asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,0,0,%2}],[%3];"::"r"(addr(dest)),"l"(map),"r"(plane),"r"(addr(barrier)):"memory");
}
__device__ __forceinline__ void copies(const CUtensorMap* map,uint64_t* barriers,unsigned char* shared,int bytes,int repeats,bool resident) {
 for(int i=0;i<2;++i)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(addr(barriers+i)):"memory");
 asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 int slot=0,phase=0;
 #pragma unroll 1
 for(int r=0;r<repeats;++r) {
  if(r>=2)wait_copy(barriers+slot,phase^1);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(addr(barriers+slot)),"r"(2*bytes):"memory");
  int packet=resident?slot:r;
  int plane=(int(blockIdx.x)*Packets+packet)*2;
  auto* dest=shared+CopyOffset+slot*2*bytes;
  tma(dest,map,barriers+slot,plane);tma(dest+bytes,map,barriers+slot,plane+1);
  if(++slot==2) { slot=0;phase^=1; }
 }
 wait_copy(barriers,phase^1);wait_copy(barriers+1,phase^1);
}
template<int N,int Depth>
__global__ __launch_bounds__(256,1) void overlap(const __grid_constant__ CUtensorMap map,int bytes,int repeats,int mode,bool resident,
    Stamp* traces,Prelude* preludes,float* sums,unsigned* errors) {
 // The128B-swizzled K-major descriptor advances1KiB per8-row group.
 // A's64 rows occupy8KiB;B's176-row address span is22KiB,not its5.5KiB
 // logical payload. Both spans must precede the disjoint TMA destinations.
 static_assert(MatrixBOffset>=64/8*1024);
 static_assert(MatrixBOffset+N/8*1024<=CopyOffset);
 extern __shared__ __align__(1024) unsigned char shared[];
 __shared__ __align__(8) uint64_t barriers[2];
 __shared__ unsigned wrong[256];
 __shared__ float totals[256];
 __shared__ unsigned long long copy_sum,expected_sum;
 __shared__ unsigned copy_xor,expected_xor;
 float accum[2][N/2];
 #pragma unroll
 for(int bank=0;bank<2;++bank) {
  #pragma unroll
  for(int i=0;i<N/2;++i) { accum[bank][i]=0;asm volatile("":"+f"(accum[bank][i])); }
 }
 for(int i=threadIdx.x;i<CopyOffset/2;i+=blockDim.x)reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u;
 if(threadIdx.x==0) {
  // This prefill makes the resident domain a small, shared-L2 working set.
  // It also keeps pre-timer copy work identical in all three modes.
  if(resident)copies(&map,barriers,shared,bytes,2,true);
  unsigned state=blockIdx.x+1;uint64_t n0=nanos(),c0=clock64();
  do { for(int i=0;i<128;++i)asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;":"+r"(state)); } while(nanos()-n0<3000000);
  asm volatile(""::"r"(state):"memory");preludes[blockIdx.x]={clock64()-c0,nanos()-n0};
 }
 unsigned base=addr(shared);uint64_t a=make_desc(base,16,1024),b=make_desc(base+MatrixBOffset,16,1024);
 __syncthreads();Stamp s{};
 if(threadIdx.x==0 && mode!=0) { s.n0=nanos();s.c0=clock64();copies(&map,barriers,shared,bytes,repeats,resident);s.c1=clock64();s.n1=nanos(); }
 if(threadIdx.x>=128 && mode!=1) {
  wg_fence();s.n0=nanos();s.c0=clock64();
  #pragma unroll 1
  for(int r=0;r<repeats;++r) {
   #pragma unroll
   for(int bank=0;bank<2;++bank) {
    #pragma unroll
    for(int k=0;k<4;++k) { if constexpr(N==128)wgmma_ss_n128_kmajor(accum[bank],a,b,1);else wgmma_ss_n176_kmajor(accum[bank],a,b,1); }
    wg_commit();asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(Depth-1):"memory");
   }
  }
  wg_wait0();s.c1=clock64();s.n1=nanos();
 }
 // No scalar numerical checking or global output while a peer is timed.
 __syncthreads();unsigned bad=0;float total=0;
 if(threadIdx.x>=128 && mode!=1) {
  #pragma unroll
  for(int bank=0;bank<2;++bank) {
   #pragma unroll
   for(int i=0;i<N/2;++i) { float v=accum[bank][i];bad+=v!=16.f*repeats;total+=v; }
  }
 }
 totals[threadIdx.x]=total;wrong[threadIdx.x]=bad;
 if(threadIdx.x==0 && mode!=0)traces[blockIdx.x*5]=s;
 if(threadIdx.x>=128 && threadIdx.x%32==0 && mode!=1)traces[blockIdx.x*5+1+(threadIdx.x-128)/32]=s;
 __syncthreads();
 if(threadIdx.x==0) { total=0;bad=0;for(int i=0;i<256;++i) { total+=totals[i];bad+=wrong[i]; }sums[blockIdx.x]=total;errors[blockIdx.x]=bad; }
 for(int slot=0;slot<2;++slot) {
  if(threadIdx.x==0) { copy_sum=expected_sum=0;copy_xor=expected_xor=0; }
  __syncthreads();unsigned long long actual=0,expected=0;unsigned ax=0,ex=0;
  if(mode!=0) {
   int last=repeats-2+slot,packet=resident?slot:last;
   auto* data=reinterpret_cast<const unsigned*>(shared+CopyOffset+slot*2*bytes);
   size_t start=(size_t(blockIdx.x)*Packets+packet)*2*bytes/4;
   for(int w=threadIdx.x;w<2*bytes/4;w+=blockDim.x) { unsigned v=data[w],e=pattern(start+w);actual+=v;expected+=e;ax^=v;ex^=e; }
  }
  atomicAdd(&copy_sum,actual);atomicAdd(&expected_sum,expected);atomicXor(&copy_xor,ax);atomicXor(&expected_xor,ex);__syncthreads();
  if(threadIdx.x==0 && (copy_sum!=expected_sum||copy_xor!=expected_xor))++errors[blockIdx.x];
  __syncthreads();
 }
}
CUtensorMap descriptor(void* data,int bytes) {
 void* fn=nullptr;cudaDriverEntryPointQueryResult status;CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&fn,cudaEnableDefault,&status));
 if(!fn||status!=cudaDriverEntryPointSuccess)std::exit(3);
 auto encode=reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
 cuuint64_t dims[4]={128,2,cuuint64_t(bytes/256),132*Packets*2},strides[3]={128,256,cuuint64_t(bytes)};
 cuuint32_t box[4]={128,2,cuuint32_t(bytes/256),1},steps[4]={1,1,1,1};CUtensorMap map;
 if(encode(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,4,data,dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)std::exit(4);
 return map;
}
template<int N,int Depth>void run(const CUtensorMap& map,int bytes,int grid,int repeats,bool resident,int mode,unsigned char* eviction,
    Stamp* traces,Prelude* preludes,float* sums,unsigned* errors) {
 auto fn=overlap<N,Depth>;CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));int occupancy;
 CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,fn,256,Shared));if(occupancy!=1)std::exit(5);
 for(int trial=-1;trial<3;++trial) {
  CHECK(cudaMemset(eviction,trial+17,256ull*1024*1024));CHECK(cudaDeviceSynchronize());
  fn<<<grid,256,Shared>>>(map,bytes,repeats,mode,resident,traces,preludes,sums,errors);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  std::vector<Stamp> t(grid*5);std::vector<Prelude> p(grid);std::vector<float> v(grid);std::vector<unsigned> e(grid);
  CHECK(cudaMemcpy(t.data(),traces,t.size()*sizeof(Stamp),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(p.data(),preludes,p.size()*sizeof(Prelude),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(v.data(),sums,v.size()*sizeof(float),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(e.data(),errors,e.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
  for(int c=0;c<grid;++c) {
   if(e[c]||v[c]!=(mode==1?0.f:float(128*N*16*repeats))) { std::fprintf(stderr,"numeric N%d depth%d mode%d bytes%d grid%d resident%d repeats%d trial%d cta%d errors%u sum%.9g\n",N,Depth,mode,bytes,grid,int(resident),repeats,trial,c,e[c],v[c]);std::exit(6); }
   for(int actor=0;actor<5;++actor) {
    if((mode==0&&actor==0)||(mode==1&&actor!=0))continue;auto s=t[c*5+actor];
    if(s.c1<=s.c0||s.n1<s.n0||p[c].ns<3000000||!p[c].cycles)std::exit(7);
    if(trial>=0)std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu\n",N,Depth,bytes,grid,repeats,int(resident),mode,trial,c,actor,s.c0,s.c1,s.n0,s.n1,p[c].cycles,p[c].ns);
   }
  }
 }
}
int main() {
 int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 8;
 Stamp* t;Prelude* p;float* v;unsigned* e;unsigned char* eviction;
 CHECK(cudaMalloc(&t,132*5*sizeof(Stamp)));CHECK(cudaMalloc(&p,132*sizeof(Prelude)));CHECK(cudaMalloc(&v,132*sizeof(float)));CHECK(cudaMalloc(&e,132*sizeof(unsigned)));CHECK(cudaMalloc(&eviction,256ull*1024*1024));
 std::puts("columns,depth,bytes,grid,repeats,resident,mode,trial,cta,actor,c0,c1,n0,n1,prelude_cycles,prelude_ns");
 for(int bytes:{32768,45056}) {
  unsigned* input;size_t words=size_t(132)*Packets*2*bytes/4;CHECK(cudaMalloc(&input,words*4));initialize<<<4096,256>>>(input,words);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());auto map=descriptor(input,bytes);
  for(int grid:{1,132})for(int repeats:{64,128,256})for(bool resident:{false,true})for(int mode:{0,1,2}) {
   run<128,1>(map,bytes,grid,repeats,resident,mode,eviction,t,p,v,e);run<128,4>(map,bytes,grid,repeats,resident,mode,eviction,t,p,v,e);
   run<176,1>(map,bytes,grid,repeats,resident,mode,eviction,t,p,v,e);run<176,4>(map,bytes,grid,repeats,resident,mode,eviction,t,p,v,e);
  }
  CHECK(cudaFree(input));
 }
 CHECK(cudaFree(t));CHECK(cudaFree(p));CHECK(cudaFree(v));CHECK(cudaFree(e));CHECK(cudaFree(eviction));
}
