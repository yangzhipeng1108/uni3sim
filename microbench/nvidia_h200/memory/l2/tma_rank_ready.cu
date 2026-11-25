// Independent same-address 2D/4D TMA readiness. No Attention/math operator.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);} } while(0)
struct Stamp { unsigned long long c0,c1,n0,n1; unsigned sum; };
__device__ __forceinline__ unsigned long long cycles() { unsigned long long v;asm volatile("mov.u64 %0,%%clock64;":"=l"(v)::"memory");return v; }
__device__ __forceinline__ unsigned long long nanos() { unsigned long long v;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(v)::"memory");return v; }
__global__ void initialize(unsigned short* data,int count) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) data[i]=unsigned(i*17u+13u)&65535u;
}
__global__ void evict(uint4* data,int count) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) data[i]=make_uint4(i,i+1,i+2,i+3);
}
template<int Rank,int Prefetch>
__global__ void __launch_bounds__(128,1) rank_ready(const __grid_constant__ CUtensorMap map,
    Stamp* output,int heads,int rows) {
  extern __shared__ __align__(128) unsigned short tile[];
  __shared__ __align__(8) unsigned long long barrier;
  __shared__ unsigned checksum;
  unsigned dst=unsigned(__cvta_generic_to_shared(tile));
  unsigned bar=unsigned(__cvta_generic_to_shared(&barrier));
  if(threadIdx.x==0) {
    checksum=0;
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar):"memory");
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
  }
  __syncthreads();
  if(threadIdx.x==0) {
    if constexpr(Prefetch) {
      asm volatile("prefetch.tensormap [%0];"::"l"(&map):"memory");
      auto begin=cycles();while(cycles()-begin<256) {}
    }
    auto n0=nanos(),c0=cycles();
    unsigned bytes=unsigned(rows*128*2);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar),"r"(bytes):"memory");
    int y=blockIdx.x*256,h=blockIdx.x%heads;
    if constexpr(Rank==2) {
      int x=h*128;
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0],[%1,{%2,%3}],[%4];"::"r"(dst),"l"(&map),"r"(x),"r"(y),"r"(bar):"memory");
    } else {
      asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0],[%1,{0,%2,%3,0}],[%4];"::"r"(dst),"l"(&map),"r"(h),"r"(y),"r"(bar):"memory");
    }
    asm volatile("{.reg .pred p; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0; @!p bra WAIT_%=;}"::"r"(bar):"memory");
    auto c1=cycles(),n1=nanos();output[blockIdx.x]={c0,c1,n0,n1,0};
  }
  __syncthreads();
  unsigned sum=0;for(int i=threadIdx.x;i<rows*128;i+=blockDim.x) sum+=tile[i];
  atomicAdd(&checksum,sum);__syncthreads();
  if(threadIdx.x==0) output[blockIdx.x].sum=checksum;
}
template<int Rank,int Prefetch>
static void pair(unsigned short* input,uint4* flush,Stamp* output,int heads,int rows,int grid,int trial) {
  constexpr unsigned long long total_rows=132*256;
  CUtensorMap map;
  cuuint64_t dims[4]={Rank==2?128ull*heads:128ull,Rank==2?total_rows:static_cast<unsigned long long>(heads),total_rows,1};
  cuuint64_t strides[3]={Rank==2?256ull*heads:256ull,256ull*heads,256ull*heads*total_rows};
  cuuint32_t box[4]={128,Rank==2?static_cast<unsigned>(rows):1u,static_cast<unsigned>(rows),1};
  cuuint32_t step[4]={1,1,1,1};
  auto code=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT16,Rank,input,dims,strides,box,step,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(code!=CUDA_SUCCESS) {std::fprintf(stderr,"tensor map error %d rank%d heads%d rows%d\n",code,Rank,heads,rows);std::exit(3);}
  CHECK(cudaFuncSetAttribute(rank_ready<Rank,Prefetch>,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
  std::vector<unsigned> expected(grid,0);
  for(int c=0;c<grid;++c) for(int y=0;y<rows;++y) for(int x=0;x<128;++x) {
    unsigned i=((c*256+y)*heads+c%heads)*128+x;expected[c]+=(i*17u+13u)&65535u;
  }
  evict<<<132,256>>>(flush,384*1024*1024/16);CHECK(cudaDeviceSynchronize());
  for(int state=0;state<2;++state) {
    rank_ready<Rank,Prefetch><<<grid,128,196608>>>(map,output,heads,rows);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<Stamp> stamps(grid);CHECK(cudaMemcpy(stamps.data(),output,grid*sizeof(Stamp),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) {
      const auto s=stamps[c];if(s.sum!=expected[c]) {std::fprintf(stderr,"full tile checksum mismatch\n");std::exit(4);}
      std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%u\n",Rank,Prefetch,heads,rows,grid,trial,state,c,s.c0,s.c1,s.n0,s.n1,s.sum);
    }
  }
}
int main(int argc,char** argv) {
  bool counter=argc==2 && !std::strcmp(argv[1],"counter");
  if(argc!=1 && !counter) return 1;
  int sms=0;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));
  if(sms!=132) {std::fprintf(stderr,"expected132SMs,found%d\n",sms);return 5;}
  unsigned short* input;uint4* flush;Stamp* output;
  CHECK(cudaMalloc(&input,132ull*256*32*128*2));CHECK(cudaMalloc(&flush,384*1024*1024));CHECK(cudaMalloc(&output,132*sizeof(Stamp)));
  std::puts("rank,prefetch,heads,rows,grid,trial,state,cta,c0,c1,n0,n1,checksum");
  for(int heads:{1,32}) {
    if(counter && heads!=32) continue;
    initialize<<<132,256>>>(input,132*256*heads*128);CHECK(cudaDeviceSynchronize());
    for(int rows:{1,32,128,192}) for(int grid:{1,32,132}) {
      if(counter && (rows!=128 || grid!=132)) continue;
      for(int trial=0;trial<(counter?1:5);++trial) {
        // Alternate the rank pair order;each rank has its own data eviction.
        if(trial%2==0) {pair<2,0>(input,flush,output,heads,rows,grid,trial);pair<4,0>(input,flush,output,heads,rows,grid,trial);}
        else {pair<4,0>(input,flush,output,heads,rows,grid,trial);pair<2,0>(input,flush,output,heads,rows,grid,trial);}
        if(!counter) {pair<2,1>(input,flush,output,heads,rows,grid,trial);pair<4,1>(input,flush,output,heads,rows,grid,trial);}
      }
    }
  }
  CHECK(cudaFree(input));CHECK(cudaFree(flush));CHECK(cudaFree(output));
}
