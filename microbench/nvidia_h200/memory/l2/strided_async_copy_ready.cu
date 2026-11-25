// Independent head-interleaved cold/repeated copies. No attention or matrix
// work. Measure both SM cycles and SI globaltimer to avoid a clock assumption.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int Smem=128*1024,FlushBytes=384*1024*1024,InputBytes=12*1024*1024;
__host__ __device__ unsigned pattern(unsigned x) {
  x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);
}
__global__ void initialize(unsigned* data,int count) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) data[i]=pattern(i);
}
__global__ void evict(uint4* data,int count) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
    data[i]=make_uint4(i,i+1,i+2,i+3);
}
__device__ unsigned long long global_ns() {
  unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t)::"memory");return t;
}
template<bool Tma>
__global__ void ready(const __grid_constant__ CUtensorMap map,const uint4* input,
                      unsigned long long* ticks,unsigned* checksums,int bytes,int heads) {
  extern __shared__ __align__(128) unsigned char shared[];
  __shared__ __align__(8) unsigned long long barrier;
  unsigned dst=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  unsigned bar=static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
  if(threadIdx.x==0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar):"memory");
    asm volatile("fence.proxy.async.shared::cta;" :::"memory");
  }
  __syncthreads();
  int head=blockIdx.x%heads,y0=blockIdx.x/heads*256;
  if(!Tma || threadIdx.x==0) {
    auto n0=global_ns();auto c0=clock64();
    if constexpr(Tma) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::"r"(bar),"r"(bytes):"memory");
      int x=head*128;
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0],[%1,{%2,%3}],[%4];" ::"r"(dst),"l"(&map),"r"(x),"r"(y0),"r"(bar):"memory");
      asm volatile("{.reg .pred p; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0; "
                   "@!p bra WAIT_%=;}" ::"r"(bar):"memory");
    } else {
      for(int i=threadIdx.x;i<bytes/16;i+=128) {
        auto source=input+((y0+i/16)*heads+head)*16+i%16;
        unsigned destination=dst+i*16;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::"r"(destination),"l"(source):"memory");
      }
      asm volatile("cp.async.commit_group;cp.async.wait_group 0;" :::"memory");
    }
    auto c1=clock64();auto n1=global_ns();
    if(threadIdx.x%32==0) {
      int i=(blockIdx.x*4+threadIdx.x/32)*2;ticks[i]=c1-c0;ticks[i+1]=n1-n0;
    }
  }
  __syncthreads();
  if(threadIdx.x==0) checksums[blockIdx.x]=*reinterpret_cast<unsigned*>(shared)
      ^*reinterpret_cast<unsigned*>(shared+bytes-4);
}
template<bool Tma>
void cell(int grid,int heads,int bytes,int trials,uint4* input,uint4* flush,
          unsigned long long* ticks,unsigned* checksums) {
  CUtensorMap map;cuuint64_t dims[2]={static_cast<unsigned>(128*heads),static_cast<unsigned>(((grid+heads-1)/heads)*256)};
  cuuint64_t strides[1]={static_cast<unsigned>(256*heads)};
  cuuint32_t box[2]={bytes==128?64u:128u,bytes==128?1u:static_cast<unsigned>(bytes/256)},elements[2]={1,1};
  auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT16,2,input,dims,strides,box,elements,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(status!=CUDA_SUCCESS) {fprintf(stderr,"tensor map error %d\n",status);exit(3);}
  CHECK(cudaFuncSetAttribute(ready<Tma>,cudaFuncAttributeMaxDynamicSharedMemorySize,Smem));
  for(int trial=0;trial<trials;++trial) {
    evict<<<132,256>>>(flush,FlushBytes/16);
    for(int state=0;state<2;++state) {
      ready<Tma><<<grid,128,Smem>>>(map,input,ticks,checksums,bytes,heads);
      CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
      std::vector<unsigned long long> host(grid*8);std::vector<unsigned> sums(grid);
      CHECK(cudaMemcpy(host.data(),ticks,host.size()*sizeof(*ticks),cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(sums.data(),checksums,sums.size()*sizeof(*checksums),cudaMemcpyDeviceToHost));
      for(int c=0;c<grid;++c) {
        int y=c/heads*256,head=c%heads,last=bytes/4-1;
        unsigned first_index=(y*heads+head)*64,last_index=((y+last/64)*heads+head)*64+last%64;
        if(sums[c]!=(pattern(first_index)^pattern(last_index))) {fprintf(stderr,"copy checksum mismatch\n");exit(4);}
        int warp=0;
        if constexpr(!Tma) for(int w=1;w<4;++w) if(host[(c*4+w)*2+1]>host[(c*4+warp)*2+1]) warp=w;
        int i=(c*4+warp)*2;
        if(!host[i] || !host[i+1]) {fprintf(stderr,"empty interval\n");exit(5);}
        printf("%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n",Tma,heads,grid,bytes,trial,state,c,host[i],host[i+1]);
      }
    }
  }
}
int main(int argc,char**) {
  uint4 *input,*flush;unsigned long long* ticks;unsigned* sums;
  CHECK(cudaMalloc(&input,InputBytes));CHECK(cudaMalloc(&flush,FlushBytes));
  CHECK(cudaMalloc(&ticks,132*8*sizeof(*ticks)));CHECK(cudaMalloc(&sums,132*sizeof(*sums)));
  initialize<<<132,256>>>(reinterpret_cast<unsigned*>(input),InputBytes/4);
  printf("tma,heads,grid,bytes,trial,state,cta,cycles,global_ns\n");
  if(argc==2) {
    cell<false>(132,32,32768,1,input,flush,ticks,sums);cell<true>(132,32,32768,1,input,flush,ticks,sums);
  } else {
    for(int heads:{1,8,32,64}) for(int grid:{1,8,32,132}) for(int bytes:{128,8192,32768,45056,65536}) {
      cell<false>(grid,heads,bytes,7,input,flush,ticks,sums);cell<true>(grid,heads,bytes,7,input,flush,ticks,sums);
    }
  }
  CHECK(cudaFree(input));CHECK(cudaFree(flush));CHECK(cudaFree(ticks));CHECK(cudaFree(sums));
}
