// Independent 2D TMA cold/repeated copy-ready probe, without math kernels.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2); } } while(0)
__global__ void evict(uint4* data,int count) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
    data[i]=make_uint4(i,i+1,i+2,i+3);
}
__global__ void ready(const __grid_constant__ CUtensorMap map,
                      unsigned long long* ticks,int bytes) {
  extern __shared__ __align__(128) unsigned char shared[];
  __shared__ __align__(8) unsigned long long barrier;
  unsigned dst=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  unsigned bar=static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
  __syncthreads();
  if(threadIdx.x==0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar):"memory");
    asm volatile("fence.proxy.async.shared::cta;" :::"memory");
    auto begin=clock64();
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::
                 "r"(bar),"r"(bytes):"memory");
    int y=blockIdx.x*256;
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0],[%1,{0,%2}],[%3];" ::
                 "r"(dst),"l"(&map),"r"(y),"r"(bar):"memory");
    asm volatile("{.reg .pred p; WAIT_%=: "
                 "mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0; "
                 "@!p bra WAIT_%=;}" ::"r"(bar):"memory");
    ticks[blockIdx.x]=clock64()-begin;
  }
}
int main() {
  constexpr int flush_bytes=384*1024*1024;
  uint4 *input,*flush;unsigned long long* ticks;
  CHECK(cudaMalloc(&input,132*65536));CHECK(cudaMalloc(&flush,flush_bytes));
  CHECK(cudaMalloc(&ticks,132*sizeof(*ticks)));CHECK(cudaMemset(input,1,132*65536));
  CHECK(cudaFuncSetAttribute(ready,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024));
  printf("grid,bytes,trial,state,cta,cycles\n");
  for(int grid:{1,8,32,132}) for(int bytes:{128,512,2048,8192,16384,32768,65536}) {
    CUtensorMap map;
    cuuint64_t dims[2]={128,132*256},strides[1]={256};
    cuuint32_t box[2]={bytes==128?64u:128u,bytes==128?1u:static_cast<unsigned>(bytes/256)};
    cuuint32_t element_strides[2]={1,1};
    auto result=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT16,2,input,
      dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(result!=CUDA_SUCCESS) {fprintf(stderr,"tensor map error %d\n",result);return 3;}
    for(int trial=0;trial<5;++trial) {
      evict<<<132,256>>>(flush,flush_bytes/16);
      for(int state=0;state<2;++state) {
        ready<<<grid,128,128*1024>>>(map,ticks,bytes);CHECK(cudaDeviceSynchronize());
        std::vector<unsigned long long> host(grid);
        CHECK(cudaMemcpy(host.data(),ticks,host.size()*sizeof(*ticks),cudaMemcpyDeviceToHost));
        for(int c=0;c<grid;++c) printf("%d,%d,%d,%d,%d,%llu\n",grid,bytes,trial,state,c,host[c]);
      }
    }
  }
  CHECK(cudaFree(input));CHECK(cudaFree(flush));CHECK(cudaFree(ticks));
}
