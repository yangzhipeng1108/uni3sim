// Independent ready-predecessor griddepcontrol.wait latency. No operator,
// operand stream, matrix, reduction or unfinished predecessor is executed.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){ \
 fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int SharedBytes=128*1024;
__device__ unsigned long long global_ns() {
  unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t));return t;
}
template<int Waits>
__global__ void __launch_bounds__(256,1) ready_wait(unsigned long long* trace) {
  extern __shared__ volatile unsigned scratch[];
  if(threadIdx.x==0) scratch[0]=1;
  __syncthreads();
  const unsigned long long c0=clock64(),n0=global_ns();
  #pragma unroll 1
  for(int i=0;i<Waits;++i) asm volatile("griddepcontrol.wait;" :::"memory");
  const unsigned long long c1=clock64(),n1=global_ns();
  if(threadIdx.x==0) {trace[blockIdx.x*2]=c1-c0;trace[blockIdx.x*2+1]=n1-n0;}
}
template<int Waits> void launch(int grid,int threads,bool extended,unsigned long long* trace) {
  if(extended) {
    cudaLaunchAttribute attribute{};
    attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t config{};config.gridDim=dim3(grid);config.blockDim=dim3(threads);
    config.dynamicSmemBytes=SharedBytes;config.attrs=&attribute;config.numAttrs=1;
    CHECK(cudaLaunchKernelEx(&config,ready_wait<Waits>,trace));
  } else ready_wait<Waits><<<grid,threads,SharedBytes>>>(trace);
}
template<int Waits> void run(int grid,int threads,bool extended,unsigned long long* trace) {
  CHECK(cudaFuncSetAttribute(ready_wait<Waits>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  for(int i=0;i<8;++i) launch<Waits>(grid,threads,extended,trace);
  CHECK(cudaDeviceSynchronize());
  for(int trial=0;trial<3;++trial) {
    launch<Waits>(grid,threads,extended,trace);CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid*2);
    CHECK(cudaMemcpy(host.data(),trace,host.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) {
      if(!host[c*2]) {fprintf(stderr,"invalid cycle interval\n");exit(3);}
      printf("%d,%d,%d,%d,%d,%d,%llu,%llu\n",Waits,extended,grid,threads,trial,c,
        host[c*2],host[c*2+1]);
    }
  }
}
int main() {
  unsigned long long* trace;CHECK(cudaMalloc(&trace,132*2*sizeof(*trace)));
  printf("waits,extended,grid,threads,trial,cta,cycles,global_ns\n");
  for(int grid:{1,32,132}) for(int threads:{32,128,256}) for(bool extended:{false,true}) {
    run<0>(grid,threads,extended,trace);run<1>(grid,threads,extended,trace);
    run<4>(grid,threads,extended,trace);run<64>(grid,threads,extended,trace);
  }
  CHECK(cudaFree(trace));
}
