// Independent async instruction-batch enqueue versus retirement. No operator,
// global operand stream, softmax or scalar pressure. Timers are checked in SASS.
#include "enqueue_wgmma_helper.cuh"
#include <cmath>
constexpr int Samples=32;
constexpr int Warmup=64;
__device__ unsigned long long cycles_now() {
  unsigned long long t;asm volatile("mov.u64 %0,%%clock64;":"=l"(t)::"memory");return t;
}
template<int Matrices,bool RS>
__global__ void __launch_bounds__(256,1)
enqueue_window(unsigned long long* trace,float* sink) {
  extern __shared__ __align__(1024) unsigned char shared[];
  for(int i=threadIdx.x;i<kSmemBytes/2;i+=blockDim.x)
    reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u+((i*17)&127);
  unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  uint64_t a=make_desc(base,16,1024),b=make_desc(base+65536,16,1024);
  constexpr int Inputs=Matrices>0?Matrices:1;
  float accum[64];unsigned fragments[Inputs][4];
  #pragma unroll
  for(int i=0;i<64;++i) accum[i]=0;
  #pragma unroll
  for(int k=0;k<Inputs;++k) {
    #pragma unroll
    for(int i=0;i<4;++i) fragments[k][i]=0x3e003e00u+((threadIdx.x*17+k*4+i)&127);
  }
  __syncthreads();
  #pragma unroll 1
  for(int r=-Warmup;r<Samples;++r) {
    wg_fence();
    auto c0=cycles_now();
    #pragma unroll
    for(int k=0;k<Matrices;++k) {
      if constexpr(RS) wgmma_rs_n128_mnmajor(accum,fragments[k],b,1);
      else wgmma_ss_n128_kmajor(accum,a,b,1);
    }
    wg_commit();
    auto c1=cycles_now();
    wg_wait0();
    auto c2=cycles_now();
    if(r>=0 && threadIdx.x%128==0) {
      int i=((blockIdx.x*(blockDim.x/128)+threadIdx.x/128)*Samples+r)*2;
      trace[i]=c1-c0;trace[i+1]=c2-c0;
    }
  }
  if(threadIdx.x%128==0) sink[blockIdx.x*(blockDim.x/128)+threadIdx.x/128]=accum[0];
}
template<int Matrices,bool RS>
void run(int grid,int groups,unsigned long long* trace,float* sink) {
  auto kernel=enqueue_window<Matrices,RS>;
  CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmemBytes));
  for(int trial=0;trial<3;++trial) {
    kernel<<<grid,groups*128,kSmemBytes>>>(trace,sink);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid*groups*Samples*2);std::vector<float> values(grid*groups);
    CHECK(cudaMemcpy(host.data(),trace,host.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values.data(),sink,values.size()*sizeof(*sink),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) for(int wg=0;wg<groups;++wg) {
      if(!std::isfinite(values[c*groups+wg])) {fprintf(stderr,"invalid accumulator\n");exit(3);}
      for(int sample=0;sample<Samples;++sample) {
        int i=((c*groups+wg)*Samples+sample)*2;
        if(!host[i] || host[i+1]<=host[i]) {fprintf(stderr,"invalid timing order\n");exit(3);}
        printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n",Matrices,RS,grid,groups,trial,c,wg,sample,host[i],host[i+1]);
      }
    }
  }
}
template<int Matrices> void types(int grid,int groups,unsigned long long* trace,float* sink) {
  run<Matrices,false>(grid,groups,trace,sink);run<Matrices,true>(grid,groups,trace,sink);
}
int main() {
  unsigned long long* trace;float* sink;
  CHECK(cudaMalloc(&trace,132*2*Samples*2*sizeof(*trace)));CHECK(cudaMalloc(&sink,132*2*sizeof(*sink)));
  printf("matrices,rs,grid,groups,trial,cta,wg,sample,enqueue_cycles,total_cycles\n");
  for(int grid:{1,132}) for(int groups:{1,2}) {
    types<0>(grid,groups,trace,sink);types<1>(grid,groups,trace,sink);
    types<4>(grid,groups,trace,sink);types<8>(grid,groups,trace,sink);types<16>(grid,groups,trace,sink);
  }
  CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
