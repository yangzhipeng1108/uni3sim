// Independent two-row FP32 max/quad-shuffle and add dependency chains.
// No scores, normalization, matrix, memory-copy loop, or full softmax.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(2); } } while(0)

template<int Width, int Mode>
__global__ void chain(unsigned long long* ticks, float* sink, float seed) {
  float a=seed+threadIdx.x*.001f, b=seed-threadIdx.x*.001f;
  __syncthreads();
  unsigned long long start=clock64();
  #pragma unroll 1
  for(int rep=0;rep<512;++rep) {
    if constexpr(Mode==1 || Mode==2) {
      #pragma unroll
      for(int i=0;i<Width;++i) {
        if constexpr(Mode==1) {
          asm volatile("max.f32 %0,%0,%1;" : "+f"(a) : "f"(seed));
          asm volatile("max.f32 %0,%0,%1;" : "+f"(b) : "f"(seed));
        } else {
          asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(a) : "f"(seed));
          asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(b) : "f"(seed));
        }
      }
    }
    if constexpr(Mode==1 || Mode==3) {
      #pragma unroll
      for(int offset=2;offset>0;offset/=2) {
        float x=__shfl_xor_sync(0xffffffff,a,offset);
        float y=__shfl_xor_sync(0xffffffff,b,offset);
        if constexpr(Mode==1) {
          asm volatile("max.f32 %0,%0,%1;" : "+f"(a) : "f"(x));
          asm volatile("max.f32 %0,%0,%1;" : "+f"(b) : "f"(y));
        } else {
          asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(a) : "f"(x));
          asm volatile("add.rn.f32 %0,%0,%1;" : "+f"(b) : "f"(y));
        }
      }
    }
    asm volatile("" ::: "memory");
  }
  unsigned long long elapsed=clock64()-start;
  ticks[blockIdx.x*128+threadIdx.x]=elapsed;
  // All admitted modes keep finite values; no exponential is executed.
  sink[blockIdx.x*128+threadIdx.x]=a+b;
}

template<int Width,int Mode> void run(unsigned long long* ticks,float* sink) {
  CHECK(cudaFuncSetAttribute(chain<Width,Mode>,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024));
  for(int trial=0;trial<3;++trial) {
    chain<Width,Mode><<<132,128,128*1024>>>(ticks,sink,.0001f);
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(132*128);
    CHECK(cudaMemcpy(host.data(),ticks,host.size()*sizeof(*ticks),cudaMemcpyDeviceToHost));
    for(int c=0;c<132;++c) {
      unsigned long long maximum=0;
      for(int t=0;t<128;++t) if(host[c*128+t]>maximum) maximum=host[c*128+t];
      printf("%d,%d,%d,%d,%llu\n",Width,Mode,trial,c,maximum);
    }
  }
}
template<int W> void width(unsigned long long* ticks,float* sink) {
  run<W,0>(ticks,sink); run<W,1>(ticks,sink);
  run<W,2>(ticks,sink);
}
int main() {
  unsigned long long* ticks;float* sink;
  CHECK(cudaMalloc(&ticks,132*128*sizeof(*ticks)));
  CHECK(cudaMalloc(&sink,132*128*sizeof(*sink)));
  // Warm clocks with a register-only dependency chain before sampling.
  CHECK(cudaFuncSetAttribute(chain<32,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024));
  for(int i=0;i<64;++i) chain<32,2><<<132,128,128*1024>>>(ticks,sink,.0001f);
  CHECK(cudaDeviceSynchronize());
  printf("width,mode,trial,cta,cycles\n");
  width<20>(ticks,sink); width<32>(ticks,sink); width<40>(ticks,sink);
  width<44>(ticks,sink); width<48>(ticks,sink);
  CHECK(cudaFree(ticks));CHECK(cudaFree(sink));
}
