// Independent primitive partition: runtime mode retains identical compiled
// register allocation for isolated and paired work. No Attention or KV stream.
#include "partition_wgmma_helper.cuh"
#include <cmath>
constexpr int Repeats=128;
__device__ unsigned long long global_ns() {
  unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t)::"memory");return t;
}
template<int Kind,int Matrices,int Values,bool Exp>
__device__ __forceinline__ float primitive_loop(uint64_t a,uint64_t b) {
  float accum[64],values[Values];
  #pragma unroll
  for(int i=0;i<64;++i) accum[i]=0;
  #pragma unroll
  for(int i=0;i<Values;++i) values[i]=-.5f-.0001f*i;
  #pragma unroll 1
  for(int r=0;r<Repeats;++r) {
    if constexpr(Kind!=1) {
      wg_fence();
      #pragma unroll
      for(int k=0;k<Matrices;++k) wgmma_ss_n128_kmajor(accum,a,b,1);
      wg_commit();
    }
    if constexpr(Kind!=0) {
      unsigned bits=unsigned(r)*9973u+threadIdx.x;
      float bias=-__uint_as_float((bits&0x007fffffu)|(Exp?0x3f800000u:0x3e000000u));
      #pragma unroll
      for(int i=0;i<Values;++i) {
        if constexpr(Exp) {
          float shifted;
          asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;":"=f"(shifted):
            "f"(values[i]),"f"(values[(i+17)%Values]),"f"(bias));
          asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(values[i]):"f"(shifted));
        } else {
          asm volatile("fma.rn.ftz.f32 %0,%0,%1,%2;":"+f"(values[i]):
            "f"(values[(i+17)%Values]),"f"(bias));
        }
      }
    }
    if constexpr(Kind!=1) wg_wait0();
  }
  float sum=accum[0];for(float v:values) sum+=v;
  return sum;
}
template<int Matrices,int Values,bool Exp>
__global__ void __launch_bounds__(256,1)
partition_pair(int mode,unsigned long long* trace,float* sink) {
  extern __shared__ __align__(1024) unsigned char shared[];
  for(int i=threadIdx.x;i<kSmemBytes/2;i+=blockDim.x)
    reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u+((i*17)&127);
  unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  uint64_t a=make_desc(base,16,1024),b=make_desc(base+65536,16,1024);
  const int wg=threadIdx.x/128;
  __syncthreads();
  const unsigned long long c0=clock64(),n0=global_ns();
  float value=0;
  if(mode==3) {
    if(wg==0) value=primitive_loop<2,Matrices,Values,Exp>(a,b);
  } else if(wg==0) {
    if(mode==0 || mode==2) value=primitive_loop<0,Matrices,Values,Exp>(a,b);
  } else {
    if(mode==1 || mode==2) value=primitive_loop<1,Matrices,Values,Exp>(a,b);
  }
  __syncthreads();
  const unsigned long long c1=clock64(),n1=global_ns();
  if(threadIdx.x%128==0) {
    int i=(blockIdx.x*2+wg)*2;
    trace[i]=c1-c0;trace[i+1]=n1-n0;
    sink[i/2]=value;
  }
}
template<int Matrices,int Values,bool Exp>
void run(int grid,unsigned long long* trace,float* sink) {
  auto kernel=partition_pair<Matrices,Values,Exp>;
  CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmemBytes));
  cudaFuncAttributes attributes;CHECK(cudaFuncGetAttributes(&attributes,kernel));
  for(int mode=0;mode<4;++mode) for(int trial=-1;trial<3;++trial) {
    kernel<<<grid,256,kSmemBytes>>>(mode,trace,sink);CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid*4);std::vector<float> values(grid*2);
    CHECK(cudaMemcpy(host.data(),trace,host.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values.data(),sink,values.size()*sizeof(*sink),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) for(int wg=0;wg<2;++wg) {
      int i=(c*2+wg)*2;
      if(!host[i] || !host[i+1] || !std::isfinite(values[i/2])) {fprintf(stderr,"invalid primitive result\n");exit(3);}
      printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n",Matrices,Values,Exp,grid,mode,trial,c,wg,
        attributes.numRegs,host[i],host[i+1]);
    }
  }
}
template<int Matrices,int Values> void types(int grid,unsigned long long* trace,float* sink) {
  run<Matrices,Values,false>(grid,trace,sink);run<Matrices,Values,true>(grid,trace,sink);
}
int main() {
  unsigned long long* trace;float* sink;
  CHECK(cudaMalloc(&trace,132*4*sizeof(*trace)));CHECK(cudaMalloc(&sink,132*2*sizeof(*sink)));
  printf("matrices,values,exp,grid,mode,trial,cta,wg,registers,cycles,global_ns\n");
  for(int grid:{1,132}) {
    types<4,16>(grid,trace,sink);types<4,64>(grid,trace,sink);types<4,128>(grid,trace,sink);
    types<8,16>(grid,trace,sink);types<8,64>(grid,trace,sink);types<8,128>(grid,trace,sink);
    types<16,16>(grid,trace,sink);types<16,64>(grid,trace,sink);types<16,128>(grid,trace,sink);
  }
  CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
