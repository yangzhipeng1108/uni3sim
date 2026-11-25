// Independent Tensor/scalar resource coupling. No Attention, normalization,
// max/reduction, output matrix or global operand stream.
#include "overlap_wgmma_helper.cuh"
#include <cmath>
constexpr int Repeats=256;
__device__ unsigned long long global_ns() {
  unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t));return t;
}
template<int Mode,bool RS,bool Exp>
__global__ void __launch_bounds__(256,1) resource_pair(unsigned long long* trace,float* sink) {
  extern __shared__ __align__(1024) unsigned char shared[];
  for(int i=threadIdx.x;i<kSmemBytes/2;i+=blockDim.x)
    reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u+((i*17)&127);
  unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  uint64_t a=make_desc(base,16,1024),b=make_desc(base+65536,16,1024);
  float accum[64],values[64];unsigned fragments[8][4];
  #pragma unroll
  for(int i=0;i<64;++i) {accum[i]=0;values[i]=-.5f-.0001f*i;}
  #pragma unroll
  for(int k=0;k<8;++k) {
    #pragma unroll
    for(int i=0;i<4;++i) fragments[k][i]=0x3e003e00u+((threadIdx.x*17+k*4+i)&127);
  }
  __syncthreads();
  const unsigned long long c0=clock64(),n0=global_ns();
  #pragma unroll 1
  for(int r=0;r<Repeats;++r) {
    if constexpr(Mode!=1) {
      wg_fence();
      #pragma unroll
      for(int k=0;k<8;++k) {
        if constexpr(RS) wgmma_rs_n128_mnmajor(accum,fragments[k],b,1);
        else wgmma_ss_n128_kmajor(accum,a,b,1);
      }
      wg_commit();
      if constexpr(Mode==3) wg_wait0();
    }
    if constexpr(Mode!=0) {
      unsigned bits=unsigned(r)*9973u+threadIdx.x;
      // The serial control must consume retired matrix state. A bare wait0
      // can be moved past independent scalar instructions by the compiler.
      if constexpr(Mode==3) bits=__float_as_uint(accum[0]);
      float bias=-__uint_as_float((bits&0x007fffffu)|(Exp?0x3f800000u:0x3e000000u));
      #pragma unroll
      for(int i=0;i<64;++i) {
        if constexpr(Exp) {
          float shifted;
          asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;":"=f"(shifted):
            "f"(values[i]),"f"(values[(i+17)%64]),"f"(bias));
          asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(values[i]):"f"(shifted));
        } else {
          asm volatile("fma.rn.ftz.f32 %0,%0,%1,%2;":"+f"(values[i]):
            "f"(values[(i+17)%64]),"f"(bias));
        }
      }
    }
    if constexpr(Mode==0 || Mode==2) wg_wait0();
  }
  const unsigned long long c1=clock64(),n1=global_ns();
  if(threadIdx.x%128==0) {
    int i=(blockIdx.x*(blockDim.x/128)+threadIdx.x/128)*2;
    trace[i]=c1-c0;trace[i+1]=n1-n0;
    float sum=accum[0];for(float v:values) sum+=v;
    sink[i/2]=sum;
  }
}
template<int Mode,bool RS,bool Exp> void run(int grid,int groups,unsigned long long* trace,float* sink) {
  CHECK(cudaFuncSetAttribute(resource_pair<Mode,RS,Exp>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmemBytes));
  resource_pair<Mode,RS,Exp><<<grid,groups*128,kSmemBytes>>>(trace,sink);CHECK(cudaDeviceSynchronize());
  for(int trial=0;trial<3;++trial) {
    resource_pair<Mode,RS,Exp><<<grid,groups*128,kSmemBytes>>>(trace,sink);CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid*groups*2);std::vector<float> values(grid*groups);
    CHECK(cudaMemcpy(host.data(),trace,host.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values.data(),sink,values.size()*sizeof(*sink),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) for(int wg=0;wg<groups;++wg) {
      int i=(c*groups+wg)*2;
      if(!host[i] || !host[i+1] || !std::isfinite(values[i/2])) {fprintf(stderr,"invalid primitive result\n");exit(3);}
      printf("%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n",Mode,RS,Exp,grid,groups,trial,c,host[i],host[i+1]);
    }
  }
}
template<bool RS,bool Exp> void modes(int grid,int groups,unsigned long long* trace,float* sink) {
  run<0,RS,Exp>(grid,groups,trace,sink);run<1,RS,Exp>(grid,groups,trace,sink);
  run<2,RS,Exp>(grid,groups,trace,sink);run<3,RS,Exp>(grid,groups,trace,sink);
}
int main() {
  unsigned long long* trace;float* sink;
  CHECK(cudaMalloc(&trace,132*2*2*sizeof(*trace)));CHECK(cudaMalloc(&sink,132*2*sizeof(*sink)));
  printf("mode,rs,exp,grid,groups,trial,cta,cycles,global_ns\n");
  for(int grid:{1,132}) for(int groups:{1,2}) {
    modes<false,false>(grid,groups,trace,sink);modes<false,true>(grid,groups,trace,sink);
    modes<true,false>(grid,groups,trace,sink);modes<true,true>(grid,groups,trace,sink);
  }
  CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
