// Independent Tensor/SFU resource pressure. Only clock frequency is derived;
// no Attention, global operand stream, normalization, or operator price.
#include "clock_wgmma_helper.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
constexpr int Chunks=8;
__device__ unsigned long long global_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;
}
template<int Mix>
__global__ void __launch_bounds__(256,1) pressure(unsigned long long* trace,float* sink,int repeats) {
  extern __shared__ __align__(1024) unsigned char shared[];
  for(int i=threadIdx.x;i<kSmemBytes/2;i+=256)
    reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u+((i*17)&127);
  __syncthreads();
  unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  uint64_t a=make_desc(base,16,1024),b=make_desc(base+65536,16,1024);
  float accum[64];
  #pragma unroll
  for(int i=0;i<64;++i) accum[i]=0;
  float seed=-.5f, sums[4]={0,0,0,0};
  for(int chunk=0;chunk<Chunks;++chunk) {
    const unsigned long long c0=clock64();
    const unsigned long long n0=global_ns();
    for(int rep=0;rep<repeats;++rep) {
      if constexpr(Mix!=3) {
        wg_fence();
        #pragma unroll
        for(int k=0;k<16;++k) wgmma_ss_n128_kmajor(accum,a,b,1);
        wg_commit();wg_wait0();
      }
      if constexpr(Mix!=0) {
        constexpr int Count=Mix==2?128:64;
        #pragma unroll
        for(int i=0;i<Count;++i) {
          float x,p;
          asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;":"=f"(x):
            "f"(seed),"f"(.5f),"f"(-.0001f*i));
          asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(p):"f"(x));
          asm volatile("add.rn.ftz.f32 %0,%0,%1;":"+f"(sums[i%4]):"f"(p));
        }
        seed=-__uint_as_float((__float_as_uint(sums[0])&0x007fffffu)|0x3f000000u);
      }
    }
    const unsigned long long c1=clock64();
    const unsigned long long n1=global_ns();
    if(threadIdx.x%128==0) {
      int i=((blockIdx.x*2+threadIdx.x/128)*Chunks+chunk)*2;
      trace[i]=c1-c0;trace[i+1]=n1-n0;
    }
  }
  if(threadIdx.x%128==0) sink[blockIdx.x*2+threadIdx.x/128]=accum[0]+sums[0];
}
template<int Mix> void run(int grid,unsigned long long* trace,float* sink,int repeats,int idle_us) {
  CHECK(cudaFuncSetAttribute(pressure<Mix>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmemBytes));
  pressure<Mix><<<grid,256,kSmemBytes>>>(trace,sink,repeats);CHECK(cudaDeviceSynchronize());
  for(int trial=0;trial<3;++trial) {
    if(idle_us) std::this_thread::sleep_for(std::chrono::microseconds(idle_us));
    pressure<Mix><<<grid,256,kSmemBytes>>>(trace,sink,repeats);CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid*2*Chunks*2);
    CHECK(cudaMemcpy(host.data(),trace,host.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) for(int wg=0;wg<2;++wg) for(int chunk=0;chunk<Chunks;++chunk) {
      int i=((c*2+wg)*Chunks+chunk)*2;
      if(!host[i]||!host[i+1]) {fprintf(stderr,"zero clock interval\n");exit(3);}
      printf("%d,%d,%d,%d,%d,%d,%llu,%llu,%.6f,%d,%d\n",Mix,grid,trial,c,wg,chunk,
             host[i],host[i+1],1000.*host[i]/host[i+1],repeats,idle_us);
    }
  }
}
int main(int argc,char** argv) {
  if(argc!=1 && argc!=3) {fprintf(stderr,"usage: probe [repeats idle_us]\n");return 2;}
  int repeats=1024,idle_us=0;
  if(argc==3) {
    char* end;
    long r=strtol(argv[1],&end,10);
    if(*end || r<1 || r>65536) return 2;
    long idle=strtol(argv[2],&end,10);
    if(*end || idle<0 || idle>100000) return 2;
    repeats=static_cast<int>(r);idle_us=static_cast<int>(idle);
  }
  unsigned long long* trace;float* sink;
  CHECK(cudaMalloc(&trace,132*2*Chunks*2*sizeof(*trace)));
  CHECK(cudaMalloc(&sink,132*2*sizeof(float)));
  printf("mix,grid,trial,cta,wg,chunk,sm_cycles,global_ns,mhz,repeats,idle_us\n");
  for(int grid:{1,32,132}) {
    if(argc==3 && grid==32) continue;
    run<0>(grid,trace,sink,repeats,idle_us);
    if(argc==1) {
      run<1>(grid,trace,sink,repeats,idle_us);
      run<2>(grid,trace,sink,repeats,idle_us);
    }
    run<3>(grid,trace,sink,repeats,idle_us);
  }
  CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
