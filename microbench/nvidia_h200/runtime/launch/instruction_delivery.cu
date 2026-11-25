// Independent instruction-footprint and per-warp stream-delivery probe.
// Constant total arithmetic work. No matrix, normalization, copy or Attention.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}while(0)
struct Values {float v[8];};
#include "instruction_streams.cuh"
constexpr int SharedBytes=128*1024;
__device__ unsigned long long global_ns() {
  unsigned long long v;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(v)::"memory");return v;
}
template<int Threads,int Ops,int Split>
__global__ void __launch_bounds__(Threads,1)
instruction_delivery(unsigned long long* trace,float* sink,int iterations,float a,float b) {
  Values values;
  #pragma unroll
  for(int j=0;j<8;++j) values.v[j]=__fmaf_rn(float(threadIdx.x),.00001f,float(j)*.01f);
  __syncthreads();
  auto n0=global_ns();auto c0=clock64();
  if constexpr(Split==0) stream<Ops,0>(values,iterations,a,b);
  else {
    if(threadIdx.x<128) stream<Ops,1>(values,iterations,a,b);
    else stream<Ops,0>(values,iterations,a,b);
  }
  auto c1=clock64();auto n1=global_ns();
  if(threadIdx.x%32==0) {
    int i=(blockIdx.x*(Threads/32)+threadIdx.x/32)*4;
    trace[i]=c0;trace[i+1]=c1;trace[i+2]=n0;trace[i+3]=n1;
  }
  #pragma unroll
  for(int j=0;j<8;++j) sink[(blockIdx.x*Threads+threadIdx.x)*8+j]=values.v[j];
}
std::vector<float> expected(int threads,int total,int split,float a,float b) {
  std::vector<float> values(threads*8);
  for(int t=0;t<threads;++t) for(int j=0;j<8;++j) {
    float x=std::fma(float(t),.00001f,float(j)*.01f);
    for(int i=0;i<total/8;++i) x=split&&t<128?x+b:std::fma(a,x,b);
    values[t*8+j]=x;
  }
  return values;
}
template<int Threads,int Ops,int Split>
void run(int grid,int total,int trials,unsigned long long* trace,float* sink) {
  auto kernel=instruction_delivery<Threads,Ops,Split>;
  CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  float a=.9999f,b=.0001f;
  auto reference=expected(Threads,total,Split,a,b);
  for(int trial=-2;trial<trials;++trial) {
    kernel<<<grid,Threads,SharedBytes>>>(trace,sink,total/Ops,a,b);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> times(grid*(Threads/32)*4);
    std::vector<float> values(grid*Threads*8);
    CHECK(cudaMemcpy(times.data(),trace,times.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values.data(),sink,values.size()*sizeof(*sink),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) {
      for(int t=0;t<Threads;++t) for(int j=0;j<8;++j) {
        if(values[(c*Threads+t)*8+j]!=reference[t*8+j]) {
          std::fprintf(stderr,"instruction checksum mismatch %d %d %d\n",c,t,j);std::exit(3);
        }
      }
      for(int w=0;w<Threads/32;++w) {
        int i=(c*(Threads/32)+w)*4;
        std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu\n",Threads,Ops,Split,grid,total,trial,c,w,times[i],times[i+1],times[i+2],times[i+3]);
      }
    }
  }
}
template<int Threads,int Ops> void modes(int grid,int total,unsigned long long* trace,float* sink) {
  run<Threads,Ops,0>(grid,total,5,trace,sink);
  run<Threads,Ops,1>(grid,total,5,trace,sink);
}
template<int Threads> void widths(int grid,int total,unsigned long long* trace,float* sink) {
  modes<Threads,64>(grid,total,trace,sink);modes<Threads,256>(grid,total,trace,sink);
  modes<Threads,512>(grid,total,trace,sink);modes<Threads,1024>(grid,total,trace,sink);
  modes<Threads,2048>(grid,total,trace,sink);
}
int main(int argc,char** argv) {
  unsigned long long* trace;float* sink;
  CHECK(cudaMalloc(&trace,132*12*4*sizeof(*trace)));CHECK(cudaMalloc(&sink,132*384*8*sizeof(*sink)));
  std::printf("threads,ops,split,grid,total,trial,cta,warp,begin_cycles,end_cycles,begin_ns,end_ns\n");
  if(argc==2) {
    int ops=std::atoi(argv[1]);
    if(ops==64) run<384,64,1>(132,65536,8,trace,sink);
    else if(ops==512) run<384,512,1>(132,65536,8,trace,sink);
    else if(ops==2048) run<384,2048,1>(132,65536,8,trace,sink);
    else {std::fprintf(stderr,"invalid counter footprint\n");return 4;}
  } else {
    for(int grid:{1,32,132}) for(int total:{32768,65536}) {
      widths<128>(grid,total,trace,sink);widths<384>(grid,total,trace,sink);
    }
  }
  CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
