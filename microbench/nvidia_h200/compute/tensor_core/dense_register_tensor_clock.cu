// Independent register-only Tensor activity and clock64/globaltimer probe.
// No global operand stream, matrix output, production shape or provider ABI.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#define CHECK(call) do { cudaError_t status=(call); if(status!=cudaSuccess) { \
  fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(status));exit(2); } } while(0)
__device__ unsigned long long global_ns() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}
__device__ void tensor_step(float (&c)[4],unsigned a,unsigned b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%4,%4,%4},{%5,%5},{%0,%1,%2,%3};"
      :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a),"r"(b));
}
__global__ __launch_bounds__(256,1) void tensor_clock(int rounds,int patterned,
    unsigned long long* clocks,float* output) {
  extern __shared__ unsigned reserve[];
  if(threadIdx.x==0) reserve[0]=blockIdx.x;
  __syncthreads();
  unsigned a=patterned ? (0x34003400u | (threadIdx.x%32)*0x10001u) : 0x3c003c00u;
  unsigned b=patterned ? (0x34003400u | ((threadIdx.x+7)%32)*0x10001u) : 0x3c003c00u;
  float acc[16][4]={};
  unsigned long long begin=clock64(),start=global_ns();
#pragma unroll 1
  for(int r=0;r<rounds;++r) {
#pragma unroll
    for(int phase=0;phase<4;++phase) {
#pragma unroll
      for(int fragment=0;fragment<16;++fragment) tensor_step(acc[fragment],a,b);
    }
  }
  unsigned long long stop=global_ns(),end=clock64();
  float sum=0;
#pragma unroll
  for(int fragment=0;fragment<16;++fragment) for(int j=0;j<4;++j) sum+=acc[fragment][j];
  if(threadIdx.x%32==0) {
    int index=blockIdx.x*8+threadIdx.x/32;
    clocks[index*2]=end-begin;clocks[index*2+1]=stop-start;output[index]=sum;
  }
}
int main() {
  CHECK(cudaFuncSetAttribute(tensor_clock,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
  cudaFuncAttributes attr;CHECK(cudaFuncGetAttributes(&attr,tensor_clock));if(attr.localSizeBytes)return 3;
  unsigned long long *clocks;float *output;
  CHECK(cudaMalloc(&clocks,132*8*2*sizeof(unsigned long long)));CHECK(cudaMalloc(&output,132*8*sizeof(float)));
  auto warm=std::chrono::steady_clock::now();
  do {tensor_clock<<<132,256,196608>>>(2048,1,clocks,output);CHECK(cudaDeviceSynchronize());}
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
  for(int grid : {32,132})for(int patterned : {0,1})for(int rounds : {512,2048}) {
    std::vector<double> frequencies;
    std::vector<unsigned long long> values(grid*16);std::vector<float> sums(grid*8);
    for(int trial=0;trial<5;++trial) {
      tensor_clock<<<grid,256,196608>>>(rounds,patterned,clocks,output);CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(values.data(),clocks,values.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(sums.data(),output,sums.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int i=0;i<grid*8;++i) {
        if(!values[i*2+1]||!std::isfinite(sums[i])||sums[i]<=0)return 4;
        if(!patterned&&sums[i]!=float(rounds*4*16*64))return 5;
        frequencies.push_back(1000.0*values[i*2]/values[i*2+1]);
      }
    }
    std::sort(frequencies.begin(),frequencies.end());size_t count=frequencies.size();
    printf("{\"grid\":%d,\"rounds\":%d,\"patterned\":%d,\"median_sm_mhz\":%.3f,"
        "\"p10_sm_mhz\":%.3f,\"p90_sm_mhz\":%.3f,\"registers\":%d,\"ones_check_passed\":%d}\n",
        grid,rounds,patterned,frequencies[count/2],frequencies[count/10],frequencies[count*9/10],attr.numRegs,!patterned);
  }
  CHECK(cudaFree(clocks));CHECK(cudaFree(output));
}
