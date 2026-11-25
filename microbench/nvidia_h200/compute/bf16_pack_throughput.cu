// Independent bounded register-only packed FP32->BF16 conversion/LOP3 capacity.
// Matched one/two LOP3 feedback controls separate width-dependent service.
// Positive second inputs put the low16 feedback bits below half an ULP;
// both controls preserve the same period-two BF16 outputs without overflow.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);}} while(0)
struct Stamp {unsigned long long c0,c1,n0,n1;unsigned sum;};
__device__ __forceinline__ unsigned long long cycles(){unsigned long long x;asm volatile("mov.u64 %0,%%clock64;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ unsigned long long nanos(){unsigned long long x;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(x)::"memory");return x;}
__host__ __device__ unsigned a_bits(unsigned tid,int i){return 0x3e800000u|((tid*12347u+unsigned(i)*9181u)&0x7fffffu);}
__host__ __device__ unsigned b_bits(unsigned tid,int i){return 0x3e000000u|((tid*9311u+unsigned(i)*17747u)&0x7fffffu);}
__host__ __device__ unsigned rne(unsigned bits){return (bits+0x7fffu+((bits>>16)&1))>>16;}
__host__ __device__ unsigned expected(unsigned tid,int i,int repetitions){return ((a_bits(tid,i)&0xffff0000u)^(((repetitions-1)&1)?0x00010000u:0u))|((b_bits(tid,i)>>16)&65535u);}
template<int Pairs,int Feedback>
__global__ void __launch_bounds__(256,1) pack_capacity(Stamp* stamps,unsigned* errors,int repetitions){
  // Exact BF16 inputs form a period-two bounded recurrence: each packed
  // conversion feeds its own next FP32 input via one XOR or an AND plus XOR.
  // All independent streams remain live;no exp/overflow or idempotent loop.
  unsigned state[Pairs],packed[Pairs];float second[Pairs];
  unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
  #pragma unroll
  for(int i=0;i<Pairs;++i){state[i]=a_bits(tid,i)&0xffff0000u;second[i]=__uint_as_float(b_bits(tid,i)&0xffff0000u);}
  #pragma unroll
  for(int i=0;i<Pairs;++i)asm volatile(""::"r"(state[i]),"f"(second[i]):"memory");
  __shared__ volatile unsigned sinks[256];
  __syncthreads();
  unsigned long long n0=0,c0=0;
  if(threadIdx.x%32==0){n0=nanos();c0=cycles();}
  __syncwarp();
  #pragma unroll 1
  for(int r=0;r<repetitions;++r){
    #pragma unroll
    for(int i=0;i<Pairs;++i){
      asm volatile("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(packed[i]):"f"(__uint_as_float(state[i])),"f"(second[i]));
      if constexpr(Feedback==1)state[i]=packed[i]^0x00010000u;
      else state[i]=(packed[i]&0xffff0000u)^0x00010000u;
    }
  }
  unsigned sum=0;
  #pragma unroll
  for(int i=0;i<Pairs;++i)sum+=packed[i];
  // The stores depend on every result;the CTA barrier precedes ending clocks.
  // This fixed final checksum/barrier cost cancels in repetition slopes.
  sinks[threadIdx.x]=sum;__syncthreads();
  if(threadIdx.x%32==0){auto c1=cycles(),n1=nanos();stamps[blockIdx.x*(blockDim.x/32)+threadIdx.x/32]={c0,c1,n0,n1,sum};}
  unsigned wrong=0;
  #pragma unroll
  for(int i=0;i<Pairs;++i)wrong+=(packed[i]!=expected(tid,i,repetitions));
  if(wrong)atomicAdd(errors,wrong);
}
template<int Pairs,int Feedback>
void run(Stamp* output,unsigned* errors,int threads,int grid,int trial,int repetitions){
  CHECK(cudaFuncSetAttribute(pack_capacity<Pairs,Feedback>,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
  pack_capacity<Pairs,Feedback><<<grid,threads,196608>>>(output,errors,repetitions);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  unsigned wrong;CHECK(cudaMemcpy(&wrong,errors,sizeof(wrong),cudaMemcpyDeviceToHost));
  if(wrong){std::fprintf(stderr,"packedBF16 CPU-bit reference mismatch: %u\n",wrong);std::exit(3);}
  int warps=threads/32;std::vector<Stamp> stamps(grid*warps);CHECK(cudaMemcpy(stamps.data(),output,stamps.size()*sizeof(Stamp),cudaMemcpyDeviceToHost));
  for(int c=0;c<grid;++c)for(int w=0;w<warps;++w){auto s=stamps[c*warps+w];unsigned sum=0;for(int i=0;i<Pairs;++i)sum+=expected(c*threads+w*32,i,repetitions);
    if(s.sum!=sum){std::fprintf(stderr,"warp checksum mismatch\n");std::exit(4);}
    std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%u\n",Pairs,Feedback,threads,grid,trial,repetitions,c,w,s.c0,s.c1,s.n0,s.n1,s.sum);
  }
}
int main(){
  int sms=0;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;
  Stamp* output;unsigned* errors;CHECK(cudaMalloc(&output,132*8*sizeof(Stamp)));CHECK(cudaMalloc(&errors,sizeof(unsigned)));CHECK(cudaMemset(errors,0,sizeof(unsigned)));
  std::puts("pairs,feedback,threads,grid,trial,repetitions,cta,warp,c0,c1,n0,n1,checksum");
  for(int threads:{128,256})for(int grid:{1,32,132})for(int trial=0;trial<3;++trial)for(int reps:{128,256,512}){
    run<16,1>(output,errors,threads,grid,trial,reps);run<16,2>(output,errors,threads,grid,trial,reps);
    run<32,1>(output,errors,threads,grid,trial,reps);run<32,2>(output,errors,threads,grid,trial,reps);
    run<44,1>(output,errors,threads,grid,trial,reps);run<44,2>(output,errors,threads,grid,trial,reps);
    run<48,1>(output,errors,threads,grid,trial,reps);run<48,2>(output,errors,threads,grid,trial,reps);
    run<64,1>(output,errors,threads,grid,trial,reps);run<64,2>(output,errors,threads,grid,trial,reps);
  }
  CHECK(cudaFree(output));CHECK(cudaFree(errors));
}
