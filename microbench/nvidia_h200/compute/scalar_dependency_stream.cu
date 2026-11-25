// Independent finite unrolled MAX/ADD streams: no row reduction,exp,Tensor,
// sorting,Attention or repeated-loop controller. Every input is live before timing.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
constexpr int Shared=128*1024,Pool=8;
struct Reading{unsigned long long c0,c1;unsigned sm;};
__host__ __device__ float operand(int tid,int i){return float(1+(tid%31)*Pool+i)*(1.f/32768);}
template<int Threads,int Kind,int Chains,int Levels>
__global__ void __launch_bounds__(Threads,1) stream(const float* inputs,float* outputs,Reading* readings){
 extern __shared__ unsigned char allocation[];
 unsigned tid=threadIdx.x;float input[Pool],v[Chains];
 volatile __shared__ float ready[Threads];float total=0;
 #pragma unroll
 for(int i=0;i<Pool;++i){asm volatile("ld.global.volatile.f32 %0,[%1];":"=f"(input[i]):"l"(inputs+tid*Pool+i):"memory");total+=input[i];}
 #pragma unroll
 for(int c=0;c<Chains;++c){v[c]=-float(c+1);asm volatile("":"+f"(v[c]));}
 ready[tid]=total;__syncthreads();
 unsigned long long c0=clock64();
 #pragma unroll
 for(int i=0;i<Levels;++i){
  #pragma unroll
  for(int c=0;c<Chains;++c){
   if constexpr(Kind==0)asm volatile("max.f32 %0,%0,%1;":"+f"(v[c]):"f"(input[(i+c)%Pool]));
   else asm volatile("add.rn.f32 %0,%0,%1;":"+f"(v[c]):"f"(input[(i+c)%Pool]));
  }
 }
 // The predicate consumes every chain before the end clock;it is a common
 // control frame,not a primitive price. Nonfinite/positive MAX is validated below.
 float sum=0;
 #pragma unroll
 for(int c=0;c<Chains;++c)sum+=v[c];
 asm volatile("{.reg .pred p;setp.nan.f32 p,%0,%0;@p trap;}"::"f"(sum):"memory");
 unsigned long long c1=clock64();
 #pragma unroll
 for(int c=0;c<Chains;++c)outputs[(blockIdx.x*Threads+tid)*Chains+c]=v[c];
 if(tid%32==0){unsigned sm;asm volatile("mov.u32 %0,%%smid;":"=r"(sm));readings[blockIdx.x*(Threads/32)+tid/32]={c0,c1,sm};}
}
template<int T,int K,int C,int N>void capture(int grid,int trial,const float* inputs){
 auto fn=stream<T,K,C,N>;CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));int resident;
 CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,T,Shared));if(resident!=1)exit(3);
 float* output;Reading* readings;CHECK(cudaMalloc(&output,grid*T*C*sizeof(float)));CHECK(cudaMalloc(&readings,grid*(T/32)*sizeof(Reading)));
 fn<<<grid,T,Shared>>>(inputs,output,readings);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 std::vector<float> host(grid*T*C);std::vector<Reading> clocks(grid*(T/32));CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(float),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(clocks.data(),readings,clocks.size()*sizeof(Reading),cudaMemcpyDeviceToHost));
 for(int b=0;b<grid;++b)for(int tid=0;tid<T;++tid)for(int c=0;c<C;++c){float expected=-float(c+1);for(int i=0;i<N;++i){if constexpr(K==0)expected=fmaxf(expected,operand(tid,(i+c)%Pool));else expected+=operand(tid,(i+c)%Pool);}if(!std::isfinite(host[(b*T+tid)*C+c])||host[(b*T+tid)*C+c]!=expected){fprintf(stderr,"Numeric failure\n");exit(4);}}
 for(int b=0;b<grid;++b)for(int w=0;w<T/32;++w){auto r=clocks[b*(T/32)+w];if(r.c1<=r.c0)exit(5);printf("%d,%d,%d,%d,%d,%d,%d,%d,%u,%llu,%llu\n",T,K,C,N,grid,trial,b,w,r.sm,r.c0,r.c1);}
 CHECK(cudaFree(output));CHECK(cudaFree(readings));
}
template<int T,int K,int C,int N>void runs(const float* input){capture<T,K,C,N>(132,-1,input);for(int grid:{1,132})for(int trial=0;trial<3;++trial){capture<T,K,C,N>(grid,-2-trial,input);capture<T,K,C,N>(grid,trial,input);}}
template<int T,int K,int C>void widths(const float* input){runs<T,K,C,16>(input);runs<T,K,C,32>(input);runs<T,K,C,64>(input);runs<T,K,C,128>(input);}
template<int T,int K>void chains(const float* input){widths<T,K,1>(input);widths<T,K,2>(input);widths<T,K,4>(input);widths<T,K,8>(input);}
template<int T>void kinds(const float* input){chains<T,0>(input);chains<T,1>(input);}
int main(){cudaDeviceProp p;CHECK(cudaGetDeviceProperties(&p,0));if(p.major!=9||p.multiProcessorCount!=132)return 6;std::vector<float> h(256*Pool);for(int t=0;t<256;++t)for(int i=0;i<Pool;++i)h[t*Pool+i]=operand(t,i);float* input;CHECK(cudaMalloc(&input,h.size()*sizeof(float)));CHECK(cudaMemcpy(input,h.data(),h.size()*sizeof(float),cudaMemcpyHostToDevice));puts("threads,kind,chains,levels,grid,trial,cta,warp,sm,c0,c1");kinds<32>(input);kinds<128>(input);kinds<256>(input);CHECK(cudaFree(input));return 0;}
