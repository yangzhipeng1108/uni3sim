// Independent rolling TMA copy window;no Attention,math or synthetic consumer.
// The warm flag requests prewarming/reuse;actual L2 residency requires counters.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
constexpr int Sms=132,MaxSteps=256,Width=128,MaxRows=128,MaxDepth=4;
struct Stamp{unsigned long long c0,c8,c1,n0,n8,n1;unsigned checksum,errors;};
__device__ __forceinline__ unsigned long long ticks(){unsigned long long x;asm volatile("mov.u64 %0,%%clock64;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ unsigned long long nanos(){unsigned long long x;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(x)::"memory");return x;}
__host__ __device__ unsigned value_at(unsigned index){return (index*17u+13u)&65535u;}
__global__ void initialize(unsigned short* input,size_t count){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)input[i]=value_at(unsigned(i));}
__global__ void evict(uint4* data,int count){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)data[i]=make_uint4(i,i+1,i+2,i+3);}
__global__ void prewarm(const unsigned short* input,unsigned* sink,int rows){
 unsigned sum=0;
 for(int tile=0;tile<8;++tile)for(int i=threadIdx.x;i<rows*Width;i+=blockDim.x){
  unsigned word,index=((blockIdx.x*MaxSteps+tile)*MaxRows)*Width+i;
  // L1 bypass;only the eight reusable tiles per CTA are brought into L2.
  asm volatile("ld.global.cg.u16 %0,[%1];":"=r"(word):"l"(input+index):"memory");sum+=word;
 }
 sink[blockIdx.x*blockDim.x+threadIdx.x]=sum;
}
__device__ __forceinline__ void wait_bar(unsigned bar,int phase){asm volatile("{.reg .pred p;WAIT_%=:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;@!p bra WAIT_%=;}"::"r"(bar),"r"(phase):"memory");}
template<int Rows>
__device__ __forceinline__ void issue(const CUtensorMap* map,unsigned dst,unsigned bar,int tile){
 int y=(blockIdx.x*MaxSteps+tile)*MaxRows;
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar),"r"(Rows*Width*2):"memory");
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,%2}],[%3];"::"r"(dst),"l"(map),"r"(y),"r"(bar):"memory");
}
template<int Rows,int Depth>
__global__ void __launch_bounds__(128,1) tma_stream(const __grid_constant__ CUtensorMap map,Stamp* output,int steps,int warm){
 extern __shared__ __align__(128) unsigned short tiles[];
 __shared__ __align__(8) unsigned long long barriers[MaxDepth];
 __shared__ unsigned errors;
 unsigned base=unsigned(__cvta_generic_to_shared(tiles)),bar=unsigned(__cvta_generic_to_shared(barriers));
 if(threadIdx.x==0){for(int s=0;s<Depth;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar+s*8):"memory");errors=0;asm volatile("fence.proxy.async.shared::cta;":::"memory");}
 __syncthreads();
 if(threadIdx.x==0){
  Stamp r{};r.n0=nanos();r.c0=ticks();
  #pragma unroll 1
  for(int i=0;i<Depth;++i)issue<Rows>(&map,base+i*Rows*Width*2,bar+i*8,warm?i%8:i);
  for(int i=0;i<steps;++i){
   int slot=i%Depth,phase=(i/Depth)&1;wait_bar(bar+slot*8,phase);
   unsigned tile=warm?i%8:i,index=((blockIdx.x*MaxSteps+tile)*MaxRows)*Width;
   // Check head andtail of every transfer before its slot can be reused.
   unsigned first=tiles[slot*Rows*Width],last=tiles[(slot+1)*Rows*Width-1];
   errors+=(first!=value_at(index))+(last!=value_at(index+Rows*Width-1));
   if(i==7){r.c8=ticks();r.n8=nanos();}
   int next=i+Depth;
   if(next<steps)issue<Rows>(&map,base+slot*Rows*Width*2,bar+slot*8,warm?next%8:next);
  }
  r.c1=ticks();r.n1=nanos();output[blockIdx.x]=r;
 }
 __syncthreads();
 // Full finalcontents of every slot validated outside the measured window.
 unsigned wrong=0,sum=0;
 for(int slot=0;slot<Depth;++slot){int i=steps-Depth+slot,tile=warm?i%8:i;unsigned index=((blockIdx.x*MaxSteps+tile)*MaxRows)*Width;
  for(int j=threadIdx.x;j<Rows*Width;j+=blockDim.x){unsigned v=tiles[slot*Rows*Width+j];wrong+=(v!=value_at(index+j));sum+=v;}
 }
 if(wrong)atomicAdd(&errors,wrong);
 if(threadIdx.x==0)output[blockIdx.x].checksum=sum;
 __syncthreads();if(threadIdx.x==0)output[blockIdx.x].errors=errors;
}
template<int Rows,int Depth>
void run(unsigned short* input,uint4* flush,Stamp* output,unsigned* sink,int grid,int steps,int warm,int trial){
 CUtensorMap map;cuuint64_t dims[2]={Width,Sms*MaxSteps*MaxRows},stride=Width*2;cuuint32_t box[2]={Width,Rows},step[2]={1,1};
 auto code=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT16,2,input,dims,&stride,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 if(code!=CUDA_SUCCESS){std::fprintf(stderr,"descriptor error %d\n",code);std::exit(3);}
 CHECK(cudaFuncSetAttribute(tma_stream<Rows,Depth>,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
 evict<<<Sms,256>>>(flush,384*1024*1024/16);CHECK(cudaDeviceSynchronize());
 if(warm){prewarm<<<grid,256>>>(input,sink,Rows);CHECK(cudaDeviceSynchronize());}
 tma_stream<Rows,Depth><<<grid,128,196608>>>(map,output,steps,warm);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 std::vector<Stamp> stamps(grid);CHECK(cudaMemcpy(stamps.data(),output,grid*sizeof(Stamp),cudaMemcpyDeviceToHost));
 for(int c=0;c<grid;++c){auto r=stamps[c];if(r.errors||!(r.c0<r.c8&&r.c8<r.c1&&r.n0<r.n8&&r.n8<r.n1)){std::fprintf(stderr,"stream numeric/window gate\n");std::exit(4);}
  unsigned expected=0;for(int s=0;s<Depth;++s){int i=steps-Depth+s,tile=warm?i%8:i;unsigned base=((c*MaxSteps+tile)*MaxRows)*Width;for(int j=0;j<Rows*Width;j+=128)expected+=value_at(base+j);}
  if(r.checksum!=expected){std::fprintf(stderr,"host slot checksum gate\n");std::exit(5);}
  std::printf("%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%u\n",Rows,Depth,grid,steps,warm,trial,c,r.c0,r.c8,r.c1,r.n0,r.n8,r.n1,r.checksum);
 }
}
int main(int argc,char**argv){
 bool counter=argc==2&&!std::strcmp(argv[1],"counter");if(argc!=1&&!counter)return 1;
 int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=Sms)return 6;
 unsigned short* input;uint4*flush;Stamp*output;unsigned*sink;size_t count=size_t(Sms)*MaxSteps*MaxRows*Width;
 CHECK(cudaMalloc(&input,count*2));CHECK(cudaMalloc(&flush,384*1024*1024));CHECK(cudaMalloc(&output,Sms*sizeof(Stamp)));CHECK(cudaMalloc(&sink,Sms*256*sizeof(unsigned)));
 initialize<<<Sms,256>>>(input,count);CHECK(cudaDeviceSynchronize());
 std::puts("rows,depth,grid,steps,warm,trial,cta,c0,c8,c1,n0,n8,n1,checksum");
 if(counter){run<128,1>(input,flush,output,sink,132,256,0,0);run<128,1>(input,flush,output,sink,132,256,1,0);run<128,4>(input,flush,output,sink,132,256,0,0);run<128,4>(input,flush,output,sink,132,256,1,0);}
 else for(int grid:{1,32,132})for(int steps:{64,128,256})for(int warm:{0,1})for(int trial=0;trial<3;++trial){
  run<32,1>(input,flush,output,sink,grid,steps,warm,trial);run<32,2>(input,flush,output,sink,grid,steps,warm,trial);run<32,4>(input,flush,output,sink,grid,steps,warm,trial);
  run<128,1>(input,flush,output,sink,grid,steps,warm,trial);run<128,2>(input,flush,output,sink,grid,steps,warm,trial);run<128,4>(input,flush,output,sink,grid,steps,warm,trial);
 }
 CHECK(cudaFree(input));CHECK(cudaFree(flush));CHECK(cudaFree(output));CHECK(cudaFree(sink));
}
