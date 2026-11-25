// Independent register-pool CTA envelope with explicit store-drain controls.
// No Attention, matrix operations, KV, reductions or production scheduler.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "outer_pressure.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
constexpr int SharedBytes=128*1024;
struct Stamp{unsigned long long n0,n1,c0,c1;};
__device__ unsigned long long nanos(){unsigned long long x;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(x)::"memory");return x;}
template<int Threads,bool High,bool Drain>
__global__ void __launch_bounds__(Threads,1) boundary_envelope(unsigned seed,Stamp* output,float* sink,int hold){
 __shared__ Stamp stamp;
 if(threadIdx.x==0){stamp.n0=nanos();stamp.c0=clock64();}
 __syncthreads();
 if constexpr(High){
  if(threadIdx.x<128){
   if constexpr(Threads==256)asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
   else asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
  }else{
   if constexpr(Threads==256)asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
   else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  }
 }
 float value=0;
 if(threadIdx.x>=128){value=float(threadIdx.x)+float(seed&31);if constexpr(High)value=pressure<Threads>(value);}
 auto begin=clock64();
 while(clock64()-begin<static_cast<unsigned>(hold))asm volatile("add.rn.ftz.f32 %0,%0,0f00000000;":"+f"(value));
 if(threadIdx.x>=128)sink[blockIdx.x*(Threads-128)+threadIdx.x-128]=value;
 if constexpr(Drain)__threadfence();
 __syncthreads();
 if(threadIdx.x==0){stamp.c1=clock64();stamp.n1=nanos();}
 // Peers remain resident through the single elected endpoint timestamp.
 __syncthreads();
 if(threadIdx.x==0)output[blockIdx.x]=stamp;
}
template<int Threads,bool High,bool Drain>
void run(int grid,bool extended,int hold,int& id,Stamp* output,float* sink){
 auto kernel=boundary_envelope<Threads,High,Drain>;cudaFuncAttributes attr;CHECK(cudaFuncGetAttributes(&attr,kernel));
 const int pool=((attr.numRegs+7)/8)*8*Threads,required=Threads==256?39936:64512;
 if(High&&pool<required){std::fprintf(stderr,"initial register pool too small\n");std::exit(3);}
 CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
 for(int trial=-2;trial<3;++trial){
  if(extended){cudaLaunchAttribute a{};a.id=cudaLaunchAttributeProgrammaticStreamSerialization;a.val.programmaticStreamSerializationAllowed=1;cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(Threads);cfg.dynamicSmemBytes=SharedBytes;cfg.attrs=&a;cfg.numAttrs=1;CHECK(cudaLaunchKernelEx(&cfg,kernel,17u,output,sink,hold));}
  else kernel<<<grid,Threads,SharedBytes>>>(17u,output,sink,hold);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  std::vector<Stamp> clocks(grid);std::vector<float> values(grid*(Threads-128));
  CHECK(cudaMemcpy(clocks.data(),output,clocks.size()*sizeof(Stamp),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(values.data(),sink,values.size()*sizeof(float),cudaMemcpyDeviceToHost));
  for(int c=0;c<grid;++c){
   for(int t=128;t<Threads;++t){unsigned count=Threads==256?224:144;float expected=float(t+17);if(High)expected=float(count*(t+17)+count*(count-1)/2);if(values[c*(Threads-128)+t-128]!=expected){std::fprintf(stderr,"numeric mismatch\n");std::exit(4);}}
   auto s=clocks[c];if(!(s.n0<s.n1&&s.c0<s.c1)){std::fprintf(stderr,"timestamp order\n");std::exit(5);}
   std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu\n",id,grid,Threads,int(High),int(Drain),int(extended),hold,trial,attr.numRegs,s.n0,s.n1,s.c1-s.c0);
  }
  ++id;
 }
}
template<int T,bool D>void pair(int grid,bool ex,int hold,int& id,Stamp* o,float* s){run<T,false,D>(grid,ex,hold,id,o,s);run<T,true,D>(grid,ex,hold,id,o,s);}
int main(){
 int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 6;
 Stamp* output;float* sink;CHECK(cudaMalloc(&output,132*sizeof(Stamp)));CHECK(cudaMalloc(&sink,132*256*sizeof(float)));int id=0;
 std::puts("launch_id,grid,threads,high,drain,extended,hold_cycles,trial,registers,begin_ns,end_ns,body_cycles");
 for(int grid:{1,132})for(bool ex:{false,true})for(int hold:{0,4096,32768}){pair<256,false>(grid,ex,hold,id,output,sink);pair<256,true>(grid,ex,hold,id,output,sink);pair<384,false>(grid,ex,hold,id,output,sink);pair<384,true>(grid,ex,hold,id,output,sink);}
 CHECK(cudaFree(output));CHECK(cudaFree(sink));
}
