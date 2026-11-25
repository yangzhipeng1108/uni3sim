// Hardware clock only: independent 48-KiB cp.async ring plus SS WGMMA.
// No routing, expert histogram, weight tensor, GEMM output or model input.
#include "wgmma_ss_helper.cuh"
#include <algorithm>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_bf16.h>
constexpr int Threads=256, Panel=49152, Stages=4, Shared=Panel*Stages, MaximumGrid=65536;
__device__ uint64_t timer_ns(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__global__ void initialize(unsigned* data,size_t words){
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=blockDim.x*gridDim.x){
  unsigned x=unsigned(i)*1664525u+1013904223u;x^=x>>13;x*=0x85ebca6bu;
  unsigned lo=0x3e00u|(x&127u)|((x>>8)&0x8000u);
  unsigned hi=0x3e00u|((x>>16)&127u)|((x>>2)&0x8000u);data[i]=lo|(hi<<16);
 }
}
__device__ void issue(unsigned char* to,const unsigned char* backing,int panels,int round){
 int panel=(int(blockIdx.x)*17+round)%panels;
 for(int v=threadIdx.x;v<Panel/16;v+=Threads){unsigned addr=unsigned(__cvta_generic_to_shared(to+v*16));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(addr),"l"(backing+size_t(panel)*Panel+v*16):"memory");}
 asm volatile("cp.async.commit_group;":::"memory");
}
template<int Mode,int Rounds>
__global__ __launch_bounds__(Threads,1) void panel_clock(const unsigned char* backing,int panels,uint64_t* stamps,float* output){
 extern __shared__ __align__(1024) unsigned char sm[];
 auto c0=clock64();auto n0=timer_ns();float d[128];
#pragma unroll
 for(int j=0;j<128;++j)d[j]=0;
 if constexpr(Mode==1){for(int j=threadIdx.x;j<Shared/4;j+=Threads)reinterpret_cast<unsigned*>(sm)[j]=0x3e803e80u;__syncthreads();}
 else{for(int s=0;s<Stages-1;++s)issue(sm+s*Panel,backing,panels,s);}
 unsigned checksum=0;
 for(int round=0;round<Rounds;++round){int slot=round%Stages;
  if constexpr(Mode!=1){if(round<Rounds-Stages+1)asm volatile("cp.async.wait_group 2;":::"memory");else asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();}
  const auto* panel=sm+slot*Panel;
  checksum+=reinterpret_cast<const volatile unsigned*>(panel)[threadIdx.x];
  if constexpr(Mode!=0){wg_fence();auto base=unsigned(__cvta_generic_to_shared(panel));uint64_t a[4],b[4];
#pragma unroll
   for(int k=0;k<4;++k){a[k]=make_desc(base+k*32,16,1024);b[k]=make_desc(base+16384+k*32,16,1024);}
   wgmma_ss_n256_group_wait1(d,a,b,round!=0);
  }
  if constexpr(Mode!=1){__syncthreads();int next=round+Stages-1;if(next<Rounds)issue(sm+(next%Stages)*Panel,backing,panels,next);}
 }
 if constexpr(Mode!=0)wg_wait0();
 auto c1=clock64();auto n1=timer_ns();
 if(threadIdx.x==0){stamps[blockIdx.x*4]=c0;stamps[blockIdx.x*4+1]=c1;stamps[blockIdx.x*4+2]=n0;stamps[blockIdx.x*4+3]=n1;
  float sum=0;
#pragma unroll
  for(int j=0;j<128;++j)sum+=d[j];output[blockIdx.x]=sum+float(checksum&255);}
}
template<int Mode,int Rounds>
void measure(const unsigned char* backing,int panels,int grid,uint64_t* stamps,float* output,int rotation){
 CHECK(cudaFuncSetAttribute(panel_clock<Mode,Rounds>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 for(int w=0;w<4;++w)panel_clock<Mode,Rounds><<<grid,Threads,Shared>>>(backing,panels,stamps,output);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 std::vector<uint64_t> values(size_t(grid)*4);std::vector<float> samples(grid);
 for(int trial=0;trial<63;++trial){panel_clock<Mode,Rounds><<<grid,Threads,Shared>>>(backing,panels,stamps,output);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(values.data(),stamps,values.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(samples.data(),output,grid*sizeof(float),cudaMemcpyDeviceToHost));
  uint64_t cycles=0,ns=0;std::vector<double> hz;hz.reserve(grid);
  for(int i=0;i<grid;++i){auto dc=values[i*4+1]-values[i*4];auto dt=values[i*4+3]-values[i*4+2];if(!dc||!dt||!std::isfinite(samples[i]))std::exit(3);cycles+=dc;ns+=dt;hz.push_back(double(dc)*1000/dt);}
  std::sort(hz.begin(),hz.end());std::printf("{\"mode\":%d,\"rounds\":%d,\"grid\":%d,\"panel_count\":%d,\"backing_bytes\":%d,\"trial\":%d,\"sum_cycles\":%llu,\"sum_ns\":%llu,\"weighted_mhz\":%.9g,\"p10_mhz\":%.9g,\"median_mhz\":%.9g,\"p90_mhz\":%.9g}\n",Mode,Rounds,grid,panels,panels*Panel,trial,(unsigned long long)cycles,(unsigned long long)ns,double(cycles)*1000/ns,hz[grid/10],hz[grid/2],hz[grid*9/10]);
 }
}
template<int R> void modes(const unsigned char*b,int p,int g,uint64_t*t,float*s,int rotation){measure<0,R>(b,p,g,t,s,rotation);measure<1,R>(b,p,g,t,s,rotation);measure<2,R>(b,p,g,t,s,rotation);}
int main(int argc,char**argv){if(argc!=2)return 2;int rotation=std::atoi(argv[1]);if(rotation<0||rotation>1)return 2;cudaDeviceProp p;CHECK(cudaGetDeviceProperties(&p,0));if(p.major!=9||p.minor!=0||p.multiProcessorCount!=132)return 2;
 std::printf("{\"device\":\"%s\",\"sms\":%d,\"shared_bytes\":%d,\"panel_bytes\":%d}\n",p.name,p.multiProcessorCount,Shared,Panel);
 unsigned char*backing;uint64_t*stamps;float*out;constexpr int large=(128*1024*1024)/Panel;
 CHECK(cudaMalloc(&backing,size_t(large)*Panel));CHECK(cudaMalloc(&stamps,size_t(MaximumGrid)*4*sizeof(uint64_t)));CHECK(cudaMalloc(&out,MaximumGrid*sizeof(float)));
 initialize<<<1056,256>>>(reinterpret_cast<unsigned*>(backing),size_t(large)*Panel/4);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 CHECK(cudaFuncSetAttribute(panel_clock<2,128>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 auto heat_start=std::chrono::steady_clock::now();
 do {panel_clock<2,128><<<2112,Threads,Shared>>>(backing,large,stamps,out);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());} while(std::chrono::duration<double>(std::chrono::steady_clock::now()-heat_start).count()<2.0);
 for(int panels:{(16*1024*1024)/Panel,large})for(int grid:{132,2112,MaximumGrid}){modes<8>(backing,panels,grid,stamps,out,rotation);modes<32>(backing,panels,grid,stamps,out,rotation);modes<128>(backing,panels,grid,stamps,out,rotation);}
 CHECK(cudaFree(backing));CHECK(cudaFree(stamps));CHECK(cudaFree(out));return 0;}
