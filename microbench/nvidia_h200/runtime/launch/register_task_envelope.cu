// Independent CTA resource/parameter launch envelope, not an operator.
// NSYS measures the full kernel; globaltimer encloses the synthetic body.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "envelope_pressure.cuh"
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){ \
 fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int SharedBytes=128*1024;
template<int Words> struct Payload {unsigned long long words[Words];};
__device__ unsigned long long global_ns() {
  unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t)::"memory");return t;
}
template<int Threads,bool High,int Words>
__global__ void __launch_bounds__(Threads,1)
register_envelope(const __grid_constant__ Payload<Words> payload,
                  unsigned long long* trace,float* sink,int hold) {
  auto n0=global_ns();auto c0=clock64();
  extern __shared__ volatile unsigned scratch[];
  if(threadIdx.x==0) scratch[0]=1;
  if constexpr(High) {
    if(threadIdx.x<128) {
      if constexpr(Threads==256) asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
      if constexpr(Threads==384) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
      if constexpr(Threads==512) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    } else {
      if constexpr(Threads==256) asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
      if constexpr(Threads==384) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
      if constexpr(Threads==512) asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
    }
  }
  float value=0;
  if(threadIdx.x>=128) {
    value=float(threadIdx.x)+float((payload.words[0]^payload.words[Words-1])&255);
    if constexpr(High) value=pressure<Threads>(value);
  }
  auto spin=clock64();
  while(clock64()-spin<static_cast<unsigned>(hold)) {
    asm volatile("add.rn.ftz.f32 %0,%0,0f00000000;":"+f"(value));
  }
  if(threadIdx.x>=128) sink[blockIdx.x*(Threads-128)+threadIdx.x-128]=value;
  __syncthreads();
  auto c1=clock64();auto n1=global_ns();
  if(threadIdx.x==0) {
    trace[blockIdx.x*3]=n0;trace[blockIdx.x*3+1]=n1;trace[blockIdx.x*3+2]=c1-c0;
  }
}
template<int Threads,bool High,int Words>
void run(int grid,bool extended,int hold,int& launch_id,unsigned long long* trace,float* sink) {
  auto kernel=register_envelope<Threads,High,Words>;
  cudaFuncAttributes attributes;CHECK(cudaFuncGetAttributes(&attributes,kernel));
  const int pool=((attributes.numRegs+7)/8)*8*Threads;
  const int required=Threads==256?39936:(Threads==384?64512:65536);
  if(High && pool<required) {fprintf(stderr,"insufficient initial register pool\n");exit(3);}
  CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  Payload<Words> payload{};for(int i=0;i<Words;++i) payload.words[i]=i*9973u+17;
  for(int trial=-2;trial<3;++trial) {
    if(extended) {
      cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute.val.programmaticStreamSerializationAllowed=1;
      cudaLaunchConfig_t config{};config.gridDim=dim3(grid);config.blockDim=dim3(Threads);
      config.dynamicSmemBytes=SharedBytes;config.attrs=&attribute;config.numAttrs=1;
      CHECK(cudaLaunchKernelEx(&config,kernel,payload,trace,sink,hold));
    } else kernel<<<grid,Threads,SharedBytes>>>(payload,trace,sink,hold);
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid*3);
    CHECK(cudaMemcpy(host.data(),trace,host.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu\n",
      launch_id,grid,Threads,High,Words*8,extended,hold,trial,attributes.numRegs,
      host[c*3],host[c*3+1],host[c*3+2]);
    ++launch_id;
  }
}
template<int Threads,int Words>
void modes(int grid,bool extended,int hold,int& id,unsigned long long* trace,float* sink) {
  run<Threads,false,Words>(grid,extended,hold,id,trace,sink);
  run<Threads,true,Words>(grid,extended,hold,id,trace,sink);
}
int main() {
  unsigned long long* trace;float* sink;
  CHECK(cudaMalloc(&trace,132*3*sizeof(*trace)));CHECK(cudaMalloc(&sink,132*384*sizeof(*sink)));
  int id=0;
  printf("launch_id,grid,threads,high,payload_bytes,extended,hold_cycles,trial,registers,begin_ns,end_ns,body_cycles\n");
  for(int grid:{1,132}) for(bool extended:{false,true}) for(int hold:{0,4096,32768}) {
    modes<256,1>(grid,extended,hold,id,trace,sink);modes<256,128>(grid,extended,hold,id,trace,sink);
    modes<256,480>(grid,extended,hold,id,trace,sink);
    modes<384,1>(grid,extended,hold,id,trace,sink);modes<384,128>(grid,extended,hold,id,trace,sink);
    modes<384,480>(grid,extended,hold,id,trace,sink);
    modes<512,1>(grid,extended,hold,id,trace,sink);modes<512,128>(grid,extended,hold,id,trace,sink);
    modes<512,480>(grid,extended,hold,id,trace,sink);
  }
  CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
