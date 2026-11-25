// Independent TMA-write / Tensor-read contention. Copy and math use disjoint
// data, with no producer/consumer dependency, matrix output or GEMM program.
#include "arbitration_helper.cuh"
#include <algorithm>
#include <cmath>
#include <chrono>
constexpr int kGrid=132;
constexpr int kMathBytes=131072;
constexpr int kCopyBytes=65536;
constexpr int kSharedBytes=kMathBytes+kCopyBytes+64;
__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}
__global__ __launch_bounds__(384,1) void arbitration_probe(
    const __grid_constant__ CUtensorMap map,int mode,int rounds,int rows,
    int cached,int patterned,unsigned long long* times,float* sums) {
  extern __shared__ __align__(1024) unsigned char shared[];
  for(int i=threadIdx.x*4;i<kMathBytes;i+=blockDim.x*4) {
    unsigned bits=0x3c003c00u;
    if(patterned) {
      // Independent fixed seeds, signed powers of two: exact FP16 values,
      // finite FP32 accumulations, with activity beyond an all-one matrix.
      unsigned state=unsigned(i)*747796405u+unsigned(patterned)*2891336453u;
      state=((state>>((state>>28)+4))^state)*277803737u;
      state=(state>>22)^state;
      unsigned low=0x3000u+((state&3u)<<10)|((state&4u)<<13);
      unsigned high=0x3000u+(((state>>3)&3u)<<10)|(((state>>3)&4u)<<13);
      bits=low|(high<<16);
    }
    *reinterpret_cast<unsigned*>(shared+i)=bits;
  }
  auto* full=reinterpret_cast<uint64_t*>(shared+kMathBytes+kCopyBytes);
  const int box_bytes=rows*128,depth=kCopyBytes/box_bytes;
  if(threadIdx.x==256) {
    for(int i=0;i<depth;++i)mbar_init(full+i,1);
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncthreads();
  int role=threadIdx.x/128;
  unsigned long long cycle_start=clock64(),start=global_ns();
  float sum=0;
  if(role<2&&mode!=1) {
    float d[64]={};
    unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
    base+=role*32768;
    wg_fence();
    for(int round=0;round<rounds;++round) {
#pragma unroll
      for(int step=0;step<4;++step) {
        uint64_t a=make_desc(base+step*32,16,1024);
        uint64_t b=make_desc(base+65536+step*32,16,1024);
        wgmma_ss_n128_kmajor(d,a,b,(round|step)!=0);
      }
      wg_commit();wg_wait0();
    }
#pragma unroll
    for(int i=0;i<64;++i)sum+=d[i];
  }
  if(threadIdx.x==256&&mode!=0) {
    for(int round=0;round<rounds;++round) {
      int slot=round%depth,phase=(round/depth)&1;
      if(round>=depth)mbar_wait(full+slot,phase^1);
      mbar_expect_tx(full+slot,box_bytes);
      tma_load_2d(shared+kMathBytes+slot*box_bytes,&map,full+slot,
          round*128,cached?0:blockIdx.x*rows);
    }
    for(int slot=0;slot<depth;++slot)mbar_wait(full+slot,((rounds/depth)-1)&1);
  }
  unsigned long long stop=global_ns(),cycle_stop=clock64();
  if(threadIdx.x%128==0) {
    int index=(blockIdx.x*3+role)*4;
    times[index]=start;times[index+1]=stop;
    times[index+2]=cycle_stop-cycle_start;times[index+3]=stop-start;
    sums[blockIdx.x*3+role]=sum;
  }
}
int main(int argc,char**) {
  const bool activity_controls=argc==2;
  CHECK(cudaFuncSetAttribute(arbitration_probe,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes));
  cudaFuncAttributes attr;CHECK(cudaFuncGetAttributes(&attr,arbitration_probe));
  if(attr.localSizeBytes)return 3;
  uint8_t* input;unsigned long long* times;float* sums;
  CHECK(cudaMalloc(&input,65536ull*kGrid*256));
  CHECK(cudaMemset(input,1,65536ull*kGrid*256));
  CHECK(cudaMalloc(&times,kGrid*3*4*sizeof(unsigned long long)));
  CHECK(cudaMalloc(&sums,kGrid*3*sizeof(float)));
  for(int patterned=0;patterned<=(activity_controls?3:0);++patterned) {
  auto warm_map=make_map(input,kGrid*256,65536,128);
  auto warm=std::chrono::steady_clock::now();
  do {arbitration_probe<<<kGrid,384,kSharedBytes>>>(warm_map,0,512,128,1,patterned,times,sums);CHECK(cudaDeviceSynchronize());}
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.0);
  std::vector<float> short_checks;
  for(int rows : {128,256})for(int cached : {0,1})for(int rounds : {128,512})for(int mode : {0,1,2}) {
    if(activity_controls&&(rows!=128||cached!=0||mode!=0))continue;
    auto map=make_map(input,kGrid*256,65536,rows);
    std::vector<double> math_ns,math_cycles,copy_ns,mhz,span;
    std::vector<unsigned long long> host(kGrid*3*4);std::vector<float> check(kGrid*3);
    for(int trial=0;trial<9;++trial) {
      arbitration_probe<<<kGrid,384,kSharedBytes>>>(map,mode,rounds,rows,cached,patterned,times,sums);
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(host.data(),times,host.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(check.data(),sums,check.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(int cta=0;cta<kGrid;++cta) {
        unsigned long long begin=~0ull,end=0;
        for(int role=0;role<3;++role) {
          if((mode==0&&role==2)||(mode==1&&role<2))continue;
          int index=(cta*3+role)*4;
          if(host[index+3]==0)return 4;
          if(role<2) {
            float value=check[cta*3+role];
            if(!std::isfinite(value))return 5;
            if(!patterned&&value!=float(rounds*4096))return 5;
            if(patterned&&rounds==512&&value!=4*short_checks[cta*3+role])return 6;
          }
          if(trial>=2) {
            if(role<2) {math_ns.push_back(host[index+3]);math_cycles.push_back(host[index+2]);mhz.push_back(1000.0*host[index+2]/host[index+3]);}
            else copy_ns.push_back(host[index+3]);
          }
          begin=std::min(begin,host[index]);end=std::max(end,host[index+1]);
        }
        if(trial>=2)span.push_back(end-begin);
      }
      if(patterned&&rounds==128)short_checks=check;
    }
    auto median=[](std::vector<double>& values){if(values.empty())return 0.0;std::sort(values.begin(),values.end());return values[values.size()/2];};
    printf("{\"rows\":%d,\"cached\":%d,\"rounds\":%d,\"mode\":%d,\"patterned\":%d,"
        "\"math_ns_p50\":%.3f,\"math_cycles_p50\":%.3f,\"copy_ns_p50\":%.3f,\"span_ns_p50\":%.3f,"
        "\"math_sm_mhz\":%.3f,\"registers\":%d,\"numerical_check_passed\":true}\n",
        rows,cached,rounds,mode,patterned,median(math_ns),median(math_cycles),median(copy_ns),median(span),median(mhz),attr.numRegs);
  }
  }
  CHECK(cudaFree(input));CHECK(cudaFree(times));CHECK(cudaFree(sums));
}
