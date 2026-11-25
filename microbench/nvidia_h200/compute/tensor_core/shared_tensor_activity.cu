// Independent on-chip Tensor activity, width and accumulator-chain controls.
// No GEMM matrices, TMA, output matrix, production shapes or library calls.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#ifdef OPERAND_DISTRIBUTION_CONTROLS
#include <cuda_fp16.h>
#include <cstring>
#include <random>
#endif
#include "shared_tensor_activity_helper.cuh"
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2); }} while(0)
#ifndef TENSOR_ACTIVE_CTAS
#define TENSOR_ACTIVE_CTAS 132
#endif
static_assert(TENSOR_ACTIVE_CTAS>0 && TENSOR_ACTIVE_CTAS<=132);
constexpr int kGrid=TENSOR_ACTIVE_CTAS,kLaunchGrid=132,kThreads=256,kShared=196608;
#ifdef OPERAND_DISTRIBUTION_CONTROLS
// Disjoint A/B arenas allow different distributions without role overlap at
// the largest measured accumulator footprint. No timed global input stream.
constexpr int kOperandBOffset=kShared/2;
struct DistributionControl { int gaussian_uniform, exponent_shift; };
static double initialize_operands(unsigned* device,int seed,DistributionControl control) {
  std::vector<unsigned short> host(kShared/2);
  std::mt19937 rng(unsigned(seed)+3571u);
  std::normal_distribution<float> normal(0.0f,1.0f);
  std::uniform_real_distribution<float> uniform(-1.0f,1.0f);
  double max_a=0,max_b=0;
  for(int i=0;i<kShared/2;++i) {
    float value=1.0f;
    if(seed) {
      if(control.gaussian_uniform) {
        // cuBLAS computes C^T = W * X^T: native M64 operand A is W.
        value=i<kShared/4?std::ldexp(uniform(rng),-control.exponent_shift):normal(rng);
      } else {
        unsigned offset=unsigned(i/2)*4;
        unsigned state=offset*747796405u+unsigned(seed)*2891336453u;
        state=((state>>((state>>28)+4))^state)*277803737u;state=(state>>22)^state;
        unsigned bits=i%2?((0x3000u+(((state>>3)&3u)<<10))|(((state>>3)&4u)<<13)|((state>>16)&1023u)):
            ((0x3000u+((state&3u)<<10))|((state&4u)<<13)|((state>>6)&1023u));
        host[i]=static_cast<unsigned short>(bits);
        __half half;std::memcpy(&half,&host[i],2);value=__half2float(half);
      }
    }
    __half half=__float2half_rn(value);std::memcpy(&host[i],&half,2);
    double magnitude=std::fabs(double(__half2float(half)));
    if(i<kShared/4)max_a=std::max(max_a,magnitude);else max_b=std::max(max_b,magnitude);
  }
  CHECK(cudaMemcpy(device,host.data(),kShared,cudaMemcpyHostToDevice));
  return max_a*max_b;
}
#else
constexpr int kOperandBOffset=65536;
#endif
__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}

template<int N,int Acc>
__global__ __launch_bounds__(kThreads,1) void tensor_activity_probe(
    int seed,int rounds,unsigned long long* times,float* sums,int dense_fraction
#ifdef OPERAND_DISTRIBUTION_CONTROLS
    ,const unsigned* operands
#endif
    ) {
  if(blockIdx.x>=kGrid)return;
  extern __shared__ __align__(1024) unsigned char shared[];
  for(int i=threadIdx.x*4;i<kShared;i+=blockDim.x*4) {
#ifdef OPERAND_DISTRIBUTION_CONTROLS
    *reinterpret_cast<unsigned*>(shared+i)=operands[i/4];
#else
    unsigned bits=0x3c003c00u;
    if(seed) {
      unsigned state=unsigned(i)*747796405u+unsigned(seed)*2891336453u;
      state=((state>>((state>>28)+4))^state)*277803737u;
      state=(state>>22)^state;
      unsigned low=(0x3000u+((state&3u)<<10))|((state&4u)<<13);
      unsigned high=(0x3000u+(((state>>3)&3u)<<10))|(((state>>3)&4u)<<13);
      if(dense_fraction) {
        low|=(state>>6)&1023u;
        high|=(state>>16)&1023u;
      }
      bits=low|(high<<16);
    }
    *reinterpret_cast<unsigned*>(shared+i)=bits;
#endif
  }
  __syncthreads();
  float accumulators[Acc][N/2]={};
  unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  base+=(threadIdx.x/128)*32768;
  unsigned long long c0=0,t0=0;
#ifdef SUSTAINED_DEVICE_ACTIVITY
  // Repeat fresh accumulator windows entirely on device before sampling.
  // Approximately 96 Mi cycles at the independently established 4096
  // FLOP/SM/cycle; no host launch/upload gap can interrupt this prelude.
  // Each window's first MMA reinitializes D, avoiding warmup saturation.
  int windows=1+(96*1024*1024)/(4*Acc*N*rounds);
#else
  int windows=1;
#endif
  for(int window=0;window<windows;++window) {
  if(window==windows-1) {c0=clock64();t0=global_ns();}
  wg_fence();
  for(int round=0;round<rounds;++round) {
#pragma unroll
    for(int step=0;step<4;++step) {
#pragma unroll
      for(int chain=0;chain<Acc;++chain) {
        uint64_t a=make_desc(base+chain*8192+step*32,16,1024);
        uint64_t b=make_desc(base+kOperandBOffset+chain*8192+step*32,16,1024);
        MatrixRequest<N>::issue(accumulators[chain],a,b,(round|step)!=0);
      }
    }
    wg_commit();wg_wait0();
  }
  }
  unsigned long long t1=global_ns(),c1=clock64();
  float sum=0;
#pragma unroll
  for(int chain=0;chain<Acc;++chain)
#pragma unroll
    for(int i=0;i<N/2;++i)sum+=accumulators[chain][i];
  sums[blockIdx.x*kThreads+threadIdx.x]=sum;
  if(threadIdx.x%128==0) {
    int index=(blockIdx.x*2+threadIdx.x/128)*2;
    times[index]=c1-c0;times[index+1]=t1-t0;
  }
}

template<int N,int Acc> void measure(unsigned long long* times,float* sums) {
  CHECK(cudaFuncSetAttribute(tensor_activity_probe<N,Acc>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,kShared));
  cudaFuncAttributes attr;
  CHECK(cudaFuncGetAttributes(&attr,tensor_activity_probe<N,Acc>));
  if(attr.localSizeBytes) { std::fprintf(stderr,"spills N%d Acc%d\n",N,Acc);std::exit(3); }
#ifdef OPERAND_DISTRIBUTION_CONTROLS
  unsigned* operands=nullptr;CHECK(cudaMalloc(&operands,kShared));
  const DistributionControl controls[]={{0,0},{1,0},{1,4},{1,6},{1,8}};
  for(DistributionControl control : controls) {
    int dense_fraction=1;
    initialize_operands(operands,1,control);
#define OPERAND_ARGUMENT ,operands
#else
#define OPERAND_ARGUMENT
#ifdef MANTISSA_ACTIVITY_CONTROLS
  for(int dense_fraction : {0,1}) {
#else
  for(int dense_fraction : {0}) {
#endif
#endif
#ifndef DISTRIBUTION_POINT_WARMUP
  auto start=std::chrono::steady_clock::now();
  do {tensor_activity_probe<N,Acc><<<kLaunchGrid,kThreads,kShared>>>(1,512,times,sums,dense_fraction OPERAND_ARGUMENT);CHECK(cudaDeviceSynchronize());}
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<2.0);
#endif
  std::vector<unsigned long long> host(kGrid*4);
  std::vector<float> check(kGrid*kThreads),short_check;
  for(int seed : {0,1,2,3})for(int rounds : {128,512}) {
#ifdef OPERAND_DISTRIBUTION_CONTROLS
    double bound_product=initialize_operands(operands,seed,control);
    // FP16 products are exact in FP32. Bound sequential accumulation and
    // the terminal sum using the standard gamma_n roundoff factors.
    constexpr double epsilon=0x1p-24;
    double rounding_factor=1.0/((1-rounds*64*epsilon)*(1-(N/2)*Acc*epsilon));
#else
    double bound_product=4.0;
    double rounding_factor=1.0;
#endif
#ifdef DISTRIBUTION_POINT_WARMUP
    if(seed) {
      auto point_start=std::chrono::steady_clock::now();
      do {tensor_activity_probe<N,Acc><<<kLaunchGrid,kThreads,kShared>>>(seed,rounds,times,sums,dense_fraction OPERAND_ARGUMENT);CHECK(cudaDeviceSynchronize());}
      while(std::chrono::duration<double>(std::chrono::steady_clock::now()-point_start).count()<2.0);
    }
#endif
    std::vector<double> cycles,ns,mhz;
    for(int trial=0;trial<9;++trial) {
      tensor_activity_probe<N,Acc><<<kLaunchGrid,kThreads,kShared>>>(seed,rounds,times,sums,dense_fraction OPERAND_ARGUMENT);
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(host.data(),times,host.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(check.data(),sums,check.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for(size_t i=0;i<check.size();++i) {
        if(!std::isfinite(check[i]))std::exit(4);
        if(seed==0&&check[i]!=float(rounds*64*(N/2)*Acc))std::exit(5);
        if(!dense_fraction&&rounds==512&&check[i]!=4*short_check[i])std::exit(6);
        // Random fractions need rounded FP32 accumulation, unlike the exact
        // dyadic legacy controls. Check finiteness and the analytical bound:
        // Per-input maxima (legacy: |a|,|b|<2), 64 products per round.
        if(dense_fraction&&std::fabs(check[i])>float(rounding_factor*bound_product*rounds*64*(N/2)*Acc))std::exit(8);
      }
      if(rounds==128)short_check=check;
      if(trial>=2)for(int i=0;i<kGrid*2;++i) {
        if(!host[2*i]||!host[2*i+1])std::exit(7);
        cycles.push_back(host[2*i]);ns.push_back(host[2*i+1]);
        mhz.push_back(1000.0*host[2*i]/host[2*i+1]);
      }
    }
    auto median=[](std::vector<double>& v){std::sort(v.begin(),v.end());return v[v.size()/2];};
    std::printf("{\"instruction_n\":%d,\"accumulator_chains\":%d,\"seed\":%d,"
      "\"rounds\":%d,\"grid_ctas\":%d,\"launched_grid_ctas\":132,\"threads_per_cta\":%d,"
      "\"flops_per_round_per_sm\":%d,\"cycles_p50\":%.3f,\"ns_p50\":%.3f,"
      "\"clock_mhz\":%.3f,\"registers\":%d,\"operand_pattern\":\"%s\","
#ifdef OPERAND_DISTRIBUTION_CONTROLS
      "\"weight_exponent_shift\":%d,\"measured_input_product_bound\":%.9g,"
#endif
      "\"numerical_check_passed\":true}\n",
      N,Acc,seed,rounds,kGrid,kThreads,2*4*Acc*2*64*N*16,
      median(cycles),median(ns),median(mhz),attr.numRegs,
#ifdef OPERAND_DISTRIBUTION_CONTROLS
      control.gaussian_uniform?"gaussian_input_uniform_weight":"random_sign_exponent_and_mantissa",
      control.exponent_shift,bound_product);
#else
      dense_fraction?"random_sign_exponent_and_mantissa":"signed_power_of_two");
#endif
  }
  }
#ifdef OPERAND_DISTRIBUTION_CONTROLS
  CHECK(cudaFree(operands));
#endif
}

int main() {
  unsigned long long* times;float* sums;
  CHECK(cudaMalloc(&times,kGrid*4*sizeof(unsigned long long)));
  CHECK(cudaMalloc(&sums,kGrid*kThreads*sizeof(float)));
#ifdef DISTRIBUTION_PILOT_CONTROLS
  measure<64,3>(times,sums);measure<96,3>(times,sums);measure<128,2>(times,sums);
#else
#ifdef EXTENDED_CONTROLS
  measure<64,2>(times,sums);measure<64,4>(times,sums);measure<64,5>(times,sums);
  measure<96,2>(times,sums);measure<96,3>(times,sums);measure<96,4>(times,sums);
  measure<128,1>(times,sums);measure<128,2>(times,sums);
#else
  measure<64,1>(times,sums);measure<64,3>(times,sums);
  measure<96,1>(times,sums);measure<96,3>(times,sums);
  measure<128,1>(times,sums);measure<128,3>(times,sums);
#endif
#endif
  CHECK(cudaFree(times));CHECK(cudaFree(sums));
}
