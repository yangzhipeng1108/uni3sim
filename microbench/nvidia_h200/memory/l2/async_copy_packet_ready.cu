// Independent finite cp.async packet response, including issue/wait/shared use.
// No Attention, matrix, reduction or production-duration input. Diagnostic only.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(2); } } while(0)
using U64=unsigned long long;
constexpr int THREADS=256, SHARED=196608, SAMPLES=21;
struct Record { U64 c0,c1,g0,g1,outer_c,outer_g; unsigned sm,error; };
struct Prelude { U64 cycles,ns; unsigned sm; };

__device__ __forceinline__ U64 gt() {
  U64 value; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory"); return value;
}
__device__ __forceinline__ unsigned smid() {
  unsigned value; asm volatile("mov.u32 %0, %%smid;":"=r"(value)); return value;
}
__host__ __device__ unsigned pattern(int sample,int cta,int tid,int word) {
  return 0x3e800000u ^ (sample*104729u+cta*1009u+tid*17u+word*3u);
}
template<int WORDS> __device__ size_t offset(int tid,int pitch) {
  constexpr int row_threads=WORDS==1?8:32;
  return size_t(tid/row_threads)*pitch+size_t(tid%row_threads)*WORDS;
}
template<int WORDS> __global__ void initialize(unsigned* input,size_t span,int pitch,int grid) {
  int cta=blockIdx.x%grid,sample=blockIdx.x/grid,tid=threadIdx.x;
  unsigned* ptr=input+(size_t(sample)*grid+cta)*span+offset<WORDS>(tid,pitch);
  for(int w=0;w<WORDS;++w) ptr[w]=pattern(sample,cta,tid,w);
}
// Touch 512KiB private to each resident CTA through .ca. The source for the
// probe is separate and fresh per sample; subsequent warm_source uses .cg.
__global__ void evict_l1(const unsigned* input,unsigned* sink) {
  extern __shared__ unsigned reserved[];
  unsigned value=0;
  const unsigned* ptr=input+size_t(blockIdx.x)*131072;
  for(int i=threadIdx.x;i<131072;i+=THREADS) {
    unsigned word; asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(word):"l"(ptr+i):"memory");
    value^=word;
  }
  if(threadIdx.x==0) sink[blockIdx.x]=value;
}
template<int WORDS> __global__ void warm_source(const unsigned* input,size_t span,
                                               int pitch,int grid,int sample,unsigned* sink) {
  const unsigned* ptr=input+(size_t(sample)*grid+blockIdx.x)*span+offset<WORDS>(threadIdx.x,pitch);
  unsigned value=0;
  for(int w=0;w<WORDS;++w) {
    unsigned word; asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(word):"l"(ptr+w):"memory");
    value^=word;
  }
  if(threadIdx.x==0) sink[blockIdx.x]=value;
}
__global__ void prelude(Prelude* output) {
  extern __shared__ unsigned reserved[];
  U64 g0=gt(),c0=clock64();
  while(gt()-g0<3000000ull) asm volatile("":::"memory");
  U64 c1=clock64(),g1=gt();
  if(threadIdx.x==0) output[blockIdx.x]={c1-c0,g1-g0,smid()};
}

template<int WORDS,bool COPY> __global__ void packet_probe(
    const unsigned* input,size_t span,int pitch,int grid,int sample,Record* output) {
  extern __shared__ __align__(16) unsigned shared[];
  int tid=threadIdx.x;
  U64 outer_g0=gt(),outer_c0=clock64();
  unsigned expected=pattern(sample,blockIdx.x,tid,0);
  // Both arms consume identical shared values. COPY must overwrite sentinel.
  for(int w=0;w<WORDS;++w)
    shared[tid*WORDS+w]=COPY?0xdeadbeefu:pattern(sample,blockIdx.x,tid,w);
  const unsigned* ptr=input+(size_t(sample)*grid+blockIdx.x)*span+offset<WORDS>(tid,pitch);
  unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(shared+tid*WORDS));
  // An untimed residency delay permits the short full-grid packets to
  // coexist. Actual SMID uniqueness is checked; no placement is inferred.
  U64 residency_start=gt();
  while(gt()-residency_start<50000ull) asm volatile("":::"memory");
  __syncthreads();
  U64 g0=gt(),c0;
  // An actual clock-result control dependency precedes the copy. No claim
  // that this frame is free or that clock64 is an instantaneous SI clock.
  asm volatile("{ .reg .pred p; .reg .u64 ptrcheck;\n"
               "add.u64 ptrcheck,%1,%2; setp.eq.u64 p,ptrcheck,0; @p trap;\n"
               "setp.eq.u32 p,%3,0; @p trap; mov.u64 %0, %%clock64;\n"
               "setp.eq.u64 p,%0,0; @p trap; }":"=l"(c0):
               "l"(ptr),"l"(static_cast<U64>(address)),"r"(expected):"memory");
  if constexpr(COPY) {
    if constexpr(WORDS==1)
      asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(address),"l"(ptr):"memory");
    else
      asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(address),"l"(ptr):"memory");
  }
  asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");
  unsigned actual;
  asm volatile("ld.volatile.shared.u32 %0,[%1];":"=r"(actual):"r"(address):"memory");
  U64 c1;
  // The end clock depends on the actual consumed shared value through a
  // real trap branch. Textual LDS order alone would not prove completion.
  asm volatile("{ .reg .pred p; setp.ne.u32 p,%1,%2; @p trap;\n"
               "mov.u64 %0, %%clock64; }":"=l"(c1):"r"(actual),"r"(expected):"memory");
  U64 g1=gt();
  __syncthreads();
  unsigned error=0;
  for(int w=0;w<WORDS;++w) {
    unsigned value=static_cast<volatile unsigned*>(shared)[tid*WORDS+w];
    error|=value!=pattern(sample,blockIdx.x,tid,w);
  }
  unsigned any=__syncthreads_or(error);
  U64 outer_c1=clock64(),outer_g1=gt();
  if(tid%32==0) output[blockIdx.x*8+tid/32]=
    {c0,c1,g0,g1,outer_c1-outer_c0,outer_g1-outer_g0,smid(),any};
}

template<int WORDS,bool COPY> void launch(const unsigned* input,size_t span,int pitch,
                                          int grid,int sample,Record* output) {
  void* args[]={&input,&span,&pitch,&grid,&sample,&output};
  CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(packet_probe<WORDS,COPY>),
                                   dim3(grid),dim3(THREADS),args,SHARED));
}
template<int WORDS> int run(int sparse,int grid,int trial,bool counter,
                             unsigned* eviction,unsigned* sink) {
  int pitch=sparse?4096:(WORDS==1?8:128);
  constexpr int rows=WORDS==1?32:8;
  size_t span=size_t(rows-1)*pitch+(WORDS==1?8:128);
  // Keep each new source region 128B aligned, including dense ca packets.
  span=(span+31)/32*32;
  int samples=counter?1:SAMPLES;
  unsigned* input;Record* output;Prelude* calibration;
  CHECK(cudaMalloc(&input,size_t(samples)*grid*span*4));
  CHECK(cudaMalloc(&output,grid*8*sizeof(Record)));
  CHECK(cudaMalloc(&calibration,132*sizeof(Prelude)));
  initialize<WORDS><<<samples*grid,THREADS>>>(input,span,pitch,grid);
  prelude<<<132,THREADS,SHARED>>>(calibration);
  CHECK(cudaDeviceSynchronize());
  std::vector<Prelude> clocks(132);
  CHECK(cudaMemcpy(clocks.data(),calibration,132*sizeof(Prelude),cudaMemcpyDeviceToHost));
  std::set<unsigned> clock_sms;
  for(int i=0;i<132;++i) {
    if(clocks[i].ns<3000000 || !clocks[i].cycles) return 3;
    clock_sms.insert(clocks[i].sm);
    printf("P,%d,%d,%d,%d,%d,%u,%llu,%llu\n",WORDS,sparse,grid,trial,i,
           clocks[i].sm,clocks[i].cycles,clocks[i].ns);
  }
  if(clock_sms.size()!=132) return 4;
  for(int sample=0;sample<samples;++sample) {
    // Alternate order with sample+trial, keeping every result/control.
    for(int order=0;order<(counter?1:2);++order) {
      bool copy=counter || ((order+sample+trial)%2==0);
      evict_l1<<<132,THREADS,SHARED>>>(eviction,sink);
      warm_source<WORDS><<<grid,THREADS>>>(input,span,pitch,grid,sample,sink);
      if(copy) launch<WORDS,true>(input,span,pitch,grid,sample,output);
      else launch<WORDS,false>(input,span,pitch,grid,sample,output);
      CHECK(cudaDeviceSynchronize());
      std::vector<Record> host(grid*8);
      CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Record),cudaMemcpyDeviceToHost));
      std::set<unsigned> sms;
      for(int c=0;c<grid;++c) for(int w=0;w<8;++w) {
        const Record& r=host[c*8+w];sms.insert(r.sm);
        if(r.error || r.c1<=r.c0 || r.g1<r.g0 || !r.outer_g || !r.outer_c) return 5;
        printf("R,%d,%d,%d,%d,%d,%d,%d,%d,%u,%llu,%llu,%llu,%llu,%llu,%llu,%u\n",
               WORDS,sparse,grid,trial,sample,int(copy),c,w,r.sm,
               r.c0,r.c1,r.g0,r.g1,r.outer_c,r.outer_g,r.error);
      }
      if(sms.size()!=size_t(grid)) return 6;
    }
  }
  CHECK(cudaFree(calibration));CHECK(cudaFree(output));CHECK(cudaFree(input));
  return 0;
}
int main(int argc,char** argv) {
  if(argc!=6) return 1;
  int words=atoi(argv[1]),sparse=atoi(argv[2]),grid=atoi(argv[3]),trial=atoi(argv[4]),counter=atoi(argv[5]);
  if((words!=1 && words!=4)||(sparse!=0 && sparse!=1)||(grid!=1 && grid!=132)||trial<0||trial>2||(counter!=0 && counter!=1)) return 1;
  cudaDeviceProp p;CHECK(cudaGetDeviceProperties(&p,0));
  if(p.major!=9 || p.multiProcessorCount!=132 || !p.cooperativeLaunch) return 7;
  for(auto kernel:{packet_probe<1,true>,packet_probe<1,false>,packet_probe<4,true>,packet_probe<4,false>}) {
    CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED));
    int active=0;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,THREADS,SHARED));
    if(active!=1) return 8;
  }
  CHECK(cudaFuncSetAttribute(prelude,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED));
  CHECK(cudaFuncSetAttribute(evict_l1,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED));
  unsigned* eviction;unsigned* sink;
  CHECK(cudaMalloc(&eviction,size_t(132)*524288));CHECK(cudaMalloc(&sink,132*4));
  CHECK(cudaMemset(eviction,7,size_t(132)*524288));
  printf("I,%s,%d,%d,%d,%d,%d,%d,%d\n",p.name,p.multiProcessorCount,THREADS,SHARED,words,sparse,grid,trial);
  int code=words==1?run<1>(sparse,grid,trial,counter,eviction,sink):run<4>(sparse,grid,trial,counter,eviction,sink);
  CHECK(cudaFree(sink));CHECK(cudaFree(eviction));
  printf("END,%d\n",code);return code;
}
