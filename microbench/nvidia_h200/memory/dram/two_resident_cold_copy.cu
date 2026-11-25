// Independent cold same-address byte requests. No matrix/router/operator output.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
constexpr int Threads=256,Panel=16384,Row=128,Rows=128,Slots=3,Shared=98304;
struct Times {unsigned long long start,end,cycles;unsigned sm;};
__host__ __device__ unsigned pattern(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__device__ unsigned long long timer(){unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
__global__ void fill(unsigned* x,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)x[i]=pattern(unsigned(i));}
__global__ void evict(unsigned* x,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)x[i]=unsigned(i);}
__device__ unsigned reduce(unsigned value,unsigned* temp){
 for(int d=16;d;d/=2)value^=__shfl_xor_sync(0xffffffff,value,d);
 if(threadIdx.x%32==0)temp[threadIdx.x/32]=value;
 __syncthreads();if(threadIdx.x==0){value=0;for(int i=0;i<8;++i)value^=temp[i];}return value;
}
// Independent direct-global reference samples every vector's first word over
// the complete source extent. It never uses shared copy slots or their order.
__global__ void reference(const unsigned* input,unsigned* expected,int stride,int fanout,size_t unique_bytes){
 __shared__ unsigned temp[8];unsigned value=0;
 size_t base=size_t(blockIdx.x/fanout)*Rows*stride/4;
 for(int v=threadIdx.x;v<Rows*stride/16;v+=Threads)value^=input[base+size_t(v)*4];
 for(int v=threadIdx.x;v<stride/16;v+=Threads)value^=input[unique_bytes/4+size_t(v)*4];
 value=reduce(value,temp);if(threadIdx.x==0)expected[blockIdx.x]=value;
}
template<int Stride> __global__ __launch_bounds__(Threads)
void cold_copy(const unsigned char* input,unsigned* sink,Times* times,int fanout,size_t unique_bytes){
 extern __shared__ __align__(16) unsigned char shared[];
 for(int v=threadIdx.x;v<Shared;v+=Threads)shared[v]=0;
 __syncthreads();unsigned long long start=timer(),clock_start=clock64();
 constexpr int Rounds=Stride/Row;
 size_t base=size_t(blockIdx.x/fanout)*Rows*Stride;
 auto issue=[&](int round,int slot){
  if(round<Rounds){
   #pragma unroll
   for(int v=threadIdx.x;v<Panel/16;v+=Threads){
    unsigned dst=unsigned(__cvta_generic_to_shared(shared+Slots*Panel+slot*Panel+v*16));
    const unsigned char* src=input+unique_bytes+round*Row+(v%(Row/16))*16;
    int bytes=v<Row/16?16:0;
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(src),"r"(bytes):"memory");
   }
  }
  asm volatile("cp.async.commit_group;":::"memory");
  if(round<Rounds){
   #pragma unroll
   for(int v=threadIdx.x;v<Panel/16;v+=Threads){
    unsigned dst=unsigned(__cvta_generic_to_shared(shared+slot*Panel+v*16));
    const unsigned char* src=input+base+(v/(Row/16))*Stride+round*Row+(v%(Row/16))*16;
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");
   }
  }
  asm volatile("cp.async.commit_group;":::"memory");
 };
 for(int slot=0;slot<Slots;++slot)issue(slot,slot);
 unsigned value=0;
 #pragma unroll 1
 for(int round=0;round<Rounds;++round){
  asm volatile("cp.async.wait_group 4;":::"memory");__syncthreads();
  volatile unsigned* b=reinterpret_cast<volatile unsigned*>(shared+(round%Slots)*Panel);
  volatile unsigned* a=reinterpret_cast<volatile unsigned*>(shared+Slots*Panel+(round%Slots)*Panel);
  #pragma unroll
  for(int v=threadIdx.x;v<Panel/16;v+=Threads){value^=b[v*4];value^=a[v*4];}
  __syncthreads();issue(round+Slots,round%Slots);
 }
 asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 unsigned long long cycles=clock64()-clock_start,end=timer();
 // All 256 lanes contribute. Reduction is outside the timed copy body.
 value=reduce(value,reinterpret_cast<unsigned*>(shared));
 if(threadIdx.x==0){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));sink[blockIdx.x]=value;times[blockIdx.x]={start,end,cycles,sm};}
}
template<int Stride> void run(int grid,int fanout,int trials){
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=132)throw std::runtime_error("H200 geometry required");
 CHECK(cudaFuncSetAttribute(cold_copy<Stride>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,cold_copy<Stride>,Threads,Shared));
 if(resident!=2)throw std::runtime_error("two resident CTAs required");
 size_t unique_bytes=size_t((grid+fanout-1)/fanout)*Rows*Stride;
 if(unique_bytes>size_t(48)*1024*1024*1024)throw std::runtime_error("frozen backing bound exceeded");
 unsigned char* input;unsigned *flush,*sink,*expected;Times* times;
 CHECK(cudaMalloc(&input,unique_bytes+Stride));CHECK(cudaMalloc(&flush,128*1024*1024));
 CHECK(cudaMalloc(&sink,grid*sizeof(unsigned)));CHECK(cudaMalloc(&expected,grid*sizeof(unsigned)));CHECK(cudaMalloc(&times,grid*sizeof(Times)));
 fill<<<4096,256>>>(reinterpret_cast<unsigned*>(input),(unique_bytes+Stride)/4);
 reference<<<grid,Threads>>>(reinterpret_cast<unsigned*>(input),expected,Stride,fanout,unique_bytes);
 CHECK(cudaDeviceSynchronize());CHECK(cudaGetLastError());
 std::vector<unsigned> truth(grid),got(grid);std::vector<Times> host(grid);
 CHECK(cudaMemcpy(truth.data(),expected,grid*sizeof(unsigned),cudaMemcpyDeviceToHost));
 unsigned cpu_first=0;
 for(int v=0;v<Rows*Stride/16;++v)cpu_first^=pattern(unsigned(size_t(v)*4));
 for(int v=0;v<Stride/16;++v)cpu_first^=pattern(unsigned(unique_bytes/4+size_t(v)*4));
 if(truth[0]!=cpu_first)throw std::runtime_error("independent CPU/direct-global reference mismatch");
 if(std::all_of(truth.begin(),truth.end(),[](unsigned x){return x==0;}))throw std::runtime_error("nondiscriminating checksum fixture");
 std::vector<std::array<double,3>> samples;
 for(int trial=-3;trial<trials;++trial){
  evict<<<4096,256>>>(flush,128*1024*1024/4);CHECK(cudaDeviceSynchronize());
  cold_copy<Stride><<<grid,Threads,Shared>>>(input,sink,times,fanout,unique_bytes);
  CHECK(cudaDeviceSynchronize());CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(got.data(),sink,grid*sizeof(unsigned),cudaMemcpyDeviceToHost));
  if(got!=truth)throw std::runtime_error("all-lane byte-copy checksum failure");
  CHECK(cudaMemcpy(host.data(),times,grid*sizeof(Times),cudaMemcpyDeviceToHost));
  unsigned long long first=~0ULL,last=0,total_ns=0,total_cycles=0;int sms[132]={};std::vector<unsigned long long> spans;
  for(const auto& t:host){if(t.sm>=132||t.end<=t.start)throw std::runtime_error("invalid CTA interval");first=std::min(first,t.start);last=std::max(last,t.end);total_ns+=t.end-t.start;total_cycles+=t.cycles;++sms[t.sm];spans.push_back(t.end-t.start);}
  if(std::any_of(std::begin(sms),std::end(sms),[](int x){return x==0;}))throw std::runtime_error("all132SM coverage required");
  std::sort(spans.begin(),spans.end());
  if(trial>=0)samples.push_back({double(last-first),double(spans[spans.size()/2]),double(total_cycles)/total_ns*1000.0});
 }
 std::printf("{\"stride\":%d,\"grid\":%d,\"fanout\":%d,\"panel_bytes\":16384,\"a_destination_bytes\":16384,\"valid_a_bytes_per_round\":128,\"threads\":256,\"slots\":3,\"shared_bytes\":98304,\"resident_ctas\":2,\"rounds\":%d,\"unique_b_bytes\":%llu,\"all_lane_checksum_pass\":true,\"distinct_sms\":132,\"samples\":[",Stride,grid,fanout,Stride/Row,static_cast<unsigned long long>(unique_bytes));
 for(size_t i=0;i<samples.size();++i)std::printf("%s{\"body_ns\":%.12g,\"median_cta_ns\":%.12g,\"weighted_mhz\":%.12g}",i?",":"",samples[i][0],samples[i][1],samples[i][2]);
 std::puts("]}");CHECK(cudaFree(times));CHECK(cudaFree(expected));CHECK(cudaFree(sink));CHECK(cudaFree(flush));CHECK(cudaFree(input));
}
int main(int argc,char** argv){try{
 if(argc!=5)throw std::runtime_error("stride grid fanout1..16 trials1or63 required");
 int s=std::atoi(argv[1]),g=std::atoi(argv[2]),f=std::atoi(argv[3]),n=std::atoi(argv[4]);
 if(g<132||g>133120||f<1||f>16||(f&(f-1))||(n!=1&&n!=63))throw std::runtime_error("unfrozen physical cell");
 #define CELL(S) if(s==S){run<S>(g,f,n);return 0;}
 CELL(1024) CELL(2048) CELL(3072) CELL(4096) CELL(5760) CELL(8192) CELL(12288) CELL(14336) CELL(16384) CELL(32768)
 throw std::runtime_error("unfrozen row stride");
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
