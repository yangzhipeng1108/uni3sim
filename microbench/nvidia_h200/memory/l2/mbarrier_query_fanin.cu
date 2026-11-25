// Fixed-state instruction mechanism, NOT a DMA completion observer.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
constexpr int Threads=288,Shared=98304,Rounds=257;
struct Result{unsigned long long sum,max,cycles,ns;unsigned sm,checksum;};
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
struct Query{unsigned long long ticks;unsigned sum;};
template<int N> __device__ __forceinline__ Query query(unsigned address,unsigned expected){
 unsigned sum=0;unsigned long long before;asm volatile("mov.u64 %0, %%clock64;":"=l"(before)::"memory");
 #pragma unroll
 for(int i=0;i<N;++i){unsigned value;
  asm volatile("{.reg .pred p;mbarrier.test_wait.parity.shared.b64 p,[%1],0;selp.u32 %0,1,0,p;}":"=r"(value):"r"(address):"memory");
  sum+=value;
 }
 unsigned long long after;
 // Only a numerically correct predicate permits the end-clock read. This
 // prevents the compiler from moving that read before predicate completion.
 // Failed queries retain after=0 and fail the all-lane checksum as well.
 asm volatile("{.reg .pred valid;mov.u64 %0,0;setp.eq.u32 valid,%1,%2;@valid mov.u64 %0,%%clock64;}":"=l"(after):"r"(sum),"r"(expected):"memory");
 return {after-before,sum};
}
template<int N> __global__ __launch_bounds__(Threads)
void query_fanin(Result* output,int distinct,int ready){
 extern __shared__ __align__(16) unsigned char data[];
 auto events=reinterpret_cast<unsigned long long*>(data);
 unsigned lane=threadIdx.x%32;
 if(threadIdx.x<32){unsigned a=unsigned(__cvta_generic_to_shared(events+threadIdx.x));asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(a):"memory");}
 __syncthreads();
 if(ready&&threadIdx.x<32){unsigned a=unsigned(__cvta_generic_to_shared(events+threadIdx.x));unsigned long long state;asm volatile("mbarrier.arrive.shared.b64 %0,[%1];":"=l"(state):"r"(a):"memory");}
 __syncthreads();cooperative_groups::this_grid().sync();
 if(threadIdx.x>=256){
  unsigned address=unsigned(__cvta_generic_to_shared(events+(lane%distinct)));
  unsigned long long nt=timer();unsigned long long ct=clock64();unsigned long long sum=0,maximum=0;unsigned checksum=0;
  #pragma unroll 1
  for(int r=0;r<Rounds;++r){auto q=query<N>(address,unsigned(ready*N));sum+=q.ticks;maximum=max(maximum,q.ticks);checksum+=q.sum;}
  unsigned long long cycles=clock64()-ct;unsigned long long ns=timer()-nt;
  unsigned failure=__ballot_sync(0xffffffff,checksum!=unsigned(ready*N*Rounds));
  if(lane==0){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));output[blockIdx.x]={sum,maximum,cycles,ns,sm,failure};}
 }
}
template<int N>void run(int grid,int distinct,int ready,int samples){
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));if(prop.major!=9||prop.multiProcessorCount!=132||!prop.cooperativeLaunch)throw std::runtime_error("cooperative H200 required");
 CHECK(cudaFuncSetAttribute(query_fanin<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,query_fanin<N>,Threads,Shared));if(resident!=2)throw std::runtime_error("changed two-resident resources");
 Result* output;CHECK(cudaMalloc(&output,grid*sizeof(Result)));std::vector<Result> host(grid);
 auto launch=[&](){void* args[]={&output,&distinct,&ready};CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(query_fanin<N>),grid,Threads,args,Shared));CHECK(cudaDeviceSynchronize());CHECK(cudaGetLastError());};
 for(int i=0;i<6;++i)launch();
 printf("{\"grid\":%d,\"distinct_addresses\":%d,\"ready\":%d,\"queries_per_bracket\":%d,\"rounds\":257,\"threads\":288,\"shared_bytes\":98304,\"resident_ctas\":2,\"samples\":[",grid,distinct,ready,N);
 for(int trial=0;trial<samples;++trial){launch();CHECK(cudaMemcpy(host.data(),output,grid*sizeof(Result),cudaMemcpyDeviceToHost));int sms[132]={};double mean=0,max_ns=0,min_clock=1e20,max_clock=0;
  for(const auto& h:host){if(h.sm>=132||h.checksum||!h.sum||!h.max||!h.cycles||!h.ns)throw std::runtime_error("all-lane numerical or clock failure");++sms[h.sm];double ns_per_cycle=double(h.ns)/h.cycles;mean+=h.sum*ns_per_cycle/(Rounds*N);max_ns=std::max(max_ns,h.max*ns_per_cycle);double mhz=1000/ns_per_cycle;min_clock=std::min(min_clock,mhz);max_clock=std::max(max_clock,mhz);}
  for(int n:sms)if(n!=grid/132)throw std::runtime_error("uneven SM placement");
  printf("%s{\"mean_query_ns\":%.12g,\"max_bracket_ns\":%.12g,\"min_cta_clock_mhz\":%.12g,\"max_cta_clock_mhz\":%.12g,\"all_lane_numeric_pass\":true,\"distinct_sms\":132}",trial?",":"",mean/grid,max_ns,min_clock,max_clock);
 }puts("]}");CHECK(cudaFree(output));
}
int main(int argc,char**argv){try{if(argc!=6)throw std::runtime_error("grid distinct ready queries samples required");int g=atoi(argv[1]),d=atoi(argv[2]),r=atoi(argv[3]),q=atoi(argv[4]),n=atoi(argv[5]);if((g!=132&&g!=264)||(d!=1&&d!=4&&d!=32)||(r!=0&&r!=1)||(n!=1&&n!=17))throw std::runtime_error("unfrozen instruction cell");if(q==1)run<1>(g,d,r,n);else if(q==128)run<128>(g,d,r,n);else throw std::runtime_error("unfrozen query count");return 0;}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
