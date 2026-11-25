// One-shot completion census. No matrix, production route, or pricing.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
constexpr int Producers=256,Threads=288,Tiers=65;
struct Result {unsigned long long before,after,origin,cal_cycles,cal_ns;unsigned ready,initial,checksum,sm,delay_sink;};
__host__ __device__ unsigned pattern(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void fill_input(unsigned* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)p[i]=pattern(unsigned(i));}
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
// This is a finite arithmetic delay, not a readiness polling loop. Its
// duration is never assumed: actual clock brackets are returned for every
// sample. The asm prevents folding the dependency chain into a closed form.
__device__ unsigned delay_chain(unsigned count,unsigned x){
 #pragma unroll 1
 for(unsigned i=0;i<count;++i)asm volatile("add.u32 %0,%0,1;":"+r"(x));
 return x;
}
__device__ unsigned query_once(unsigned address,unsigned phase){
 unsigned ready;
 asm volatile("{.reg .pred p;mbarrier.test_wait.parity.shared.b64 p,[%1],%2;selp.u32 %0,1,0,p;}":"=r"(ready):"r"(address),"r"(phase):"memory");
 return ready;
}
template<int B,int L> __global__ __launch_bounds__(Threads)
void deadline_group_observer(const unsigned char* input,Result* output,int rounds,int reverse){
 extern __shared__ __align__(16) unsigned char shared[];
 auto events=reinterpret_cast<unsigned long long*>(shared+4*B);
 auto stamps=events+32;auto gate=reinterpret_cast<unsigned*>(stamps+1);
 unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
 if(threadIdx.x<32){unsigned a=unsigned(__cvta_generic_to_shared(events+threadIdx.x));asm volatile("mbarrier.init.shared.b64 [%0],33;"::"r"(a):"memory");}
 __syncthreads();
 auto issue=[&](int slot){
  #pragma unroll
  for(int v=threadIdx.x;v<B/16;v+=Producers){unsigned dst=unsigned(__cvta_generic_to_shared(shared+slot*B+v*16));const unsigned char* src=input+(size_t(blockIdx.x)*4+slot)*B+v*16;asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");}
  asm volatile("cp.async.commit_group;":::"memory");
 };
 if(warp<8){for(int g=0;g<4;++g)issue(g);asm volatile("cp.async.wait_group 0;":::"memory");}
 cooperative_groups::this_grid().sync();
 auto cal_ns_start=timer();auto cal_start=clock64();
 #pragma unroll 1
 for(int r=0;r<rounds;++r){
  int tier=reverse?Tiers-1-r%Tiers:r%Tiers;
  if(threadIdx.x==256)*gate=0;
  __syncthreads();
  unsigned initial=0,ready=0,sink=0;unsigned long long before=0,after=0,origin=0;
  if(warp<8){
   while(*reinterpret_cast<volatile unsigned*>(gate)==0){}
   #pragma unroll
   for(int g=0;g<L;++g){issue(g);unsigned a=unsigned(__cvta_generic_to_shared(events+warp*4+g));asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"::"r"(a):"memory");if(lane==0){unsigned long long state;asm volatile("mbarrier.arrive.shared.b64 %0,[%1];":"=l"(state):"r"(a):"memory");}}
   asm volatile("cp.async.wait_group 0;":::"memory");
  }else{
   unsigned a=unsigned(__cvta_generic_to_shared(events+lane));
   initial=query_once(a,r&1);
   // Full-warp ballot orders every initial query before the leader release.
   unsigned initial_mask=__ballot_sync(0xffffffff,initial!=0&&lane%4<L);
   origin=clock64();
   if(lane==0){*stamps=origin;*reinterpret_cast<volatile unsigned*>(gate)=initial_mask?2:1;}
   sink=delay_chain(unsigned(tier)*8,unsigned(r));
   // Dependency ties the finite delay to the query phase; no timed shared
   // load/store or readiness backedge is admitted inside this bracket.
   unsigned phase=(unsigned(r)+sink-unsigned(r)-unsigned(tier)*8)&1;
   before=clock64();
   ready=query_once(a,phase);
   after=clock64();
  }
  __syncthreads();
  if(warp==8){
   origin=*stamps;
   unsigned mask=__ballot_sync(0xffffffff,ready!=0);
   unsigned bad=__ballot_sync(0xffffffff,initial!=0&&lane%4<L);
   unsigned long long lo=before,hi=after;
   for(int d=16;d;d/=2){lo=min(lo,__shfl_xor_sync(0xffffffff,lo,d));hi=max(hi,__shfl_xor_sync(0xffffffff,hi,d));}
   if(lane==0){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));output[size_t(blockIdx.x)*rounds+r]={lo,hi,origin,0,0,mask,bad,0,sm,sink};}
  }
  __syncthreads();
 }
 auto cal_cycles=clock64()-cal_start;auto cal_ns=timer()-cal_ns_start;
 unsigned sum=0;volatile unsigned* words=reinterpret_cast<volatile unsigned*>(shared);
 for(int w=threadIdx.x;w<4*B/4;w+=Threads)sum^=words[w];
 for(int d=16;d;d/=2)sum^=__shfl_xor_sync(0xffffffff,sum,d);
 if(lane==0)atomicXor(&output[size_t(blockIdx.x)*rounds].checksum,sum);
 __syncthreads();
 if(threadIdx.x==256)for(int r=0;r<rounds;++r){output[size_t(blockIdx.x)*rounds+r].cal_cycles=cal_cycles;output[size_t(blockIdx.x)*rounds+r].cal_ns=cal_ns;}
}
template<int B,int L> void run(int grid,int repeats,int reverse){
 constexpr int Shared=6*B;constexpr int Rounds=Tiers*4;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=132||!prop.cooperativeLaunch)throw std::runtime_error("cooperative H200 required");
 CHECK(cudaFuncSetAttribute(deadline_group_observer<B,L>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,deadline_group_observer<B,L>,Threads,Shared));
 if(resident!=(B==16384?2:1))throw std::runtime_error("changed resident capacity");
 size_t bytes=size_t(grid)*4*B;unsigned char* input;Result* output;CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&output,size_t(grid)*Rounds*sizeof(Result)));
 fill_input<<<132,256>>>(reinterpret_cast<unsigned*>(input),bytes/4);CHECK(cudaDeviceSynchronize());std::vector<Result> host(size_t(grid)*Rounds);
 auto launch=[&](int rounds){void* args[]={&input,&output,&rounds,&reverse};CHECK(cudaMemset(output,0,size_t(grid)*Rounds*sizeof(Result)));CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(deadline_group_observer<B,L>),grid,Threads,args,Shared));CHECK(cudaDeviceSynchronize());CHECK(cudaGetLastError());};
 for(int i=0;i<6;++i)launch(Rounds);
 for(int trial=0;trial<repeats;++trial){launch(Rounds);CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Result),cudaMemcpyDeviceToHost));
  struct Tier {unsigned long long lo=~0ULL,hi=0,width=0;unsigned first=0,all=0,count=0;};Tier tiers[Tiers];
  double min_mhz=1e20,max_mhz=0;int sms[132]={};unsigned all_mask=L==1?0x11111111u:L==2?0x33333333u:0xffffffffu;
  for(int b=0;b<grid;++b){unsigned expected=0;for(size_t w=size_t(b)*4*B/4;w<size_t(b+1)*4*B/4;++w)expected^=pattern(unsigned(w));if(host[size_t(b)*Rounds].checksum!=expected)throw std::runtime_error("all-lane checksum failure");
   for(int r=0;r<Rounds;++r){auto h=host[size_t(b)*Rounds+r];int tier=reverse?Tiers-1-r%Tiers:r%Tiers;
    if(h.initial||h.before<h.origin||h.after<h.before||!h.cal_cycles||!h.cal_ns||h.sm>=132||h.delay_sink!=unsigned(r)+tier*8)throw std::runtime_error("invalid causal query or delay chain");
    auto& q=tiers[tier];q.lo=std::min(q.lo,h.before-h.origin);q.hi=std::max(q.hi,h.after-h.origin);q.width=std::max(q.width,h.after-h.before);++q.count;q.first+=(h.ready&0x11111111u)==0x11111111u;q.all+=(h.ready&all_mask)==all_mask;
    double mhz=1000.0*h.cal_cycles/h.cal_ns;min_mhz=std::min(min_mhz,mhz);max_mhz=std::max(max_mhz,mhz);
   }++sms[host[size_t(b)*Rounds].sm];
  }
  for(int n:sms)if(n!=grid/132)throw std::runtime_error("uneven resident placement");
  std::printf("{\"B_bytes\":%d,\"grid\":%d,\"lookahead\":%d,\"threads\":288,\"shared_bytes\":%d,\"resident_ctas\":%d,\"rounds\":%d,\"tiers\":65,\"trial\":%d,\"reverse\":%d,\"all_lane_checksum_pass\":true,\"first_query_pending_all\":true,\"distinct_sms\":132,\"min_cta_calibration_mhz\":%.12g,\"max_cta_calibration_mhz\":%.12g,\"rows\":[",B,grid,L,Shared,resident,Rounds,trial,reverse,min_mhz,max_mhz);
  for(int t=0;t<Tiers;++t){auto q=tiers[t];std::printf("%s[%d,%llu,%llu,%u,%u,%u,%llu]",t?",":"",t,q.lo,q.hi,q.first,q.all,q.count,q.width);}
  std::printf("]}\n");
 }
 CHECK(cudaFree(output));CHECK(cudaFree(input));
}
int main(int argc,char** argv){try{if(argc!=6)throw std::runtime_error("B grid lookahead repeats reverse required");int b=std::atoi(argv[1]),g=std::atoi(argv[2]),l=std::atoi(argv[3]),n=std::atoi(argv[4]),rev=std::atoi(argv[5]);if((n!=1&&n!=17)||(rev!=0&&rev!=1)||!((b==16384&&(g==132||g==264))||(b==32768&&g==132)))throw std::runtime_error("unfrozen physical cell");
#define CELL(B,L) if(b==B&&l==L){run<B,L>(g,n,rev);return 0;}
CELL(16384,1) CELL(16384,2) CELL(16384,4) CELL(32768,1) CELL(32768,2) CELL(32768,4)
throw std::runtime_error("unfrozen lookahead");}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
