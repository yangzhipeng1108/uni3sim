// CTA-wide hardware event joins; uniform queries, no operator or prices.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
constexpr int Producers=256, Threads=288, Rounds=127;
struct Result { unsigned long long response,enqueue,poll,calibration_cycles,calibration_ns; unsigned checksum,sm,initial_failures; };
__host__ __device__ unsigned pattern(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void fill_input(unsigned* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)p[i]=pattern(unsigned(i));}
__device__ unsigned long long timer(){unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
struct Observation { unsigned stamp,gap,early; };
__device__ Observation observe(unsigned first,unsigned all,unsigned address,unsigned phase,unsigned armed,unsigned lane){
 Observation o;
 // Each query address is identical in every observer lane. The hardware
 // events already join all256 producer threads; no per-lane event scatter.
 asm volatile("{.reg .pred p0,p1,bad,leader,ready,uniform_ready;"
  ".reg .b32 now,previous,delta,gap,flag;"
  "mbarrier.test_wait.parity.shared.b64 p0,[%3],%6;"
  "mbarrier.test_wait.parity.shared.b64 p1,[%4],%6;"
  "or.pred bad,p0,p1;selp.u32 %2,1,0,bad;selp.u32 flag,2,1,bad;"
  "mov.u32 gap,0;mov.u32 previous,%%clock;setp.eq.u32 leader,%8,0;"
  "@leader st.shared.u32 [%7],flag;"
  "uniform_poll:"
  "mbarrier.test_wait.parity.shared.b64 ready,[%5],%6;"
  "vote.sync.all.pred uniform_ready,ready,0xffffffff;"
  "mov.u32 now,%%clock;sub.u32 delta,now,previous;max.u32 gap,gap,delta;"
  "mov.u32 previous,now;@!uniform_ready bra.uni uniform_poll;"
  "mov.u32 %0,now;mov.u32 %1,gap;}"
  :"=r"(o.stamp),"=r"(o.gap),"=r"(o.early):"r"(first),"r"(all),"r"(address),"r"(phase),"r"(armed),"r"(lane):"memory");
 return o;
}
__device__ void drain_event(unsigned address,unsigned phase){
 unsigned ready;
 do {asm volatile("{.reg .pred p;mbarrier.test_wait.parity.shared.b64 p,[%1],%2;selp.u32 %0,1,0,p;}":"=r"(ready):"r"(address),"r"(phase):"memory");}while(!ready);
}
template<int B,int Lookahead,int Endpoint> __global__ __launch_bounds__(Threads)
void cta_group_observer(const unsigned char* input,Result* output,int rounds){
 extern __shared__ __align__(16) unsigned char shared[];
 // Every endpoint receives256deferred arrivals plus one ordinary arrival.
 // First tracks group0; All tracks every copy issued before its marker.
 // Constant two markers per producer are retained even for Lookahead1.
 static_assert(Producers==256);
 auto events=reinterpret_cast<unsigned long long*>(shared+4*B);
 auto starts=events+2;auto enqueued=starts+8;auto observed=enqueued+8;
 auto poll_gaps=observed+32;auto early_seen=reinterpret_cast<unsigned*>(poll_gaps+32);auto armed=early_seen+32;
 unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
 if(threadIdx.x<2){unsigned a=unsigned(__cvta_generic_to_shared(events+threadIdx.x));asm volatile("mbarrier.init.shared.b64 [%0],257;"::"r"(a):"memory");}
 __syncthreads();
 auto issue=[&](int slot){
  #pragma unroll
  for(int v=threadIdx.x;v<B/16;v+=Producers){
   unsigned dst=unsigned(__cvta_generic_to_shared(shared+slot*B+v*16));
   const unsigned char* src=input+(size_t(blockIdx.x)*4+slot)*B+v*16;
   asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
 };
 if(warp<8){for(int g=0;g<4;++g)issue(g);asm volatile("cp.async.wait_group 0;":::"memory");}
 cooperative_groups::this_grid().sync();
 auto calibration_start=timer();auto calibration_clock_start=clock64();
 unsigned long long sum_response=0,sum_enqueue=0,max_poll=0;unsigned initial_failures=0;
 #pragma unroll 1
 for(int r=0;r<rounds;++r){
  if(threadIdx.x==256)*armed=0;
  __syncthreads();
  if(warp<8){
   while(*reinterpret_cast<volatile unsigned*>(armed)==0){}
   auto start=clock64();if(lane==0)starts[warp]=start;
   issue(0);
   unsigned first=unsigned(__cvta_generic_to_shared(events));
   asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"::"r"(first):"memory");
   if(threadIdx.x==0){unsigned long long state;asm volatile("mbarrier.arrive.shared.b64 %0,[%1];":"=l"(state):"r"(first):"memory");}
   #pragma unroll
   for(int g=1;g<Lookahead;++g)issue(g);
   unsigned all=unsigned(__cvta_generic_to_shared(events+1));
   asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"::"r"(all):"memory");
   if(threadIdx.x==0){unsigned long long state;asm volatile("mbarrier.arrive.shared.b64 %0,[%1];":"=l"(state):"r"(all):"memory");}
   auto end=clock64();if(lane==0)enqueued[warp]=end;
   asm volatile("cp.async.wait_group 0;":::"memory");
  }else{
   unsigned first=unsigned(__cvta_generic_to_shared(events));
   unsigned all=unsigned(__cvta_generic_to_shared(events+1));
   auto result=observe(first,all,Endpoint?all:first,r&1,unsigned(__cvta_generic_to_shared(armed)),lane);
   unsigned long long current=clock64();
   observed[lane]=current-unsigned(unsigned(current)-result.stamp);
   poll_gaps[lane]=result.gap;early_seen[lane]=result.early;
   // Both phases finish before reuse. These drains are outside the measured
   // polling/stamp interval and identical in the two endpoint programs.
   drain_event(first,r&1);drain_event(all,r&1);
  }
  __syncthreads();
  if(threadIdx.x==256){
   unsigned long long enqueue=0,stamp=observed[0];
   unsigned long long origin=starts[0];for(int w=1;w<8;++w)if(starts[w]<origin)origin=starts[w];
   for(int w=0;w<8;++w)if(enqueued[w]>enqueue)enqueue=enqueued[w];
   for(int i=0;i<32;++i){
    if(observed[i]!=stamp)++initial_failures;
    if(poll_gaps[i]>max_poll)max_poll=poll_gaps[i];
    initial_failures+=early_seen[i];
   }
   if(stamp<=origin)++initial_failures;
   sum_response+=stamp-origin;sum_enqueue+=enqueue-origin;
  }
  __syncthreads();
 }
 unsigned long long calibration_cycles=clock64()-calibration_clock_start;
 auto calibration_ns=timer()-calibration_start;
 unsigned sum=0;
 volatile unsigned* words=reinterpret_cast<volatile unsigned*>(shared);
 for(int w=threadIdx.x;w<4*B/4;w+=Threads)sum^=words[w];
 for(int d=16;d;d/=2)sum^=__shfl_xor_sync(0xffffffff,sum,d);
 if(lane==0){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));output[blockIdx.x*9+warp]={sum_response,sum_enqueue,max_poll,calibration_cycles,calibration_ns,sum,sm,initial_failures};}
}
template<int B,int L,int E> void run(int grid,int trials){
 constexpr int Shared=6*B;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=132||!prop.cooperativeLaunch)throw std::runtime_error("cooperative H200 required");
 CHECK(cudaFuncSetAttribute(cta_group_observer<B,L,E>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,cta_group_observer<B,L,E>,Threads,Shared));
 if(resident!=(B==16384?2:1))throw std::runtime_error("physical occupancy changed");
 size_t bytes=size_t(grid)*4*B;unsigned char* input;Result* output;
 CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&output,size_t(grid)*9*sizeof(Result)));
 fill_input<<<132,256>>>(reinterpret_cast<unsigned*>(input),bytes/4);CHECK(cudaDeviceSynchronize());
 std::vector<Result> host(grid*9);std::vector<unsigned> expected(grid*9,0);
 for(int b=0;b<grid;++b)for(int t=0;t<Threads;++t)for(int w=t;w<4*B/4;w+=Threads)expected[b*9+t/32]^=pattern(unsigned(size_t(b)*4*B/4+w));
 auto launch=[&](int rounds){void* args[]={&input,&output,&rounds};CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(cta_group_observer<B,L,E>),grid,Threads,args,Shared));CHECK(cudaDeviceSynchronize());CHECK(cudaGetLastError());};
 for(int warm=0;warm<6;++warm)launch(1024);
 std::vector<std::array<double,6>> samples;
 for(int trial=0;trial<trials;++trial){
  launch(Rounds);CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Result),cudaMemcpyDeviceToHost));
  double response=0,enqueue=0,poll=0,clock_sum=0,min_clock=1e20,max_clock=0;int sms[132]={};
  for(size_t i=0;i<host.size();++i){const auto& h=host[i];
   if(h.checksum!=expected[i]||h.sm>=132)throw std::runtime_error("numerical or SM identity failure");
   if(i%9==8){
    if(h.initial_failures)throw std::runtime_error("observer began after an event completed");
    if(!h.response||!h.enqueue||!h.poll||!h.calibration_cycles||!h.calibration_ns||h.calibration_cycles>=(1ULL<<31))
     throw std::runtime_error("invalid completion endpoints or clock calibration");
    double ns_per_cycle=double(h.calibration_ns)/h.calibration_cycles;
    response+=h.response*ns_per_cycle;enqueue+=h.enqueue*ns_per_cycle;
    poll=std::max(poll,h.poll*ns_per_cycle);
    double mhz=1000.0/ns_per_cycle;clock_sum+=mhz;
    min_clock=std::min(min_clock,mhz);max_clock=std::max(max_clock,mhz);
   }
   ++sms[h.sm];
  }
  for(int n:sms)if(n!=grid/132*9)throw std::runtime_error("uneven resident placement");
  double scale=double(grid)*Rounds;samples.push_back({response/scale,enqueue/scale,poll,clock_sum/grid,min_clock,max_clock});
 }
 std::printf("{\"B_bytes\":%d,\"lookahead\":%d,\"endpoint\":%d,\"barrier_arrivals\":257,\"grid\":%d,\"rounds\":127,\"threads\":288,\"producer_threads\":256,\"shared_bytes\":%d,\"resident_ctas\":%d,\"all_lane_checksum_pass\":true,\"distinct_sms\":132,\"first_poll_pending_all\":true,\"start_policy\":\"observer_pending_before_producer_release\",\"samples\":[",B,L,E,grid,Shared,resident);
 for(size_t i=0;i<samples.size();++i)std::printf("%s{\"response_ns\":%.12g,\"enqueue_ns\":%.12g,\"max_poll_ns\":%.12g,\"calibration_mhz\":%.12g,\"min_cta_calibration_mhz\":%.12g,\"max_cta_calibration_mhz\":%.12g}",i?",":"",samples[i][0],samples[i][1],samples[i][2],samples[i][3],samples[i][4],samples[i][5]);
 std::printf("]}\n");CHECK(cudaFree(output));CHECK(cudaFree(input));
}
int main(int argc,char** argv){try{
 if(argc!=6)throw std::runtime_error("B grid lookahead endpoint trials required");
 int b=std::atoi(argv[1]),g=std::atoi(argv[2]),l=std::atoi(argv[3]),e=std::atoi(argv[4]),n=std::atoi(argv[5]);
 if((e!=0&&e!=1)||(n!=1&&n!=63)||!((b==16384&&(g==132||g==264))||(b==32768&&g==132)))throw std::runtime_error("unfrozen physical cell");
 #define CELL(B,L) if(b==B&&l==L){if(e==0)run<B,L,0>(g,n);else run<B,L,1>(g,n);return 0;}
 CELL(16384,1) CELL(16384,2) CELL(16384,4) CELL(32768,1) CELL(32768,2) CELL(32768,4)
 throw std::runtime_error("unfrozen lookahead");
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
