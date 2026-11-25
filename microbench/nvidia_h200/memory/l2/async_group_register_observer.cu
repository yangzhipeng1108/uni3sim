// Register-only clock64 observer; independent byte events, no matrix or prices.
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
struct Result { unsigned long long first,all,enqueue,poll,calibration_cycles,calibration_ns; unsigned checksum,sm,initial_failures; };
__host__ __device__ unsigned pattern(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void fill_input(unsigned* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)p[i]=pattern(unsigned(i));}
__device__ unsigned long long timer(){unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
struct Observation { unsigned stamp,gap,early; };
__device__ Observation observe(unsigned address,unsigned phase,unsigned valid,unsigned armed,unsigned lane){
 Observation o;
 // The polling loop has one uniform warp backedge and only register state.
 // 32-bit subtraction is modular; each bounded event interval is far below
 // the 2^32-cycle wrap period. The caller reconstructs the current 64-bit
 // clock epoch only after the join, outside the timing loop.
 asm volatile("{.reg .pred ready,pending,update,late,again,nready,leader;"
  ".reg .b32 now,previous,delta,stamp,gap,early,live,release_value;"
  "mov.u32 stamp,0;mov.u32 gap,0;mov.u32 early,0;"
  "setp.ne.u32 pending,%5,0;"
  // The initial event check precedes release of every producer. Start the
  // gap clock before publishing that release, so its entire cost is covered.
  "mbarrier.test_wait.parity.shared.b64 ready,[%3],%4;"
  "and.pred late,pending,ready;@late mov.u32 early,1;"
  "selp.u32 release_value,2,1,late;"
  "mov.u32 previous,%%clock;setp.eq.u32 leader,%7,0;"
  "@leader st.shared.u32 [%6],release_value;"
  "poll_loop:"
  "mbarrier.test_wait.parity.shared.b64 ready,[%3],%4;"
  "mov.u32 now,%%clock;sub.u32 delta,now,previous;"
  "@pending max.u32 gap,gap,delta;"
  "and.pred update,pending,ready;@update mov.u32 stamp,now;"
  "not.pred nready,ready;and.pred pending,pending,nready;"
  "mov.u32 previous,now;"
  "vote.sync.ballot.b32 live,pending,0xffffffff;"
  "setp.ne.u32 again,live,0;@again bra.uni poll_loop;"
  "mov.u32 %0,stamp;mov.u32 %1,gap;mov.u32 %2,early;}"
  :"=r"(o.stamp),"=r"(o.gap),"=r"(o.early):"r"(address),"r"(phase),"r"(valid),"r"(armed),"r"(lane):"memory");
 return o;
}
template<int B,int Lookahead> __global__ __launch_bounds__(Threads)
void register_group_observer(const unsigned char* input,Result* output,int rounds){
 extern __shared__ __align__(16) unsigned char shared[];
 // Four private copy panels occupy 4B; event metadata fits within the
 // unchanged physical 6B reservation. Each event joins one producer warp.
 auto events=reinterpret_cast<unsigned long long*>(shared+4*B);
 auto starts=events+32;auto enqueued=starts+8;auto observed=enqueued+8;
 auto poll_gaps=observed+32;auto early_seen=reinterpret_cast<unsigned*>(poll_gaps+32);auto armed=early_seen+32;
 unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
 if(threadIdx.x<32){unsigned a=unsigned(__cvta_generic_to_shared(events+threadIdx.x));asm volatile("mbarrier.init.shared.b64 [%0],33;"::"r"(a):"memory");}
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
 unsigned long long sum_first=0,sum_all=0,sum_enqueue=0,max_poll=0;unsigned initial_failures=0;
 #pragma unroll 1
 for(int r=0;r<rounds;++r){
  if(threadIdx.x==256)*armed=0;
  __syncthreads();
  if(warp<8){
   while(*reinterpret_cast<volatile unsigned*>(armed)==0){}
   auto start=clock64();if(lane==0)starts[warp]=start;
   #pragma unroll
   for(int g=0;g<Lookahead;++g){
    issue(g);
    unsigned a=unsigned(__cvta_generic_to_shared(events+warp*4+g));
    // 32 deferred arrivals plus one ordinary arrival per event. The fixed
    // noinc count resets every phase; no timestamp executes on this path.
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"::"r"(a):"memory");
    if(lane==0){unsigned long long state;asm volatile("mbarrier.arrive.shared.b64 %0,[%1];":"=l"(state):"r"(a):"memory");}
   }
   auto end=clock64();if(lane==0)enqueued[warp]=end;
   asm volatile("cp.async.wait_group 0;":::"memory");
  }else{
   unsigned g=lane%4;
   unsigned a=unsigned(__cvta_generic_to_shared(events+lane));
   auto result=observe(a,r&1,g<Lookahead,unsigned(__cvta_generic_to_shared(armed)),lane);
   unsigned long long current=clock64();
   observed[lane]=current-unsigned(unsigned(current)-result.stamp);
   poll_gaps[lane]=result.gap;early_seen[lane]=result.early;
  }
  __syncthreads();
  if(threadIdx.x==256){
   unsigned long long first=0,all=0,enqueue=0;
   // A CTA operand panel is ready only after all eight producer warps.
   // Retain each warp's timestamps, then join endpoints at CTA scope.
   unsigned long long origin=starts[0];for(int w=1;w<8;++w)if(starts[w]<origin)origin=starts[w];
   for(int w=0;w<8;++w){
    if(observed[w*4]>first)first=observed[w*4];
    for(int g=0;g<Lookahead;++g)if(observed[w*4+g]>all)all=observed[w*4+g];
    if(enqueued[w]>enqueue)enqueue=enqueued[w];
   }
   sum_first+=first-origin;sum_all+=all-origin;sum_enqueue+=enqueue-origin;
   for(int i=0;i<32;++i){
    if(i%4<Lookahead&&poll_gaps[i]>max_poll)max_poll=poll_gaps[i];
    initial_failures+=early_seen[i];
   }
  }
  __syncthreads();
 }
 unsigned long long calibration_cycles=clock64()-calibration_clock_start;
 auto calibration_ns=timer()-calibration_start;
 unsigned sum=0;
 volatile unsigned* words=reinterpret_cast<volatile unsigned*>(shared);
 for(int w=threadIdx.x;w<4*B/4;w+=Threads)sum^=words[w];
 for(int d=16;d;d/=2)sum^=__shfl_xor_sync(0xffffffff,sum,d);
 if(lane==0){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));output[blockIdx.x*9+warp]={sum_first,sum_all,sum_enqueue,max_poll,calibration_cycles,calibration_ns,sum,sm,initial_failures};}
}
template<int B,int L> void run(int grid,int trials){
 constexpr int Shared=6*B;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=132||!prop.cooperativeLaunch)throw std::runtime_error("cooperative H200 required");
 CHECK(cudaFuncSetAttribute(register_group_observer<B,L>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,register_group_observer<B,L>,Threads,Shared));
 if(resident!=(B==16384?2:1))throw std::runtime_error("physical occupancy changed");
 size_t bytes=size_t(grid)*4*B;unsigned char* input;Result* output;
 CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&output,size_t(grid)*9*sizeof(Result)));
 fill_input<<<132,256>>>(reinterpret_cast<unsigned*>(input),bytes/4);CHECK(cudaDeviceSynchronize());
 std::vector<Result> host(grid*9);std::vector<unsigned> expected(grid*9,0);
 for(int b=0;b<grid;++b)for(int t=0;t<Threads;++t)for(int w=t;w<4*B/4;w+=Threads)expected[b*9+t/32]^=pattern(unsigned(size_t(b)*4*B/4+w));
 auto launch=[&](int rounds){void* args[]={&input,&output,&rounds};CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(register_group_observer<B,L>),grid,Threads,args,Shared));CHECK(cudaDeviceSynchronize());CHECK(cudaGetLastError());};
 for(int warm=0;warm<6;++warm)launch(1024);
 std::vector<std::array<double,7>> samples;
 for(int trial=0;trial<trials;++trial){
  launch(Rounds);CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Result),cudaMemcpyDeviceToHost));
  double first=0,all=0,enqueue=0,poll=0,clock_sum=0,min_clock=1e20,max_clock=0;int sms[132]={};
  for(size_t i=0;i<host.size();++i){const auto& h=host[i];
   if(h.checksum!=expected[i]||h.sm>=132)throw std::runtime_error("numerical or SM identity failure");
   if(i%9==8){
    if(h.initial_failures)throw std::runtime_error("observer began after an event completed");
    if(!h.first||h.first>h.all||!h.enqueue||!h.poll||!h.calibration_cycles||!h.calibration_ns||h.calibration_cycles>=(1ULL<<31))
     throw std::runtime_error("invalid completion endpoints or clock calibration");
    double ns_per_cycle=double(h.calibration_ns)/h.calibration_cycles;
    first+=h.first*ns_per_cycle;all+=h.all*ns_per_cycle;enqueue+=h.enqueue*ns_per_cycle;
    poll=std::max(poll,h.poll*ns_per_cycle);
    double mhz=1000.0/ns_per_cycle;clock_sum+=mhz;
    min_clock=std::min(min_clock,mhz);max_clock=std::max(max_clock,mhz);
   }
   ++sms[h.sm];
  }
  for(int n:sms)if(n!=grid/132*9)throw std::runtime_error("uneven resident placement");
  double scale=double(grid)*Rounds;samples.push_back({first/scale,all/scale,enqueue/scale,poll,clock_sum/grid,min_clock,max_clock});
 }
 std::printf("{\"B_bytes\":%d,\"lookahead\":%d,\"grid\":%d,\"rounds\":127,\"threads\":288,\"producer_threads\":256,\"shared_bytes\":%d,\"resident_ctas\":%d,\"all_lane_checksum_pass\":true,\"distinct_sms\":132,\"first_poll_pending_all\":true,\"start_policy\":\"observer_pending_before_producer_release\",\"samples\":[",B,L,grid,Shared,resident);
 for(size_t i=0;i<samples.size();++i)std::printf("%s{\"first_ns\":%.12g,\"all_ns\":%.12g,\"enqueue_ns\":%.12g,\"max_poll_ns\":%.12g,\"calibration_mhz\":%.12g,\"min_cta_calibration_mhz\":%.12g,\"max_cta_calibration_mhz\":%.12g}",i?",":"",samples[i][0],samples[i][1],samples[i][2],samples[i][3],samples[i][4],samples[i][5],samples[i][6]);
 std::printf("]}\n");CHECK(cudaFree(output));CHECK(cudaFree(input));
}
int main(int argc,char** argv){try{
 if(argc!=5)throw std::runtime_error("B grid lookahead trials required");
 int b=std::atoi(argv[1]),g=std::atoi(argv[2]),l=std::atoi(argv[3]),n=std::atoi(argv[4]);
 if((n!=1&&n!=63)||!((b==16384&&(g==132||g==264))||(b==32768&&g==132)))throw std::runtime_error("unfrozen physical cell");
 #define CELL(B,L) if(b==B&&l==L){run<B,L>(g,n);return 0;}
 CELL(16384,1) CELL(16384,2) CELL(16384,4) CELL(32768,1) CELL(32768,2) CELL(32768,4)
 throw std::runtime_error("unfrozen lookahead");
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
