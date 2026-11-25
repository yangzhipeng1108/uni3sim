// Independent observer warp tags asynchronous group completion. No matrix.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
constexpr int Producers=256, Threads=288, Rounds=127;
struct Result { unsigned long long first,all,enqueue,poll; unsigned checksum,sm; };
__host__ __device__ unsigned pattern(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void fill_input(unsigned* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)p[i]=pattern(unsigned(i));}
__device__ unsigned long long timer(){unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
__device__ bool completed(unsigned address,unsigned phase){unsigned done;asm volatile("{.reg .pred p; mbarrier.test_wait.parity.shared.b64 p,[%1],%2; selp.b32 %0,1,0,p;}":"=r"(done):"r"(address),"r"(phase):"memory");return done;}
template<int B,int Lookahead> __global__ __launch_bounds__(Threads)
void group_observer(const unsigned char* input,Result* output,int rounds){
 extern __shared__ __align__(16) unsigned char shared[];
 // Four private copy panels occupy 4B; event metadata fits within the
 // unchanged physical 6B reservation. Each event joins one producer warp.
 auto events=reinterpret_cast<unsigned long long*>(shared+4*B);
 auto starts=events+32;auto enqueued=starts+8;auto observed=enqueued+8;
 auto poll_gaps=observed+32;
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
 unsigned long long sum_first=0,sum_all=0,sum_enqueue=0,max_poll=0;
 #pragma unroll 1
 for(int r=0;r<rounds;++r){
  if(warp<8){
   auto start=timer();if(lane==0)starts[warp]=start;
   #pragma unroll
   for(int g=0;g<Lookahead;++g){
    issue(g);
    unsigned a=unsigned(__cvta_generic_to_shared(events+warp*4+g));
    // 32 deferred arrivals plus one ordinary arrival per event. The fixed
    // noinc count resets every phase; no timestamp executes on this path.
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"::"r"(a):"memory");
    if(lane==0){unsigned long long state;asm volatile("mbarrier.arrive.shared.b64 %0,[%1];":"=l"(state):"r"(a):"memory");}
   }
   auto end=timer();if(lane==0)enqueued[warp]=end;
   asm volatile("cp.async.wait_group 0;":::"memory");
  }else{
   unsigned g=lane%4;bool done=g>=Lookahead;
   unsigned a=unsigned(__cvta_generic_to_shared(events+lane));
   auto previous=timer();unsigned long long gap=0;
   while(__ballot_sync(0xffffffff,!done)){
    bool ready=!done&&completed(a,r&1);auto now=timer();
    if(!done&&now-previous>gap)gap=now-previous;
    if(ready){observed[lane]=now;done=true;}
    previous=now;
   }
   poll_gaps[lane]=gap;
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
   for(int i=0;i<32;++i)if(i%4<Lookahead&&poll_gaps[i]>max_poll)max_poll=poll_gaps[i];
  }
  __syncthreads();
 }
 unsigned sum=0;
 volatile unsigned* words=reinterpret_cast<volatile unsigned*>(shared);
 for(int w=threadIdx.x;w<4*B/4;w+=Threads)sum^=words[w];
 for(int d=16;d;d/=2)sum^=__shfl_xor_sync(0xffffffff,sum,d);
 if(lane==0){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));output[blockIdx.x*9+warp]={sum_first,sum_all,sum_enqueue,max_poll,sum,sm};}
}
template<int B,int L> void run(int grid,int trials){
 constexpr int Shared=6*B;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=132||!prop.cooperativeLaunch)throw std::runtime_error("cooperative H200 required");
 CHECK(cudaFuncSetAttribute(group_observer<B,L>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,group_observer<B,L>,Threads,Shared));
 if(resident!=(B==16384?2:1))throw std::runtime_error("physical occupancy changed");
 size_t bytes=size_t(grid)*4*B;unsigned char* input;Result* output;
 CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&output,size_t(grid)*9*sizeof(Result)));
 fill_input<<<132,256>>>(reinterpret_cast<unsigned*>(input),bytes/4);CHECK(cudaDeviceSynchronize());
 std::vector<Result> host(grid*9);std::vector<unsigned> expected(grid*9,0);
 for(int b=0;b<grid;++b)for(int t=0;t<Threads;++t)for(int w=t;w<4*B/4;w+=Threads)expected[b*9+t/32]^=pattern(unsigned(size_t(b)*4*B/4+w));
 auto launch=[&](int rounds){void* args[]={&input,&output,&rounds};CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(group_observer<B,L>),grid,Threads,args,Shared));CHECK(cudaDeviceSynchronize());CHECK(cudaGetLastError());};
 for(int warm=0;warm<6;++warm)launch(1024);
 std::vector<std::array<double,4>> samples;
 for(int trial=0;trial<trials;++trial){
  launch(Rounds);CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Result),cudaMemcpyDeviceToHost));
  unsigned long long first=0,all=0,enqueue=0,poll=0;int sms[132]={};
  for(size_t i=0;i<host.size();++i){const auto& h=host[i];
   if(h.checksum!=expected[i]||h.sm>=132)throw std::runtime_error("numerical or SM identity failure");
   if(i%9==8){if(!h.first||h.first>h.all||!h.enqueue||!h.poll)throw std::runtime_error("invalid completion endpoints");first+=h.first;all+=h.all;enqueue+=h.enqueue;if(h.poll>poll)poll=h.poll;}
   ++sms[h.sm];
  }
  for(int n:sms)if(n!=grid/132*9)throw std::runtime_error("uneven resident placement");
  double scale=double(grid)*Rounds;samples.push_back({first/scale,all/scale,enqueue/scale,double(poll)});
 }
 std::printf("{\"B_bytes\":%d,\"lookahead\":%d,\"grid\":%d,\"rounds\":127,\"threads\":288,\"producer_threads\":256,\"shared_bytes\":%d,\"resident_ctas\":%d,\"all_lane_checksum_pass\":true,\"distinct_sms\":132,\"samples\":[",B,L,grid,Shared,resident);
 for(size_t i=0;i<samples.size();++i)std::printf("%s{\"first_ns\":%.12g,\"all_ns\":%.12g,\"enqueue_ns\":%.12g,\"max_poll_ns\":%.12g}",i?",":"",samples[i][0],samples[i][1],samples[i][2],samples[i][3]);
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
