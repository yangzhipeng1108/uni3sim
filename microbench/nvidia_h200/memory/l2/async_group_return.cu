// Independent cp.async group-return ordering, with private hot byte streams.
// No matrix, routed indices, activation, production output, or fitted price.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
constexpr int Threads=256,Rounds=127;
struct Result { unsigned long long first,all,gap; unsigned checksum,sm; };
__host__ __device__ unsigned pattern(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void fill_input(unsigned* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)p[i]=pattern(unsigned(i));}
__device__ unsigned long long timer(){unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
template<int B,int Lookahead> __global__ __launch_bounds__(Threads)
void group_return(const unsigned char* input,Result* output,int rounds){
 extern __shared__ __align__(16) unsigned char shared[];
 auto issue=[&](int slot){
  #pragma unroll
  for(int v=threadIdx.x;v<B/16;v+=Threads){
   unsigned dst=unsigned(__cvta_generic_to_shared(shared+slot*B+v*16));
   const unsigned char* src=input+(size_t(blockIdx.x)*4+slot)*B+v*16;
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(dst),"l"(src):"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
 };
 #pragma unroll
 for(int g=0;g<4;++g)issue(g);
 asm volatile("cp.async.wait_group 0;":::"memory");
 cooperative_groups::this_grid().sync();
 const bool leader=threadIdx.x%32==0;
 unsigned long long sum_first=0,sum_all=0,sum_gap=0;
 #pragma unroll 1
 for(int r=0;r<rounds;++r){
  issue(0);unsigned long long first_issue=0,last_issue=0,first_done=0,last_done=0;
  if(leader)first_issue=timer();
  #pragma unroll
  for(int g=1;g<Lookahead;++g)issue(g);
  if(leader)last_issue=timer();
  asm volatile("cp.async.wait_group %0;"::"n"(Lookahead-1):"memory");
  __syncwarp();
  if(leader)first_done=timer();
  if constexpr(Lookahead>1){
   asm volatile("cp.async.wait_group 0;":::"memory");__syncwarp();
   if(leader)last_done=timer();
  }else if(leader)last_done=first_done;
  if(leader){sum_first+=first_done-first_issue;sum_all+=last_done-first_issue;sum_gap+=last_issue-first_issue;}
  __syncwarp();
 }
 __syncthreads();
 unsigned sum=0;
 volatile unsigned* words=reinterpret_cast<volatile unsigned*>(shared);
 for(int w=threadIdx.x;w<4*B/4;w+=Threads)sum^=words[w];
 for(int d=16;d;d/=2)sum^=__shfl_xor_sync(0xffffffff,sum,d);
 if(leader){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));output[blockIdx.x*8+threadIdx.x/32]={sum_first,sum_all,sum_gap,sum,sm};}
}
template<int B,int L> void run(int grid,int trials){
 constexpr int Shared=6*B;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=132)throw std::runtime_error("H200 required");
 CHECK(cudaFuncSetAttribute(group_return<B,L>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,group_return<B,L>,Threads,Shared));
 if(resident!=(B==16384?2:1)||!prop.cooperativeLaunch)throw std::runtime_error("physical occupancy or cooperative launch changed");
 size_t bytes=size_t(grid)*4*B;unsigned char* input;Result* output;
 CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&output,size_t(grid)*8*sizeof(Result)));
 fill_input<<<132,256>>>(reinterpret_cast<unsigned*>(input),bytes/4);CHECK(cudaDeviceSynchronize());
 std::vector<Result> host(grid*8);std::vector<unsigned> expected(grid*8,0);
 for(int b=0;b<grid;++b)for(int t=0;t<Threads;++t)for(int w=t;w<4*B/4;w+=Threads)expected[b*8+t/32]^=pattern(unsigned(size_t(b)*4*B/4+w));
 auto launch=[&](int rounds){void* args[]={&input,&output,&rounds};CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(group_return<B,L>),grid,Threads,args,Shared));CHECK(cudaDeviceSynchronize());CHECK(cudaGetLastError());};
 for(int warm=0;warm<6;++warm)launch(2048);
 std::vector<std::array<double,3>> samples;
 for(int trial=0;trial<trials;++trial){
  launch(Rounds);CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Result),cudaMemcpyDeviceToHost));
  unsigned long long first=0,all=0,gap=0;int sms[132]={};
  for(size_t i=0;i<host.size();++i){const auto& h=host[i];
   if(h.checksum!=expected[i]||h.sm>=132||!h.first||h.first>h.all||h.gap>h.first)throw std::runtime_error("numerical,return ordering or SM identity failure");
   first+=h.first;all+=h.all;gap+=h.gap;++sms[h.sm];
  }
  for(int n:sms)if(n!=grid/132*8)throw std::runtime_error("uneven physical resident placement");
  double scale=double(host.size())*Rounds;
  samples.push_back({first/scale,all/scale,gap/scale});
 }
 std::printf("{\"B_bytes\":%d,\"lookahead\":%d,\"grid\":%d,\"rounds\":127,\"threads\":256,\"shared_bytes\":%d,\"resident_ctas\":%d,\"warm_kernels\":6,\"warm_groups_per_kernel\":4,\"all_lane_checksum_pass\":true,\"distinct_sms\":132,\"samples\":[",B,L,grid,Shared,resident);
 for(size_t i=0;i<samples.size();++i)std::printf("%s{\"first_ns\":%.12g,\"all_ns\":%.12g,\"issue_gap_ns\":%.12g}",i?",":"",samples[i][0],samples[i][1],samples[i][2]);
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
