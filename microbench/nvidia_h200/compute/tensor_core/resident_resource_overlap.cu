// Independent resource coexistence. Each CTA does either a hot byte-copy ring
// OR shared-resident matrix instructions. No CTA executes a GEMM pipeline;
// there are no routed rows, expert weights or mathematical matrix outputs.
#include "wgmma_ss_helper.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
constexpr int Threads=256,Panel=32768,Slots=3,Shared=98304,SMs=132;
struct Record {uint64_t begin,end,cycles;unsigned sm,role,entry,checksum;float matrix_sum;};
__device__ uint64_t timer(){uint64_t v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}
__host__ __device__ unsigned word(unsigned i){i^=i>>16;i*=0x7feb352du;i^=i>>15;i*=0x846ca68bu;return i^(i>>16);}
__global__ void initialize(unsigned* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)p[i]=word(unsigned(i));}
__device__ void issue(unsigned char* sm,const unsigned char* p,int round,int slot){
 #pragma unroll
 for(int operand=0;operand<2;++operand){
  #pragma unroll
  for(int v=threadIdx.x;v<Panel/2/16;v+=Threads){
   unsigned address=unsigned(__cvta_generic_to_shared(sm+slot*Panel+operand*(Panel/2)+v*16));
   auto* from=p+(size_t(blockIdx.x)*4+round%4)*Panel+operand*(Panel/2)+v*16;
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(from):"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
 }
}
__global__ __launch_bounds__(Threads,2) void resource_overlap(
 const unsigned char* input,unsigned* entries,Record* records,int mode,int rounds,unsigned pause_ns){
 extern __shared__ __align__(1024) unsigned char sm[];
 __shared__ unsigned role,entry,smid;
 if(threadIdx.x==0){asm volatile("mov.u32 %0, %%smid;":"=r"(smid));entry=atomicAdd(entries+smid,1u);role=mode==2?(entry%2):unsigned(mode==1||mode==4);}
 for(int v=threadIdx.x;v<Shared/4;v+=Threads)reinterpret_cast<unsigned*>(sm)[v]=0x3e803e80u;
 __syncthreads();
 asm volatile("fence.proxy.async.shared::cta;":::"memory");
 float d[64];
 #pragma unroll
 for(int j=0;j<64;++j)d[j]=0;
 auto base=unsigned(__cvta_generic_to_shared(sm));uint64_t a[4],b[4];
 #pragma unroll
 // Repeated identical descriptors avoid descriptor-register overwrites while
 // asynchronous matrix groups are outstanding. This measures coexistence,
 // not a production K layout or a replacement matrix service rate.
 for(int k=0;k<4;++k){a[k]=make_desc(base,16,1024);b[k]=make_desc(base+16384,16,1024);}
 auto begin=timer();auto c0=clock64();
 if(role==0){
  #pragma unroll
  for(int s=0;s<Slots;++s)issue(sm,input,s,s);
  #pragma unroll 1
  for(int r=0;r<rounds;++r){
   if(r<rounds-Slots)asm volatile("cp.async.wait_group 4;":::"memory");else asm volatile("cp.async.wait_group 0;":::"memory");
   __syncthreads();__syncthreads();
   if(r+Slots<rounds)issue(sm,input,r+Slots,r%Slots);
  }
  asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 }else{
  wg_fence();
  #pragma unroll 1
  for(int r=0;r<rounds;++r){
   wgmma_ss_n128_group_wait1(d,a,b,1);
   if(pause_ns)asm volatile("nanosleep.u32 %0;"::"r"(pause_ns):"memory");
  }
  wg_wait0();
 }
 auto c1=clock64();auto end=timer();
 if(threadIdx.x==0){
  float sum=0;
  #pragma unroll
  for(int j=0;j<64;++j)sum+=d[j];
  unsigned checksum=role==0?*reinterpret_cast<volatile unsigned*>(sm+((rounds-1)%Slots)*Panel):0;
  records[blockIdx.x]={begin,end,uint64_t(c1-c0),smid,role,entry,checksum,sum};
 }
}
int main(int argc,char** argv){
 if(argc!=5&&argc!=6)return 2;int mode=std::atoi(argv[1]),rounds=std::atoi(argv[2]),trials=std::atoi(argv[3]),rotation=std::atoi(argv[4]);
 unsigned pause_ns=argc==6?unsigned(std::atoi(argv[5])):0;
 if(pause_ns!=0&&pause_ns!=1024)return 2;
 if(mode<0||mode>4||(rounds!=256&&rounds!=1024)||trials<1||trials>63||rotation<0||rotation>1)return 2;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));if(!std::strstr(prop.name,"H200")||prop.multiProcessorCount!=SMs)return 2;
 CHECK(cudaFuncSetAttribute(resource_overlap,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,resource_overlap,Threads,Shared));if(resident!=2)return 3;
 cudaFuncAttributes attr{};CHECK(cudaFuncGetAttributes(&attr,resource_overlap));if(attr.localSizeBytes)return 3;
 const int grid=mode>=3?SMs:2*SMs;const size_t bytes=size_t(grid)*4*Panel;
 unsigned char* input;unsigned* entries;Record* output;
 CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&entries,SMs*4));CHECK(cudaMalloc(&output,grid*sizeof(Record)));
 initialize<<<132,256>>>(reinterpret_cast<unsigned*>(input),bytes/4);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 std::vector<Record> host(grid);
 for(int trial=-2;trial<trials;++trial){
  CHECK(cudaMemset(entries,0,SMs*4));resource_overlap<<<grid,Threads,Shared>>>(input,entries,output,mode,rounds,pause_ns);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(host.data(),output,grid*sizeof(Record),cudaMemcpyDeviceToHost));
  int roles[2]={};int bysm[SMs][2]={};uint64_t first=~uint64_t(0),last=0,cycles=0,ns=0;
  for(int block=0;block<grid;++block){const auto& r=host[block];if(r.sm>=SMs||r.role>1||r.end<=r.begin||!r.cycles)return 4;
   if(r.role==0&&r.checksum!=word(unsigned((size_t(block)*4+(rounds-1)%4)*(Panel/4))))return 4;
   if(r.role==1&&std::abs(r.matrix_sum-float(256*rounds))>0.5f)return 4;
   ++roles[r.role];++bysm[r.sm][r.role];first=std::min(first,r.begin);last=std::max(last,r.end);cycles+=r.cycles;ns+=r.end-r.begin;
  }
  for(int sm=0;sm<SMs;++sm){
   int expected=mode>=3?1:2;if(bysm[sm][0]+bysm[sm][1]!=expected)return 5;
   if(mode==2&&(bysm[sm][0]!=1||bysm[sm][1]!=1))return 5;
  }
  if(trial>=0){std::printf("{\"mode\":%d,\"rounds\":%d,\"rotation\":%d,\"trial\":%d,\"grid\":%d,\"resident_ctas\":%d,\"registers_per_thread\":%d,\"copy_ctas\":%d,\"matrix_ctas\":%d,\"body_ns\":%llu,\"weighted_mhz\":%.9g,\"ctas\":[",mode,rounds,rotation,trial,grid,resident,attr.numRegs,roles[0],roles[1],(unsigned long long)(last-first),double(cycles)/ns*1000.);
   for(int block=0;block<grid;++block){auto& r=host[block];std::printf("%s[%u,%u,%u,%llu,%llu,%llu]",block?",":"",r.sm,r.role,r.entry,(unsigned long long)r.begin,(unsigned long long)r.end,(unsigned long long)r.cycles);}
   std::printf("]}\n");}
 }
 CHECK(cudaFree(output));CHECK(cudaFree(entries));CHECK(cudaFree(input));return 0;
}
