// Independent request-layer accounting. No matrix, routing, or operator output.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
constexpr int Threads=256,Rounds=257,AB=16384;
__host__ __device__ unsigned value(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void fill(unsigned* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=value(unsigned(i));}
__device__ unsigned smid(){unsigned x;asm volatile("mov.u32 %0, %%smid;":"=r"(x));return x;}
struct Result{unsigned checksum,sm;};
template<int B,int Slots,int Stride>
__global__ __launch_bounds__(Threads) void hot_requests(const unsigned char* p,int fanout,int skew,Result* out){
 extern __shared__ __align__(16) unsigned char shared[];
 const int local=blockIdx.x%fanout;
 unsigned long long start=clock64(),delay=fanout==1?0:static_cast<unsigned long long>(local)*skew/(fanout-1);
 while(clock64()-start<delay)asm volatile("":::"memory");
 constexpr int Rows=B/128,Shared=Slots*(AB+B);
 auto issue=[&](int round,int slot){
  if(round>=Rounds)return;
  #pragma unroll
  for(int v=threadIdx.x;v<AB/16;v+=Threads){
   unsigned dst=unsigned(__cvta_generic_to_shared(shared+Slots*B+slot*AB+v*16));
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;"::"r"(dst),"l"(p):"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
  #pragma unroll
  for(int v=threadIdx.x;v<B/16;v+=Threads){
   size_t offset=(size_t(blockIdx.x/fanout)*Rows+v/8)*Stride+(round%4)*128+(v%8)*16;
   unsigned dst=unsigned(__cvta_generic_to_shared(shared+slot*B+v*16));
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(dst),"l"(p+offset):"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
 };
 for(int s=0;s<Slots;++s)issue(s,s);
 #pragma unroll 1
 for(int r=0;r<Rounds;++r){
  asm volatile("cp.async.wait_group %0;"::"n"(2*(Slots-1)):"memory");
  __syncthreads();__syncthreads();issue(r+Slots,r%Slots);
 }
 asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 unsigned sum=0;
 volatile unsigned* last=reinterpret_cast<volatile unsigned*>(shared+((Rounds-1)%Slots)*B);
 for(int w=threadIdx.x;w<B/4;w+=Threads)sum^=last[w];
 volatile unsigned* azero=reinterpret_cast<volatile unsigned*>(shared+Slots*B);
 for(int w=threadIdx.x;w<Slots*AB/4;w+=Threads)sum^=azero[w];
 for(int delta=16;delta;delta/=2)sum^=__shfl_xor_sync(0xffffffff,sum,delta);
 __syncthreads();
 unsigned* reduction=reinterpret_cast<unsigned*>(shared+Shared-32);
 if(threadIdx.x%32==0)reduction[threadIdx.x/32]=sum;
 __syncthreads();
 if(threadIdx.x==0){unsigned total=0;for(int w=0;w<8;++w)total^=reduction[w];out[blockIdx.x]={total,smid()};}
}
template<int B,int Slots,int Stride> void run(int grid,int fanout,int skew){
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=132)throw std::runtime_error("H200 geometry required");
 constexpr int Shared=Slots*(AB+B),Rows=B/128;
 CHECK(cudaFuncSetAttribute(hot_requests<B,Slots,Stride>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,hot_requests<B,Slots,Stride>,Threads,Shared));
 if(resident!=(B==16384?2:1))throw std::runtime_error("changed physical occupancy");
 size_t bytes=size_t((grid+fanout-1)/fanout)*Rows*Stride;
 unsigned char* input;Result* result;CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&result,grid*sizeof(Result)));
 fill<<<4096,256>>>(reinterpret_cast<unsigned*>(input),bytes/4);CHECK(cudaDeviceSynchronize());
 std::vector<Result> host(grid);
 for(int trial=0;trial<4;++trial){
  hot_requests<B,Slots,Stride><<<grid,Threads,Shared>>>(input,fanout,skew,result);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(host.data(),result,grid*sizeof(Result),cudaMemcpyDeviceToHost));
  int sms[132]={};
  for(int block=0;block<grid;++block){unsigned expected=0;
   for(int word=0;word<B/4;++word){size_t offset=(size_t(block/fanout)*Rows+word/32)*Stride+((Rounds-1)%4)*128+(word%32)*4;expected^=value(unsigned(offset/4));}
   if(host[block].checksum!=expected||host[block].sm>=132)throw std::runtime_error("checksum or SM identity failed");++sms[host[block].sm];
  }
  int count=0;for(int x:sms)count+=x>0;if(count!=132)throw std::runtime_error("incomplete SM coverage");
 }
 std::printf("{\"B_bytes\":%d,\"slots\":%d,\"stride\":%d,\"grid\":%d,\"fanout\":%d,\"skew_cycles\":%d,\"rounds\":%d,\"threads\":256,\"shared_bytes\":%d,\"resident_ctas\":%d,\"distinct_sms\":132,\"all_lane_checksum_pass\":true}\n",B,Slots,Stride,grid,fanout,skew,Rounds,Shared,resident);
 CHECK(cudaFree(result));CHECK(cudaFree(input));
}
int main(int argc,char** argv){try{
 if(argc!=6)throw std::runtime_error("B stride grid fanout skew required");
 int b=std::atoi(argv[1]),stride=std::atoi(argv[2]),grid=std::atoi(argv[3]),fanout=std::atoi(argv[4]),skew=std::atoi(argv[5]);
 if((fanout!=1&&fanout!=2&&fanout!=4&&fanout!=8&&fanout!=16)||(skew!=0&&skew!=512)||grid<132||grid>528)throw std::runtime_error("unfrozen input");
 #define CELL(B,S,T) if(b==B&&stride==T){if(grid!=132*(B==16384?2:1)&&grid!=264*(B==16384?2:1))throw std::runtime_error("unfrozen grid");run<B,S,T>(grid,fanout,skew);return 0;}
 CELL(16384,3,12288) CELL(16384,3,32768) CELL(32768,4,1536) CELL(32768,4,4096)
 throw std::runtime_error("unfrozen program");
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
