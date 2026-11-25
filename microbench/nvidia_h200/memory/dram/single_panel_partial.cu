// Independent single-panel byte progress. No matrix, routing, or GEMM output.
// Shared reservation reproduces resident CTA count, not native register use.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(10);}}while(0)
constexpr size_t SourceBytes=size_t(3)*1024*1024*1024;
struct Record{unsigned sm,checksum;unsigned long long begin,end,cycles;};
__host__ __device__ unsigned pattern(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
__global__ void initialize(unsigned* x,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)x[i]=pattern(unsigned(i));}
__device__ void cp16(unsigned char* dst,const unsigned char* src,int valid){unsigned address=unsigned(__cvta_generic_to_shared(dst));asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(address),"l"(src),"r"(valid):"memory");}
template<int P,int RowsA,int Row,int Threads,int Shared>
__global__ __launch_bounds__(Threads) void partial_single_panel(const unsigned char* src,Record* out,int stride){
 extern __shared__ __align__(128) unsigned char shared[];
 const size_t base=size_t(blockIdx.x)*(P/Row)*stride;
 unsigned sum=0;unsigned long long begin=timer(),start=clock64();
 #pragma unroll 1
 for(int round=0;round<stride/Row;++round){
  #pragma unroll
  for(int v=threadIdx.x;v<RowsA*Row/16;v+=Threads)cp16(shared+P+v*16,src+SourceBytes+round*Row+(v%(Row/16))*16,v<Row/16?16:0);
  asm volatile("cp.async.commit_group;":::"memory");
  #pragma unroll
  for(int v=threadIdx.x;v<P/16;v+=Threads){size_t offset=base+(v/(Row/16))*stride+round*Row+(v%(Row/16))*16;cp16(shared+v*16,src+offset,16);}
  asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");__syncthreads();
  volatile unsigned* b=reinterpret_cast<volatile unsigned*>(shared);volatile unsigned* a=reinterpret_cast<volatile unsigned*>(shared+P);
  #pragma unroll
  for(int v=threadIdx.x;v<P/16;v+=Threads)sum+=b[v*4];
  #pragma unroll
  for(int v=threadIdx.x;v<RowsA*Row/16;v+=Threads)sum+=a[v*4];
  __syncthreads();
 }
 unsigned long long cycles=clock64()-start,end=timer();
 if(threadIdx.x==0){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));out[blockIdx.x]={sm,sum,begin,end,cycles};}
}
template<int P,int RowsA,int Row,int Threads,int Shared,int Resident>
void run(const unsigned char* src,Record* out,unsigned* eviction,int l2,int grid,int stride,int samples,int process,int config){
 auto kernel=partial_single_panel<P,RowsA,Row,Threads,Shared>;
 CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,Threads,Shared));if(resident!=Resident)exit(4);
 cudaFuncAttributes attr{};CHECK(cudaFuncGetAttributes(&attr,kernel));std::vector<Record> host(grid);
 if(size_t(grid)*(P/Row)*stride>SourceBytes)exit(7);
 for(int trial=-3;trial<samples;++trial){
  initialize<<<132,256>>>(eviction,2*size_t(l2)/4);CHECK(cudaDeviceSynchronize());
  kernel<<<grid,Threads,Shared>>>(src,out,stride);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(host.data(),out,grid*sizeof(Record),cudaMemcpyDeviceToHost));
  unsigned long long begin=~0ull,end=0,cycles=0,span=0;std::vector<int> sms(132,0);
  for(int b=0;b<grid;++b){unsigned expected=0;size_t base=size_t(b)*(P/Row)*stride;
   for(int r=0;r<stride/Row;++r){for(int v=0;v<P/16;v+=Threads){size_t offset=base+(v/(Row/16))*stride+r*Row+(v%(Row/16))*16;expected+=pattern(unsigned(offset/4));}for(int v=0;v<RowsA*Row/16;v+=Threads)if(v<Row/16)expected+=pattern(unsigned((SourceBytes+r*Row+(v%(Row/16))*16)/4));}
   const auto& x=host[b];if(x.sm>=132||x.checksum!=expected||x.end<=x.begin||x.cycles==0)exit(5);
   ++sms[x.sm];begin=std::min(begin,x.begin);end=std::max(end,x.end);cycles+=x.cycles;span+=x.end-x.begin;
  }
  if(std::count_if(sms.begin(),sms.end(),[](int n){return n>0;})!=132)exit(6);
  if(trial<0)continue;
  printf("{\"config\":%d,\"panel\":%d,\"rows_a\":%d,\"row\":%d,\"threads\":%d,\"shared\":%d,\"resident\":%d,\"registers\":%d,\"grid\":%d,\"stride\":%d,\"rounds\":%d,\"trial\":%d,\"process\":%d,\"body_ns\":%llu,\"clock64_hz\":%.9f,\"distinct_sms\":132,\"max_ctas_per_sm\":%d}\n",config,P,RowsA,Row,Threads,Shared,resident,attr.numRegs,grid,stride,stride/Row,trial,process,end-begin,1e9*double(cycles)/span,*std::max_element(sms.begin(),sms.end()));
 }
}
int main(int argc,char**argv){
 if(argc!=6)return 1;int config=atoi(argv[1]),grid=atoi(argv[2]),stride=atoi(argv[3]),samples=atoi(argv[4]),process=atoi(argv[5]);
 if(config<0||config>4||grid<264||grid>4168||(stride!=1024&&stride!=1536&&stride!=4096)||samples<1||samples>63||process<0||process>1)return 2;
 if(stride==1024&&config!=4)return 2;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));if(prop.major!=9||prop.multiProcessorCount!=132||!strstr(prop.name,"H200"))return 3;
 unsigned char* src;unsigned* eviction;Record* out;
 CHECK(cudaMalloc(&src,SourceBytes+8192));CHECK(cudaMalloc(&eviction,2*size_t(prop.l2CacheSize)));CHECK(cudaMalloc(&out,grid*sizeof(Record)));
 initialize<<<4096,256>>>(reinterpret_cast<unsigned*>(src),(SourceBytes+8192)/4);CHECK(cudaDeviceSynchronize());
 #define RUN(C,P,A,R,T,S,N) if(config==C)run<P,A,R,T,S,N>(src,out,eviction,prop.l2CacheSize,grid,stride,samples,process,config)
 RUN(0,16384,16,256,128,26624,8);
 RUN(1,32768,16,512,128,40960,5);
 RUN(2,32768,16,256,128,49152,4);
 RUN(3,65536,32,256,128,98304,2);
 RUN(4,65536,16,256,256,98304,2);
 CHECK(cudaFree(src));CHECK(cudaFree(eviction));CHECK(cudaFree(out));
}
