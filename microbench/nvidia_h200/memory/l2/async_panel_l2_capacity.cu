// Independent byte-copy capacity: no matrix, routing or production output.
// 256 threads, three 32-KiB slots, shared allocation admitting two CTAs/SM.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(1);} } while(0)
constexpr int Threads=256,Panel=32768,Stages=3,Shared=Panel*Stages;
struct Result { unsigned long long cycles,ns; unsigned checksum,smid; };
__device__ unsigned long long timer_ns(){unsigned long long v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v)::"memory");return v;}
__device__ void copy_panel(unsigned char* dst,const unsigned char* src){
  #pragma unroll
  for(int i=0;i<Panel/(Threads*16);++i){
    unsigned off=(i*Threads+threadIdx.x)*16;
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(dst+off));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(addr),"l"(src+off):"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
}
__global__ __launch_bounds__(Threads,2) void async_panel_l2(
    const unsigned char* input,Result* results,int panels,int rounds){
  extern __shared__ __align__(16) unsigned char storage[];
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  auto ns=timer_ns();auto cycles=clock64();
  for(int r=0;r<Stages;++r)copy_panel(storage+r*Panel,input+((blockIdx.x*17+r)%panels)*Panel);
  unsigned checksum=0;
  #pragma unroll 1
  for(int r=0;r<rounds;++r){
    if(r<rounds-Stages)asm volatile("cp.async.wait_group 1;":::"memory");
    else asm volatile("cp.async.wait_group 0;":::"memory");
    __syncthreads();
    int slot=r%Stages;
    checksum+=reinterpret_cast<volatile unsigned*>(storage+slot*Panel)[threadIdx.x];
    __syncthreads();
    int next=r+Stages;
    if(next<rounds)copy_panel(storage+slot*Panel,input+((blockIdx.x*17+next)%panels)*Panel);
  }
  auto end=clock64();auto endns=timer_ns();
  if(threadIdx.x==0)results[blockIdx.x]=Result{static_cast<unsigned long long>(end-cycles),endns-ns,checksum,sm};
}
__global__ void fill(unsigned* input,int words){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=blockDim.x*gridDim.x)input[i]=static_cast<unsigned>(i/(Panel/4)+1);
}
int positive(const char* p){char* end;long n=std::strtol(p,&end,10);if(*end||n<=0||n>1000000)std::exit(2);return n;}
int capacity_main(int argc,char** argv);
int entropy_main(int argc,char** argv);
int main(int argc,char** argv){
 if(argc==7 && std::strcmp(argv[1],"entropy")==0)return entropy_main(argc,argv);
 if(argc==7 && std::strcmp(argv[1],"capacity")==0)return capacity_main(argc,argv);
 if(argc!=5)return 2;
 int grid=positive(argv[1]),mib=positive(argv[2]),rounds=positive(argv[3]),trials=positive(argv[4]);
 if((mib!=8&&mib!=32)||(rounds!=256&&rounds!=1024)||trials>9)return 2;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(std::strcmp(prop.name,"NVIDIA H200")||prop.multiProcessorCount!=132)return 2;
 CHECK(cudaFuncSetAttribute(async_panel_l2,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,async_panel_l2,Threads,Shared));
 if(resident!=2)return 3;
 int bytes=mib*1024*1024,panels=bytes/Panel;
 unsigned char* input;Result* output;CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&output,grid*sizeof(Result)));
 fill<<<132,256>>>(reinterpret_cast<unsigned*>(input),bytes/4);CHECK(cudaDeviceSynchronize());
 cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
 std::vector<Result> host(grid);std::vector<float> samples;
 unsigned long long sumcycles=0,sumns=0;
 for(int trial=-2;trial<trials;++trial){
  async_panel_l2<<<grid,Threads,Shared>>>(input,output,panels,1024);CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(begin));async_panel_l2<<<grid,Threads,Shared>>>(input,output,panels,rounds);
  CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));CHECK(cudaGetLastError());
  float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));
  CHECK(cudaMemcpy(host.data(),output,grid*sizeof(Result),cudaMemcpyDeviceToHost));
  unsigned long long cy=0,ns=0;
  for(int block=0;block<grid;++block){
   unsigned expected=0;for(int r=0;r<rounds;++r)expected+=(block*17+r)%panels+1;
   if(host[block].checksum!=expected)return 4;
   cy+=host[block].cycles;ns+=host[block].ns;
  }
  if(trial>=0){samples.push_back(ms);sumcycles+=cy;sumns+=ns;}
 }
 std::printf("{\"grid\":%d,\"backing_mib\":%d,\"rounds\":%d,\"threads\":%d,\"shared_bytes\":%d,\"resident_ctas\":%d,\"logical_bytes\":%llu,\"sum_cycles\":%llu,\"sum_ns\":%llu,\"samples_ms\":[",grid,mib,rounds,Threads,Shared,resident,static_cast<unsigned long long>(grid)*rounds*Panel,sumcycles,sumns);
 for(int i=0;i<trials;++i)std::printf("%s%.9g",i?",":"",samples[i]);
 std::printf("]}\n");
 CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaFree(output));CHECK(cudaFree(input));
}

// Paired independent ingress/control experiment. A and B are private strided
// byte streams. Consumer=0 measures copies and synchronization only; the final
// checksum is deliberately outside the timing window. No matrix operation.
struct CapacityResult { unsigned long long begin,end,cycles; unsigned checksum,sm; };
__host__ __device__ unsigned capacity_pattern(unsigned i){i^=i>>16;i*=0x7feb352du;i^=i>>15;i*=0x846ca68bu;return i^(i>>16);}
// A deterministic, approximately normal scalar distribution from twelve
// independent uniform bytes. This is an entropy contrast, not an operator.
__host__ __device__ unsigned entropy_bf16(unsigned index){
 int sum=0;
 for(unsigned j=0;j<3;++j){unsigned h=capacity_pattern(index*3+j);
  for(int b=0;b<4;++b)sum+=int((h>>(b*8))&255);}
 float value=float(sum-1530)*(1.0f/256.0f);
 unsigned bits;std::memcpy(&bits,&value,sizeof(bits));
 return (bits+0x7fffu+((bits>>16)&1u))>>16;
}
__host__ __device__ unsigned capacity_value(unsigned index,int pattern){
 if(pattern==0)return capacity_pattern(index);
 if(pattern==1)return entropy_bf16(index*2)|(entropy_bf16(index*2+1)<<16);
 return 0x3f803f80u;
}
__global__ void capacity_fill(unsigned* p,size_t words,int pattern){
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=size_t(blockDim.x)*gridDim.x)p[i]=capacity_value(unsigned(i),pattern);
}
template<int Stride> __device__ void capacity_issue(unsigned char* sm,const unsigned char* p,int round,int slot){
 constexpr int Half=Panel/2,Row=128,Rows=Half/Row;
 #pragma unroll
 for(int operand=0;operand<2;++operand){
  #pragma unroll
  for(int v=threadIdx.x;v<Half/16;v+=Threads){
   size_t off=(size_t(blockIdx.x)*2+operand)*Rows*Stride+(v/(Row/16))*Stride+(round%4)*Row+(v%(Row/16))*16;
   unsigned dst=unsigned(__cvta_generic_to_shared(sm+slot*Panel+operand*Half+v*16));
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(dst),"l"(p+off):"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
 }
}
template<int Stride,bool Consumer> __global__ __launch_bounds__(Threads,2)
void copy_capacity(const unsigned char* p,CapacityResult* out,int rounds){
 extern __shared__ __align__(16) unsigned char sm[];
 for(int v=threadIdx.x;v<Shared/4;v+=Threads)reinterpret_cast<unsigned*>(sm)[v]=0;
 __syncthreads();
 auto begin=timer_ns();auto c0=clock64();
 #pragma unroll
 for(int slot=0;slot<Stages;++slot)capacity_issue<Stride>(sm,p,slot,slot);
 unsigned checksum=0;
 #pragma unroll 1
 for(int r=0;r<rounds;++r){
  if(r<rounds-Stages)asm volatile("cp.async.wait_group 4;":::"memory");
  else asm volatile("cp.async.wait_group 0;":::"memory");
  __syncthreads();
  if constexpr(Consumer){
   volatile unsigned* panel=reinterpret_cast<volatile unsigned*>(sm+(r%Stages)*Panel);
   #pragma unroll
   for(int v=threadIdx.x;v<Panel/16;v+=Threads)checksum^=panel[v*4];
  }
  __syncthreads();
  if(r+Stages<rounds)capacity_issue<Stride>(sm,p,r+Stages,r%Stages);
 }
 asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 auto c1=clock64();auto end=timer_ns();
 if constexpr(!Consumer){
  volatile unsigned* panel=reinterpret_cast<volatile unsigned*>(sm+((rounds-1)%Stages)*Panel);
  #pragma unroll
  for(int v=threadIdx.x;v<Panel/16;v+=Threads)checksum^=panel[v*4];
 }
 if(threadIdx.x==0){unsigned smid;asm volatile("mov.u32 %0, %%smid;":"=r"(smid));out[blockIdx.x]={begin,end,static_cast<unsigned long long>(c1-c0),checksum,smid};}
}
template<int Stride,bool Consumer> int run_capacity(int grid,int rounds,int trials,int pattern=0){
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9 || prop.multiProcessorCount!=132 || !std::strstr(prop.name,"H200"))return 2;
 CHECK(cudaFuncSetAttribute(copy_capacity<Stride,Consumer>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,copy_capacity<Stride,Consumer>,Threads,Shared));if(resident!=2)return 3;
 size_t bytes=size_t(grid)*256*Stride;unsigned char* input;CapacityResult* result;
 CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&result,size_t(grid)*sizeof(CapacityResult)));
 capacity_fill<<<4096,256>>>(reinterpret_cast<unsigned*>(input),bytes/4,pattern);CHECK(cudaDeviceSynchronize());
 std::vector<CapacityResult> host(grid);
 for(int trial=-2;trial<trials;++trial){
  copy_capacity<Stride,Consumer><<<grid,Threads,Shared>>>(input,result,rounds);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(host.data(),result,size_t(grid)*sizeof(CapacityResult),cudaMemcpyDeviceToHost));
  unsigned long long first=~0ULL,last=0,total_ns=0,total_cycles=0;int sms[132]={};
  for(int block=0;block<grid;++block){
   unsigned expected=0;
   for(int r=Consumer?0:rounds-1;r<rounds;++r)for(int v=0;v<Panel/16;v+=Threads){
    int operand=v/(Panel/2/16),j=v%(Panel/2/16);
    size_t offset=(size_t(block)*2+operand)*128*Stride+(j/8)*Stride+(r%4)*128+(j%8)*16;
    expected^=capacity_value(unsigned(offset/4),pattern);
   }
   if(host[block].checksum!=expected || host[block].sm>=132 || host[block].end<=host[block].begin)return 4;
   first=std::min(first,host[block].begin);last=std::max(last,host[block].end);total_ns+=host[block].end-host[block].begin;total_cycles+=host[block].cycles;++sms[host[block].sm];
  }
  int distinct=0,maximum=0;for(int x:sms){distinct+=x>0;maximum=std::max(maximum,x);}
  if(trial>=0)std::printf("{\"grid\":%d,\"row_stride_bytes\":%d,\"rounds\":%d,\"consumer\":%d,\"pattern\":%d,\"trial\":%d,\"resident_ctas_per_sm\":%d,\"logical_bytes\":%llu,\"body_ns\":%llu,\"weighted_mhz\":%.9g,\"distinct_sms\":%d,\"maximum_ctas_per_sm\":%d}\n",grid,Stride,rounds,int(Consumer),pattern,trial,resident,static_cast<unsigned long long>(grid)*rounds*Panel,last-first,double(total_cycles)/total_ns*1000.,distinct,maximum);
 }
 CHECK(cudaFree(result));CHECK(cudaFree(input));return 0;
}
int capacity_main(int argc,char** argv){
 if(std::strcmp(argv[5],"0") && std::strcmp(argv[5],"1"))return 2;
 int grid=positive(argv[2]),stride=positive(argv[3]),rounds=positive(argv[4]),consumer=std::atoi(argv[5]),trials=positive(argv[6]);
 if((grid!=132&&grid!=264&&grid!=528)||(rounds!=256&&rounds!=1024)||(consumer!=0&&consumer!=1)||trials>63)return 2;
 #define CAPACITY(S,C) if(stride==S&&consumer==C)return run_capacity<S,C>(grid,rounds,trials);
 CAPACITY(512,0) CAPACITY(512,1) CAPACITY(4096,0) CAPACITY(4096,1) CAPACITY(5760,0) CAPACITY(5760,1) CAPACITY(12288,0) CAPACITY(12288,1) CAPACITY(32768,0) CAPACITY(32768,1)
 return 2;
}
int entropy_main(int argc,char** argv){
 int grid=positive(argv[2]),stride=positive(argv[3]),rounds=positive(argv[4]),trials=positive(argv[6]);
 if(std::strcmp(argv[5],"0")&&std::strcmp(argv[5],"1")&&std::strcmp(argv[5],"2"))return 2;
 int pattern=std::atoi(argv[5]);
 if((grid!=132&&grid!=264&&grid!=528)||(rounds!=256&&rounds!=1024)||trials>63)return 2;
 if(stride==512)return run_capacity<512,false>(grid,rounds,trials,pattern);
 if(stride==12288)return run_capacity<12288,false>(grid,rounds,trials,pattern);
 return 2;
}
