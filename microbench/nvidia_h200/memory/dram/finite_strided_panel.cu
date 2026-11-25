// Independent byte-request progress: no matrix, router, or operator output.
// Static power-of-two row programs avoid division inside the measured loop.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#define CUDA_OK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#ifndef MIDLAYER_FINITE_MAX_GRID
#define MIDLAYER_FINITE_MAX_GRID 9184
#endif
#ifndef MIDLAYER_FINITE_SOURCE_GIB
#define MIDLAYER_FINITE_SOURCE_GIB 12
#endif
constexpr int MaxGrid=MIDLAYER_FINITE_MAX_GRID;
int last_resident_ctas=0,last_shared_reservation=0;
constexpr std::size_t SourceBytes=std::size_t(MIDLAYER_FINITE_SOURCE_GIB)*1024*1024*1024;
constexpr std::size_t WarmBytes=64*1024*1024;
__device__ unsigned smid(){unsigned s;asm volatile("mov.u32 %0, %smid;":"=r"(s));return s;}
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %globaltimer;":"=l"(t));return t;}
__device__ void cp16(unsigned char* dst,const unsigned char* src){unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(dst));asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(a),"l"(src):"memory");}
__device__ void cp16_masked(unsigned char* dst,const unsigned char* src,int bytes){unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(dst));asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(a),"l"(src),"r"(bytes):"memory");}
__host__ __device__ unsigned pattern(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
__global__ void initialize_source(unsigned* x,std::size_t n){for(std::size_t i=threadIdx.x+blockIdx.x*blockDim.x;i<n;i+=gridDim.x*blockDim.x)x[i]=pattern(static_cast<unsigned>(i));}
__global__ void evict_cache(unsigned* x,std::size_t n){unsigned v=threadIdx.x+blockIdx.x*blockDim.x;for(std::size_t i=v;i<n;i+=gridDim.x*blockDim.x)x[i]=v+static_cast<unsigned>(i);}
struct Times{unsigned long long start,end,cycles;unsigned sm;};
template<int P> __host__ __device__ constexpr int threads(){return P>=32768?256:128;}
template<int P,int Row,int AMode=0> __host__ __device__ constexpr int companion(){return (AMode==4 || AMode==5)?128*Row:P>=32768?0:16*Row;}
template<int P,int S,int Row,int Stride,bool Copy,int AMode=0>
__device__ void body(const unsigned char* src,unsigned* sink,Times* times,int rounds,int tier){
 constexpr int Threads=threads<P>(),Rows=P/Row,Columns=(Stride+Row-1)/Row,Shared=S*(P+companion<P,Row,AMode>());
 extern __shared__ __align__(16) unsigned char shared[];
 for(int i=threadIdx.x;i<Shared;i+=Threads)shared[i]=0;
 __syncthreads();unsigned long long begin=timer(),cycle_begin=clock64();
 int bands=(rounds+Columns-1)/Columns;
 int reuse=tier>=5?1<<(tier-5):1;
 std::size_t base=static_cast<std::size_t>(blockIdx.x/reuse)*bands*Rows*Stride;
 std::size_t warm_base=SourceBytes+static_cast<std::size_t>(blockIdx.x)*((4+Columns-1)/Columns)*Rows*Stride;
 auto issue=[&](int round,int slot){
  if constexpr(Copy){if(round<rounds){
   if constexpr(AMode>0){
    constexpr int ABytes=companion<P,Row,AMode>();
    #pragma unroll
    for(int v=threadIdx.x;v<ABytes/16;v+=Threads){
     bool valid=(AMode==3 || AMode==4 || ((AMode==2 || AMode==5) && v<Row/16)) && round*Row+(v%(Row/16))*16<Stride;
     std::size_t offset=SourceBytes+(v/(Row/16))*Stride+round*Row+(v%(Row/16))*16;
     cp16_masked(shared+S*P+slot*ABytes+v*16,src+offset,valid?16:0);
    }
   }
   if constexpr(AMode>0)asm volatile("cp.async.commit_group;":::"memory");
   int hot_round=round%4;
   #pragma unroll
   for(int v=threadIdx.x;v<P/16;v+=Threads){
    bool cold=tier>=5 || tier==0 || (tier==2&&v<P/16/16) || (tier==3&&v<P/16/4) || (tier==4&&v<P/16/2);
    std::size_t row_offset=(v/(Row/16))*Stride+(v%(Row/16))*16;
    std::size_t offset=cold?base+(round/Columns)*Rows*Stride+(round%Columns)*Row+row_offset:warm_base+(hot_round/Columns)*Rows*Stride+(hot_round%Columns)*Row+row_offset;
    if constexpr(Stride%Row==0)cp16(shared+slot*P+v*16,src+offset);
    else cp16_masked(shared+slot*P+v*16,src+offset,((round%Columns)*Row+(v%(Row/16))*16<Stride)?16:0);
   }
  }}
  if constexpr(AMode>0){
   if constexpr(!Copy)asm volatile("cp.async.commit_group;":::"memory");
   else if(round>=rounds)asm volatile("cp.async.commit_group;":::"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
 };
 for(int slot=0;slot<S;++slot)issue(slot,slot);
 unsigned sum=0;
 #pragma unroll 1
 for(int round=0;round<rounds;++round){
  asm volatile("cp.async.wait_group %0;"::"n"((AMode>0?2:1)*(S-1)):"memory");__syncthreads();
  volatile unsigned* data=reinterpret_cast<volatile unsigned*>(shared+(round%S)*P);
  #pragma unroll
  for(int v=threadIdx.x;v<P/16;v+=Threads)sum^=data[v*4];
  if constexpr(AMode>0){
   volatile unsigned* adata=reinterpret_cast<volatile unsigned*>(shared+S*P+(round%S)*companion<P,Row,AMode>());
   #pragma unroll
   for(int v=threadIdx.x;v<companion<P,Row,AMode>()/16;v+=Threads)sum^=adata[v*4];
  }
  __syncthreads();issue(round+S,round%S);
 }
 asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 unsigned long long cycles=clock64()-cycle_begin,end=timer();
 if(threadIdx.x==0){times[blockIdx.x]={begin,end,cycles,smid()};sink[blockIdx.x]=sum;}
}
template<int P,int S,int Row,int Stride,int AMode=0>__global__ __launch_bounds__(threads<P>()) void copy_panels(const unsigned char*x,unsigned*out,Times*t,int r,int tier){body<P,S,Row,Stride,true,AMode>(x,out,t,r,tier);}
template<int P,int S,int Row,int Stride,int AMode=0>__global__ __launch_bounds__(threads<P>()) void control_panels(const unsigned char*x,unsigned*out,Times*t,int r,int tier){body<P,S,Row,Stride,false,AMode>(x,out,t,r,tier);}
__global__ __launch_bounds__(256) void warmup_private48(const unsigned char*x,unsigned*out,Times*t){body<49152,4,128,128,true>(x,out,t,4,1);}
__global__ __launch_bounds__(256) void prewarm_panels(const unsigned char*x,unsigned*out,Times*t){body<32768,3,128,128,true>(x,out,t,4,1);}
template<int P,int S,int Row,int Stride,int AMode=0>void launch(bool copy,int g,const unsigned char*x,unsigned*out,Times*t,int r,int tier){
 constexpr int Shared=(P==16384 && S==1 && Row==256)?26624:S*(P+companion<P,Row,AMode>());
 static int copy_resident=-1,control_resident=-1;
 CUDA_OK(cudaFuncSetAttribute(copy_panels<P,S,Row,Stride,AMode>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 CUDA_OK(cudaFuncSetAttribute(control_panels<P,S,Row,Stride,AMode>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 if(copy_resident<0)CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&copy_resident,copy_panels<P,S,Row,Stride,AMode>,threads<P>(),Shared));
 if(control_resident<0)CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&control_resident,control_panels<P,S,Row,Stride,AMode>,threads<P>(),Shared));
 last_resident_ctas=copy?copy_resident:control_resident;last_shared_reservation=Shared;
 if constexpr(P==16384 && S==1 && Row==256){if(last_resident_ctas!=8)throw std::runtime_error("single-panel residency must equal8");}
 if(copy)copy_panels<P,S,Row,Stride,AMode><<<g,threads<P>(),Shared>>>(x,out,t,r,tier);
 else control_panels<P,S,Row,Stride,AMode><<<g,threads<P>(),Shared>>>(x,out,t,r,tier);
 CUDA_OK(cudaGetLastError());
}
struct Cell{int panel,slots,grid,rounds,row,stride;int tier;int a_mode=0;};
void dispatch(bool copy,Cell c,const unsigned char*x,unsigned*out,Times*t){
 #define ACASE(P,S,R,T,A) if(c.panel==P&&c.slots==S&&c.row==R&&c.stride==T&&c.a_mode==A){launch<P,S,R,T,A>(copy,c.grid,x,out,t,c.rounds,c.tier);return;}
#ifdef MIDLAYER_COALESCED_SCOPE_ONLY
 ACASE(32768,4,128,1024,5) ACASE(32768,4,128,1536,5) ACASE(32768,4,128,4096,5)
 throw std::runtime_error("unsupported coalesced-scope physical program");
#else
 ACASE(4096,3,128,4096,1) ACASE(4096,3,128,4096,2) ACASE(4096,3,128,4096,3)
 ACASE(8192,4,128,4096,1) ACASE(8192,4,128,4096,2) ACASE(8192,4,128,4096,3)
 ACASE(16384,3,256,4096,1) ACASE(16384,3,256,4096,2) ACASE(16384,3,256,4096,3)
 ACASE(32768,4,128,1024,5) ACASE(32768,4,128,1536,5) ACASE(32768,4,128,4096,5)
 ACASE(32768,4,128,1024,4) ACASE(32768,4,128,1536,4) ACASE(32768,4,128,4096,4)
 ACASE(16384,2,128,1024,2) ACASE(16384,2,128,4096,2)
 ACASE(16384,1,256,1536,2) ACASE(16384,1,256,4096,2)
 ACASE(4096,3,128,1024,2)
 ACASE(8192,3,128,512,2) ACASE(8192,3,128,1024,2) ACASE(8192,3,128,1536,2) ACASE(8192,3,128,4096,2)
 ACASE(8192,4,128,512,2) ACASE(8192,4,128,1024,2) ACASE(8192,4,128,1536,2)
 ACASE(16384,3,128,1024,2) ACASE(16384,3,128,4096,2)
 ACASE(16384,3,256,1024,2) ACASE(16384,3,256,2048,2)
 ACASE(16384,3,256,5760,2) ACASE(16384,3,256,8192,2)
 ACASE(16384,3,256,12288,2) ACASE(16384,3,256,14336,2)
 if(c.a_mode!=0)throw std::runtime_error("unsupported companion program");
 #define CASE(P,S,R,T) if(c.panel==P&&c.slots==S&&c.row==R&&c.stride==T){launch<P,S,R,T>(copy,c.grid,x,out,t,c.rounds,c.tier);return;}
 #define GEOMETRY(P,S) CASE(P,S,128,256) CASE(P,S,128,4096) CASE(P,S,256,256) CASE(P,S,256,4096)
 GEOMETRY(4096,2) GEOMETRY(4096,3) GEOMETRY(4096,4)
 GEOMETRY(8192,2) GEOMETRY(8192,3) GEOMETRY(8192,4)
 GEOMETRY(16384,2) GEOMETRY(16384,3) GEOMETRY(16384,4)
 CASE(32768,3,128,128) CASE(49152,4,128,128)
 CASE(8192,3,128,512) CASE(8192,4,128,512)
 CASE(8192,3,128,1024) CASE(8192,4,128,1024)
 CASE(8192,3,128,1536) CASE(8192,4,128,1536)
 CASE(16384,3,128,1024)
 #undef GEOMETRY
 #undef CASE
 throw std::runtime_error("unsupported static request program");
#endif
 #undef ACASE
}
int main(int argc,char**argv){try{
 int mode=argc>=2?std::atoi(argv[1]):0,rotation=argc>=3?std::atoi(argv[2]):0;
 if((mode==44?argc!=5:argc>3)||mode<0||mode>44||rotation<0||rotation>1)throw std::runtime_error("usage: probe [profile] [0|1 process]; profile44also needs physical-cells.csv and1|63samples");
 int scope_samples=mode==44?std::atoi(argv[4]):0;
 if(mode==44&&scope_samples!=1&&scope_samples!=63)throw std::runtime_error("scope profile requires1or63samples");
 cudaDeviceProp prop{};CUDA_OK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=132||!std::strstr(prop.name,"H200"))throw std::runtime_error("H200 device geometry required");
 std::fprintf(stderr,"device=%s sm=%d cc=%d.%d\n",prop.name,prop.multiProcessorCount,prop.major,prop.minor);
 unsigned char*input;unsigned*eviction;unsigned*sink;Times*times;
 CUDA_OK(cudaMalloc(&input,SourceBytes+WarmBytes));CUDA_OK(cudaMalloc(&eviction,128*1024*1024));CUDA_OK(cudaMalloc(&sink,MaxGrid*sizeof(unsigned)));CUDA_OK(cudaMalloc(&times,MaxGrid*sizeof(Times)));
 initialize_source<<<4096,256>>>(reinterpret_cast<unsigned*>(input),(SourceBytes+WarmBytes)/4);
 for(int i=0;i<128;++i)evict_cache<<<4096,256>>>(eviction,128*1024*1024/4);CUDA_OK(cudaDeviceSynchronize());
 std::vector<Cell>cells;
 if(mode==44){
  std::ifstream input(argv[3]);if(!input)throw std::runtime_error("missing physical cell input");
  int stride,grid,fanout;char comma1,comma2;
  while(input>>stride>>comma1>>grid>>comma2>>fanout){
   if(comma1!=','||comma2!=','||(stride!=1024&&stride!=1536&&stride!=4096)||grid<132||grid>MaxGrid||fanout<1||fanout>16||(fanout&(fanout-1)))throw std::runtime_error("invalid coalesced physical cell");
   if(std::size_t((grid+fanout-1)/fanout)*256*stride>SourceBytes)throw std::runtime_error("coalesced cell exceeds unique source allocation");
   int power=0;while((1<<power)<fanout)++power;
   cells.push_back({32768,4,grid,stride/128,128,stride,5+power,5});
  }
  if(!input.eof()||cells.empty())throw std::runtime_error("unclosed physical cell input");
 }
 else if(mode==1)cells.push_back({16384,3,128,16,256,4096,0});
 else if(mode==2)cells.push_back({8192,4,192,16,128,4096,0});
 else if(mode==3)cells.push_back({32768,3,264,1024,128,128,1});
 else if(mode==4)cells.push_back({32768,3,264,256,128,128,0});
 else if(mode==5)cells.push_back({32768,3,264,256,128,128,2});
 else if(mode==6)cells.push_back({32768,3,264,256,128,128,3});
 else if(mode==7){for(int g:{132,264})for(int r:{64,256,1024})cells.push_back({49152,4,g,r,128,128,1});}
 else if(mode==8)cells.push_back({49152,4,132,1024,128,128,1});
 else if(mode==9){
  for(int g:{64,128,160,192,256,320,384,512,640,1024,1280}){
   cells.push_back({4096,3,g,32,128,4096,0});
   for(int s:{3,4}){cells.push_back({8192,s,g,4,128,512,0});cells.push_back({8192,s,g,8,128,1024,0});cells.push_back({8192,s,g,12,128,1536,0});cells.push_back({8192,s,g,32,128,4096,0});}
   cells.push_back({16384,3,g,8,128,1024,0});
   cells.push_back({16384,3,g,32,128,4096,0});
   cells.push_back({16384,3,g,16,256,4096,0});
  }
 }
 else if(mode==10)cells.push_back({4096,3,320,32,128,4096,0});
 else if(mode==11)cells.push_back({8192,4,192,32,128,4096,0});
 else if(mode==12)cells.push_back({16384,3,160,32,128,4096,0});
 else if(mode==13)cells.push_back({16384,3,128,16,256,4096,0});
 else if(mode==14){for(int a_mode:{0,1,2,3}){cells.push_back({4096,3,320,32,128,4096,0,a_mode});cells.push_back({8192,4,192,32,128,4096,0,a_mode});cells.push_back({16384,3,128,16,256,4096,0,a_mode});}}
 else if(mode>=15 && mode<=17)cells.push_back({4096,3,320,32,128,4096,0,mode-14});
 else if(mode==18 || mode==23){
  for(int g:{64,128,132,160,180,192,256,320,360,384,512,640,768,896,1024,1280}){
   for(int stride:{1024,4096})cells.push_back({4096,3,g,stride/128,128,stride,0,2});
   for(int slots:{3,4})for(int stride:{512,1024,1536,4096})cells.push_back({8192,slots,g,stride/128,128,stride,0,2});
   for(int stride:{1024,4096})cells.push_back({16384,3,g,stride/128,128,stride,0,2});
   for(int stride:{1024,2048,4096,5760,8192,12288,14336})cells.push_back({16384,3,g,(stride+255)/256,256,stride,0,2});
  }
 }
 else if(mode==19)cells.push_back({16384,3,360,23,256,5760,0,2});
 else if(mode==20)cells.push_back({8192,3,256,12,128,1536,0,2});
 else if(mode==21)cells.push_back({16384,3,160,32,128,4096,0,2});
 else if(mode==22)cells.push_back({16384,3,320,32,256,8192,0,2});
 else if(mode==37){for(int stride:{1024,1536,4096})for(int g:{132,1530,2040,4592,9184})for(int factor=0;factor<5;++factor)cells.push_back({32768,4,g,stride/128,128,stride,5+factor,5});}
 else if(mode==38)cells.push_back({32768,4,1530,32,128,4096,6,5});
 else if(mode==39)cells.push_back({32768,4,2040,12,128,1536,7,5});
 else if(mode==40)cells.push_back({32768,4,4592,32,128,4096,6,5});
 else if(mode==41)cells.push_back({32768,4,9184,8,128,1024,6,5});
 else if(mode==42)cells.push_back({32768,4,132,32,128,4096,9,5});
 else if(mode==43)cells.push_back({32768,4,4592,32,128,4096,5,5});
 else if(mode==33){for(int stride:{1024,1536,4096})for(int g:{132,264,528,1056,1530,2040,2112,4096,4592,8192,9184})cells.push_back({32768,4,g,stride/128,128,stride,0,4});}
 else if(mode==34)cells.push_back({32768,4,1530,32,128,4096,0,4});
 else if(mode==35)cells.push_back({32768,4,2040,12,128,1536,0,4});
 else if(mode==36)cells.push_back({32768,4,4592,32,128,4096,0,4});
 else if(mode==24){for(int stride:{1536,4096})for(int g:{528,1056,1272,1320,1584,1696,2048,2112,3168})cells.push_back({16384,1,g,stride/256,256,stride,0,2});}
 else if(mode==25){for(int g:{132,264})for(int rounds:{4,6,8,12,16,24,32,48,64})cells.push_back({49152,4,g,rounds,128,128,1,0});}
 else if(mode==26)cells.push_back({16384,1,1272,16,256,4096,0,2});
 else if(mode==27)cells.push_back({16384,1,1696,6,256,1536,0,2});
 else if(mode==28)cells.push_back({49152,4,132,8,128,128,1,0});
 else if(mode==29)cells.push_back({49152,4,132,32,128,128,1,0});
 else if(mode==30){for(int stride:{1024,4096})for(int g:{128,160,192,256,320,384,512,640,768,1024,1280})cells.push_back({16384,2,g,stride/128,128,stride,0,2});}
 else if(mode==31)cells.push_back({16384,2,320,32,128,4096,0,2});
 else if(mode==32)cells.push_back({16384,2,640,8,128,1024,0,2});
 else{
  for(int p:{4096,8192,16384})for(int s:{2,3,4})for(int g:{64,128,192,256,512,1024})for(int r:{4,16,64,256})for(int row:{128,256})for(int stride:{256,4096})cells.push_back({p,s,g,r,row,stride,0});
  for(int tier:{0,1,2,3,4})for(int g:{132,264})for(int r:{64,256,1024})cells.push_back({32768,3,g,r,128,128,tier});
 }
 std::puts("panel_bytes,slots,threads,grid,rounds,row_bytes,row_stride,tier,mode,trial,logical_bytes,body_ns,median_cta_ns,weighted_mhz,distinct_sms,max_ctas_per_sm,companion_mode,shared_reservation_bytes,resident_ctas_per_sm,request_fanout");
 std::vector<Times>observed(MaxGrid);std::vector<unsigned>checksums(MaxGrid);
 for(std::size_t pos=0;pos<cells.size();++pos){const Cell c=cells[mode==44?pos:(pos*37)%cells.size()];
  for(int trial=-3;trial<(mode==44?scope_samples:mode>=23?63:(mode>=9?21:5));++trial)for(int m=0;m<2;++m){bool copy=((m+trial+2+rotation)%2)==0;
   if(c.tier!=1){evict_cache<<<4096,256>>>(eviction,128*1024*1024/4);CUDA_OK(cudaDeviceSynchronize());}
   if(c.tier>=2 && c.tier<=4){
    CUDA_OK(cudaFuncSetAttribute(prewarm_panels,cudaFuncAttributeMaxDynamicSharedMemorySize,98304));
    prewarm_panels<<<c.grid,256,98304>>>(input,sink,times);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
   }
   if(c.tier==1 && c.panel==49152){
    CUDA_OK(cudaFuncSetAttribute(warmup_private48,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
    warmup_private48<<<c.grid,256,196608>>>(input,sink,times);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
   }
   dispatch(copy,c,input,sink,times);CUDA_OK(cudaDeviceSynchronize());
   CUDA_OK(cudaMemcpy(observed.data(),times,c.grid*sizeof(Times),cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(checksums.data(),sink,c.grid*sizeof(unsigned),cudaMemcpyDeviceToHost));
   int columns=(c.stride+c.row-1)/c.row,bands=(c.rounds+columns-1)/columns,rows=c.panel/c.row,threads=c.panel>=32768?256:128;
   for(int b=0;b<c.grid;++b){unsigned expected=0;
    if(copy)for(int round=0;round<c.rounds;++round)for(int v=0;v<c.panel/16;v+=threads){
     bool cold=c.tier>=5||c.tier==0||(c.tier==2&&v<c.panel/16/16)||(c.tier==3&&v<c.panel/16/4)||(c.tier==4&&v<c.panel/16/2);
     int source_round=cold?round:round%4;
     std::size_t base=cold?static_cast<std::size_t>(b/(c.tier>=5?1<<(c.tier-5):1))*bands*rows*c.stride:SourceBytes+static_cast<std::size_t>(b)*((4+columns-1)/columns)*rows*c.stride;
     std::size_t offset=base+(source_round/columns)*rows*c.stride+(source_round%columns)*c.row+(v/(c.row/16))*c.stride+(v%(c.row/16))*16;
     if((source_round%columns)*c.row+(v%(c.row/16))*16<c.stride)expected^=pattern(static_cast<unsigned>(offset/4));
    }
    if(copy && c.a_mode>0)for(int round=0;round<c.rounds;++round)for(int v=0;v<((c.a_mode==4 || c.a_mode==5)?128:16)*c.row/16;v+=threads){
     bool valid=(c.a_mode==3 || c.a_mode==4 || ((c.a_mode==2 || c.a_mode==5) && v<c.row/16)) && round*c.row+(v%(c.row/16))*16<c.stride;
     if(valid){std::size_t offset=SourceBytes+(v/(c.row/16))*c.stride+round*c.row+(v%(c.row/16))*16;expected^=pattern(static_cast<unsigned>(offset/4));}
    }
    if(checksums[b]!=expected)throw std::runtime_error("copy/control checksum mismatch");
   }
   if(trial<0)continue;
   std::vector<unsigned long long>spans;unsigned long long first=~0ULL,last=0,total_ns=0,total_cycles=0;int counts[256]={};
   for(int i=0;i<c.grid;++i){const auto t=observed[i];if(t.sm>=256||t.end<=t.start)throw std::runtime_error("invalid timer/SM identity");first=std::min(first,t.start);last=std::max(last,t.end);spans.push_back(t.end-t.start);total_ns+=t.end-t.start;total_cycles+=t.cycles;++counts[t.sm];}
   std::sort(spans.begin(),spans.end());int distinct=0,max_count=0;for(int v:counts){distinct+=v>0;max_count=std::max(max_count,v);}
   std::printf("%d,%d,%d,%d,%d,%d,%d,%s,%s,%d,%llu,%llu,%llu,%.6f,%d,%d,%d,%d,%d,%d\n",c.panel,c.slots,c.panel>=32768?256:128,c.grid,c.rounds,c.row,c.stride,c.tier>=5?"cold_shared_requests":c.tier==0?"cold_unique":c.tier==1?"warm_private":c.tier==2?"mixed_sixteenth":c.tier==3?"mixed_quarter":"mixed_half",copy?"copy":"control",trial,static_cast<unsigned long long>(c.panel/c.row)*c.grid*((c.rounds/columns)*c.stride+std::min((c.rounds%columns)*c.row,c.stride)),last-first,spans[spans.size()/2],double(total_cycles)/total_ns*1000.,distinct,max_count,c.a_mode,last_shared_reservation,last_resident_ctas,c.tier>=5?1<<(c.tier-5):1);
  }
 }
 CUDA_OK(cudaFree(times));CUDA_OK(cudaFree(sink));CUDA_OK(cudaFree(eviction));CUDA_OK(cudaFree(input));
 }catch(const std::exception&e){std::fprintf(stderr,"error: %s\n",e.what());return 1;}}
