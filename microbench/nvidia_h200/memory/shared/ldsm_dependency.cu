// Independent LDSM x4 load/use-address bundles. No async copy or matrix math.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
constexpr unsigned Constant=0x12341234u,Shared=196608,SMs=132,Unroll=1;
struct Record{unsigned long long begin,end,begin_ns,end_ns;unsigned sm,valid;};
__device__ unsigned long long ns_clock(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
template<int Chains> __global__ __launch_bounds__(1024,1)
void ldsm_dependency(Record* output,unsigned inner){
 extern __shared__ __align__(16) unsigned data[];
 unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
 for(unsigned i=threadIdx.x;i<Shared/4;i+=blockDim.x)data[i]=Constant;
 __syncthreads();cooperative_groups::this_grid().sync();
 unsigned base[Chains],address[Chains],checksum[Chains];
 #pragma unroll
 for(int c=0;c<Chains;++c){base[c]=unsigned(__cvta_generic_to_shared(data+warp*(Chains*128)+c*128+lane*4));address[c]=base[c];checksum[c]=0;}
 unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
 auto begin_ns=ns_clock();unsigned long long begin=clock64();
 #pragma unroll 1
 for(unsigned i=0;i<inner;++i){
  #pragma unroll
  for(int u=0;u<Unroll;++u){
   unsigned fragment[Chains][4];
   #pragma unroll
   for(int c=0;c<Chains;++c)asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(fragment[c][0]),"=r"(fragment[c][1]),"=r"(fragment[c][2]),"=r"(fragment[c][3]):"r"(address[c]):"memory");
   #pragma unroll
   for(int c=0;c<Chains;++c){
    unsigned x=fragment[c][0]^fragment[c][1]^fragment[c][2]^fragment[c][3];
    asm volatile("add.u32 %0,%1,%2;":"=r"(address[c]):"r"(base[c]),"r"(x));
    asm volatile("add.u32 %0,%0,%1;":"+r"(checksum[c]):"r"(fragment[c][0]));
   }
  }
 }
 // Each final store address depends on ALL four loaded register results.
 // It forces their actual consumption before the end-clock issue. Tail
 // stores/loop/checksum arithmetic remain in the full measured bundle bound.
 #pragma unroll
 for(int c=0;c<Chains;++c)asm volatile("st.shared.u32 [%0],%1;"::"r"(address[c]),"r"(checksum[c]):"memory");
 unsigned long long end=clock64();auto end_ns=ns_clock();
 bool valid=true;
 #pragma unroll
 for(int c=0;c<Chains;++c)valid=valid&&address[c]==base[c]&&checksum[c]==Constant*unsigned(inner*Unroll);
 unsigned mask=__ballot_sync(0xffffffff,valid);
 if(lane==0)output[blockIdx.x*(blockDim.x/32)+warp]={begin,end,begin_ns,end_ns,sm,mask==0xffffffff};
}
template<int Chains> void execute(unsigned warps,unsigned inner,unsigned samples){
 int device=0,count=0;CHECK(cudaGetDeviceCount(&count));if(count!=1)throw std::runtime_error("exactly one H200 required");
 cudaDeviceProp prop;CHECK(cudaGetDeviceProperties(&prop,device));
 if(!strstr(prop.name,"H200")||prop.multiProcessorCount!=SMs||!prop.cooperativeLaunch)throw std::runtime_error("132SM cooperative H200 required");
 auto kernel=ldsm_dependency<Chains>;
 CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int active;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,warps*32,Shared));
 if(active!=1)throw std::runtime_error("one resident CTA required");
 Record* output;CHECK(cudaMalloc(&output,sizeof(Record)*SMs*warps));std::vector<Record> host(SMs*warps);
 void* args[]={&output,&inner};
 printf("{\"chains\":%d,\"warps\":%u,\"inner\":%u,\"bundles_per_chain\":%u,\"samples\":[",Chains,warps,inner,inner*Unroll);
 for(unsigned s=0;s<samples+6;++s){
  CHECK(cudaLaunchCooperativeKernel((void*)kernel,SMs,warps*32,args,Shared));CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(host.data(),output,sizeof(Record)*host.size(),cudaMemcpyDeviceToHost));
  unsigned seen[SMs]={};
  for(unsigned b=0;b<SMs;++b)for(unsigned w=0;w<warps;++w){auto& v=host[b*warps+w];
   if(!v.valid||v.sm>=SMs||v.end<=v.begin||v.end_ns<=v.begin_ns||v.sm!=host[b*warps].sm)throw std::runtime_error("invalid warp data");
   if(w==0)++seen[v.sm];}
  for(unsigned sm=0;sm<SMs;++sm)if(seen[sm]!=1)throw std::runtime_error("not exactly one CTA per SM");
  if(s<6)continue;
  printf("%s[",s==6?"":",");
  for(unsigned b=0;b<SMs;++b){
   auto first=host[b*warps];unsigned long long min_c=~0ull,max_c=0,min_n=~0ull,max_n=0,sum_c=0,sum_n=0,min_start=~0ull,max_start=0;
   for(unsigned w=0;w<warps;++w){auto& v=host[b*warps+w];auto c=v.end-v.begin,n=v.end_ns-v.begin_ns;
    min_c=std::min(min_c,c);max_c=std::max(max_c,c);min_n=std::min(min_n,n);max_n=std::max(max_n,n);sum_c+=c;sum_n+=n;min_start=std::min(min_start,v.begin);max_start=std::max(max_start,v.begin);}
   printf("%s[%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu]",b?",":"",first.sm,min_c,max_c,min_n,max_n,sum_c,sum_n,min_start,max_start);
  }
  printf("]");
 }
 printf("]}\n");CHECK(cudaFree(output));
}
int main(int argc,char** argv){try{
 if(argc==2&&!strcmp(argv[1],"--cpu-only")){
  for(unsigned w=0;w<32;++w)for(unsigned c=0;c<4;++c)for(unsigned m=0;m<4;++m){unsigned occupied[32]={};
   for(unsigned lane=m*8;lane<(m+1)*8;++lane)for(unsigned word=0;word<4;++word)++occupied[((w*512+c*128+lane*4+word)%32)];
   for(auto v:occupied)if(v!=1)throw std::runtime_error("bank fixture failed");}
  printf("{\"passed\":true,\"gpu_executed\":false,\"ldsm_x4_groups\":512}\n");return 0;
 }
 if(argc!=5)throw std::runtime_error("chains warps inner samples required");
 unsigned chains=std::stoul(argv[1]),warps=std::stoul(argv[2]),inner=std::stoul(argv[3]),samples=std::stoul(argv[4]);
 if((warps!=1&&warps!=4&&warps!=8&&warps!=16&&warps!=32)||(inner!=256&&inner!=1024)||samples<1||samples>17)throw std::runtime_error("unsupported physical primitive geometry");
 if(chains==1)execute<1>(warps,inner,samples);else if(chains==4)execute<4>(warps,inner,samples);else throw std::runtime_error("chains must be1or4");
 return 0;
 }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 2;}}
