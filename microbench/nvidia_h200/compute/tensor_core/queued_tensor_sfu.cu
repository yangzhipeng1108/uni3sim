// Independent queued SS/RS instructions and bounded EX2 streams on separate WGs.
// Legacy K4/two-C or full-A K8/one-C;no scores,max,sum,normalization orAttention.
#include "queued_wgmma_helper.cuh"
#include <algorithm>
#include <cmath>
#ifndef QUEUED_WGMMA_K_STEPS
#define QUEUED_WGMMA_K_STEPS 4
#endif
#ifndef QUEUED_POST_JOIN
#define QUEUED_POST_JOIN 0
#endif
#ifndef QUEUED_ACCUMULATOR_BANKS
#define QUEUED_ACCUMULATOR_BANKS 2
#endif
#ifndef QUEUED_BRACKET_CLOCK
#define QUEUED_BRACKET_CLOCK 0
#endif
#ifndef QUEUED_COVER_SCALAR
#define QUEUED_COVER_SCALAR 0
#endif
#ifndef QUEUED_COVER_SAMPLE
#define QUEUED_COVER_SAMPLE 0
#endif
static_assert(!QUEUED_COVER_SCALAR || (QUEUED_WGMMA_K_STEPS == 8 && QUEUED_POST_JOIN && !QUEUED_BRACKET_CLOCK));
static_assert(!QUEUED_COVER_SAMPLE || QUEUED_COVER_SCALAR);
static_assert(QUEUED_BRACKET_CLOCK == 0 || QUEUED_BRACKET_CLOCK == 1);
static_assert(QUEUED_WGMMA_K_STEPS == 4 || QUEUED_WGMMA_K_STEPS == 8);
static_assert(QUEUED_POST_JOIN == 0 || QUEUED_POST_JOIN == 1);
static_assert((QUEUED_WGMMA_K_STEPS == 4 && QUEUED_ACCUMULATOR_BANKS == 2)
           || (QUEUED_WGMMA_K_STEPS == 8 && QUEUED_ACCUMULATOR_BANKS == 1));
__device__ __forceinline__ unsigned validation_join(){
 if constexpr(QUEUED_POST_JOIN){
  extern __shared__ __align__(1024) unsigned char shared[];
  unsigned address=unsigned(__cvta_generic_to_shared(shared))+200704,word;
  asm volatile("bar.sync 5,256;ld.volatile.shared.b32 %0,[%1];":"=r"(word):"r"(address):"memory");
  return word^0x3f003f00u;
 }else{return 0;}
}
struct ClockBracket{unsigned long long before,cycle,after;};
struct Stamp{unsigned long long c0,c8,c1,n0,n8,n1;
#if QUEUED_COVER_SCALAR
 unsigned long long c32,n32,c128,n128;
#endif
#if QUEUED_BRACKET_CLOCK
 ClockBracket start,stop;
#endif
};
// Each read feeds a nonzero check and branch around trap before the next read.
// This brackets clock64 in globaltimer rather than assuming an adjacent read order.
__device__ __forceinline__ ClockBracket bracket_clock(){
 ClockBracket s{};
 asm volatile("{.reg .pred p,q;mov.u64 %0,%%globaltimer;"
  "setp.ne.u64 p,%0,0;@!p trap;mov.u64 %1,%%clock64;"
  "setp.ne.u64 q,%1,0;@!q trap;mov.u64 %2,%%globaltimer;}"
  :"=l"(s.before),"+l"(s.cycle),"+l"(s.after)::"memory");
 return s;
}
__device__ __forceinline__ unsigned long long nanos(){unsigned long long v;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(v)::"memory");return v;}
__device__ __forceinline__ void cover_arrive(){
 if constexpr(QUEUED_COVER_SCALAR)asm volatile("bar.arrive 4,256;":::"memory");
}
__device__ __forceinline__ void cover_wait(){
 if constexpr(QUEUED_COVER_SCALAR)asm volatile("bar.sync 4,256;":::"memory");
}
template<int Depth>__device__ __forceinline__ void queue_wait(){asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(Depth-1):"memory");}
__host__ __device__ float bf16_value(unsigned bits){unsigned word=bits<<16;float v;memcpy(&v,&word,4);return v;}
__host__ __device__ float matrix_value(bool rs,int repeats){
 if(!rs || QUEUED_WGMMA_K_STEPS==8)return float(16*QUEUED_WGMMA_K_STEPS*repeats)*.25f;
 float sum=0;for(int k=0;k<QUEUED_WGMMA_K_STEPS;++k)for(int j=0;j<4;++j)sum+=bf16_value(0x3f00u+(QUEUED_WGMMA_K_STEPS==8?k:k*4+j)*16);
 return float(2*repeats)*sum;
}
__host__ float matrix_total(bool rs,int repeats){return float(QUEUED_ACCUMULATOR_BANKS*128*64)*matrix_value(rs,repeats);}
template<bool RS,int Depth>__device__ __forceinline__ Stamp matrix(uint64_t a0,uint64_t b0,unsigned fragment_base,int repeats,float& checksum,unsigned& bad){
 float accum[QUEUED_ACCUMULATOR_BANKS][64];unsigned fragments[QUEUED_WGMMA_K_STEPS][4];uint64_t ad[QUEUED_WGMMA_K_STEPS],bd[QUEUED_WGMMA_K_STEPS];
 #pragma unroll
 for(int bank=0;bank<QUEUED_ACCUMULATOR_BANKS;++bank){
  #pragma unroll
  for(int i=0;i<64;++i){accum[bank][i]=0;asm volatile("":"+f"(accum[bank][i]));}
 }
 #pragma unroll
 for(int k=0;k<QUEUED_WGMMA_K_STEPS;++k){
  ad[k]=a0+((k*2048)>>4);bd[k]=b0+((k*4096)>>4);
  #pragma unroll
  for(int j=0;j<4;++j){
   if constexpr(QUEUED_WGMMA_K_STEPS==8 && RS){
    unsigned address=fragment_base+(k*4+j)*4;
    asm volatile("ld.volatile.shared.b32 %0,[%1];":"=r"(fragments[k][j]):"r"(address):"memory");
   }else{unsigned v=0x3f00u+(k*4+j)*16;fragments[k][j]=v|(v<<16);asm volatile("":"+r"(fragments[k][j]));}
  }
 }
 Stamp s{};wg_fence();s.n0=nanos();s.c0=clock64();
 #pragma unroll 1
 for(int r=0;r<repeats;++r){
  #pragma unroll
  for(int bank=0;bank<QUEUED_ACCUMULATOR_BANKS;++bank){
   #pragma unroll
   for(int k=0;k<QUEUED_WGMMA_K_STEPS;++k){if constexpr(RS)wgmma_rs_n128_mnmajor(accum[bank],fragments[k],bd[k],1);else wgmma_ss_n128_kmajor(accum[bank],ad[k],bd[k],1);}
   wg_commit();queue_wait<Depth>();
  }
  if(r==7){s.c8=clock64();s.n8=nanos();cover_arrive();}
#if QUEUED_COVER_SAMPLE
  if(r==31){s.c32=clock64();s.n32=nanos();}
  if(r==127){s.c128=clock64();s.n128=nanos();}
#endif
#if QUEUED_BRACKET_CLOCK
  if(r==7)s.start=bracket_clock();
  if(r==31)s.stop=bracket_clock();
#endif
 }
 wg_wait0();s.c1=clock64();s.n1=nanos();unsigned token=validation_join();checksum=0;bad=0;
 float expected=matrix_value(RS,repeats);
 #pragma unroll
 for(int bank=0;bank<QUEUED_ACCUMULATOR_BANKS;++bank){
  #pragma unroll
  for(int i=0;i<64;++i){float checked=accum[bank][i]+float(token);checksum+=checked;
   if constexpr(QUEUED_WGMMA_K_STEPS==8)bad+=!std::isfinite(checked)||checked!=expected;
   else bad+=!std::isfinite(checked)||checked<=0;
  }
 }
 return s;
}
template<int Width>__device__ __forceinline__ Stamp scalar(int repeats,float& checksum,unsigned& bad){
 float values[Width];
 #pragma unroll
 for(int i=0;i<Width;++i){values[i]=1.f+(i+threadIdx.x)*(1.f/1024);asm volatile("":"+f"(values[i]));}
 Stamp s{};cover_wait();s.n0=nanos();s.c0=clock64();
 #pragma unroll 1
 for(int r=0;r<repeats;++r){
  #pragma unroll
  for(int i=0;i<Width;++i)asm volatile("{.reg .f32 input;neg.f32 input,%0;ex2.approx.ftz.f32 %0,input;}":"+f"(values[i]));
  if(r==7){s.c8=clock64();s.n8=nanos();}
#if QUEUED_BRACKET_CLOCK
  if(r==512/Width-1)s.start=bracket_clock();
  if(r==2048/Width-1)s.stop=bracket_clock();
#endif
 }
 s.c1=clock64();s.n1=nanos();unsigned token=validation_join();checksum=0;bad=0;
 #pragma unroll
 for(int i=0;i<Width;++i){float checked=values[i]+float(token);checksum+=checked;bad+=!std::isfinite(checked)||fabsf(checked-.641185744504986f)>2e-6f;}
 return s;
}
template<bool RS,int Depth,int Width>__global__ void __launch_bounds__(256,1) queued(int mode,int repeats,Stamp* output,float* checks,unsigned* errors){
 extern __shared__ __align__(1024) unsigned char shared[];
 __shared__ float sums[256];__shared__ unsigned wrong[256];
 for(int i=threadIdx.x;i<204800/2;i+=blockDim.x)reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u;
 unsigned base=unsigned(__cvta_generic_to_shared(shared));uint64_t a=make_desc(base,16,1024),b=make_desc(base+65536,16,1024);int wg=threadIdx.x/128;Stamp s{};float sum=0;unsigned bad=0;
 __syncthreads();
 if(wg==0&&mode!=1)s=matrix<RS,Depth>(a,b,base+196608,(QUEUED_COVER_SCALAR?4:1)*repeats,sum,bad);
 if(wg==1&&mode!=0)s=scalar<Width>(repeats,sum,bad);
 if(wg==0&&mode==1){cover_arrive();validation_join();}
 if(wg==1&&mode==0){cover_wait();validation_join();}
 sums[threadIdx.x]=sum;wrong[threadIdx.x]=bad;__syncthreads();
 if(threadIdx.x%128==0){float total=0;unsigned error=0;for(int i=0;i<128;++i){total+=sums[wg*128+i];error+=wrong[wg*128+i];}
  int index=blockIdx.x*2+wg;output[index]=s;checks[index]=total;errors[index]=error;
 }
}
template<bool RS,int Depth,int Width>void run(int grid,int repeats,int mode,int trial,Stamp* output,float* checks,unsigned* errors){
 auto kernel=queued<RS,Depth,Width>;CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,204800));kernel<<<grid,256,204800>>>(mode,repeats,output,checks,errors);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 std::vector<Stamp> stamps(grid*2);std::vector<float> sums(grid*2);std::vector<unsigned> bad(grid*2);CHECK(cudaMemcpy(stamps.data(),output,stamps.size()*sizeof(Stamp),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(sums.data(),checks,sums.size()*sizeof(float),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(bad.data(),errors,bad.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
 for(int c=0;c<grid;++c)for(int wg=0;wg<2;++wg){if((mode==0&&wg==1)||(mode==1&&wg==0))continue;auto s=stamps[c*2+wg];float sum=sums[c*2+wg],expected=wg==0?matrix_total(RS,(QUEUED_COVER_SCALAR?4:1)*repeats):float(128*Width)*.641185744504986f;
  if(bad[c*2+wg]||!(s.c0<s.c8&&s.c8<s.c1&&s.n0<s.n8&&s.n8<s.n1)||fabsf(sum-expected)>(wg==0?0.f:float(128*Width)*5e-6f)){std::fprintf(stderr,"numeric/window gate:rs%d,d%d,w%d,mode%d,wg%d,sum%.9g,expected%.9g\n",int(RS),Depth,Width,mode,wg,sum,expected);std::exit(3);}
  std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%.9g",int(RS),Depth,Width,grid,repeats,mode,trial,c,wg,s.c0,s.c8,s.c1,s.n0,s.n8,s.n1,sum);
#if QUEUED_BRACKET_CLOCK
  if(!(s.start.before>0&&s.start.before<=s.start.after&&s.start.after<s.stop.before&&s.stop.before<=s.stop.after&&s.start.cycle>0&&s.start.cycle<s.stop.cycle))std::exit(4);
  std::printf(",%llu,%llu,%llu,%llu,%llu,%llu",s.start.before,s.start.cycle,s.start.after,s.stop.before,s.stop.cycle,s.stop.after);
#endif
#if QUEUED_COVER_SCALAR
  if(QUEUED_COVER_SAMPLE&&wg==0&&!(s.c8<s.c32&&s.c32<s.c128&&s.c128<s.c1&&s.n8<s.n32&&s.n32<s.n128&&s.n128<s.n1))std::exit(6);
  std::printf(",%llu,%llu,%llu,%llu",s.c32,s.n32,s.c128,s.n128);
#endif
  std::puts("");
 }
}
template<bool RS,int Depth>void widths(int grid,int repeats,int mode,int trial,Stamp* o,float* c,unsigned*e){run<RS,Depth,32>(grid,repeats,mode,trial,o,c,e);run<RS,Depth,64>(grid,repeats,mode,trial,o,c,e);run<RS,Depth,128>(grid,repeats,mode,trial,o,c,e);}
int main(){int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;Stamp* o;float*c;unsigned*e;CHECK(cudaMalloc(&o,132*2*sizeof(Stamp)));CHECK(cudaMalloc(&c,132*2*sizeof(float)));CHECK(cudaMalloc(&e,132*2*sizeof(unsigned)));
 std::printf("register_a,depth,width,grid,repeats,mode,trial,cta,wg,c0,c8,c1,n0,n8,n1,checksum");
#if QUEUED_BRACKET_CLOCK
 std::printf(",b0,bc0,a0,b1,bc1,a1");
#endif
#if QUEUED_COVER_SCALAR
 std::printf(",c32,n32,c128,n128");
#endif
 std::puts("");
 for(int grid:{1,132})for(int repeats:{64*(QUEUED_COVER_SCALAR?4:1),128*(QUEUED_COVER_SCALAR?4:1),256*(QUEUED_COVER_SCALAR?4:1)})for(int mode:{0,1,2})for(int trial=0;trial<3;++trial){widths<false,1>(grid,repeats,mode,trial,o,c,e);widths<false,4>(grid,repeats,mode,trial,o,c,e);widths<true,1>(grid,repeats,mode,trial,o,c,e);widths<true,4>(grid,repeats,mode,trial,o,c,e);}
 CHECK(cudaFree(o));CHECK(cudaFree(c));CHECK(cudaFree(e));}
