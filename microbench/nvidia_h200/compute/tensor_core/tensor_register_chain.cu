// Independent RS WGMMA plus two MAX/ADD chains with distinct live inputs.
// No scores,EX2,normalization,globaloperandstream orAttention.
#include "chain_wgmma_helper.cuh"
#include <cmath>
struct Reading{unsigned long long c0,c1,n0,n1;float a,b,sum;};
__device__ __forceinline__ unsigned long long nanos(){unsigned long long x;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(x)::"memory");return x;}
__host__ __device__ float initial_a(int tid){return 1.f+tid*(1.f/1024);}
__host__ __device__ float initial_b(int tid){return 2.f-tid*(1.f/1024);}
__host__ __device__ float expected_a(int tid,int kind,int repeats,int width,int mode){float x=initial_a(tid);return mode==0?x:(kind==0?x:x+float(repeats*width*(width+1+2*(tid&31))/2)*(1.f/32768));}
__host__ __device__ float expected_b(int tid,int kind,int repeats,int width,int mode){float x=initial_b(tid);return mode==0?x:(kind==0?x:x+float(repeats*width*(3*width+1+2*(tid&31))/2)*(1.f/32768));}
template<int Mode,int Kind,int Steps,int Width>
__device__ __forceinline__ Reading primitive(uint64_t b,int repeats){
 float accum[64],a=initial_a(threadIdx.x),v=initial_b(threadIdx.x),left[Width],right[Width];unsigned fragments[Steps][4];
 #pragma unroll
 for(int i=0;i<Width;++i){left[i]=float(i+1+(threadIdx.x&31))*(1.f/32768);right[i]=float(i+Width+1+(threadIdx.x&31))*(1.f/32768);asm volatile("":"+f"(left[i]),"+f"(right[i]));}
 #pragma unroll
 for(int i=0;i<64;++i){accum[i]=0;asm volatile("":"+f"(accum[i]));}
 #pragma unroll
 for(int k=0;k<Steps;++k){
  #pragma unroll
  for(int j=0;j<4;++j){fragments[k][j]=0x3e003e00u+((threadIdx.x*17+k*4+j)&127);asm volatile("":"+r"(fragments[k][j]));}
 }
 Reading r{};r.n0=nanos();r.c0=clock64();
 #pragma unroll 1
 for(int rep=0;rep<repeats;++rep){
  if constexpr(Mode!=1){wg_fence();
   #pragma unroll
   for(int k=0;k<Steps;++k)wgmma_rs_n128_mnmajor(accum,fragments[k],b,1);
   wg_commit();
  }
  if constexpr(Mode!=0){
   #pragma unroll
   for(int i=0;i<Width;++i){
    if constexpr(Kind==0){asm volatile("max.f32 %0,%0,%1;":"+f"(a):"f"(left[i]));asm volatile("max.f32 %0,%0,%1;":"+f"(v):"f"(right[i]));}
    else{asm volatile("add.rn.f32 %0,%0,%1;":"+f"(a):"f"(left[i]));asm volatile("add.rn.f32 %0,%0,%1;":"+f"(v):"f"(right[i]));}
   }
  }
  if constexpr(Mode!=1)wg_wait0();
 }
 r.c1=clock64();r.n1=nanos();r.a=a;r.b=v;float sum=0;
 #pragma unroll
 for(int i=0;i<64;++i)sum+=accum[i];
 #pragma unroll
 for(int k=0;k<Steps;++k){
  #pragma unroll
  for(int j=0;j<4;++j)sum+=__uint_as_float(fragments[k][j]);
 }
 r.sum=sum;return r;
}
template<int Kind,int Steps,int Width>
__global__ void __launch_bounds__(256,1) tensor_chain(int mode,int active,int repeats,Reading* output,unsigned* errors){
 extern __shared__ __align__(1024) unsigned char shared[];
 for(int i=threadIdx.x;i<kSmemBytes/2;i+=blockDim.x)reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u+((i*17)&127);
 unsigned base=unsigned(__cvta_generic_to_shared(shared));uint64_t b=make_desc(base+65536,16,1024);int wg=threadIdx.x/128;
 __syncthreads();Reading r{};
 if(wg<active){
  if(mode==0)r=primitive<0,Kind,Steps,Width>(b,repeats);
  else if(mode==1)r=primitive<1,Kind,Steps,Width>(b,repeats);
  else r=primitive<2,Kind,Steps,Width>(b,repeats);
  unsigned wrong=(r.a!=expected_a(threadIdx.x,Kind,repeats,Width,mode))+(r.b!=expected_b(threadIdx.x,Kind,repeats,Width,mode))+(!std::isfinite(r.sum)||r.sum<=0);
  if(wrong)atomicAdd(errors,wrong);
 }
 __syncthreads();if(wg<active&&threadIdx.x%128==0)output[blockIdx.x*2+wg]=r;
}
template<int Kind,int Steps,int Width>
void run(int grid,int active,int repeats,int mode,int trial,Reading* output,unsigned* errors){
 auto kernel=tensor_chain<Kind,Steps,Width>;CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmemBytes));
 kernel<<<grid,256,kSmemBytes>>>(mode,active,repeats,output,errors);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 unsigned wrong;CHECK(cudaMemcpy(&wrong,errors,sizeof(wrong),cudaMemcpyDeviceToHost));if(wrong){std::fprintf(stderr,"numericerror%u\n",wrong);std::exit(3);}
 std::vector<Reading> host(grid*2);CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Reading),cudaMemcpyDeviceToHost));
 for(int c=0;c<grid;++c)for(int wg=0;wg<active;++wg){auto r=host[c*2+wg];if(!(r.c0<r.c1&&r.n0<r.n1)||r.a!=expected_a(wg*128,Kind,repeats,Width,mode)||r.b!=expected_b(wg*128,Kind,repeats,Width,mode)){std::fprintf(stderr,"window/checksumerror\n");std::exit(4);}
  std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%.9g,%.9g\n",Kind,Steps,Width,grid,active,repeats,mode,trial,c,wg,r.c0,r.c1,r.n0,r.n1,r.a,r.b);
 }
}
template<int S,int W>
void kinds(int grid,int active,int repeats,int mode,int trial,Reading* output,unsigned* errors){run<0,S,W>(grid,active,repeats,mode,trial,output,errors);run<1,S,W>(grid,active,repeats,mode,trial,output,errors);}
int main(){
 int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;
 Reading* output;unsigned*errors;CHECK(cudaMalloc(&output,132*2*sizeof(Reading)));CHECK(cudaMalloc(&errors,sizeof(unsigned)));CHECK(cudaMemset(errors,0,sizeof(unsigned)));
 std::puts("kind,steps,width,grid,active,repeats,mode,trial,cta,wg,c0,c1,n0,n1,a,b");
 for(int grid:{1,132})for(int active:{1,2})for(int repeats:{64,128,256})for(int mode:{0,1,2})for(int trial=0;trial<3;++trial){
  kinds<4,64>(grid,active,repeats,mode,trial,output,errors);kinds<8,32>(grid,active,repeats,mode,trial,output,errors);kinds<8,64>(grid,active,repeats,mode,trial,output,errors);kinds<12,32>(grid,active,repeats,mode,trial,output,errors);
 }
 CHECK(cudaFree(output));CHECK(cudaFree(errors));
}
