// Independent generic-register -> WGMMA register-A boundary.
// Exact inverse ADDs touch either each live packed A word or a disjoint bank.
// No Attention, conversion, reduction, KV stream or concurrent scalar work.
#include "register_a_wgmma_helper.cuh"
struct Reading { unsigned long long c0,c1,n0,n1; unsigned errors; };
__device__ __forceinline__ unsigned long long ticks(){unsigned long long x;asm volatile("mov.u64 %0,%%clock64;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ unsigned long long nanos(){unsigned long long x;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ void identity_pair(float& x){
 asm volatile("add.rn.f32 %0,%0,0f3e800000;add.rn.f32 %0,%0,0fbe800000;":"+f"(x)::"memory");
}
template<int Steps>
__global__ void __launch_bounds__(256,1)
register_a_boundary(int mode,int active,int repeats,Reading* output){
 extern __shared__ __align__(1024) unsigned char shared[];
 __shared__ unsigned errors;
 __shared__ unsigned long long markers[8][4];
 __shared__ volatile float primer[256];
 for(int i=threadIdx.x;i<kSmemBytes/2;i+=blockDim.x)reinterpret_cast<unsigned short*>(shared)[i]=0x3d80u;
 unsigned base=unsigned(__cvta_generic_to_shared(shared));
 uint64_t b=make_desc(base+65536,16,1024);
 float accum[64],other[Steps][4]; unsigned fragments[Steps][4];
 #pragma unroll
 for(int i=0;i<64;++i){accum[i]=0;asm volatile("":"+f"(accum[i]));}
 #pragma unroll
 for(int k=0;k<Steps;++k){
  #pragma unroll
  for(int j=0;j<4;++j){
   fragments[k][j]=0x3d803d80u;
   other[k][j]=1.f+(4*k+j)*(1.f/8)+(threadIdx.x&31)*(1.f/32);
   asm volatile("":"+r"(fragments[k][j]),"+f"(other[k][j]));
  }
 }
 float primed=0;
 #pragma unroll
 for(int i=0;i<64;++i)primed+=accum[i];
 #pragma unroll
 for(int k=0;k<Steps;++k){
  #pragma unroll
  for(int j=0;j<4;++j)primed+=__uint_as_float(fragments[k][j])+other[k][j];
 }
 primer[threadIdx.x]=primed;
 if(threadIdx.x==0)errors=0;
 __syncthreads();
 int wg=threadIdx.x/128,warp=threadIdx.x/32;
 if(wg<active){
  auto n0=nanos(); auto c0=ticks();
  #pragma unroll 1
  for(int repeat=0;repeat<repeats;++repeat){
   if(mode==1){
    #pragma unroll
    for(int k=0;k<Steps;++k){
     #pragma unroll
     for(int j=0;j<4;++j){
      float word=__uint_as_float(fragments[k][j]);
      identity_pair(word);
      fragments[k][j]=__float_as_uint(word);
     }
    }
   }else{
    #pragma unroll
    for(int k=0;k<Steps;++k){
     #pragma unroll
     for(int j=0;j<4;++j)identity_pair(other[k][j]);
    }
   }
   wg_fence();
   #pragma unroll
   for(int k=0;k<Steps;++k)wgmma_rs_n128_mnmajor(accum,fragments[k],b,1);
   wg_commit();wg_wait0();
  }
  auto c1=ticks(); auto n1=nanos();
  unsigned late=unsigned(c1>>63),tag=(threadIdx.x&31)^late,wrong=late;
  float expected=float((repeats^late)*Steps)*(1.f/16);
  #pragma unroll
  for(int i=0;i<64;++i)wrong+=accum[i]!=expected;
  #pragma unroll
  for(int k=0;k<Steps;++k){
   #pragma unroll
   for(int j=0;j<4;++j){
    wrong+=fragments[k][j]!=(0x3d803d80u^late);
    wrong+=other[k][j]!=(1.f+(4*k+j)*(1.f/8)+tag*(1.f/32));
   }
  }
  if(wrong)atomicAdd(&errors,wrong);
  if((threadIdx.x&31)==0){markers[warp][0]=c0;markers[warp][1]=c1;markers[warp][2]=n0;markers[warp][3]=n1;}
 }
 __syncthreads();
 if(threadIdx.x==0){
  Reading r{~0ull,0,~0ull,0,errors};
  for(int w=0;w<active*4;++w){r.c0=min(r.c0,markers[w][0]);r.c1=max(r.c1,markers[w][1]);r.n0=min(r.n0,markers[w][2]);r.n1=max(r.n1,markers[w][3]);}
  output[blockIdx.x]=r;
 }
}
template<int Steps>
void run(int grid,int active,int repeats,Reading* output){
 auto kernel=register_a_boundary<Steps>;
 CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmemBytes));
 for(int trial=-1;trial<3;++trial)for(int mode=0;mode<2;++mode){
  kernel<<<grid,256,kSmemBytes>>>(mode,active,repeats,output);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  std::vector<Reading> host(grid);CHECK(cudaMemcpy(host.data(),output,grid*sizeof(Reading),cudaMemcpyDeviceToHost));
  for(int c=0;c<grid;++c){auto r=host[c];if(r.errors||r.c0>=r.c1||r.n0>=r.n1){std::fprintf(stderr,"numeric/window failure steps%d mode%d grid%d cta%d errors%u\n",Steps,mode,grid,c,r.errors);std::exit(4);}
   if(trial>=0)std::printf("%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu\n",Steps,grid,active,repeats,mode,trial,c,r.c0,r.c1,r.n0,r.n1);
  }
 }
}
int main(){
 int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;
 Reading* output;CHECK(cudaMalloc(&output,132*sizeof(Reading)));
 std::puts("k,grid,active,repeats,mode,trial,cta,c0,c1,n0,n1");
 for(int grid:{1,132})for(int active:{1,2})for(int repeats:{64,128,256}){
  run<4>(grid,active,repeats,output);run<8>(grid,active,repeats,output);run<12>(grid,active,repeats,output);
 }
 CHECK(cudaFree(output));
}
