// Independent cache-resident cp.async and static-operand SS WGMMA streams.
// No softmax, Attention, Q/K/V dependency chain, or production input.
#include "copy_tensor_wgmma_helper.cuh"
#include <cmath>
constexpr int Repeats=128, Panels=4, CopyOffset=128*1024, CopySlot=32*1024;
__device__ unsigned long long global_ns() {
 unsigned long long t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t)::"memory");return t;
}
__host__ __device__ unsigned pattern(size_t i) {return 0x3f000000u|unsigned((i*9973u+17)&0x7fffffu);}
template<int Bytes> __device__ __forceinline__ void copy_loop(const unsigned* input,unsigned shared_base) {
 #pragma unroll 1
 for(int r=0;r<Repeats;++r) {
  size_t source=(size_t(blockIdx.x)*Panels+r%Panels)*(Bytes/4);
  unsigned target=shared_base+CopyOffset+(r%2)*CopySlot;
  #pragma unroll
  for(int vector=threadIdx.x;vector<Bytes/16;vector+=128) {
   asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::
       "r"(target+vector*16),"l"(input+source+vector*4):"memory");
  }
  asm volatile("cp.async.commit_group;" :::"memory");
  if(r>=1) asm volatile("cp.async.wait_group 1;" :::"memory");
 }
 asm volatile("cp.async.wait_all;" :::"memory");
}
template<int Matrices> __device__ __forceinline__ float matrix_loop(uint64_t a,uint64_t b) {
 float accum[64];
 #pragma unroll
 for(int i=0;i<64;++i) accum[i]=0;
 #pragma unroll 1
 for(int r=0;r<Repeats;++r) {
  wg_fence();
  #pragma unroll
  for(int m=0;m<Matrices;++m) wgmma_ss_n128_kmajor(accum,a,b,1);
  wg_commit();wg_wait0();
 }
 float sum=0;
 #pragma unroll
 for(int i=0;i<64;++i) sum+=accum[i];
 return sum;
}
template<int Bytes,int Matrices,int MatrixWGs>
__global__ void __launch_bounds__((MatrixWGs+1)*128,1)
copy_tensor(int mode,const unsigned* input,unsigned long long* trace,unsigned* errors,float* sink) {
 extern __shared__ __align__(1024) unsigned char shared[];
 for(int i=threadIdx.x;i<kSmemBytes/2;i+=blockDim.x)
  reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u+((i*17)&127);
 unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
 uint64_t a=make_desc(base,16,1024),b=make_desc(base+65536,16,1024);
 const int wg=threadIdx.x/128;
 __syncthreads();auto c0=clock64();auto n0=global_ns();float value=0;
 if(mode==3) {
  if(wg==0) copy_loop<Bytes>(input,base);
  __syncthreads();
  if(wg>0) value=matrix_loop<Matrices>(a,b);
 } else if(wg==0) {
  if(mode==0 || mode==2) copy_loop<Bytes>(input,base);
 } else {
  if(mode==1 || mode==2) value=matrix_loop<Matrices>(a,b);
 }
 __syncthreads();auto n1=global_ns();auto c1=clock64();
 if(threadIdx.x%128==0) {int i=(blockIdx.x*(MatrixWGs+1)+wg)*2;trace[i]=c1-c0;trace[i+1]=n1-n0;}
 if(wg==0 && mode!=1) {
  unsigned bad=0;
  for(int slot=0;slot<2;++slot) {
   int last=Repeats-2+slot;
   const unsigned* data=reinterpret_cast<const unsigned*>(shared+CopyOffset+slot*CopySlot);
   size_t source=(size_t(blockIdx.x)*Panels+last%Panels)*(Bytes/4);
   for(int word=threadIdx.x;word<Bytes/4;word+=128) if(data[word]!=pattern(source+word)) ++bad;
  }
  if(bad) atomicAdd(errors+blockIdx.x,bad);
 }
 if(!isfinite(value)) atomicAdd(errors+blockIdx.x,1);
 if(threadIdx.x%128==0) sink[blockIdx.x*(MatrixWGs+1)+wg]=value;
}
template<int Bytes,int Matrices,int WGs>
void run(int grid,int mode,unsigned* input,unsigned long long* trace,unsigned* errors,float* sink,int warm,int trials) {
 auto kernel=copy_tensor<Bytes,Matrices,WGs>;cudaFuncAttributes attr;CHECK(cudaFuncGetAttributes(&attr,kernel));
 CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,kSmemBytes));
 for(int trial=-warm;trial<trials;++trial) {
  CHECK(cudaMemset(errors,0,grid*sizeof(*errors)));
  kernel<<<grid,(WGs+1)*128,kSmemBytes>>>(mode,input,trace,errors,sink);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  std::vector<unsigned long long> times(grid*(WGs+1)*2);std::vector<unsigned> bad(grid);
  CHECK(cudaMemcpy(times.data(),trace,times.size()*sizeof(*trace),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(bad.data(),errors,bad.size()*sizeof(*errors),cudaMemcpyDeviceToHost));
  for(int c=0;c<grid;++c) {
   if(bad[c]) {fprintf(stderr,"stream checksum failure\n");exit(3);}
   for(int wg=0;wg<=WGs;++wg) {
    int i=(c*(WGs+1)+wg)*2;
    if(!times[i] || !times[i+1]) {fprintf(stderr,"invalid clock interval\n");exit(4);}
    if(wg>0 && mode!=0 && times[i]<Repeats*Matrices*32ull) {fprintf(stderr,"MMA interval below conservative instruction-work floor\n");exit(5);}
    printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n",Bytes,Matrices,WGs,grid,mode,trial,c,wg,attr.numRegs,times[i],times[i+1]);
   }
  }
 }
}
template<int Bytes,int Matrices> void wgs(int grid,unsigned* input,unsigned long long* trace,unsigned* errors,float* sink) {
 for(int mode=0;mode<4;++mode) {
  run<Bytes,Matrices,1>(grid,mode,input,trace,errors,sink,1,3);
  run<Bytes,Matrices,2>(grid,mode,input,trace,errors,sink,1,3);
 }
}
template<int Bytes> void matrices(int grid,unsigned* input,unsigned long long* trace,unsigned* errors,float* sink) {
 wgs<Bytes,4>(grid,input,trace,errors,sink);wgs<Bytes,8>(grid,input,trace,errors,sink);wgs<Bytes,16>(grid,input,trace,errors,sink);
}
int main(int argc,char** argv) {
 constexpr size_t words=132*Panels*32768/4;
 unsigned *input,*errors;unsigned long long* trace;float* sink;
 CHECK(cudaMalloc(&input,words*4));CHECK(cudaMalloc(&errors,132*4));CHECK(cudaMalloc(&trace,132*3*2*sizeof(*trace)));CHECK(cudaMalloc(&sink,132*3*sizeof(*sink)));
 std::vector<unsigned> values(words);for(size_t i=0;i<words;++i) values[i]=pattern(i);
 CHECK(cudaMemcpy(input,values.data(),words*4,cudaMemcpyHostToDevice));
 printf("bytes,matrices,matrix_wgs,grid,mode,trial,cta,wg,registers,cycles,global_ns\n");
 if(argc==2) run<32768,8,2>(132,atoi(argv[1]),input,trace,errors,sink,8,1);
 else for(int grid:{1,32,132}) {matrices<8192>(grid,input,trace,errors,sink);matrices<16384>(grid,input,trace,errors,sink);matrices<32768>(grid,input,trace,errors,sink);}
 CHECK(cudaFree(input));CHECK(cudaFree(errors));CHECK(cudaFree(trace));CHECK(cudaFree(sink));
}
