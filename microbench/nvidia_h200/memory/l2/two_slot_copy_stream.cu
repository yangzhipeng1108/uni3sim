// Independent warm two-slot K/V-sized copy stream. No Attention/math work.
// One 128-thread producer, two synthetic consumer WGs, one CTA/SM.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x);if(e!=cudaSuccess) { \
  fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2); } } while(0)
constexpr int Rows=128, Dim=128, Heads=8, Panels=4, Repeats=128;
constexpr int OperandBytes=Rows*Dim*2, StageBytes=2*OperandBytes;

__device__ void wait_bar(unsigned address,int phase) {
  asm volatile("{.reg .pred p; WAIT_%=: "
    "mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra WAIT_%=;}" ::
    "r"(address),"r"(phase):"memory");
}
__global__ void copy_stream(const unsigned short* key,const unsigned short* value,
                            unsigned long long* ticks,int hold) {
  extern __shared__ __align__(128) unsigned char shared[];
  __shared__ __align__(8) unsigned long long bars[4];
  unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(shared));
  unsigned bar=static_cast<unsigned>(__cvta_generic_to_shared(bars));
  if(threadIdx.x==0) {
    for(int s=0;s<2;++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],128;" ::"r"(bar+8*s));
      asm volatile("mbarrier.init.shared::cta.b64 [%0],8;" ::"r"(bar+8*(s+2)));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");
  }
  __syncthreads();
  if(threadIdx.x<128) {
    int row_group=threadIdx.x/8, chunk=threadIdx.x%8;
    for(int b=0;b<Repeats;++b) {
      int s=b%2, phase=b/2;
      if(b>=2) wait_bar(bar+8*(s+2),(phase-1)&1);
      // Eight adjacent heads share an interleaved NHD source allocation.
      // Panels repeat inside a 34-MiB chip footprint, below 60-MiB L2.
      for(int pass=0;pass<Rows/16;++pass) {
        int row=pass*16+row_group;
        long long offset=((((blockIdx.x/Heads)*Panels+b%Panels)*Rows+row)*Heads
                          +blockIdx.x%Heads)*Dim+chunk*8;
        unsigned dst=base+s*StageBytes+(row*Dim+chunk*8)*2;
        #pragma unroll
        for(int column=0;column<Dim/64;++column)
          for(int operand=0;operand<2;++operand) {
            const unsigned short* source=(operand?value:key)+offset+column*64;
            unsigned target=dst+operand*OperandBytes+column*128;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::
                         "r"(target),"l"(source):"memory");
          }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::
                   "r"(bar+8*s):"memory");
    }
    asm volatile("cp.async.wait_all;" :::"memory");
  } else {
    for(int b=0;b<Repeats;++b) {
      int s=b%2;
      wait_bar(bar+8*s,(b/2)&1);
      unsigned long long now=clock64();
      if(threadIdx.x==128) ticks[blockIdx.x*Repeats+b]=now;
      if(hold) while(clock64()-now<static_cast<unsigned>(hold)) {}
      __syncwarp();
      if(threadIdx.x%32==0)
        asm volatile("{.reg .b64 state;mbarrier.arrive.shared::cta.b64 state,[%0];}" ::
                     "r"(bar+8*(s+2)):"memory");
    }
  }
}
int main() {
  constexpr size_t elements=((132+Heads-1)/Heads)*Panels*Rows*Heads*Dim;
  unsigned short *key,*value;unsigned long long* ticks;
  CHECK(cudaMalloc(&key,elements*2));CHECK(cudaMalloc(&value,elements*2));
  CHECK(cudaMalloc(&ticks,132*Repeats*sizeof(*ticks)));
  CHECK(cudaMemset(key,1,elements*2));CHECK(cudaMemset(value,2,elements*2));
  CHECK(cudaFuncSetAttribute(copy_stream,cudaFuncAttributeMaxDynamicSharedMemorySize,2*StageBytes+1024));
  printf("grid,hold,trial,cta,mean_ii_cycles,stage_bytes\n");
  for(int grid:{1,32,132}) for(int hold:{0,1024,4096}) {
    for(int warm=0;warm<8;++warm) copy_stream<<<grid,384,2*StageBytes+1024>>>(key,value,ticks,hold);
    CHECK(cudaDeviceSynchronize());
    for(int trial=0;trial<3;++trial) {
      copy_stream<<<grid,384,2*StageBytes+1024>>>(key,value,ticks,hold);
      CHECK(cudaDeviceSynchronize());
      std::vector<unsigned long long> host(grid*Repeats);
      CHECK(cudaMemcpy(host.data(),ticks,host.size()*sizeof(*ticks),cudaMemcpyDeviceToHost));
      for(int c=0;c<grid;++c) {
        double ii=double(host[c*Repeats+Repeats-1]-host[c*Repeats+7])/(Repeats-8);
        printf("%d,%d,%d,%d,%.6f,%d\n",grid,hold,trial,c,ii,StageBytes);
      }
    }
  }
  CHECK(cudaFree(key));CHECK(cudaFree(value));CHECK(cudaFree(ticks));
}
