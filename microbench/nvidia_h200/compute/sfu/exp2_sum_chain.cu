// Independent bounded FFMA -> EX2 -> two-row local add chains.
// No row max, normalization, conversion, output rescaling, matrix, or copy.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){ \
 fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int Repeats=256;
template<int Values,int Mode>
__global__ void chain(unsigned long long* cycles,float* witness) {
  float seed=-.125f*(threadIdx.x%8+1), sum0=0,sum1=0;
  float p[2][Values];
  __syncthreads();
  auto start=clock64();
  for(int rep=0;rep<Repeats;++rep) {
    if constexpr(Mode==0) {
      sum0=seed;sum1=seed;
      asm volatile("" : "+f"(sum0),"+f"(sum1));
    } else {
      #pragma unroll
      for(int row=0;row<2;++row) {
        #pragma unroll
        for(int col=0;col<Values;++col) {
          float shifted;
          float bias=-.001f*(col+row*Values);
          asm volatile("fma.rn.ftz.f32 %0,%1,%2,%3;" :"=f"(shifted):
                       "f"(seed),"f"(.5f),"f"(bias));
          if constexpr(Mode==1)
            asm volatile("ex2.approx.ftz.f32 %0,%1;" :"=f"(p[row][col]):"f"(shifted));
          else p[row][col]=shifted;
        }
      }
      sum0=sum1=0;
      #pragma unroll
      for(int col=0;col<Values;++col) {
        asm volatile("add.rn.ftz.f32 %0,%0,%1;" :"+f"(sum0):"f"(p[0][col]));
        asm volatile("add.rn.ftz.f32 %0,%0,%1;" :"+f"(sum1):"f"(p[1][col]));
      }
    }
    // The next primitive consumes the preceding sums, but its input remains
    // in [-1,-.5): unlike the rejected old probe it can never overflow EX2.
    unsigned bits=(__float_as_uint(sum0+sum1)&0x007fffffu)|0x3f000000u;
    seed=-__uint_as_float(bits);
  }
  cycles[blockIdx.x*blockDim.x+threadIdx.x]=clock64()-start;
  witness[blockIdx.x*blockDim.x+threadIdx.x]=sum0+sum1;
}
template<int V,int Mode> void run(int grid,int groups,unsigned long long* cycles,float* witness) {
  CHECK(cudaFuncSetAttribute(chain<V,Mode>,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024));
  int threads=groups*128;
  for(int i=0;i<8;++i) chain<V,Mode><<<grid,threads,128*1024>>>(cycles,witness);
  CHECK(cudaDeviceSynchronize());
  for(int trial=0;trial<3;++trial) {
    chain<V,Mode><<<grid,threads,128*1024>>>(cycles,witness);CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> host(grid*threads);std::vector<float> values(grid*threads);
    CHECK(cudaMemcpy(host.data(),cycles,host.size()*sizeof(*cycles),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(values.data(),witness,values.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(float v:values) if(!std::isfinite(v) || (Mode==1 && (v<=0 || v>2*V))) {
      fprintf(stderr,"nonfinite/out-of-range primitive witness\n");exit(3);
    }
    for(int c=0;c<grid;++c) for(int g=0;g<groups;++g)
      printf("%d,%d,%d,%d,%d,%d,%llu,%.8g\n",V,Mode,grid,groups,trial,c,
        *std::max_element(host.begin()+c*threads+g*128,
                          host.begin()+c*threads+(g+1)*128),values[c*threads+g*128]);
  }
}
template<int V> void width(unsigned long long* cycles,float* witness) {
  for(int grid:{1,132}) for(int groups:{1,2}) {
    run<V,0>(grid,groups,cycles,witness);run<V,1>(grid,groups,cycles,witness);
    run<V,2>(grid,groups,cycles,witness);
  }
}
int main() {
  unsigned long long* cycles;float* witness;
  CHECK(cudaMalloc(&cycles,132*256*sizeof(*cycles)));
  CHECK(cudaMalloc(&witness,132*256*sizeof(*witness)));
  printf("values,mode,grid,groups,trial,cta,cycles,witness\n");
  width<20>(cycles,witness);width<32>(cycles,witness);width<40>(cycles,witness);
  width<44>(cycles,witness);width<48>(cycles,witness);
  CHECK(cudaFree(cycles));CHECK(cudaFree(witness));
}
