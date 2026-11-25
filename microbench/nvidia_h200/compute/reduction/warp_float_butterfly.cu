// Independent single-row floating warp-butterfly dependency. No operator input.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#define CHECK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
constexpr int Threads=256, Shared=128*1024;
struct Reading {unsigned long long begin,end;unsigned errors;};
template<int Mode,int Levels> __device__ __forceinline__ void tree(float& value) {
    if constexpr(Levels>1) tree<Mode,Levels-1>(value);
    float other;
    asm volatile("shfl.sync.bfly.b32 %0,%1,%2,31,-1;":"=f"(other):"f"(value),"n"(1<<(Levels-1)));
    if constexpr(Mode==0) asm volatile("max.f32 %0,%0,%1;":"+f"(value):"f"(other));
    else asm volatile("add.rn.f32 %0,%0,%1;":"+f"(value):"f"(other));
}
template<int Mode,int Levels,int Warps>
__global__ void __launch_bounds__(Threads,1) probe(int repeats,int warps,Reading* out) {
    extern __shared__ unsigned char reserved[];
    __shared__ unsigned errors[Threads];
    int warp=threadIdx.x/32,lane=threadIdx.x%32;
    if(threadIdx.x==0) reserved[0]=unsigned(blockIdx.x);
    __syncthreads();
    float value=float(lane+1)/32+float(warp)/8;
    asm volatile("":"+f"(value));
    unsigned long long begin=0,end=0;
    if(Warps==8||warp<Warps) {
        if(lane==0) begin=clock64();
        #pragma unroll 1
        for(int r=0;r<repeats;++r) {
            tree<Mode,Levels>(value);
            // A power-of-two normalization keeps the SUM recurrence finite.
            // Its fixed cost is an intercept,never part of the butterfly price.
            constexpr float norm=Mode==0?1.f:1.f/float(1<<Levels);
            asm volatile("mul.rn.f32 %0,%0,%1;":"+f"(value):"f"(norm));
        }
        if(lane==0) end=clock64();
    }
    constexpr int lanes=1<<Levels;
    int base=(lane/lanes)*lanes;
    float expected=Mode==0?float(base+lanes)/32+float(warp)/8
                         :float(2*base+lanes+1)/64+float(warp)/8;
    errors[threadIdx.x]=warp<warps&&(!std::isfinite(value)||value!=expected);
    __syncthreads();
    if(lane==0&&warp<warps) {
        unsigned count=0;for(int l=0;l<32;++l)count+=errors[warp*32+l];
        out[blockIdx.x*8+warp]={begin,end,count};
    }
}
template<int Mode,int Levels,int Warps> void capture(int grid,int repeats,int trial) {
    constexpr int warps=Warps;
    Reading* device;CHECK(cudaMalloc(&device,grid*8*sizeof(Reading)));
    CHECK(cudaFuncSetAttribute(probe<Mode,Levels,Warps>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
    int resident=0;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,probe<Mode,Levels,Warps>,Threads,Shared));
    if(resident!=1){fprintf(stderr,"Expected1CTA/SM,got%d\n",resident);exit(3);}
    CHECK(cudaMemset(device,0,grid*8*sizeof(Reading)));
    probe<Mode,Levels,Warps><<<grid,Threads,Shared>>>(repeats,warps,device);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<Reading> host(grid*8);CHECK(cudaMemcpy(host.data(),device,host.size()*sizeof(Reading),cudaMemcpyDeviceToHost));
    CHECK(cudaFree(device));
    for(int c=0;c<grid;++c) for(int w=0;w<warps;++w) {
        auto r=host[c*8+w];if(r.errors||r.end<=r.begin){fprintf(stderr,"Numeric/timer failure\n");exit(4);}
        printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n",Mode,Levels,grid,warps,repeats,trial,c,w,r.begin,r.end);
    }
}
template<int M,int L,int W> void counts() {
    for(int g:{1,132})for(int n:{64,128,256})for(int t=0;t<3;++t)capture<M,L,W>(g,n,t);
}
template<int M,int L> void all() {
    capture<M,L,8>(132,64,-1);
    counts<M,L,1>();counts<M,L,4>();counts<M,L,8>();
}
int main(){cudaDeviceProp p;CHECK(cudaGetDeviceProperties(&p,0));if(p.major!=9||p.multiProcessorCount!=132)return 5;
    puts("mode,levels,grid,warps,repeats,trial,cta,warp,begin,end");
    all<0,1>();all<0,2>();all<0,3>();all<0,4>();all<0,5>();
    all<1,1>();all<1,2>();all<1,3>();all<1,4>();all<1,5>();return 0;
}
