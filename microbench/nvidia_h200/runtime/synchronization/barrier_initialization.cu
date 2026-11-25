// Independent single fresh shared-mbarrier initialization/fence/CTA-sync.
// No repetition controller,payload,matrix,reduction,scheduler or Attention.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
constexpr int Shared=128*1024;
struct Reading{unsigned long long c0,c1,n0,n1;unsigned complete,errors;};
__device__ __forceinline__ unsigned long long ns(){unsigned long long x;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(x)::"memory");return x;}
__device__ __forceinline__ void init(unsigned p,unsigned n){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(p),"r"(n):"memory");}
__device__ __forceinline__ void arrive(unsigned p){asm volatile("{.reg .b64 s;mbarrier.arrive.shared::cta.b64 s,[%0];}"::"r"(p):"memory");}
__device__ __forceinline__ unsigned complete(unsigned p){unsigned v;asm volatile("{.reg .pred q;mbarrier.test_wait.parity.shared::cta.b64 q,[%1],0;selp.b32 %0,1,0,q;}":"=r"(v):"r"(p):"memory");return v;}
template<int Threads,int Mode,int Groups>
__global__ void __launch_bounds__(Threads,1) single_protocol(unsigned arrivals,Reading* output){
    extern __shared__ __align__(16) unsigned char shared[];
    constexpr int Count=4*Groups-2;
    unsigned base=unsigned(__cvta_generic_to_shared(shared));
    __syncthreads();
    unsigned long long c0=0,c1=0,n0=0,n1=0;
    if(threadIdx.x==0){n0=ns();c0=clock64();}
    if constexpr(Mode==1){
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        #pragma unroll
        for(int g=0;g<Groups;++g){
            if(threadIdx.x==0){
                #pragma unroll
                for(int b=0;b<(g==0?2:4);++b)init(base+unsigned((g==0?b:2+4*(g-1)+b)*8),arrivals);
            }
            asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
        }
    }
    if constexpr(Mode!=2)__syncthreads();
    if(threadIdx.x==0){c1=clock64();n1=ns();}
    __syncthreads();
    unsigned ready=0;
    if constexpr(Mode==1){
        for(int i=0;i<Count;++i){
            unsigned p=base+unsigned(i*8);
            if(threadIdx.x<arrivals)arrive(p);
            __syncthreads();
            if(threadIdx.x==0)ready+=complete(p);
        }
    }
    if(threadIdx.x==0)output[blockIdx.x]={c0,c1,n0,n1,ready,ready!=(Mode==1?Count:0)};
}
template<int T,int M,int G>void capture(int grid,int trial,unsigned arrivals){
    auto fn=single_protocol<T,M,G>;
    CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
    int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,T,Shared));
    if(resident!=1){fprintf(stderr,"Expected1residentCTA/SM\n");exit(3);}
    Reading* device;CHECK(cudaMalloc(&device,grid*sizeof(Reading)));
    fn<<<grid,T,Shared>>>(arrivals,device);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<Reading> host(grid);CHECK(cudaMemcpy(host.data(),device,grid*sizeof(Reading),cudaMemcpyDeviceToHost));CHECK(cudaFree(device));
    for(int c=0;c<grid;++c){auto r=host[c];if(r.errors||r.c1<=r.c0||r.n1<r.n0){fprintf(stderr,"Numeric/timer failure\n");exit(4);}
        printf("%d,%d,%d,%d,%d,%u,%d,%llu,%llu,%llu,%llu,%u\n",T,M,G,grid,trial,arrivals,c,r.c0,r.c1,r.n0,r.n1,r.complete);}
}
template<int T,int M,int G>void all(){capture<T,M,G>(132,-1,1);for(int grid:{1,132})for(unsigned a:{1u,4u,128u})for(int t=0;t<3;++t)capture<T,M,G>(grid,t,a);}
template<int T,int M>void groups(){all<T,M,1>();all<T,M,2>();all<T,M,4>();all<T,M,6>();}
template<int T>void modes(){groups<T,0>();groups<T,1>();groups<T,2>();}
int main(){cudaDeviceProp p;CHECK(cudaGetDeviceProperties(&p,0));if(p.major!=9||p.multiProcessorCount!=132)return 5;
 puts("threads,mode,groups,grid,trial,arrivals,cta,c0,c1,n0,n1,complete");modes<128>();modes<256>();modes<384>();modes<512>();return 0;}
