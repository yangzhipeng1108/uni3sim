// Independent special-register timestamp reads under arithmetic pressure.
// No Tensor instructions, Attention, reductions or global operand stream.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2); } } while(0)
struct Trace { unsigned long long start, end, background_start, background_end; unsigned wrong, reads; };
__device__ __forceinline__ unsigned long long nanos() {
    unsigned long long t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t) :: "memory"); return t;
}
template<bool Global, bool Exp, int Width>
__global__ void __launch_bounds__(256,1) timer_pressure(int background, int reads, Trace* output) {
    __shared__ volatile unsigned ready;
    __shared__ unsigned long long windows[4];
    __shared__ unsigned errors[256], counts[128];
    float values[Width];
    #pragma unroll
    for(int i=0;i<Width;++i) { values[i]=1.f+(i+threadIdx.x)*(1.f/1024);asm volatile("" : "+f"(values[i])); }
    if(threadIdx.x==0) ready=0;
    __syncthreads();
    unsigned wrong=0, count=0; unsigned long long checksum=0;
    if(threadIdx.x<128) {
        while(!ready) {}
        auto start=clock64();
        unsigned long long last=0;
        #pragma unroll 1
        for(int i=0;i<reads;) {
            unsigned long long t;
            if constexpr(Global) t=nanos(); else t=clock64();
            wrong+=t<last || (t>>63)!=0;
            last=t; checksum^=t; ++count;
            // Keep each next iteration dependent on the actual read.
            // Both physical clocks remain below 2^63 in this task.
            i+=1+int(t>>63);
        }
        auto end=clock64();
        if(threadIdx.x==0) { windows[0]=start;windows[1]=end; }
        counts[threadIdx.x]=count;
        // A live read witness, without stores until every peer has finished.
        wrong+=checksum==0 && reads==1;
    } else {
        auto start=clock64();
        if(background) {
            #pragma unroll 1
            for(int rep=0;rep<4104;++rep) {
                #pragma unroll
                for(int i=0;i<Width;++i) {
                    if constexpr(Exp) asm volatile("{.reg .f32 x;neg.f32 x,%0;ex2.approx.ftz.f32 %0,x;}" : "+f"(values[i]));
                    else asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(values[i]) : "f"(.5f),"f"(.25f));
                }
                if(rep==7 && threadIdx.x==128) ready=1;
            }
        } else if(threadIdx.x==128) ready=1;
        auto end=clock64();
        if(threadIdx.x==128) { windows[2]=start;windows[3]=end; }
        if(background) {
            #pragma unroll
            for(int i=0;i<Width;++i)
                wrong+=!std::isfinite(values[i]) || fabsf(values[i]-(Exp?.641185744504986f:.5f))>2e-6f;
        }
    }
    errors[threadIdx.x]=wrong;
    __syncthreads();
    if(threadIdx.x==0) {
        unsigned total=0,n=0;
        for(int i=0;i<256;++i) total+=errors[i];
        for(int i=0;i<128;++i) n+=counts[i];
        output[blockIdx.x]={windows[0],windows[1],windows[2],windows[3],total,n};
    }
}
template<bool Global,bool Exp,int Width> void run(int grid,int n,int background,int trial,Trace* output) {
    auto kernel=timer_pressure<Global,Exp,Width>;
    CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,204800));
    kernel<<<grid,256,204800>>>(background,n,output);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<Trace> trace(grid);CHECK(cudaMemcpy(trace.data(),output,grid*sizeof(Trace),cudaMemcpyDeviceToHost));
    for(int c=0;c<grid;++c) {
        auto t=trace[c];
        if(t.wrong || t.reads!=128u*n || t.start>=t.end || (background && !(t.background_start<=t.start && t.end<=t.background_end))) {
            std::fprintf(stderr,"timer numeric/coverage gate failed:wrong%u,reads%u\n",t.wrong,t.reads);std::exit(3);
        }
        std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%u,%u\n",
            int(Global),int(Exp),Width,grid,n,background,trial,c,t.start,t.end,t.background_start,t.background_end,t.wrong,t.reads);
    }
}
template<bool Global,bool Exp>void widths(int grid,int n,int bg,int trial,Trace* out) {
    run<Global,Exp,32>(grid,n,bg,trial,out);run<Global,Exp,128>(grid,n,bg,trial,out);
}
int main() {
    int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;
    Trace* out;CHECK(cudaMalloc(&out,132*sizeof(Trace)));
    std::puts("global_timer,exp,width,grid,reads,background,trial,cta,c0,c1,bg0,bg1,wrong,validated_reads");
    for(int grid:{1,132})for(int n:{64,128,256})for(int bg:{0,1})for(int trial=0;trial<3;++trial) {
        widths<false,false>(grid,n,bg,trial,out);widths<false,true>(grid,n,bg,trial,out);
        widths<true,false>(grid,n,bg,trial,out);widths<true,true>(grid,n,bg,trial,out);
    }
    CHECK(cudaFree(out));
}
