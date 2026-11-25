// Independent finite 2-read/1-write memory bursts. No activation, GEMM or routing.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(1);} }while(0)
__device__ uint32_t hash_value(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
__global__ void initialize(uint4* a,uint4* b,size_t n) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        uint32_t s=static_cast<uint32_t>(4*i);
        a[i]=make_uint4(hash_value(s),hash_value(s+1),hash_value(s+2),hash_value(s+3));
        b[i]=make_uint4(hash_value(s+17),hash_value(s+18),hash_value(s+19),hash_value(s+20));
    }
}
template<int Threads,int Bursts>
__global__ void memory_burst(const uint4* a,const uint4* b,uint4* out) {
    size_t index=static_cast<size_t>(blockIdx.x)*Threads*Bursts+threadIdx.x;
#pragma unroll
    for(int j=0;j<Bursts;++j) {
        uint4 x,y;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w) : "l"(a+index));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(y.x),"=r"(y.y),"=r"(y.z),"=r"(y.w) : "l"(b+index));
        x.x^=y.x;x.y^=y.y;x.z^=y.z;x.w^=y.w;
        asm volatile("st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out+index),"r"(x.x),"r"(x.y),"r"(x.z),"r"(x.w));
        index+=Threads;
    }
}
using Launcher=void(*)(size_t,const uint4*,const uint4*,uint4*);
template<int T,int B> void launch(size_t n,const uint4*a,const uint4*b,uint4*c) {memory_burst<T,B><<<n/(T*B),T>>>(a,b,c);}
struct Point{int threads,bursts;Launcher launch;};
const Point points[]={
 {64,1,launch<64,1>},{96,1,launch<96,1>},{128,1,launch<128,1>},{256,1,launch<256,1>},{512,1,launch<512,1>},{1024,1,launch<1024,1>},
 {64,8,launch<64,8>},{96,8,launch<96,8>},{128,8,launch<128,8>},{256,8,launch<256,8>},{512,8,launch<512,8>},{1024,8,launch<1024,8>},
 {96,2,launch<96,2>},{96,4,launch<96,4>},{128,2,launch<128,2>},{128,4,launch<128,4>},{1024,2,launch<1024,2>},{1024,4,launch<1024,4>}};
void validate(const Point&p,size_t n,const uint4*a,const uint4*b,uint4*c) {
    p.launch(n,a,b,c);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    for(size_t index: {size_t(0),n/2,n-1}) {
        uint4 x,y,z;CHECK(cudaMemcpy(&x,a+index,16,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&y,b+index,16,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&z,c+index,16,cudaMemcpyDeviceToHost));
        if(z.x!=(x.x^y.x)||z.y!=(x.y^y.y)||z.z!=(x.z^y.z)||z.w!=(x.w^y.w)){std::fprintf(stderr,"validation failed\n");std::exit(1);}
    }
}
void measure(const Point&p,size_t requested,const uint4*a,const uint4*b,uint4*c,bool single) {
    size_t n=requested/(p.threads*p.bursts)*(p.threads*p.bursts);
    if(single){p.launch(n,a,b,c);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());return;}
    validate(p,n,a,b,c);
    for(int j=0;j<8;++j)p.launch(n,a,b,c);CHECK(cudaDeviceSynchronize());
    cudaEvent_t first,last;CHECK(cudaEventCreate(&first));CHECK(cudaEventCreate(&last));std::vector<float> samples;
    for(int sample=0;sample<7;++sample){
        CHECK(cudaEventRecord(first));for(int repeat=0;repeat<4;++repeat)p.launch(n,a,b,c);CHECK(cudaEventRecord(last));CHECK(cudaEventSynchronize(last));
        float elapsed;CHECK(cudaEventElapsedTime(&elapsed,first,last));samples.push_back(elapsed/4);
    }
    auto sorted=samples;std::sort(sorted.begin(),sorted.end());
    std::printf("{\"threads\":%d,\"bursts\":%d,\"vectors\":%zu,\"grid\":%zu,\"read_bytes\":%zu,\"write_bytes\":%zu,\"median_ms\":%.9g,\"samples_ms\":[",p.threads,p.bursts,n,n/(p.threads*p.bursts),n*32,n*16,sorted[3]);
    for(size_t j=0;j<samples.size();++j)std::printf("%s%.9g",j?",":"",samples[j]);std::puts("]}");
    CHECK(cudaEventDestroy(first));CHECK(cudaEventDestroy(last));
}
int main(int argc,char**argv){
    cudaDeviceProp prop;CHECK(cudaGetDeviceProperties(&prop,0));if(prop.major!=9||prop.minor!=0||prop.multiProcessorCount!=132){std::fprintf(stderr,"requires H200 132 SM\n");return 2;}
    std::printf("{\"device\":\"%s\",\"sms\":%d,\"l2_bytes\":%d,\"mechanism\":\"two_vector16_reads_one_vector16_write_xor_no_math\"}\n",prop.name,prop.multiProcessorCount,prop.l2CacheSize);
    size_t maximum=size_t(1)<<25;uint4*a,*b,*c;CHECK(cudaMalloc(&a,maximum*16));CHECK(cudaMalloc(&b,maximum*16));CHECK(cudaMalloc(&c,maximum*16));initialize<<<1056,256>>>(a,b,maximum);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    bool single=argc==3;
    if(single){int threads=std::atoi(argv[1]),bursts=std::atoi(argv[2]);bool found=false;for(const auto&p:points)if(p.threads==threads&&p.bursts==bursts){measure(p,maximum,a,b,c,true);found=true;}if(!found)return 3;}
    else if(argc==1){for(size_t n:{size_t(1)<<22,maximum})for(const auto&p:points)measure(p,n,a,b,c,false);}
    else return 3;
    CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(c));return 0;
}
