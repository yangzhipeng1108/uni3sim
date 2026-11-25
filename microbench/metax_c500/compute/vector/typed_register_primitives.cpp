// Independent typed register mechanisms; no model, routing, reduction or argmax.
// CPU reference: g++ -O2 -std=c++17 -DMIDLAYER_TYPED_CPU_ONLY this.cpp -o cpu
// MACA: mxcc -x maca --maca-path=/opt/maca -offload-arch=native -O3
//       -std=c++17 this.cpp -o typed_register_primitives
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef MIDLAYER_TYPED_CPU_ONLY
#include <mc_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif

namespace {
enum Kind { F32_GT, F32_EQ, F32_NAN, S64_LT, U64_GT, SHUFFLE32 };
constexpr int kUnroll = 31;
constexpr int kWave = 64;
const char* names[] = {"f32_gt", "f32_eq", "f32_nan", "s64_lt", "u64_gt", "shuffle32"};
HD float as_float(std::uint32_t v) { union { std::uint32_t u; float f; } x; x.u=v; return x.f; }
HD std::uint32_t as_bits(float v) { union { std::uint32_t u; float f; } x; x.f=v; return x.u; }
HD std::int64_t as_signed(std::uint64_t v) { union { std::uint64_t u; std::int64_t s; } x; x.u=v; return x.s; }
struct Bounds { float fl, fh, fp; std::uint64_t ul, uh, up; };
HD Bounds bounds(Kind kind, unsigned tid) {
    Bounds b;
    b.fl=-static_cast<float>(1+tid%7);
    b.fh=kind==F32_NAN ? as_float(0x7fc00001u+tid%7) : -b.fl;
    b.fp=kind==F32_EQ ? b.fh : 0.0f;
    b.ul=kind==S64_LT ? (0xfffffff000000000ull | (tid%31)) : (0x00000000ffffff00ull+tid%31);
    b.uh=kind==S64_LT ? (0x00000001fedcba00ull+tid%31) : (0xffffffff00000000ull+tid%31);
    b.up=kind==S64_LT ? 0xfffffffe80000000ull : 0x8000000080000000ull;
    // Half the lanes require low-word ordering at equal high words; the other
    // half require high-word ordering. Both outcomes remain period two.
    if(tid&2) { b.ul=b.up-1; b.uh=b.up+1; }
    return b;
}
HD std::uint64_t initial(Kind kind, unsigned tid, int chain) {
    if(kind==SHUFFLE32) return (tid<<3)|static_cast<unsigned>(chain);
    const Bounds b=bounds(kind,tid);
    const bool high=(tid+chain)&1;
    return kind<=F32_NAN ? as_bits(high ? b.fh : b.fl) : (high ? b.uh : b.ul);
}
std::uint64_t reference_step(Kind kind, std::uint64_t x, const Bounds& b) {
    const float f=as_float(static_cast<std::uint32_t>(x));
    switch(kind) {
        case F32_GT: return as_bits(f>b.fp ? b.fl : b.fh);
        case F32_EQ: return as_bits(f==b.fp ? b.fl : b.fh);
        case F32_NAN: return as_bits(std::isnan(f) ? b.fl : b.fh);
        case S64_LT: return as_signed(x)<as_signed(b.up) ? b.uh : b.ul;
        case U64_GT: return x>b.up ? b.ul : b.uh;
        case SHUFFLE32: throw std::runtime_error("shuffle reference requires a full wave");
    }
    throw std::runtime_error("unknown primitive");
}
std::uint64_t expected(Kind kind, unsigned tid, int chain, unsigned steps) {
    if(kind==SHUFFLE32) return initial(kind,tid^(steps&1),chain);
    const auto x=initial(kind,tid,chain);
    const Bounds b=bounds(kind,tid);
    if(reference_step(kind,reference_step(kind,x,b),b)!=x)
        throw std::runtime_error("synthetic stream is not period two");
    return steps&1 ? reference_step(kind,x,b) : x;
}
int self_test() {
    unsigned checks=0;
    for(int k=F32_GT;k<=U64_GT;++k)
        for(int c=0;c<4;++c)
            for(unsigned tid=0;tid<256;++tid)
                for(unsigned steps : {0u,1u,2u,31u,32u,129u}) {
                    const Kind kind=static_cast<Kind>(k);
                    auto x=initial(kind,tid,c);
                    for(unsigned s=0;s<steps;++s) x=reference_step(kind,x,bounds(kind,tid));
                    if(x!=expected(kind,tid,c,steps)) throw std::runtime_error("CPU period check failed");
                    ++checks;
                }
    for(int c=0;c<4;++c) for(unsigned steps : {0u,1u,2u,31u,32u,129u}) {
        std::uint64_t lanes[kWave];
        for(unsigned lane=0;lane<kWave;++lane) lanes[lane]=initial(SHUFFLE32,lane,c);
        for(unsigned step=0;step<steps;++step) {
            std::uint64_t next[kWave];
            for(unsigned lane=0;lane<kWave;++lane) next[lane]=lanes[lane^1];
            std::memcpy(lanes,next,sizeof(lanes));
        }
        for(unsigned lane=0;lane<kWave;++lane) {
            if(lanes[lane]!=expected(SHUFFLE32,lane,c,steps)) throw std::runtime_error("CPU shuffle check failed");
            ++checks;
        }
    }
    Bounds b=bounds(F32_GT,0);
    if(reference_step(F32_GT,0x7fc01234u,b)!=as_bits(b.fh)) throw std::runtime_error("ordered GT NaN check failed");
    ++checks;
    b=bounds(F32_EQ,0); b.fp=0.0f;
    if(reference_step(F32_EQ,0x80000000u,b)!=as_bits(b.fl)) throw std::runtime_error("signed zero EQ check failed");
    ++checks;
    b=bounds(F32_NAN,0);
    if(reference_step(F32_NAN,0x7fc01234u,b)!=as_bits(b.fl)) throw std::runtime_error("NaN check failed");
    ++checks;
    // Equal high words force low-word comparisons; differing high words also occur.
    b=bounds(S64_LT,0); b.up=0x1234567880000000ull;
    if(reference_step(S64_LT,0x123456787fffffffull,b)!=b.uh ||
       reference_step(S64_LT,0x1234567880000001ull,b)!=b.ul)
        throw std::runtime_error("signed two-word boundary check failed");
    checks+=2;
    b=bounds(U64_GT,0); b.up=0xf234567880000000ull;
    if(reference_step(U64_GT,0xf234567880000001ull,b)!=b.ul ||
       reference_step(U64_GT,0xf23456787fffffffull,b)!=b.uh)
        throw std::runtime_error("unsigned two-word boundary check failed");
    checks+=2;
    std::printf("{\"schema\":\"metax_typed_register_cpu_reference_v1\",\"checks\":%u,\"passed\":true,\"GPU_executed\":false}\n",checks);
    return 0;
}

#ifndef MIDLAYER_TYPED_CPU_ONLY
void check(mcError_t result) { if(result!=mcSuccess) throw std::runtime_error(mcGetErrorString(result)); }
struct Record { unsigned long long begin,end; };
// An empty read/write volatile register fence prevents C++ constant propagation
// through the period-two stream. It emits no intentional timed instruction.
// Final machine code must still be checked for compare/select preservation and spills.
__device__ __forceinline__ std::uint64_t fence64(std::uint64_t x) {
    std::uint32_t lo=static_cast<std::uint32_t>(x),hi=static_cast<std::uint32_t>(x>>32);
    asm volatile("" : "+r"(lo), "+r"(hi));
    return (static_cast<std::uint64_t>(hi)<<32)|lo;
}
template<int K> __device__ __forceinline__ void primitive(float& f,std::uint64_t& u,const Bounds& b) {
    if constexpr(K<=F32_NAN) {
        asm volatile("" : "+r"(f));
        if constexpr(K==F32_GT) f=f>b.fp ? b.fl : b.fh;
        else if constexpr(K==F32_EQ) f=f==b.fp ? b.fl : b.fh;
        else f=(f!=f) ? b.fl : b.fh;
        asm volatile("" : "+r"(f));
    } else if constexpr(K==SHUFFLE32) {
        std::uint32_t word=static_cast<std::uint32_t>(u);
        asm volatile("" : "+r"(word));
        word=__shfl_xor_sync(UINT64_MAX,word,1,kWave);
        asm volatile("" : "+r"(word));
        u=word;
    } else {
        u=fence64(u);
        if constexpr(K==S64_LT) u=as_signed(u)<as_signed(b.up) ? b.uh : b.ul;
        else u=u>b.up ? b.ul : b.uh;
        u=fence64(u);
    }
}
template<int K,int Chains> __global__ void probe(Record* records,std::uint64_t* values,int outer,
                                                  std::uint64_t wide_pivot) {
    const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
    Bounds b=bounds(static_cast<Kind>(K),tid);
    // Supplied at runtime: compiler may not reduce a full-width compare to sign.
    b.up=wide_pivot;
    float f[Chains]; std::uint64_t u[Chains];
    #pragma unroll
    for(int c=0;c<Chains;++c) { u[c]=initial(static_cast<Kind>(K),tid,c); f[c]=as_float(static_cast<unsigned>(u[c])); }
    __syncthreads();
    asm volatile("" ::: "memory");
    const unsigned long long begin=clock64();
    #pragma unroll 1
    for(int i=0;i<outer;++i) {
        #pragma unroll
        for(int j=0;j<kUnroll;++j) {
            #pragma unroll
            for(int c=0;c<Chains;++c) primitive<K>(f[c],u[c],b);
        }
    }
    asm volatile("" ::: "memory");
    const unsigned long long end=clock64();
    if(threadIdx.x%kWave==0) records[tid/kWave]={begin,end};
    #pragma unroll
    for(int c=0;c<Chains;++c) values[tid*Chains+c]=K<=F32_NAN ? as_bits(f[c]) : u[c];
}
template<int K> void launch(int chains,int threads,int grid,Record* records,std::uint64_t* values,int outer) {
    const auto pivot=bounds(static_cast<Kind>(K),0).up;
    if(chains==1) probe<K,1><<<grid,threads>>>(records,values,outer,pivot);
    else probe<K,4><<<grid,threads>>>(records,values,outer,pivot);
}
void run_gpu(int device,Kind kind,int chains,int threads,int grid,int outer,int samples) {
    check(mcSetDevice(device));
    mcDeviceProp_t p{}; check(mcGetDeviceProperties(&p,device));
    if(p.multiProcessorCount!=104 || p.warpSize!=kWave)
        throw std::runtime_error("requires C500 104 AP/wave64 device");
    const unsigned count=threads*grid;
    Record* records=nullptr; std::uint64_t* values=nullptr;
    check(mcMalloc(&records,count/kWave*sizeof(Record)));
    check(mcMalloc(&values,count*chains*sizeof(std::uint64_t)));
    std::vector<Record> host_records(count/kWave);
    std::vector<std::uint64_t> host_values(count*chains);
    auto invoke=[&] {
        switch(kind) {
            case F32_GT: launch<F32_GT>(chains,threads,grid,records,values,outer); break;
            case F32_EQ: launch<F32_EQ>(chains,threads,grid,records,values,outer); break;
            case F32_NAN: launch<F32_NAN>(chains,threads,grid,records,values,outer); break;
            case S64_LT: launch<S64_LT>(chains,threads,grid,records,values,outer); break;
            case U64_GT: launch<U64_GT>(chains,threads,grid,records,values,outer); break;
            case SHUFFLE32: launch<SHUFFLE32>(chains,threads,grid,records,values,outer); break;
        }
        check(mcGetLastError()); check(mcDeviceSynchronize());
    };
    invoke();
    std::printf("hardware_id,device_name,device_id,ap_count,wave_size,clock_khz,sample,primitive,chains,threads,grid,outer,wave_index,block_index,wave_in_block,begin_cycle,end_cycle,wave_bundles,checksum,validation_status\n");
    for(int s=0;s<samples;++s) {
        invoke();
        check(mcMemcpy(host_records.data(),records,host_records.size()*sizeof(Record),mcMemcpyDeviceToHost));
        check(mcMemcpy(host_values.data(),values,host_values.size()*sizeof(std::uint64_t),mcMemcpyDeviceToHost));
        for(unsigned tid=0;tid<count;++tid)
            for(int c=0;c<chains;++c)
                if(host_values[tid*chains+c]!=expected(kind,tid,c,static_cast<unsigned>(outer*kUnroll)))
                    throw std::runtime_error("GPU result differs from CPU reference");
        for(unsigned w=0;w<host_records.size();++w) {
            const auto r=host_records[w];
            if(r.end<=r.begin) throw std::runtime_error("invalid same-wave cycle interval");
            unsigned long long checksum=0;
            for(unsigned tid=w*kWave;tid<(w+1)*kWave;++tid)
                for(int c=0;c<chains;++c) checksum+=host_values[tid*chains+c];
            std::printf("metax_c500,\"%s\",%d,%d,%d,%d,%d,%s,%d,%d,%d,%d,%u,%u,%u,%llu,%llu,%llu,%llu,pass\n",
                p.name,device,p.multiProcessorCount,p.warpSize,p.clockRate,s,names[kind],chains,threads,grid,outer,w,
                w/(threads/kWave),w%(threads/kWave),r.begin,r.end,
                static_cast<unsigned long long>(outer)*kUnroll*chains,checksum);
        }
    }
    check(mcFree(values)); check(mcFree(records));
}
#endif
int integer(const char* text) {
    std::size_t end; const int value=std::stoi(text,&end);
    if(end!=std::strlen(text)) throw std::runtime_error("nonintegral argument");
    return value;
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--self-test") return self_test();
#ifndef MIDLAYER_TYPED_CPU_ONLY
        if(argc!=10 || std::string(argv[1])!="--device-id" || std::string(argv[3])!="--sample")
            throw std::runtime_error("usage: binary --device-id ID --sample primitive chains threads grid outer samples");
        const int device=integer(argv[2]); Kind kind=F32_GT; bool found=false;
        for(int k=F32_GT;k<=SHUFFLE32;++k) if(argv[4]==std::string(names[k])) {kind=static_cast<Kind>(k);found=true;}
        if(!found) throw std::runtime_error("unknown primitive");
        const int chains=integer(argv[5]),threads=integer(argv[6]),grid=integer(argv[7]),outer=integer(argv[8]),samples=integer(argv[9]);
        if(device<0 || (chains!=1 && chains!=4) || (threads!=64 && threads!=256) || grid<1 || grid>416 || outer<1 || outer>2047 || samples<1 || samples>10)
            throw std::runtime_error("argument outside declared mechanism domain");
        run_gpu(device,kind,chains,threads,grid,outer,samples); return 0;
#else
        throw std::runtime_error("CPU-only executable accepts --self-test only");
#endif
    } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
