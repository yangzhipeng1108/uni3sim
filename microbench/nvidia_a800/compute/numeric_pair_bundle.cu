// Independent finite numeric comparator + selected pair register mechanism.
// No input loads, reduction, shuffle, routing, argmax or model tensors in timer.
// The pair-max recurrence is idempotent after one step, explicitly validated.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef MIDLAYER_NUMERIC_CPU_ONLY
#define HD
#else
#include <cuda_runtime.h>
#define HD __host__ __device__
#endif
constexpr int kUnroll = 31;
HD float as_float(std::uint32_t v) { union { unsigned u; float f; } x; x.u=v; return x.f; }
HD std::uint32_t as_bits(float v) { union { unsigned u; float f; } x; x.f=v; return x.u; }
std::int64_t as_signed(std::uint64_t x) { std::int64_t s; std::memcpy(&s,&x,sizeof(x)); return s; }
struct Pair { float value; std::uint64_t index; };
struct Inputs { Pair a,b; };
HD Inputs inputs(unsigned tid,int chain) {
    Inputs x;
    switch((tid+chain)%8) {
        case 0: x.a.value=2.0f; x.b.value=1.0f; break;
        case 1: x.a.value=-2.0f; x.b.value=1.0f; break;
        case 2: x.a.value=1.0f; x.b.value=1.0f; break;
        case 3: x.a.value=-1.0f; x.b.value=-1.0f; break;
        case 4: x.a.value=0.0f; x.b.value=as_float(0x80000000u); break;
        case 5: x.a.value=as_float(0x80000000u); x.b.value=0.0f; break;
        case 6: x.a.value=-11.0f; x.b.value=11.0f; break;
        default: x.a.value=3.0f; x.b.value=3.0f; break;
    }
    switch((tid/8+chain)%8) {
        case 0: x.a.index=0x1234567880000001ull; x.b.index=0x1234567880000000ull; break;
        case 1: x.a.index=0x123456787fffffffull; x.b.index=0x1234567880000000ull; break;
        case 2: x.a.index=0x1234567900000000ull; x.b.index=0x12345678ffffffffull; break;
        case 3: x.a.index=0xffffffff80000000ull; x.b.index=0x0000000080000000ull; break;
        case 4: x.a.index=0x8000000000000001ull; x.b.index=0x7fffffffffffffffull; break;
        case 5: x.a.index=0x8000000100000000ull; x.b.index=0x80000000ffffffffull; break;
        case 6: x.a.index=17; x.b.index=19; break;
        default: x.a.index=0x00000000ffffffffull; x.b.index=0x0000000080000000ull; break;
    }
    return x;
}
Pair reference(Pair a,Pair b) {
    if(std::isnan(a.value)) return (!std::isnan(b.value)||as_signed(a.index)<as_signed(b.index)) ? a : b;
    return (a.value==b.value ? as_signed(a.index)<as_signed(b.index) : a.value>b.value) ? a : b;
}
bool same(Pair a,Pair b) {return as_bits(a.value)==as_bits(b.value)&&a.index==b.index;}
int self_test() {
    unsigned checks=0;
    for(unsigned tid=0;tid<256;++tid) for(int c=0;c<4;++c) {
        const Inputs x=inputs(tid,c);
        for(unsigned steps : {0u,1u,2u,31u,62u,129u}) {
            Pair y=x.a;
            for(unsigned i=0;i<steps;++i) y=reference(y,x.b);
            const Pair expected=steps==0 ? x.a : reference(x.a,x.b);
            if(!same(y,expected)) throw std::runtime_error("CPU idempotent reference failed");
            ++checks;
        }
    }
    // Same-high-word low32 unsigned ordering and signed high32 ordering.
    if(reference({1.0f,0x123456787fffffffull},{1.0f,0x1234567880000000ull}).index!=0x123456787fffffffull)
        throw std::runtime_error("low32 tie order failed");
    if(reference({1.0f,0x8000000000000001ull},{1.0f,0x7fffffffffffffffull}).index!=0x8000000000000001ull)
        throw std::runtime_error("signed high32 tie order failed");
    checks+=2;
    const float nan1=as_float(0x7fc00001u),nan2=as_float(0x7fc00002u);
    if(!same(reference({nan1,3},{1.0f,7}),{nan1,3}) ||
       !same(reference({1.0f,3},{nan2,7}),{nan2,7}) ||
       !same(reference({nan1,3},{nan2,7}),{nan1,3}) ||
       !same(reference({nan1,7},{nan2,3}),{nan2,3}))
        throw std::runtime_error("source NaN pair reference failed");
    checks+=4;
    std::printf("{\"schema\":\"sm80_numeric_pair_cpu_reference_v1\",\"checks\":%u,\"passed\":true,\"GPU_executed\":false,\"idempotent_after_one\":true}\n",checks);
    return 0;
}
#ifndef MIDLAYER_NUMERIC_CPU_ONLY
struct Record { unsigned long long begin,end; unsigned smid,smid_space; };
__device__ __forceinline__ void numeric_pair(Pair& a,const Pair& b) {
    unsigned decision;
    // Exact source numeric bool materialization, deliberately not reduced to
    // a new branchless predicate formula or a minimal mathematical compare.
    asm volatile("{.reg .pred pne,pge,pgt;\n"
        "setp.neu.f32 pne,%1,%2;\n"
        "setp.ge.s64 pge,%3,%4;\n"
        "setp.gt.f32 pgt,%1,%2;\n"
        "selp.b32 %0,0,0xffffffff,!pgt;\n"
        "@!pne selp.b32 %0,0,0xffffffff,pge;\n"
        "and.b32 %0,%0,1;\n}"
        : "=r"(decision) : "f"(a.value),"f"(b.value),"l"(a.index),"l"(b.index));
    asm volatile("{.reg .pred choose;\n"
        "setp.eq.u32 choose,%2,1;\n"
        "selp.f32 %0,%3,%4,choose;\n"
        "selp.b64 %1,%5,%6,choose;\n}"
        : "=f"(a.value),"=l"(a.index)
        : "r"(decision),"f"(a.value),"f"(b.value),"l"(a.index),"l"(b.index));
}
__device__ __forceinline__ void classified_pair(Pair& a,const Pair& b) {
    // Full source comparator only: preserve classification and actual branches.
    // Its numeric input domain is a synthetic capture condition, not an
    // assertion about production logits or user acceptance.
    asm volatile("{.reg .pred pna,pnb,pne,pge,pgt,choose; .reg .f32 absval; .reg .b32 decision;\n"
        "abs.f32 absval,%2;\n"
        "setp.gtu.f32 pna,absval,0f7f800000;\n"
        "@pna bra NAN_PATH;\n"
        "setp.neu.f32 pne,%2,%3;\n"
        "setp.ge.s64 pge,%4,%5;\n"
        "setp.gt.f32 pgt,%2,%3;\n"
        "selp.b32 decision,0,0xffffffff,!pgt;\n"
        "@!pne selp.b32 decision,0,0xffffffff,pge;\n"
        "and.b32 decision,decision,1;\n"
        "setp.eq.u32 choose,decision,1;\n"
        "bra JOIN_PAIR;\n"
        "NAN_PATH: abs.f32 absval,%3;\n"
        "setp.gtu.f32 pnb,absval,0f7f800000;\n"
        "setp.lt.or.s64 choose,%4,%5,!pnb;\n"
        "JOIN_PAIR: selp.f32 %0,%2,%3,choose;\n"
        "selp.b64 %1,%4,%5,choose;\n}"
        : "=f"(a.value),"=l"(a.index)
        : "f"(a.value),"f"(b.value),"l"(a.index),"l"(b.index));
}
template<int Scope,int Chains> __global__ void probe_pair(Record* records,std::uint64_t* values,int outer) {
    const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
    Pair a[Chains],b[Chains];
    #pragma unroll
    for(int c=0;c<Chains;++c) { const Inputs x=inputs(tid,c); a[c]=x.a;b[c]=x.b; }
    __syncthreads();
    unsigned long long begin,end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(begin) :: "memory");
    #pragma unroll 1
    for(int i=0;i<outer;++i) {
        #pragma unroll
        for(int j=0;j<kUnroll;++j) {
            #pragma unroll
            for(int c=0;c<Chains;++c) {
                if constexpr(Scope==0) numeric_pair(a[c],b[c]);
                else classified_pair(a[c],b[c]);
            }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end) :: "memory");
    if(threadIdx.x%32==0) {
        unsigned smid,space;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        asm volatile("mov.u32 %0, %%nsmid;" : "=r"(space));
        records[tid/32]={begin,end,smid,space};
    }
    #pragma unroll
    for(int c=0;c<Chains;++c) {
        values[(tid*Chains+c)*2]=as_bits(a[c].value);
        values[(tid*Chains+c)*2+1]=a[c].index;
    }
}
void check(cudaError_t error) { if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
void run(int scope,int chains,int threads,int grid,int outer,int samples) {
    cudaDeviceProp p;check(cudaGetDeviceProperties(&p,0));
    if(p.major!=8||p.minor!=0||p.multiProcessorCount!=108||!std::strstr(p.name,"A800"))
        throw std::runtime_error("requires owned A800 SM80/108SM");
    char bus[32];check(cudaDeviceGetPCIBusId(bus,sizeof(bus),0));
    std::fprintf(stderr,"a800_cuda_device_identity,%s,%s,%d,%d,%d\n",p.name,bus,p.major,p.minor,p.multiProcessorCount);
    const unsigned count=threads*grid;
    Record* records=nullptr; std::uint64_t* values=nullptr;
    check(cudaMalloc(&records,count/32*sizeof(Record)));
    check(cudaMalloc(&values,count*chains*2*sizeof(std::uint64_t)));
    std::vector<Record> hr(count/32);std::vector<std::uint64_t> hv(count*chains*2);
    auto invoke=[&] {
        if(scope==0) {
            if(chains==1) probe_pair<0,1><<<grid,threads>>>(records,values,outer);
            else probe_pair<0,4><<<grid,threads>>>(records,values,outer);
        } else {
            if(chains==1) probe_pair<1,1><<<grid,threads>>>(records,values,outer);
            else probe_pair<1,4><<<grid,threads>>>(records,values,outer);
        }
        check(cudaGetLastError());check(cudaDeviceSynchronize());
    };
    invoke();
    std::printf("sample,primitive,chains,threads,grid,outer,warp_index,smid,smid_space,begin_cycle,end_cycle,warp_bundles,checksum,validation_status\n");
    for(int sample=0;sample<samples;++sample) {
        invoke();
        check(cudaMemcpy(hr.data(),records,hr.size()*sizeof(Record),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hv.data(),values,hv.size()*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
        for(unsigned tid=0;tid<count;++tid) for(int c=0;c<chains;++c) {
            const auto x=inputs(tid,c); const auto expected=reference(x.a,x.b);
            if(hv[(tid*chains+c)*2]!=as_bits(expected.value)||hv[(tid*chains+c)*2+1]!=expected.index)
                throw std::runtime_error("GPU differs from finite full64 CPU reference");
        }
        for(unsigned w=0;w<hr.size();++w) {
            const auto r=hr[w];
            if(r.end<=r.begin||r.smid_space<108||r.smid>=r.smid_space) throw std::runtime_error("invalid same-SM interval");
            unsigned long long checksum=0;
            for(unsigned tid=w*32;tid<(w+1)*32;++tid) for(int c=0;c<chains;++c)
                checksum+=hv[(tid*chains+c)*2]+hv[(tid*chains+c)*2+1];
            std::printf("%d,%s,%d,%d,%d,%d,%u,%u,%u,%llu,%llu,%llu,%llu,pass\n",sample,scope==0 ? "argmax_numeric_pair" : "argmax_classified_pair_finite_inputs",chains,threads,grid,outer,w,r.smid,r.smid_space,r.begin,r.end,static_cast<unsigned long long>(outer)*kUnroll*chains,checksum);
        }
    }
    check(cudaFree(values));check(cudaFree(records));
}
#endif
int main(int argc,char** argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--self-test") return self_test();
#ifndef MIDLAYER_NUMERIC_CPU_ONLY
        if(argc!=8||std::string(argv[1])!="--sample") throw std::runtime_error("expected --sample scope chains threads grid outer samples");
        int a[6];for(int i=0;i<6;++i) {std::size_t end;a[i]=std::stoi(argv[i+2],&end);if(end!=std::strlen(argv[i+2])) throw std::runtime_error("nonintegral argument");}
        if((a[0]!=0&&a[0]!=1)||(a[1]!=1&&a[1]!=4)||a[2]<32||a[2]>256||a[2]%32||a[3]<1||a[3]>432||a[4]<1||a[4]>2047||a[5]<1||a[5]>10)
            throw std::runtime_error("outside declared mechanism domain");
        run(a[0],a[1],a[2],a[3],a[4],a[5]);return 0;
#else
        throw std::runtime_error("CPU reference accepts --self-test only");
#endif
    } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
