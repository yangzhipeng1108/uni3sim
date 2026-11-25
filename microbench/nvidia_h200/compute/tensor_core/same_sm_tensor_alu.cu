// Two independently compiled CTA roles on each SM. RS queue and MAX/ADD
// dependency chains have separate completion records, without same-WG markers.
// No Attention, normalization, global operand stream, or production shapes.
#include "same_sm_wgmma_helper.cuh"
#include <algorithm>
#include <cmath>
#include <set>

constexpr int SmSlots = 1024;
// Leave room for static shared arrays and the driver's per-CTA reservation.
constexpr int PairSharedBytes = 104 * 1024;
constexpr int Threads = 128;
struct PairState { unsigned assigned, started, done; };
struct Reading {
    unsigned sm, role, active;
    unsigned long long c0,c1,n0,n1;
    float checksum;
    unsigned errors;
};
__device__ __forceinline__ unsigned sm_id() {
    unsigned result; asm volatile("mov.u32 %0,%%smid;":"=r"(result)); return result;
}
__device__ __forceinline__ unsigned long long nanos() {
    unsigned long long result;
    asm volatile("mov.u64 %0,%%globaltimer;":"=l"(result)::"memory"); return result;
}
template<int Depth> __device__ __forceinline__ void queue_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;"::"n"(Depth-1):"memory");
}
__host__ __device__ float bf16_value(unsigned bits) {
    unsigned word=bits<<16; float value; memcpy(&value,&word,4); return value;
}
__host__ __device__ float operand(int i,int tid) {
    return float(i+1+(tid&31))*(1.f/32768);
}
__device__ __forceinline__ void pair_begin(PairState* state) {
    if(threadIdx.x==0) {
        atomicAdd(&state->started,1);
        while(atomicAdd(&state->started,0)<2) __nanosleep(256);
    }
    __syncthreads();
}
__device__ __forceinline__ void pair_end(PairState* state) {
    __syncthreads();
    if(threadIdx.x==0) {
        __threadfence();atomicAdd(&state->done,1);
        while(atomicAdd(&state->done,0)<2) __nanosleep(1024);
    }
    __syncthreads();
}
template<int Kind,int Chains,int Ops> __host__ __device__ float scalar_reference(int tid,int repeats) {
    float total=0;
    for(int chain=0;chain<Chains;++chain) {
        float value=-float(1+chain)-float(tid)*(1.f/1024);
        if constexpr(Kind==0) {
            for(int i=chain;i<Ops;i+=Chains) value=fmaxf(value,operand(i,tid));
        } else {
            // All inputs are binary fractions; this sum stays exact in FP32.
            float increment=0;
            for(int i=chain;i<Ops;i+=Chains) increment+=operand(i,tid);
            value+=float(repeats)*increment;
        }
        total+=value;
    }
    return total;
}
template<int Depth> __device__ __forceinline__ Reading matrix(uint64_t b,int repeats,PairState* state) {
    float accum[2][64]; unsigned fragments[4][4]; uint64_t descriptors[4];
    #pragma unroll
    for(int bank=0;bank<2;++bank) {
        #pragma unroll
        for(int i=0;i<64;++i) {accum[bank][i]=0;asm volatile("":"+f"(accum[bank][i]));}
    }
    #pragma unroll
    for(int k=0;k<4;++k) {
        descriptors[k]=b+((k*4096)>>4);
        #pragma unroll
        for(int j=0;j<4;++j) {
            unsigned v=0x3f00u+(k*4+j)*16;
            fragments[k][j]=v|(v<<16); asm volatile("":"+r"(fragments[k][j]));
        }
    }
    Reading r{}; wg_fence(); pair_begin(state);r.n0=nanos(); r.c0=clock64();
    #pragma unroll 1
    for(int repeat=0;repeat<repeats;++repeat) {
        #pragma unroll
        for(int bank=0;bank<2;++bank) {
            #pragma unroll
            for(int k=0;k<4;++k)
                wgmma_rs_n128_mnmajor(accum[bank],fragments[k],descriptors[k],1);
            wg_commit();queue_wait<Depth>();
        }
    }
    wg_wait0();r.c1=clock64();r.n1=nanos();
    pair_end(state);
    #pragma unroll
    for(int bank=0;bank<2;++bank) {
        #pragma unroll
        for(int i=0;i<64;++i) {
            r.checksum+=accum[bank][i];
            r.errors+=!std::isfinite(accum[bank][i])||accum[bank][i]<=0;
        }
    }
    return r;
}
template<int Kind,int Chains,int Ops> __device__ __forceinline__ Reading scalar(int repeats,PairState* state) {
    float values[Chains];
    #pragma unroll
    for(int chain=0;chain<Chains;++chain) {
        values[chain]=-float(1+chain)-float(threadIdx.x)*(1.f/1024);
        asm volatile("":"+f"(values[chain]));
    }
    unsigned feedback=threadIdx.x;
    Reading r{};pair_begin(state);r.n0=nanos();r.c0=clock64();
    #pragma unroll 1
    for(int repeat=0;repeat<repeats;++repeat) {
        #pragma unroll
        for(int i=0;i<Ops;++i) {
            float input=operand(i,threadIdx.x);
            if constexpr(Kind==0)
                asm volatile("max.f32 %0,%0,%1;":"+f"(values[i%Chains]):"f"(input));
            else
                asm volatile("add.rn.f32 %0,%0,%1;":"+f"(values[i%Chains]):"f"(input));
        }
        asm volatile("add.u32 %0,%0,1;":"+r"(feedback));
    }
    r.c1=clock64();r.n1=nanos();
    pair_end(state);
    #pragma unroll
    for(int chain=0;chain<Chains;++chain) r.checksum+=values[chain];
    r.errors=(!std::isfinite(r.checksum)
        ||r.checksum!=scalar_reference<Kind,Chains,Ops>(threadIdx.x,repeats)
        ||feedback!=threadIdx.x+repeats);
    return r;
}
template<int Kind,int Chains,int Depth,int Ops>
__global__ void __launch_bounds__(Threads,2) same_sm(
        int repeats,int scalar_scale,int mode,int target_sm,
        PairState* states,unsigned* ready,Reading* output) {
    extern __shared__ __align__(1024) unsigned char shared[];
    __shared__ unsigned role;
    __shared__ float sums[Threads];
    __shared__ unsigned errors[Threads];
    __shared__ unsigned long long clocks[4][4];
    const unsigned sm=sm_id();
    if(sm>=SmSlots) asm("trap;");
    if(threadIdx.x==0) role=atomicAdd(&states[sm].assigned,1);
    __syncthreads();
    if(role>=2) asm("trap;");
    for(int i=threadIdx.x;i<PairSharedBytes/2;i+=Threads)
        reinterpret_cast<unsigned short*>(shared)[i]=0x3f00u;
    __syncthreads();
    // Occupancy is checked on the host before launch. A full-resident grid
    // fixes each pair before any idle CTA can leave and admit another CTA.
    if(threadIdx.x==0) {
        atomicAdd(ready,1);
        while(atomicAdd(ready,0)<gridDim.x) __nanosleep(256);
    }
    __syncthreads();
    const bool active=target_sm<0||sm==unsigned(target_sm);
    Reading reading{};
    if(active&&role==0&&mode!=1) {
        unsigned base=unsigned(__cvta_generic_to_shared(shared));
        reading=matrix<Depth>(make_desc(base+65536,16,1024),repeats,&states[sm]);
    }
    if(active&&role==1&&mode!=0)
        reading=scalar<Kind,Chains,Ops>(repeats*scalar_scale,&states[sm]);
    if(!active||(role==0&&mode==1)||(role==1&&mode==0)) {
        pair_begin(&states[sm]);pair_end(&states[sm]);
    }
    sums[threadIdx.x]=reading.checksum;errors[threadIdx.x]=reading.errors;
    if(threadIdx.x%32==0) {
        const int warp=threadIdx.x/32;
        clocks[warp][0]=reading.c0;clocks[warp][1]=reading.c1;
        clocks[warp][2]=reading.n0;clocks[warp][3]=reading.n1;
    }
    __syncthreads();
    if(threadIdx.x==0) {
        reading.checksum=0;reading.errors=0;
        for(int i=0;i<Threads;++i) {reading.checksum+=sums[i];reading.errors+=errors[i];}
        for(int warp=0;warp<4;++warp) {
            reading.c0=min(reading.c0,clocks[warp][0]);
            reading.c1=max(reading.c1,clocks[warp][1]);
            reading.n0=min(reading.n0,clocks[warp][2]);
            reading.n1=max(reading.n1,clocks[warp][3]);
        }
        reading.sm=sm;reading.role=role;reading.active=active;
        output[blockIdx.x]=reading;
    }
    __syncthreads();
}
template<int Kind,int Chains,int Depth,int Ops>
int run(int repeats,int scalar_scale,int mode,int target,int trial,
        PairState* states,unsigned* ready,Reading* output) {
    auto kernel=same_sm<Kind,Chains,Depth,Ops>;
    CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,PairSharedBytes));
    CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,100));
    int resident=0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,Threads,PairSharedBytes));
    if(resident!=2) {std::fprintf(stderr,"requires exactly2 resident CTAs,got%d\n",resident);std::exit(5);}
    CHECK(cudaMemset(states,0,SmSlots*sizeof(PairState)));
    CHECK(cudaMemset(ready,0,sizeof(unsigned)));
    void* arguments[]={&repeats,&scalar_scale,&mode,&target,&states,&ready,&output};
    CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),dim3(264),dim3(Threads),
        arguments,PairSharedBytes));
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    std::vector<Reading> host(264);
    CHECK(cudaMemcpy(host.data(),output,host.size()*sizeof(Reading),cudaMemcpyDeviceToHost));
    std::set<unsigned> smids;std::set<std::pair<unsigned,unsigned>> pairs;
    float operand_sum=0;
    for(int k=0;k<4;++k)for(int j=0;j<4;++j) operand_sum+=bf16_value(0x3f00u+(k*4+j)*16);
    for(int cta=0;cta<264;++cta) {
        const auto& r=host[cta];smids.insert(r.sm);pairs.insert({r.sm,r.role});
        if(r.role>1||r.errors) {std::fprintf(stderr,"role/numeric gate\n");std::exit(6);}
        if(!r.active||(r.role==0&&mode==1)||(r.role==1&&mode==0)) continue;
        if(!(r.c0<r.c1&&r.n0<r.n1)) {std::fprintf(stderr,"window gate\n");std::exit(7);}
        float expected=0;
        if(r.role==0) expected=float(2*128*128*repeats)*operand_sum;
        else for(int tid=0;tid<Threads;++tid) expected+=scalar_reference<Kind,Chains,Ops>(tid,repeats*scalar_scale);
        if(r.checksum!=expected) {std::fprintf(stderr,"checksum gate role%u got%.9g expected%.9g\n",r.role,r.checksum,expected);std::exit(8);}
        std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%u,%u,%llu,%llu,%llu,%llu,%.9g\n",
            Kind,Chains,Depth,Ops,repeats,scalar_scale,mode,target,trial,cta,r.sm,r.role,r.c0,r.c1,r.n0,r.n1,r.checksum);
    }
    if(smids.size()!=132||pairs.size()!=264) {std::fprintf(stderr,"pair coverage gate\n");std::exit(9);}
    return int(*smids.begin());
}
template<int Chains,int Ops> void depths(int n,int scale,int mode,int target,int trial,PairState*s,unsigned*r,Reading*o) {
    run<0,Chains,1,Ops>(n,scale,mode,target,trial,s,r,o);
    run<0,Chains,4,Ops>(n,scale,mode,target,trial,s,r,o);
}
template<int Chains> void sizes(int n,int scale,int mode,int target,int trial,PairState*s,unsigned*r,Reading*o) {
    depths<Chains,0>(n,scale,mode,target,trial,s,r,o);
    depths<Chains,32>(n,scale,mode,target,trial,s,r,o);
    depths<Chains,64>(n,scale,mode,target,trial,s,r,o);
    depths<Chains,128>(n,scale,mode,target,trial,s,r,o);
}
int main() {
    int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 10;
    PairState*states;unsigned*ready;Reading*output;
    CHECK(cudaMalloc(&states,SmSlots*sizeof(PairState)));
    CHECK(cudaMalloc(&ready,sizeof(unsigned)));CHECK(cudaMalloc(&output,264*sizeof(Reading)));
    std::puts("kind,chains,depth,ops,repeats,scalar_scale,mode,target_sm,trial,cta,sm,role,c0,c1,n0,n1,checksum");
    int target=run<0,2,4,64>(8,1,0,-1,-1,states,ready,output);
    for(int active:{1,132})for(int n:{64,128,256})for(int scale:{1,8})
        for(int trial=0;trial<3;++trial)for(int index=0;index<3;++index) {
            int mode=(index+trial)%3,t=active==1?target:-1;
            sizes<2>(n,scale,mode,t,trial,states,ready,output);
            sizes<8>(n,scale,mode,t,trial,states,ready,output);
        }
    CHECK(cudaFree(states));CHECK(cudaFree(ready));CHECK(cudaFree(output));
}
