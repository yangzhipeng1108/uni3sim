// Independent cold byte-ring progress with inactive launch groups.
// No matrix, routing, activation, expert weights or operator output.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
constexpr int Threads=128, Panel=16384, APanel=4096, Slots=3, Shared=Slots*(Panel+APanel);
struct Stamp { unsigned long long start,end,cycles; unsigned sm,active; };
__host__ __device__ unsigned pattern(unsigned x) { x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; return x^(x>>16); }
__device__ unsigned long long globaltimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory"); return t; }
__device__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;":"=r"(s)); return s; }
__global__ void initialize(unsigned* x,size_t n) { for(size_t j=blockIdx.x*blockDim.x+threadIdx.x;j<n;j+=size_t(gridDim.x)*blockDim.x) x[j]=pattern(unsigned(j)); }
__global__ void evict(unsigned* x,size_t n) { for(size_t j=blockIdx.x*blockDim.x+threadIdx.x;j<n;j+=size_t(gridDim.x)*blockDim.x) x[j]=unsigned(j); }
template<int Stride> __global__ __launch_bounds__(Threads)
void cohort_copy(const unsigned char* input,unsigned* output,Stamp* stamps,int active_groups,int suffix) {
    constexpr int Rounds=Stride/256;
    int group=int(blockIdx.x)/(int(gridDim.x)/8);
    bool active=suffix?group>=8-active_groups:group<active_groups;
    if(!active) { if(threadIdx.x==0) { output[size_t(blockIdx.x)*Threads]=0; stamps[blockIdx.x]={0,0,0,smid(),0}; } return; }
    extern __shared__ __align__(16) unsigned char shared[];
    for(int j=threadIdx.x;j<Shared;j+=Threads) shared[j]=0;
    __syncthreads();
    const unsigned long long begin=globaltimer(), first_cycle=static_cast<unsigned long long>(clock64());
    auto issue=[&](int round,int slot) {
        if(round<Rounds) {
            #pragma unroll
            for(int v=threadIdx.x;v<APanel/16;v+=Threads) {
                unsigned address=unsigned(__cvta_generic_to_shared(shared+Slots*Panel+slot*APanel+v*16));
                const unsigned char* source=input+size_t(gridDim.x)*64*Stride+round*256+(v%16)*16;
                int valid=v<16?16:0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(source),"r"(valid):"memory");
            }
            #pragma unroll
            for(int v=threadIdx.x;v<Panel/16;v+=Threads) {
                unsigned address=unsigned(__cvta_generic_to_shared(shared+slot*Panel+v*16));
                const unsigned char* source=input+size_t(blockIdx.x)*64*Stride+(v/16)*Stride+round*256+(v%16)*16;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(source):"memory");
            }
        }
        // A and B belong to one committed group, including zero-fill A.
        asm volatile("cp.async.commit_group;":::"memory");
    };
    for(int slot=0;slot<Slots;++slot) issue(slot,slot);
    unsigned sum=0;
    for(int round=0;round<Rounds;++round) {
        asm volatile("cp.async.wait_group 2;":::"memory"); __syncthreads();
        volatile unsigned* b=reinterpret_cast<volatile unsigned*>(shared+(round%Slots)*Panel);
        volatile unsigned* a=reinterpret_cast<volatile unsigned*>(shared+Slots*Panel+(round%Slots)*APanel);
        #pragma unroll
        for(int v=threadIdx.x;v<Panel/16;v+=Threads) for(int word=0;word<4;++word) sum^=b[v*4+word];
        #pragma unroll
        for(int v=threadIdx.x;v<APanel/16;v+=Threads) for(int word=0;word<4;++word) sum^=a[v*4+word];
        __syncthreads(); issue(round+Slots,round%Slots);
    }
    asm volatile("cp.async.wait_group 0;":::"memory"); __syncthreads();
    const unsigned long long cycles=static_cast<unsigned long long>(clock64())-first_cycle, end=globaltimer();
    // Validate every thread; the timed region has no output stores.
    output[size_t(blockIdx.x)*Threads+threadIdx.x]=sum;
    if(threadIdx.x==0) stamps[blockIdx.x]={begin,end,cycles,smid(),1};
}
template<int Stride> void run(int grid,int active_groups,int suffix,int trials) {
    size_t input_bytes=size_t(grid)*64*Stride+Stride;
    unsigned char* input; unsigned *flush,*output; Stamp* stamps;
    CHECK(cudaMalloc(&input,input_bytes)); CHECK(cudaMalloc(&flush,128*1024*1024));
    CHECK(cudaMalloc(&output,size_t(grid)*Threads*sizeof(unsigned))); CHECK(cudaMalloc(&stamps,grid*sizeof(Stamp)));
    initialize<<<1024,256>>>(reinterpret_cast<unsigned*>(input),input_bytes/4); CHECK(cudaDeviceSynchronize());
    CHECK(cudaFuncSetAttribute(cohort_copy<Stride>,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
    int residents=0; CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residents,cohort_copy<Stride>,Threads,Shared));
    if(residents!=3) throw std::runtime_error("byte program must have three resident CTA capacity");
    std::vector<unsigned> got(size_t(grid)*Threads), truth(size_t(grid)*Threads); std::vector<Stamp> time(grid);
    // Construct the independent CPU reference once, outside every GPU sample.
    for(int block=0;block<grid;++block) {
        int group=block/(grid/8); bool active=suffix?group>=8-active_groups:group<active_groups;
        if(!active) continue;
        for(int thread=0;thread<Threads;++thread) {
            unsigned expected=0;
            for(int round=0;round<Stride/256;++round) {
                for(int v=thread;v<Panel/16;v+=Threads) {
                    size_t offset=size_t(block)*64*Stride+(v/16)*Stride+round*256+(v%16)*16;
                    for(int word=0;word<4;++word) expected^=pattern(unsigned(offset/4+word));
                }
                for(int v=thread;v<APanel/16;v+=Threads) if(v<16) {
                    size_t offset=size_t(grid)*64*Stride+round*256+(v%16)*16;
                    for(int word=0;word<4;++word) expected^=pattern(unsigned(offset/4+word));
                }
            }
            truth[size_t(block)*Threads+thread]=expected;
        }
    }
    if(std::all_of(truth.begin(),truth.end(),[](unsigned x){return x==0;})) throw std::runtime_error("nondiscriminating checksum fixture");
    std::vector<std::array<double,5>> samples;
    for(int trial=-3;trial<trials;++trial) {
        evict<<<4096,256>>>(flush,128*1024*1024/4); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(output,0,size_t(grid)*Threads*sizeof(unsigned)));
        cohort_copy<Stride><<<grid,Threads,Shared>>>(input,output,stamps,active_groups,suffix);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(got.data(),output,got.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(time.data(),stamps,time.size()*sizeof(Stamp),cudaMemcpyDeviceToHost));
        if(got!=truth) throw std::runtime_error("full-thread checksum mismatch");
        unsigned long long first=~0ULL,last=0,total_ns=0,total_cycles=0; int sm_counts[132]={},active_ctas=0;
        std::vector<std::pair<unsigned long long,int>> events[132];
        for(int block=0;block<grid;++block) {
            int group=block/(grid/8); bool active=suffix?group>=8-active_groups:group<active_groups;
            if(bool(time[block].active)!=active) throw std::runtime_error("active mask mismatch");
            if(!active) continue;
            const auto t=time[block]; if(t.sm>=132 || t.end<=t.start) throw std::runtime_error("invalid CTA clock/SM");
            first=std::min(first,t.start); last=std::max(last,t.end); total_ns+=t.end-t.start; total_cycles+=t.cycles;
            events[t.sm].push_back({t.start,1}); events[t.sm].push_back({t.end,-1});
            ++sm_counts[t.sm]; ++active_ctas;
        }
        if(active_ctas!=grid*active_groups/8) throw std::runtime_error("active CTA conservation failed");
        int distinct=0,max_count=0; for(int count:sm_counts) { distinct+=count>0; max_count=std::max(max_count,count); }
        int peak=0;
        for(auto& sm_events:events) { std::sort(sm_events.begin(),sm_events.end()); int overlap=0;
            for(const auto& item:sm_events) { overlap+=item.second; peak=std::max(peak,overlap); }
        }
        if(peak<1||peak>residents) throw std::runtime_error("invalid timed-body residency overlap");
        if(trial>=0) samples.push_back({double(last-first),double(total_cycles)/total_ns*1000, double(distinct),double(max_count),double(peak)});
    }
    std::printf("{\"stride\":%d,\"grid\":%d,\"active_groups\":%d,\"suffix\":%d,\"active_ctas\":%d,\"rounds\":%d,\"threads\":128,\"shared_bytes\":61440,\"resident_capacity\":3,\"panel_B_bytes\":16384,\"padded_A_bytes\":4096,\"valid_A_bytes\":256,\"all_thread_checksum_pass\":true,\"samples\":[",Stride,grid,active_groups,suffix,grid*active_groups/8,Stride/256);
    for(size_t j=0;j<samples.size();++j) std::printf("%s{\"body_ns\":%.12g,\"weighted_mhz\":%.12g,\"active_sms\":%.12g,\"CTA_records_per_SM_max\":%.12g,\"timed_body_overlap_peak\":%.12g}",j?",":"",samples[j][0],samples[j][1],samples[j][2],samples[j][3],samples[j][4]);
    std::puts("]}"); CHECK(cudaFree(stamps)); CHECK(cudaFree(output)); CHECK(cudaFree(flush)); CHECK(cudaFree(input));
}
int main(int argc,char** argv) { try {
    if(argc!=6) throw std::runtime_error("stride grid active_groups suffix trials required");
    int stride=std::atoi(argv[1]),grid=std::atoi(argv[2]),groups=std::atoi(argv[3]),suffix=std::atoi(argv[4]),trials=std::atoi(argv[5]);
    if((stride!=1536&&stride!=4096)||(grid!=192&&grid!=256)||(groups!=4&&groups!=5&&groups!=8)||(suffix!=0&&suffix!=1)||(trials!=1&&trials!=63)) throw std::runtime_error("unfrozen physical byte program");
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop,0));
    if(prop.major!=9||prop.multiProcessorCount!=132||!std::strstr(prop.name,"H200")) throw std::runtime_error("H200132SM required");
    if(stride==4096&&grid==192) run<4096>(grid,groups,suffix,trials);
    else if(stride==1536&&grid==256) run<1536>(grid,groups,suffix,trials);
    else throw std::runtime_error("unfrozen stride/grid pair");
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; } }
