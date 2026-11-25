// Independent warpgroup register-quota handoff; no math/copy in timed windows.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}while(0)
constexpr int MaxRepeats=256;
struct Event{unsigned long long f0,f1,r0,r1,n0,n1;};
__device__ __forceinline__ unsigned long long cycles(){unsigned long long v;asm volatile("mov.u64 %0,%%clock64;":"=l"(v)::"memory");return v;}
__device__ __forceinline__ unsigned long long nanos(){unsigned long long v;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(v)::"memory");return v;}
template<int Registers>__device__ __forceinline__ void increase(){asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"::"n"(Registers):"memory");}
template<int Registers>__device__ __forceinline__ void decrease(){asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"::"n"(Registers):"memory");}
template<int Groups,int Producer,int Consumer>
__global__ void __launch_bounds__(Groups*128,1) allocation(int mode,int repeats,Event* events,unsigned* witness){
 constexpr int Base=Groups==2?256:168;
 extern __shared__ unsigned char unused[];
 float values[120];
 #pragma unroll
 for(int i=0;i<120;++i)values[i]=float((threadIdx.x+i)%31);
 asm volatile("":"+f"(values[0]),"+f"(values[1]),"+f"(values[2]),"+f"(values[3]),"+f"(values[4]),"+f"(values[5]),"+f"(values[6]),"+f"(values[7]),"+f"(values[8]),"+f"(values[9]),"+f"(values[10]),"+f"(values[11]),"+f"(values[12]),"+f"(values[13]),"+f"(values[14]),"+f"(values[15]),"+f"(values[16]),"+f"(values[17]),"+f"(values[18]),"+f"(values[19]),"+f"(values[20]),"+f"(values[21]),"+f"(values[22]),"+f"(values[23]),"+f"(values[24]),"+f"(values[25]),"+f"(values[26]),"+f"(values[27]),"+f"(values[28]),"+f"(values[29]),"+f"(values[30]),"+f"(values[31]),"+f"(values[32]),"+f"(values[33]),"+f"(values[34]),"+f"(values[35]),"+f"(values[36]),"+f"(values[37]),"+f"(values[38]),"+f"(values[39]),"+f"(values[40]),"+f"(values[41]),"+f"(values[42]),"+f"(values[43]),"+f"(values[44]),"+f"(values[45]),"+f"(values[46]),"+f"(values[47]),"+f"(values[48]),"+f"(values[49]),"+f"(values[50]),"+f"(values[51]),"+f"(values[52]),"+f"(values[53]),"+f"(values[54]),"+f"(values[55]),"+f"(values[56]),"+f"(values[57]),"+f"(values[58]),"+f"(values[59]),"+f"(values[60]),"+f"(values[61]),"+f"(values[62]),"+f"(values[63]),"+f"(values[64]),"+f"(values[65]),"+f"(values[66]),"+f"(values[67]),"+f"(values[68]),"+f"(values[69]),"+f"(values[70]),"+f"(values[71]),"+f"(values[72]),"+f"(values[73]),"+f"(values[74]),"+f"(values[75]),"+f"(values[76]),"+f"(values[77]),"+f"(values[78]),"+f"(values[79]),"+f"(values[80]),"+f"(values[81]),"+f"(values[82]),"+f"(values[83]),"+f"(values[84]),"+f"(values[85]),"+f"(values[86]),"+f"(values[87]),"+f"(values[88]),"+f"(values[89]),"+f"(values[90]),"+f"(values[91]),"+f"(values[92]),"+f"(values[93]),"+f"(values[94]),"+f"(values[95]),"+f"(values[96]),"+f"(values[97]),"+f"(values[98]),"+f"(values[99]),"+f"(values[100]),"+f"(values[101]),"+f"(values[102]),"+f"(values[103]),"+f"(values[104]),"+f"(values[105]),"+f"(values[106]),"+f"(values[107]),"+f"(values[108]),"+f"(values[109]),"+f"(values[110]),"+f"(values[111]),"+f"(values[112]),"+f"(values[113]),"+f"(values[114]),"+f"(values[115]),"+f"(values[116]),"+f"(values[117]),"+f"(values[118]),"+f"(values[119]));
 float sum=0;
 #pragma unroll
 for(int i=0;i<120;++i)sum+=values[i];
 unsigned expected=0;for(int i=0;i<120;++i)expected+=(threadIdx.x+i)%31;
 if(sum!=float(expected))asm volatile("trap;");
 witness[blockIdx.x*Groups*128+threadIdx.x]=unsigned(sum);
 increase<Base>();
 int wg=threadIdx.x/128;
 #pragma unroll 1
 for(int rep=0;rep<repeats;++rep){
  __syncthreads();Event e{};e.n0=nanos();e.f0=cycles();
  if(mode){if(wg==0)decrease<Producer>();else increase<Consumer>();}
  e.f1=cycles();e.n1=nanos();
  __syncthreads();e.r0=cycles();
  if(mode){if(wg==0)increase<Base>();else decrease<Base>();}
  e.r1=cycles();
  __syncthreads();
  if(threadIdx.x%128==0)events[(blockIdx.x*MaxRepeats+rep)*Groups+wg]=e;
 }
}
template<int Groups,int P,int C>void run(int grid,int repeats,int mode,int trial,Event* output,unsigned* witness){
 auto kernel=allocation<Groups,P,C>;CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,196608));
 kernel<<<grid,Groups*128,196608>>>(mode,repeats,output,witness);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 std::vector<Event> events(grid*MaxRepeats*Groups);CHECK(cudaMemcpy(events.data(),output,events.size()*sizeof(Event),cudaMemcpyDeviceToHost));
 std::vector<unsigned> words(grid*Groups*128);CHECK(cudaMemcpy(words.data(),witness,words.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
 for(int cta=0;cta<grid;++cta)for(int tid=0;tid<Groups*128;++tid){unsigned expected=0;for(int i=0;i<120;++i)expected+=(tid+i)%31;if(words[cta*Groups*128+tid]!=expected)std::exit(3);}
 for(int cta=0;cta<grid;++cta){
  unsigned long long forward=0,reverse=0,ns=0,first=0,last=0,producer=0,consumers=0;
  for(int rep=0;rep<repeats;++rep){
   auto initial=events[(cta*MaxRepeats+rep)*Groups];
   auto f0=initial.f0,f1=initial.f1,r0=initial.r0,r1=initial.r1,n0=initial.n0,n1=initial.n1;
   unsigned long long consumer=0;
   for(int wg=0;wg<Groups;++wg){auto e=events[(cta*MaxRepeats+rep)*Groups+wg];if(!(e.f0<e.f1&&e.r0<e.r1&&e.n0<=e.n1)){std::fprintf(stderr,"timer ordering: %llu %llu %llu %llu %llu %llu\n",e.f0,e.f1,e.r0,e.r1,e.n0,e.n1);std::exit(4);}
    f0=std::min(f0,e.f0);f1=std::max(f1,e.f1);r0=std::min(r0,e.r0);r1=std::max(r1,e.r1);n0=std::min(n0,e.n0);n1=std::max(n1,e.n1);
    if(wg)consumer=std::max(consumer,e.f1-e.f0);else producer+=e.f1-e.f0;
   }
   consumers+=consumer;forward+=f1-f0;reverse+=r1-r0;ns+=n1-n0;
   if(rep==0)first=f1-f0;if(rep==repeats-1)last=f1-f0;
  }
  std::printf("%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",Groups,P,C,grid,repeats,mode,trial,cta,forward,reverse,ns,first,last,producer,consumers);
 }
}
int main(){int sms;CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));if(sms!=132)return 5;Event* output;unsigned*witness;CHECK(cudaMalloc(&output,132*MaxRepeats*3*sizeof(Event)));CHECK(cudaMalloc(&witness,132*384*sizeof(unsigned)));
 std::puts("groups,producer,consumer,grid,repeats,mode,trial,cta,forward_cycles,reverse_cycles,forward_ns,first_cycles,last_cycles,producer_cycles,consumer_cycles");
 for(int grid:{1,132})for(int repeats:{64,128,256})for(int mode:{0,1})for(int trial=0;trial<3;++trial){run<2,56,256>(grid,repeats,mode,trial,output,witness);run<3,40,232>(grid,repeats,mode,trial,output,witness);}
 CHECK(cudaFree(output));CHECK(cudaFree(witness));}
