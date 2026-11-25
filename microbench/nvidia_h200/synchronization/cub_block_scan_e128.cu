// Independent E128 CUB BlockScan component; no alignment/routing/operator output.
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
constexpr int Threads=1024,Experts=128,Repeats=63;
__global__ void scan(const int* input,int* output,unsigned long long* cycles){
 using BlockScan=cub::BlockScan<int32_t,Threads>;
 __shared__ typename BlockScan::TempStorage temp;
 unsigned long long start=clock64();
 int value=threadIdx.x<Experts?input[threadIdx.x]:0;
 for(int repeat=0;repeat<Repeats;++repeat){
  int result;
  BlockScan(temp).ExclusiveSum(value,result);
  if(threadIdx.x==Experts-1&&repeat==Repeats-1)output[0]=result+input[Experts-1];
  __syncthreads();
 }
 if(threadIdx.x==0)cycles[0]=clock64()-start;
}
int main(){try{
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(prop.major!=9||prop.multiProcessorCount!=132)throw std::runtime_error("H200 geometry required");
 int *input,*output;unsigned long long* cycles;
 CHECK(cudaMallocManaged(&input,Experts*sizeof(int)));CHECK(cudaMallocManaged(&output,sizeof(int)));
 CHECK(cudaMallocManaged(&cycles,sizeof(unsigned long long)));
 for(int i=0;i<Experts;++i)input[i]=(i%7)+1;
 scan<<<1,Threads>>>(input,output,cycles);CHECK(cudaDeviceSynchronize());CHECK(cudaGetLastError());
 int expected=0;for(int i=0;i<Experts;++i)expected+=(i%7)+1;
 if(output[0]!=expected)throw std::runtime_error("BlockScan CPU reference mismatch");
 if(cycles[0]==0)throw std::runtime_error("invalid BlockScan cycles");
 std::printf("{\"experts\":128,\"threads\":1024,\"repeats\":63,\"cycles\":%llu,\"cpu_reference_pass\":true}\n",cycles[0]);
 CHECK(cudaFree(cycles));CHECK(cudaFree(output));CHECK(cudaFree(input));
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
 return 0;
}
