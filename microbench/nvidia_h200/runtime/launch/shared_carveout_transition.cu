// Independent launch/cache-configuration transition. Same tiny dependency
// kernel repeats alone or alternates with a larger shared-memory allocation.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do {auto e=(x); if(e!=cudaSuccess){ \
 fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int Nodes=64, Threads=384;
template<bool Large>
__global__ void cache_transition(const unsigned* input,unsigned* output) {
  extern __shared__ unsigned large[];
  __shared__ unsigned small[32];
  unsigned* shared=Large?large:small;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned value=input[i];
  if(threadIdx.x<32) shared[threadIdx.x]=value;
  __syncthreads();
  output[i]=value+shared[threadIdx.x%32];
}
void graph(int grid,int bytes,int pattern,unsigned* input,unsigned* output) {
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
  cudaGraph_t graph;cudaGraphExec_t exec;
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  int count=0;
  for(int n=0;n<Nodes;++n) {
    if(pattern!=1) {cache_transition<false><<<grid,Threads,0,stream>>>(input,output);++count;}
    if(pattern!=0) {cache_transition<true><<<grid,Threads,bytes,stream>>>(input,output);++count;}
  }
  CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));
  cudaEvent_t start,stop;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));
  for(int trial=-2;trial<5;++trial) {
    CHECK(cudaEventRecord(start,stream));CHECK(cudaGraphLaunch(exec,stream));
    CHECK(cudaEventRecord(stop,stream));CHECK(cudaEventSynchronize(stop));
    float ms;CHECK(cudaEventElapsedTime(&ms,start,stop));
    printf("%d,%d,%d,%d,%d,%.6f\n",grid,bytes,pattern,trial,count,ms*1e6);
  }
  CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));
  CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
}
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  char* end;long bytes=strtol(argv[1],&end,10);
  if(*end || bytes<128*1024 || bytes>224*1024) return 2;
  unsigned *input,*output;CHECK(cudaMalloc(&input,132*Threads*sizeof(unsigned)));
  CHECK(cudaMalloc(&output,132*Threads*sizeof(unsigned)));CHECK(cudaMemset(input,1,132*Threads*sizeof(unsigned)));
  CHECK(cudaFuncSetAttribute(cache_transition<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  printf("grid,shared_bytes,pattern,trial,nodes,event_ns\n");
  for(int grid:{1,132}) for(int p=0;p<3;++p) graph(grid,bytes,p,input,output);
  CHECK(cudaFree(input));CHECK(cudaFree(output));
}
