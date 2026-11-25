// Production-shape work/stall observation only; never a Forward time price.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)
#define BLAS(x) do{auto e=(x);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CUBLAS %d\n",int(e));exit(3);}}while(0)
__global__ void ones(__half* p,size_t count) {
 for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x)p[i]=__float2half(1.0f);
}
int main(int argc,char** argv) {
 if(argc!=4)return 4;
 int m=std::atoi(argv[1]),n=std::atoi(argv[2]),k=std::atoi(argv[3]);
 if(m<1 || n<1 || k<1 || k>32768 || m>1024 || n>32768)return 4;
 __half *weights=nullptr,*input=nullptr,*output=nullptr;void* workspace=nullptr;
 constexpr size_t ws=32*1024*1024;
 CUDA(cudaMalloc(&weights,size_t(n)*k*2));CUDA(cudaMalloc(&input,size_t(m)*k*2));CUDA(cudaMalloc(&output,size_t(m)*n*2));CUDA(cudaMalloc(&workspace,ws));
 ones<<<4096,256>>>(weights,size_t(n)*k);ones<<<4096,256>>>(input,size_t(m)*k);CUDA(cudaDeviceSynchronize());
 cublasHandle_t blas;BLAS(cublasCreate(&blas));BLAS(cublasSetMathMode(blas,CUBLAS_TENSOR_OP_MATH));BLAS(cublasSetWorkspace(blas,workspace,ws));
 int version=0;BLAS(cublasGetVersion(blas,&version));printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"cublas_version\":%d,\"workspace_bytes\":%zu}\n",m,n,k,version,ws);fflush(stdout);
 float alpha=1,beta=0;
 auto run=[&](){BLAS(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,weights,CUDA_R_16F,k,input,CUDA_R_16F,k,&beta,output,CUDA_R_16F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));CUDA(cudaDeviceSynchronize());};
 for(int warm=0;warm<5;++warm)run();
 CUDA(cudaProfilerStart());run();CUDA(cudaProfilerStop());
 std::vector<__half> host(size_t(m)*n);CUDA(cudaMemcpy(host.data(),output,host.size()*2,cudaMemcpyDeviceToHost));
 for(auto value:host)if(__half2float(value)!=float(k)){fprintf(stderr,"output check failed\n");return 5;}
 printf("{\"all_output_elements_checked\":%zu,\"check_passed\":true,\"production_duration_collected_by_program\":false}\n",host.size());
 BLAS(cublasDestroy(blas));CUDA(cudaFree(workspace));CUDA(cudaFree(output));CUDA(cudaFree(input));CUDA(cudaFree(weights));return 0;
}
