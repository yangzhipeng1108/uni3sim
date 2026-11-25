// Launch metadata only. No production duration or fitted performance price.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cupti.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#define CUDA(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)
#define BLAS(call) do{auto e=(call);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS status%d\n",int(e));exit(3);}}while(0)
#define CUPTI(call) do{auto e=(call);if(e!=CUPTI_SUCCESS){fprintf(stderr,"CUPTI status%d\n",int(e));exit(4);}}while(0)
using ParamInfo=CUresult(*)(CUfunction,size_t,size_t*,size_t*);
using FuncName=CUresult(*)(const char**,CUfunction);
using FuncAttribute=CUresult(*)(int*,CUfunction_attribute,CUfunction);
using KernelFunction=CUresult(*)(CUfunction*,CUkernel);
static ParamInfo param_info=nullptr;
static FuncName func_name=nullptr;
static FuncAttribute func_attribute=nullptr;
static KernelFunction kernel_function=nullptr;
static const char* caller="parameter_marker";
static const char* case_id="parameter_marker";
static std::mutex output_mutex;
static thread_local bool in_callback=false;
static int parameter_records=0;
static void fail(const char* message){fprintf(stderr,"%s\n",message);exit(5);}
static void print_string(const char* s) {
  putchar('"');for(;*s;++s){if(*s=='"'||*s=='\\')putchar('\\');if(unsigned(*s)<32)fail("control character in metadata");putchar(*s);}putchar('"');
}
static void CUPTIAPI callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void* raw) {
  if(domain!=CUPTI_CB_DOMAIN_DRIVER_API || in_callback)return;
  const auto* info=static_cast<const CUpti_CallbackData*>(raw);
  if(info->callbackSite!=CUPTI_API_ENTER)return;
  CUfunction function;void **params,**extra;unsigned grid[3],block[3],shared;
  switch(id) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz: {
      auto* p=static_cast<const cuLaunchKernel_params*>(info->functionParams);
      function=p->f;params=p->kernelParams;extra=p->extra;
      grid[0]=p->gridDimX;grid[1]=p->gridDimY;grid[2]=p->gridDimZ;
      block[0]=p->blockDimX;block[1]=p->blockDimY;block[2]=p->blockDimZ;shared=p->sharedMemBytes;break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
      auto* p=static_cast<const cuLaunchKernelEx_params*>(info->functionParams);
      function=p->f;params=p->kernelParams;extra=p->extra;
      grid[0]=p->config->gridDimX;grid[1]=p->config->gridDimY;grid[2]=p->config->gridDimZ;
      block[0]=p->config->blockDimX;block[1]=p->config->blockDimY;block[2]=p->config->blockDimZ;shared=p->config->sharedMemBytes;break;
    }
    default:return;
  }
  in_callback=true;
  const char* name=nullptr;
  auto original_name_status=func_name(&name,function);
  const char* handle_kind="CUfunction";
  if(original_name_status!=CUDA_SUCCESS) {
    // cuLaunchKernel accepts both CUfunction and context-independent CUkernel
    // handles (cuda.h API contract). Normalize the latter explicitly before
    // querying context-local function metadata; preserve the observed kind.
    CUfunction context_function=nullptr;
    auto conversion=kernel_function(&context_function,reinterpret_cast<CUkernel>(function));
    if(conversion!=CUDA_SUCCESS) {
      fprintf(stderr,"name query status%d; CUkernel normalization status%d; API%s\n",int(original_name_status),int(conversion),info->functionName);
      fail("unrecognized launch handle");
    }
    function=context_function;handle_kind="CUkernel";
    if(func_name(&name,function)!=CUDA_SUCCESS)fail("normalized function name unavailable");
  }
  if(!name)fail("null function name");
  std::lock_guard<std::mutex> lock(output_mutex);
  printf("{\"caller\":");print_string(caller);printf(",\"case_id\":");print_string(case_id);printf(",\"kernel\":");print_string(name);
  printf(",\"launch_handle_kind\":");print_string(handle_kind);
  printf(",\"original_function_name_query_status\":%d",int(original_name_status));
  printf(",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"dynamic_shared_bytes\":%u",grid[0],grid[1],grid[2],block[0],block[1],block[2],shared);
  bool selected=strstr(name,"nvjet_")!=nullptr || strstr(name,"parameter_marker")!=nullptr;
  if(selected) {
    int registers=0,static_shared=0,binary_version=0;
    if(func_attribute(&registers,CU_FUNC_ATTRIBUTE_NUM_REGS,function)!=CUDA_SUCCESS
        || func_attribute(&static_shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,function)!=CUDA_SUCCESS
        || func_attribute(&binary_version,CU_FUNC_ATTRIBUTE_BINARY_VERSION,function)!=CUDA_SUCCESS)fail("function resources unavailable");
    printf(",\"registers_per_thread\":%d,\"static_shared_bytes\":%d,\"binary_version\":%d",registers,static_shared,binary_version);
    if(bool(params)==bool(extra))fail("require exactly one launch parameter representation");
    const unsigned char* packed=nullptr;size_t packed_size=0;
    if(extra) {
      bool ended=false;
      for(int index=0;index<16;++index) {
        auto key=extra[index];if(key==CU_LAUNCH_PARAM_END){ended=true;break;}
        if(key==CU_LAUNCH_PARAM_BUFFER_POINTER)packed=static_cast<const unsigned char*>(extra[++index]);
        else if(key==CU_LAUNCH_PARAM_BUFFER_SIZE)packed_size=*static_cast<const size_t*>(extra[++index]);
        else fail("unknown launch parameter token");
      }
      if(!ended || !packed || !packed_size)fail("incomplete packed launch parameters");
    }
    std::vector<unsigned char> bytes;
    printf(",\"parameter_fields\":[");size_t count=0;
    for(size_t index=0;index<1024;++index) {
      size_t offset=0,size=0;auto status=param_info(function,index,&offset,&size);
      if(status==CUDA_ERROR_INVALID_VALUE)break; // documented end of parameter indices
      if(status!=CUDA_SUCCESS || !size || offset+size>16384)fail("parameter metadata unsupported");
      if(extra && offset+size>packed_size)fail("packed launch parameter bounds");
      bytes.resize(offset+size,0);
      const void* source=extra?static_cast<const void*>(packed+offset):params[index];
      if(!source)fail("null parameter value");memcpy(bytes.data()+offset,source,size);
      printf("%s{\"offset\":%zu,\"size\":%zu}",index?",":"",offset,size);++count;
    }
    if(!count || count==1024)fail("parameter enumeration incomplete");
    printf("],\"parameter_bytes_hex\":\"");for(unsigned char byte:bytes)printf("%02x",byte);
    printf("\",\"parameter_bytes\":%zu,\"padding_between_parameter_fields\":\"unobserved_zero_markers\"",bytes.size());++parameter_records;
  }
  printf("}\n");fflush(stdout);in_callback=false;
}
__global__ void parameter_marker(int* output,unsigned value){if(threadIdx.x==0)*output=int(value);}
struct Shape {std::string id;int m,n,k;};
static std::vector<Shape> shapes(int argc,char** argv) {
  if(argc==1)return {{"GEMM_B8_M4096_K8192_N28672",32768,28672,8192}};
  if(argc!=2)fail("expected optional shape-only CSV");
  std::ifstream source(argv[1]);std::string line;
  if(!std::getline(source,line) || line!="case_id,m,n,k")fail("invalid shape header");
  std::vector<Shape> result;std::set<std::string> ids;
  while(std::getline(source,line)) {
    Shape s;char first=0,second=0;std::istringstream row(line);
    if(!std::getline(row,s.id,',') || !(row>>s.m>>first>>s.n>>second>>s.k)
        || first!=',' || second!=',' || s.m<=0 || s.n<=0 || s.k<=0
        || s.id.empty() || !ids.insert(s.id).second)fail("invalid shape row");
    row>>std::ws;if(!row.eof())fail("unexpected shape suffix");result.push_back(s);
  }
  if(result.size()!=221)fail("fixed workload requires221 shape rows");return result;
}
int main(int argc,char** argv) {
  auto workload=shapes(argc,argv);bool batch=argc==2;
  CUDA(cudaFree(nullptr));
  void* driver=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);if(!driver)fail("CUDA driver library unavailable");
  param_info=reinterpret_cast<ParamInfo>(dlsym(driver,"cuFuncGetParamInfo"));
  func_name=reinterpret_cast<FuncName>(dlsym(driver,"cuFuncGetName"));
  func_attribute=reinterpret_cast<FuncAttribute>(dlsym(driver,"cuFuncGetAttribute"));
  kernel_function=reinterpret_cast<KernelFunction>(dlsym(driver,"cuKernelGetFunction"));
  if(!param_info || !func_name || !func_attribute || !kernel_function)fail("driver parameter/name/resource query unavailable");
  CUpti_SubscriberHandle subscriber;CUPTI(cuptiSubscribe(&subscriber,callback,nullptr));
  CUPTI(cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API));
  int* marker=nullptr;CUDA(cudaMalloc(&marker,sizeof(int)));
  parameter_marker<<<1,1>>>(marker,0x51515151);CUDA(cudaDeviceSynchronize());
  int marker_value=0;CUDA(cudaMemcpy(&marker_value,marker,sizeof(int),cudaMemcpyDeviceToHost));if(marker_value!=0x51515151)fail("parameter marker failed");CUDA(cudaFree(marker));
  constexpr size_t workspace_bytes=32*1024*1024;void* workspace=nullptr;
  CUDA(cudaMalloc(&workspace,workspace_bytes));
  float alpha=1,beta=0;cublasHandle_t blas;BLAS(cublasCreate(&blas));
  int version=0;BLAS(cublasGetVersion(blas,&version));printf("{\"cublas_version\":%d,\"case_count\":%zu,\"workspace_bytes\":%zu}\n",version,workload.size(),workspace_bytes);
  BLAS(cublasSetMathMode(blas,CUBLAS_TENSOR_OP_MATH));BLAS(cublasSetWorkspace(blas,workspace,workspace_bytes));
  for(const auto& shape:workload) {
  const int m=shape.m,n=shape.n,k=shape.k;case_id=shape.id.c_str();
  printf("{\"case_id\":");print_string(case_id);printf(",\"m\":%d,\"n\":%d,\"k\":%d}\n",m,n,k);fflush(stdout);
  __half *weights=nullptr,*input=nullptr,*output=nullptr;
  CUDA(cudaMalloc(&weights,size_t(n)*k*2));CUDA(cudaMalloc(&input,size_t(m)*k*2));CUDA(cudaMalloc(&output,size_t(m)*n*2));
  CUDA(cudaMemset(weights,0,size_t(n)*k*2));CUDA(cudaMemset(input,0,size_t(m)*k*2));CUDA(cudaDeviceSynchronize());
  caller="cublasGemmEx";
  BLAS(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,weights,CUDA_R_16F,k,input,CUDA_R_16F,k,&beta,output,CUDA_R_16F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));CUDA(cudaDeviceSynchronize());
  if(!batch) {
  cublasLtHandle_t lt;BLAS(cublasLtCreate(&lt));cublasLtMatmulDesc_t operation;BLAS(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t transpose=CUBLAS_OP_T,normal=CUBLAS_OP_N;
  BLAS(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
  BLAS(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)));
  cublasLtMatrixLayout_t a,b,c;BLAS(cublasLtMatrixLayoutCreate(&a,CUDA_R_16F,k,n,k));BLAS(cublasLtMatrixLayoutCreate(&b,CUDA_R_16F,k,m,k));BLAS(cublasLtMatrixLayoutCreate(&c,CUDA_R_16F,n,m,n));
  cublasLtMatmulPreference_t preference;BLAS(cublasLtMatmulPreferenceCreate(&preference));BLAS(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
  cublasLtMatmulHeuristicResult_t heuristic;int returned=0;BLAS(cublasLtMatmulAlgoGetHeuristic(lt,operation,a,b,c,c,preference,1,&heuristic,&returned));if(returned!=1)fail("rank0 heuristic unavailable");
  caller="cublasLt_rank0";
  BLAS(cublasLtMatmul(lt,operation,&alpha,weights,a,input,b,&beta,output,c,output,c,&heuristic.algo,workspace,workspace_bytes,nullptr));CUDA(cudaDeviceSynchronize());
  BLAS(cublasLtMatmulPreferenceDestroy(preference));BLAS(cublasLtMatrixLayoutDestroy(a));BLAS(cublasLtMatrixLayoutDestroy(b));BLAS(cublasLtMatrixLayoutDestroy(c));BLAS(cublasLtMatmulDescDestroy(operation));BLAS(cublasLtDestroy(lt));
  }
  __half result;CUDA(cudaMemcpy(&result,output,sizeof(result),cudaMemcpyDeviceToHost));if(__half2float(result)!=0)fail("zero-input output check failed");
  CUDA(cudaFree(weights));CUDA(cudaFree(input));CUDA(cudaFree(output));
  }
  printf("{\"parameter_records\":%d,\"cases_checked\":%zu,\"marker_and_zero_output_check\":true,\"zero_output_check_scope\":\"first FP16 element\",\"production_duration_collected\":false}\n",parameter_records,workload.size());
  CUPTI(cuptiUnsubscribe(subscriber));BLAS(cublasDestroy(blas));CUDA(cudaFree(workspace));dlclose(driver);
}
