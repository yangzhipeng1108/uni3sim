// Synthetic BF16 GEMM for CUDA 12.9 cuBLASLt non-duration program evidence.
// This uses the same row-major TNN operation contract as the H200 heuristic
// query. Model weights, serving requests and observed durations are absent.
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, \
          cudaGetErrorString(e)); std::exit(1); } } while (0)
#define LT_CHECK(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "%s:%d cuBLASLt status %d\n", __FILE__, __LINE__, \
          static_cast<int>(e)); std::exit(2); } } while (0)

constexpr int64_t M = 16384;
constexpr int64_t N = 151936;
constexpr int64_t K = 5120;
constexpr size_t WorkspaceBytes = 32ull << 20;

int main() {
  cudaDeviceProp device = {};
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
  if (device.major != 9 || device.minor != 0 ||
      device.multiProcessorCount != 132) {
    fprintf(stderr, "expected H200 with 132 SMs\n");
    return 3;
  }
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  LT_CHECK(cublasLtCreate(&handle));
  LT_CHECK(cublasLtMatmulDescCreate(
      &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_a = CUBLAS_OP_N;
  const cublasOperation_t trans_b = CUBLAS_OP_T;
  LT_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSA,
      &trans_a, sizeof(trans_a)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSB,
      &trans_b, sizeof(trans_b)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, M, K, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, N, K, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_16BF, M, N, N));
  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  for (cublasLtMatrixLayout_t layout : {a_layout, b_layout, c_layout}) {
    LT_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order, sizeof(row_order)));
  }
  LT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &WorkspaceBytes, sizeof(WorkspaceBytes)));
  cublasLtMatmulHeuristicResult_t results[32] = {};
  int count = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      handle, operation, a_layout, b_layout, c_layout, c_layout,
      preference, 32, results, &count));
  if (count < 1 || count == 32) {
    fprintf(stderr, "invalid heuristic count %d\n", count);
    return 4;
  }
  int algo_id = 0;
  size_t written = 0;
  LT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &results[0].algo, CUBLASLT_ALGO_CONFIG_ID,
      &algo_id, sizeof(algo_id), &written));
  printf("rank0_algo_id=%d,heuristic_count=%d,cublaslt_version=%zu,"
         "workspace_bytes=%zu,M=%lld,N=%lld,K=%lld\n", algo_id, count,
         cublasLtGetVersion(), WorkspaceBytes,
         static_cast<long long>(M), static_cast<long long>(N),
         static_cast<long long>(K));
  fflush(stdout);

  void* a = nullptr;
  void* b = nullptr;
  void* c = nullptr;
  void* workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&a, static_cast<size_t>(M * K * 2)));
  CUDA_CHECK(cudaMalloc(&b, static_cast<size_t>(N * K * 2)));
  CUDA_CHECK(cudaMalloc(&c, static_cast<size_t>(M * N * 2)));
  CUDA_CHECK(cudaMalloc(&workspace, WorkspaceBytes));
  CUDA_CHECK(cudaMemset(a, 1, static_cast<size_t>(M * K * 2)));
  CUDA_CHECK(cudaMemset(b, 1, static_cast<size_t>(N * K * 2)));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  LT_CHECK(cublasLtMatmul(
      handle, operation, &alpha, a, a_layout, b, b_layout, &beta,
      c, c_layout, c, c_layout, &results[0].algo,
      workspace, WorkspaceBytes, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("completed=1\n");

  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFree(c));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(a));
  LT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  LT_CHECK(cublasLtMatrixLayoutDestroy(c_layout));
  LT_CHECK(cublasLtMatrixLayoutDestroy(b_layout));
  LT_CHECK(cublasLtMatrixLayoutDestroy(a_layout));
  LT_CHECK(cublasLtMatmulDescDestroy(operation));
  LT_CHECK(cublasLtDestroy(handle));
  return 0;
}
