// Compile the shared ingress-completion probe for the same normalized
// 128x112x128 MatMulV2 program as the MTE2-only companion.
#define GEMM_SOURCE_A_ROWS 128
#define GEMM_SOURCE_A_COLUMNS 256
#define GEMM_SOURCE_B_COLUMNS 256
#include "aic_mte2_gemm_ingress_completion_kernel.cpp"
