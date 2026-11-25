#define GEMM_SOURCE_A_ROWS 16
#define GEMM_SOURCE_A_COLUMNS 1536
#define GEMM_SOURCE_B_COLUMNS 256
#define GEMM_SOURCE_MAXIMUM_B_ROWS 224
#include "aic_mte2_gemm_source_order_kernel.cpp"
