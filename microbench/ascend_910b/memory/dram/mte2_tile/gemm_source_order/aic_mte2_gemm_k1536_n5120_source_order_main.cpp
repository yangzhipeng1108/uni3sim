// Keep host allocation geometry identical to the two specialized kernels.
#define GEMM_SOURCE_A_ROWS 128
#define GEMM_SOURCE_A_COLUMNS 256
#include "aic_mte2_gemm_source_order_main.cpp"
